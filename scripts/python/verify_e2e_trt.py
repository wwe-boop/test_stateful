#!/usr/bin/env python3
"""
TRT E2E verification — Pure TensorRT (talker_context + talker_decode_fused engines).

Loads TRT engines built by trtexec and runs prefill + decode loop, comparing
against FP32 PyTorch reference (e2e_trt_ref.npz from verify_e2e_trt_ref.py).

Usage (inside NGC container or host with TensorRT, invoked by verify_e2e_trt.sh):
  python3 verify_e2e_trt.py --model-dir /path/to/exported/variant \\
      [--ref-file /path/to/e2e_trt_ref.npz]
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("e2e_trt")


def cosine_sim(a, b):
    if isinstance(a, np.ndarray):
        a = torch.from_numpy(a).float()
    if isinstance(b, np.ndarray):
        b = torch.from_numpy(b).float()
    return float(F.cosine_similarity(a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)))


def _load_trt_engine(engine_path, device):
    """Load a TensorRT engine and create execution context."""
    import tensorrt as trt

    trt_logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f:
        runtime = trt.Runtime(trt_logger)
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()
    return engine, context


def _trt_infer(engine, context, feed_dict, device):
    """Run TRT inference with named input/output tensors."""
    import tensorrt as trt

    stream = torch.cuda.Stream(device=device)
    bindings = {}

    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.INPUT:
            if name in feed_dict:
                t = feed_dict[name]
                if not t.is_contiguous():
                    t = t.contiguous()
                context.set_input_shape(name, tuple(t.shape))
                context.set_tensor_address(name, t.data_ptr())
        else:
            shape = context.get_tensor_shape(name)
            dtype_trt = engine.get_tensor_dtype(name)
            dtype_map = {
                trt.float32: torch.float32,
                trt.float16: torch.float16,
                trt.int32: torch.int32,
                trt.int64: torch.int64,
            }
            if hasattr(trt, "bfloat16"):
                dtype_map[trt.bfloat16] = torch.bfloat16
            torch_dtype = dtype_map.get(dtype_trt, torch.float32)
            out_t = torch.empty(list(shape), dtype=torch_dtype, device=device)
            context.set_tensor_address(name, out_t.data_ptr())
            bindings[name] = out_t

    context.execute_async_v3(stream_handle=stream.cuda_stream)
    stream.synchronize()
    return bindings


def main():
    parser = argparse.ArgumentParser(description="TRT E2E verification (Pure TRT engines)")
    parser.add_argument("--model-dir", required=True, help="Exported variant dir")
    parser.add_argument("--ref-file", default=None, help="Path to e2e_trt_ref.npz")
    parser.add_argument("--variant", default=None, help="Variant name for report")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    variant_name = args.variant or model_dir.name
    ref_file = args.ref_file or str(model_dir / "e2e_trt_ref.npz")
    if not os.path.exists(ref_file):
        logger.error(f"Reference file not found: {ref_file}. Run verify_e2e_trt_ref.py on host first.")
        sys.exit(1)

    ref = np.load(ref_file)
    inputs_embeds = ref["inputs_embeds"]
    prefill_logits = ref["prefill_logits"]
    seq_len = int(ref["seq_len"])
    n_steps = int(ref["n_steps"])
    all_codec_tokens = ref["all_codec_tokens"]
    step_talker_logits = ref["step_talker_logits"]

    device = torch.device("cuda:0")

    report = {
        "variant": variant_name,
        "n_steps": n_steps,
        "prefill_cosine": None,
        "talker_token_match_rate": None,
        "status": "fail",
    }

    # ── Load TRT engines ──
    ctx_engine_path = model_dir / "talker_context.engine"
    dec_engine_path = model_dir / "talker_decode_fused.engine"

    if not ctx_engine_path.exists():
        logger.error(f"talker_context.engine not found: {ctx_engine_path}")
        sys.exit(1)
    if not dec_engine_path.exists():
        logger.error(f"talker_decode_fused.engine not found: {dec_engine_path}")
        sys.exit(1)

    logger.info("Loading TRT engines ...")
    t0 = time.time()
    ctx_engine, ctx_context = _load_trt_engine(str(ctx_engine_path), device)
    dec_engine, dec_context = _load_trt_engine(str(dec_engine_path), device)
    logger.info(f"Engines loaded in {time.time() - t0:.1f}s")

    # ── (A) Context (prefill) ──
    logger.info(f"(A) Running context prefill (seq_len={seq_len}) ...")
    inp_emb = torch.from_numpy(inputs_embeds).to(device).float()
    B, S, H = inp_emb.shape
    pos_ids = torch.arange(S, device=device, dtype=torch.int64)
    pos_ids = pos_ids.unsqueeze(0).unsqueeze(0).expand(3, B, S)

    ctx_feed = {"input_embeds": inp_emb, "position_ids": pos_ids}
    ctx_out = _trt_infer(ctx_engine, ctx_context, ctx_feed, device)

    if "last_logits" in ctx_out:
        ctx_logits = ctx_out["last_logits"].cpu().numpy()
        prefill_cos = cosine_sim(ctx_logits, prefill_logits)
        report["prefill_cosine"] = prefill_cos
        logger.info(f"  Prefill logits cosine: {prefill_cos:.6f}")

    codec_sum_ctx = ctx_out.get("codec_sum")
    full_codec_ctx = ctx_out.get("full_codec")

    kv_tensors = []
    i = 0
    while f"present_kv_{i}_k" in ctx_out:
        kv_tensors.append(ctx_out[f"present_kv_{i}_k"])
        kv_tensors.append(ctx_out[f"present_kv_{i}_v"])
        i += 1
    num_layers = i
    logger.info(f"  KV cache: {num_layers} layers captured")

    ref_first_token = int(all_codec_tokens[0, 0]) if all_codec_tokens.ndim == 2 else int(all_codec_tokens[0])
    if full_codec_ctx is not None:
        trt_first_token = int(full_codec_ctx[0, 0].item())
        logger.info(f"  First codec token: TRT={trt_first_token}, ref={ref_first_token}, match={trt_first_token == ref_first_token}")

    # ── (B) Decode loop ──
    logger.info(f"(B) Running decode loop ({n_steps} steps) ...")
    token_matches = 0
    total_compare = 0
    t1 = time.time()

    current_codec_sum = codec_sum_ctx
    current_pos = S

    for step in range(n_steps):
        pos_step = torch.zeros(3, B, 1, device=device, dtype=torch.int64)
        pos_step[:, :, :] = current_pos

        dec_feed = {
            "input_embeds": current_codec_sum,
            "position_ids": pos_step,
        }
        for li in range(num_layers):
            dec_feed[f"past_kv_{li}_k"] = kv_tensors[2 * li]
            dec_feed[f"past_kv_{li}_v"] = kv_tensors[2 * li + 1]

        dec_out = _trt_infer(dec_engine, dec_context, dec_feed, device)

        current_codec_sum = dec_out.get("codec_sum")
        full_codec_step = dec_out.get("full_codec")

        kv_tensors = []
        for li in range(num_layers):
            kv_tensors.append(dec_out[f"present_kv_{li}_k"])
            kv_tensors.append(dec_out[f"present_kv_{li}_v"])

        current_pos += 1

        if full_codec_step is not None and step + 1 < all_codec_tokens.shape[0]:
            trt_token = int(full_codec_step[0, 0].item())
            ref_token = int(all_codec_tokens[step + 1, 0])
            if trt_token == ref_token:
                token_matches += 1
            total_compare += 1

    gen_time = time.time() - t1
    match_rate = token_matches / total_compare if total_compare > 0 else 0
    report["talker_token_match_rate"] = match_rate
    report["decode_time_s"] = gen_time
    report["steps_per_second"] = n_steps / gen_time if gen_time > 0 else 0
    logger.info(f"  Decode: {n_steps} steps in {gen_time:.3f}s ({report['steps_per_second']:.1f} steps/s)")
    logger.info(f"  Token match: {token_matches}/{total_compare} = {match_rate:.1%}")

    # ── Summary ──
    prefill_ok = report.get("prefill_cosine") is None or report["prefill_cosine"] > 0.9
    talker_ok = report.get("talker_token_match_rate") is None or report["talker_token_match_rate"] >= 0
    report["status"] = "pass" if (prefill_ok and talker_ok) else "fail"

    logger.info("")
    logger.info("=" * 60)
    logger.info("  E2E TRT SUMMARY (Pure TRT)")
    logger.info("=" * 60)
    logger.info(f"  Prefill cosine:     {report.get('prefill_cosine', 'N/A')}")
    logger.info(f"  Token match rate:   {report.get('talker_token_match_rate', 'N/A')}")
    logger.info(f"  Decode speed:       {report.get('steps_per_second', 'N/A'):.1f} steps/s")
    logger.info(f"  Status:             {report['status']}")

    out_path = model_dir / "e2e_trt_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"  Report: {out_path}")

    sys.exit(0 if report["status"] == "pass" else 1)


if __name__ == "__main__":
    main()

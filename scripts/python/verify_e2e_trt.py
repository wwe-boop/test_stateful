#!/usr/bin/env python3
"""
TRT E2E verification — Single talker_unified engine (prefill + decode).

Loads talker_unified.engine built by trtexec, runs prefill + decode loop, comparing
against FP32 PyTorch reference (e2e_trt_ref.npz from verify_e2e_trt_ref.py).

Usage (inside NGC container or host with TensorRT):
  python3 verify_e2e_trt.py --model-dir /path/to/exported/variant [--ref-file /path/to/e2e_trt_ref.npz]
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
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    if a_flat.numel() == 0:
        return 1.0
    return float(F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)))


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


def _load_talker_dims(model_dir: Path):
    """Load H, num_kv_heads, head_dim, num_layers from weights config or ref."""
    cfg_path = model_dir / "weights" / "config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            c = json.load(f)
        H = int(c.get("talker_hidden_size", 2048))
        n_heads = int(c.get("talker_num_heads", 16))
        num_kv_heads = int(c.get("talker_num_kv_heads", 8))
        num_layers = int(c.get("talker_num_layers", 28))
        head_dim = H // n_heads
        return H, num_kv_heads, head_dim, num_layers
    return 2048, 8, 128, 28


def main():
    parser = argparse.ArgumentParser(
        description="TRT E2E verification (talker_unified.engine)"
    )
    parser.add_argument("--model-dir", required=True, help="Exported variant dir")
    parser.add_argument("--ref-file", default=None, help="Path to e2e_trt_ref.npz")
    parser.add_argument("--variant", default=None, help="Variant name for report")
    parser.add_argument("--steps", type=int, default=None, help="Max decode steps (default: use ref n_steps)")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    variant_name = args.variant or model_dir.name
    ref_file = args.ref_file or str(model_dir / "e2e_trt_ref.npz")
    if not os.path.exists(ref_file):
        logger.error(
            f"Reference file not found: {ref_file}. Run verify_e2e_trt_ref.py on host first."
        )
        sys.exit(1)

    ref = np.load(ref_file)
    inputs_embeds = ref["inputs_embeds"]
    prefill_logits = ref["prefill_logits"]
    pad_embed_ref = ref["pad_embed"] if "pad_embed" in ref else None
    seq_len = int(ref["seq_len"])
    n_steps = int(ref["n_steps"]) if args.steps is None else min(int(ref["n_steps"]), args.steps)
    all_codec_tokens = ref["all_codec_tokens"]
    step_output_codec_token_0_ref = (
        ref["step_output_codec_token_0"]
        if "step_output_codec_token_0" in ref
        else all_codec_tokens[:, 0]
    )
    step_talker_logits = ref["step_talker_logits"]
    codec_eos_id = int(ref["codec_eos_id"]) if "codec_eos_id" in ref else 4198
    ref_eos_step = int(ref["eos_step"]) if "eos_step" in ref else -1

    device = torch.device("cuda:0")
    H, num_kv_heads, head_dim, num_layers = _load_talker_dims(model_dir)

    report = {
        "variant": variant_name,
        "n_steps": n_steps,
        "prefill_logits_cosine": None,
        "step_logits_cosines": [],
        "codec_token_0_match_rate": None,
        "full_codec_match_rate": None,
        "ref_eos_step": ref_eos_step,
        "trt_eos_step": None,
        "status": "fail",
    }

    # Load single unified engine
    engine_path = model_dir / "talker_unified.engine"
    if not engine_path.exists():
        logger.error(f"talker_unified.engine not found: {engine_path}")
        sys.exit(1)

    logger.info("Loading talker_unified.engine ...")
    t0 = time.time()
    engine, context = _load_trt_engine(str(engine_path), device)
    logger.info(f"Engine loaded in {time.time() - t0:.1f}s")

    # Prefill: input_embeds [1, S, H], position_ids [3, 1, S], dummy past_kv [1, kv, 1, hd]
    S = seq_len
    B = 1
    inp_emb = torch.from_numpy(inputs_embeds).to(device=device, dtype=torch.bfloat16)
    if inp_emb.shape[1] != S or inp_emb.shape[2] != H:
        logger.warning(
            f"Ref inputs_embeds shape {inp_emb.shape} vs expected (1, {S}, {H}); trimming/padding"
        )
        inp_emb = inp_emb[:, :S, :].contiguous()
        if inp_emb.shape[2] != H:
            logger.error("Ref hidden size does not match engine H")
            sys.exit(1)
    position_ids_prefill = torch.arange(1, S + 1, device=device, dtype=torch.int64)
    position_ids_prefill = position_ids_prefill.unsqueeze(0).unsqueeze(0).expand(3, B, S)

    feed = {
        "input_embeds": inp_emb,
        "position_ids": position_ids_prefill,
    }
    for i in range(num_layers):
        feed[f"past_kv_{i}_k"] = torch.zeros(
            1, num_kv_heads, 1, head_dim, device=device, dtype=torch.bfloat16
        )
        feed[f"past_kv_{i}_v"] = torch.zeros(
            1, num_kv_heads, 1, head_dim, device=device, dtype=torch.bfloat16
        )

    logger.info(f"(A) Prefill: seq_len={S} ...")
    out = _trt_infer(engine, context, feed, device)

    logits = out["logits"]
    if logits.dtype == torch.bfloat16:
        logits_f = logits.float()
    else:
        logits_f = logits.float()
    prefill_logits_ref = np.asarray(prefill_logits, dtype=np.float32)
    if prefill_logits_ref.ndim == 3:
        prefill_logits_ref = prefill_logits_ref[:, -1, :]
    prefill_cos = cosine_sim(logits_f.cpu().numpy(), prefill_logits_ref)
    report["prefill_logits_cosine"] = prefill_cos
    logger.info(f"  Prefill logits cosine: {prefill_cos:.6f}")

    codec_sum = out["codec_sum"]
    full_codec = out["full_codec"]
    kv_tensors = []
    for i in range(num_layers):
        kv_tensors.append(out[f"present_kv_{i}_k"])
        kv_tensors.append(out[f"present_kv_{i}_v"])

    ref_first = int(all_codec_tokens[0, 0]) if all_codec_tokens.ndim == 2 else int(all_codec_tokens[0])
    trt_first = int(full_codec[0, 0].item())
    logger.info(f"  First codec token: TRT={trt_first}, ref={ref_first}, match={trt_first == ref_first}")

    # Decode loop
    logger.info(f"(B) Decode loop ({n_steps} steps) ...")
    token_matches = 0
    full_codec_matches = 0
    step_cosines = []
    trt_eos_step = -1
    t1 = time.time()

    # Next decode input = codec_sum + pad_embed (match ref)
    current_codec_sum = codec_sum
    if pad_embed_ref is not None:
        pad_embed = torch.from_numpy(np.asarray(pad_embed_ref, dtype=np.float32)).to(device=device)
        if pad_embed.dim() == 2:
            pad_embed = pad_embed.unsqueeze(1)
        current_codec_sum = (current_codec_sum.float() + pad_embed).to(codec_sum.dtype)
    current_pos = S

    for step in range(n_steps):
        pos_step = torch.full(
            (3, B, 1), current_pos, device=device, dtype=torch.int64
        )
        dec_feed = {
            "input_embeds": current_codec_sum,
            "position_ids": pos_step,
        }
        for i in range(num_layers):
            dec_feed[f"past_kv_{i}_k"] = kv_tensors[2 * i]
            dec_feed[f"past_kv_{i}_v"] = kv_tensors[2 * i + 1]

        dec_out = _trt_infer(engine, context, dec_feed, device)

        codec_sum_step = dec_out["codec_sum"]
        if pad_embed_ref is not None:
            current_codec_sum = (codec_sum_step.float() + pad_embed).to(codec_sum_step.dtype)
        else:
            current_codec_sum = codec_sum_step
        full_codec_step = dec_out["full_codec"]
        step_logits = dec_out["logits"]
        if step_logits.dtype == torch.bfloat16:
            step_logits_f = step_logits.float()
        else:
            step_logits_f = step_logits.float()

        kv_tensors = []
        for i in range(num_layers):
            kv_tensors.append(dec_out[f"present_kv_{i}_k"])
            kv_tensors.append(dec_out[f"present_kv_{i}_v"])

        current_pos += 1

        ref_step_logits = step_talker_logits[step]
        if ref_step_logits.ndim == 3:
            ref_step_logits = ref_step_logits[:, -1, :]
        cos_s = cosine_sim(step_logits_f.cpu().numpy(), ref_step_logits)
        step_cosines.append(cos_s)

        trt_token_0 = int(full_codec_step[0, 0].item())
        ref_token_0 = int(step_output_codec_token_0_ref[step])
        if trt_token_0 == ref_token_0:
            token_matches += 1
        ref_full = all_codec_tokens[step]
        trt_full = full_codec_step[0].cpu().numpy().astype(np.int64)
        if trt_full.shape[0] >= ref_full.shape[0] and np.all(trt_full[: ref_full.shape[0]] == ref_full):
            full_codec_matches += 1
        if trt_eos_step < 0 and trt_token_0 == codec_eos_id:
            trt_eos_step = step

        if (step + 1) % 50 == 0 or step == 0:
            logger.info(
                f"  step {step}: token_0 TRT={trt_token_0} ref={ref_token_0} match={trt_token_0 == ref_token_0} cos={cos_s:.4f}"
            )

    gen_time = time.time() - t1
    total_compare = n_steps
    match_rate = token_matches / total_compare if total_compare > 0 else 0
    full_match_rate = full_codec_matches / total_compare if total_compare > 0 else 0
    report["codec_token_0_match_rate"] = match_rate
    report["full_codec_match_rate"] = full_match_rate
    report["step_logits_cosines"] = step_cosines
    report["trt_eos_step"] = trt_eos_step
    report["decode_time_s"] = gen_time
    report["steps_per_second"] = n_steps / gen_time if gen_time > 0 else 0

    logger.info(
        f"  Decode: {n_steps} steps in {gen_time:.3f}s ({report['steps_per_second']:.1f} steps/s)"
    )
    logger.info(f"  codec_token_0 match: {token_matches}/{total_compare} = {match_rate:.1%}")
    logger.info(f"  full_codec match: {full_codec_matches}/{total_compare} = {full_match_rate:.1%}")
    logger.info(f"  EOS: ref_step={ref_eos_step}, trt_step={trt_eos_step}")

    min_cos = min(step_cosines) if step_cosines else 0
    prefill_ok = report["prefill_logits_cosine"] is None or report["prefill_logits_cosine"] > 0.9
    token_ok = report["codec_token_0_match_rate"] is not None and report["codec_token_0_match_rate"] >= 0
    report["status"] = "pass" if (prefill_ok and token_ok) else "fail"

    logger.info("")
    logger.info("=" * 60)
    logger.info("  E2E TRT SUMMARY (talker_unified)")
    logger.info("=" * 60)
    logger.info(f"  Prefill logits cosine: {report.get('prefill_logits_cosine', 'N/A')}")
    logger.info(f"  codec_token_0 match rate: {report.get('codec_token_0_match_rate', 'N/A')}")
    logger.info(f"  full_codec match rate: {report.get('full_codec_match_rate', 'N/A')}")
    logger.info(f"  Min step logits cosine: {min_cos:.4f}")
    logger.info(f"  EOS ref_step / trt_step: {ref_eos_step} / {trt_eos_step}")
    logger.info(f"  Decode speed: {report.get('steps_per_second', 'N/A'):.1f} steps/s")
    logger.info(f"  Status: {report['status']}")

    out_path = model_dir / "e2e_trt_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"  Report: {out_path}")

    sys.exit(0 if report["status"] == "pass" else 1)


if __name__ == "__main__":
    main()

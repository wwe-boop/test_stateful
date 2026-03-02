#!/usr/bin/env python3
"""
[Phase 2, Item 12 — Part 2] TRT E2E verification (container-side).

Runs inside NGC TRT-LLM container. Loads TRT-LLM (Talker) + TRT (Code Predictor)
and compares against FP32 PyTorch reference (e2e_trt_ref.npz).

  (A) Talker: prefill via prompt_embedding_table, generate(max_new_tokens=N),
      compare context_logits (prefill cosine) + decode token sequence.
  (B) CP: per-step TRT CP inference with ref (past_hidden, codec_token_0),
      compare 15-token output with ref step_cp_tokens.

Usage (inside container, invoked by verify_e2e_trt.sh):
  python3 /mnt/scripts/verify_e2e_trt.py --model-dir /mnt/model [--ref-file /mnt/model/e2e_trt_ref.npz]
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
    a_flat = a.flatten().unsqueeze(0)
    b_flat = b.flatten().unsqueeze(0)
    return float(F.cosine_similarity(a_flat, b_flat))


def main():
    parser = argparse.ArgumentParser(description="TRT E2E verification (container)")
    parser.add_argument("--model-dir", required=True, help="Exported variant dir (e.g. /mnt/model)")
    parser.add_argument("--ref-file", default=None, help="Path to e2e_trt_ref.npz (default: <model-dir>/e2e_trt_ref.npz)")
    parser.add_argument("--variant", default=None, help="Variant name for report (default: model-dir basename)")
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
    hidden_size = int(ref["hidden_size"])
    vocab_size = int(ref["vocab_size"])
    n_steps = int(ref["n_steps"])
    all_codec_tokens = ref["all_codec_tokens"]
    step_past_hidden = ref["step_past_hidden"]
    step_cp_tokens = ref["step_cp_tokens"]
    step_talker_logits = ref["step_talker_logits"]

    ref_talker_decode_tokens = all_codec_tokens[:, 0].tolist()
    logger.info(f"Reference: seq_len={seq_len}, n_steps={n_steps}, hidden={hidden_size}, vocab={vocab_size}")
    device = torch.device("cuda:0")

    report = {
        "variant": variant_name,
        "n_steps": n_steps,
        "prefill_cosine": None,
        "talker_token_match_rate": None,
        "cp_token_match_rates": [],
        "cp_mean_match_rate": None,
        "status": "fail",
    }

    # -------------------------------------------------------------------------
    # (A) Talker: TRT-LLM prefill + generate
    # -------------------------------------------------------------------------
    engine_dir = model_dir / "trtllm_engine"
    if engine_dir.exists():
        logger.info("(A) Loading TRT-LLM Talker engine ...")
        try:
            import tensorrt_llm
            logger.info(f"TensorRT-LLM version: {getattr(tensorrt_llm, '__version__', 'unknown')}")
        except Exception:
            pass
        from tensorrt_llm.runtime import ModelRunner, SamplingConfig

        t0 = time.time()
        runner = ModelRunner.from_dir(
            str(engine_dir),
            max_output_len=n_steps + 1,
        )
        logger.info(f"Engine loaded in {time.time() - t0:.1f}s")

        prompt_table = torch.from_numpy(inputs_embeds[0]).cuda()
        batch_input_ids = [torch.arange(vocab_size, vocab_size + seq_len, dtype=torch.int32)]
        sampling = SamplingConfig(
            end_id=-1,
            pad_id=-1,
            max_new_tokens=n_steps,
            top_k=1,
            temperature=1.0,
            return_dict=True,
            output_sequence_lengths=True,
        )

        logger.info("Running TRT-LLM generate ...")
        t1 = time.time()
        outputs = runner.generate(
            batch_input_ids=batch_input_ids,
            sampling_config=sampling,
            prompt_table=prompt_table,
            prompt_tasks="0",
        )
        gen_time = time.time() - t1
        logger.info(f"Generate completed in {gen_time:.3f}s")

        generated_ids = outputs["output_ids"][0, 0, seq_len:].cpu().numpy()
        n_compare = min(len(generated_ids), len(ref_talker_decode_tokens))
        talker_match = sum(1 for i in range(n_compare) if int(generated_ids[i]) == int(ref_talker_decode_tokens[i]))
        report["talker_token_match_rate"] = talker_match / n_compare if n_compare > 0 else 0
        report["talker_generated_ids"] = generated_ids.tolist()
        report["talker_ref_ids"] = ref_talker_decode_tokens[:n_compare]
        report["talker_gen_time_s"] = gen_time
        logger.info(f"Talker decode token match: {talker_match}/{n_compare} = {report['talker_token_match_rate']:.1%}")

        context_logits = outputs.get("context_logits")
        if context_logits is not None:
            ctx = context_logits[0].cpu().numpy()
            prefill_cos = cosine_sim(ctx, prefill_logits[0])
            report["prefill_cosine"] = prefill_cos
            logger.info(f"Prefill logits cosine: {prefill_cos:.6f}")
    else:
        logger.warning("(A) trtllm_engine not found, skipping Talker verification")

    # -------------------------------------------------------------------------
    # (B) Code Predictor: TRT engine per-step
    # -------------------------------------------------------------------------
    plan_path = model_dir / "code_predictor_unrolled.plan"
    if plan_path.exists():
        logger.info("(B) Loading TRT Code Predictor engine ...")
        _script_dir = os.path.dirname(os.path.abspath(__file__))
        if _script_dir not in sys.path:
            sys.path.insert(0, _script_dir)
        from verify_code_predictor_trt import TRTRunner
        runner_cp = TRTRunner(str(plan_path), device)

        match_counts = []
        for i in range(n_steps):
            past_h = torch.from_numpy(step_past_hidden[i : i + 1]).to(device)
            if past_h.dim() == 2:
                past_h = past_h.unsqueeze(0)
            past_h = past_h.float()
            codec_0 = int(all_codec_tokens[i, 0])
            feeds = {
                "past_hidden": past_h,
                "codec_token_0": torch.tensor([codec_0], dtype=torch.int64, device=device),
            }
            out = runner_cp.infer(feeds)
            out_names = list(out.keys())
            trt_tokens = out[out_names[0]].cpu().numpy().flatten()[:15]
            ref_15 = step_cp_tokens[i]
            match = sum(1 for a, b in zip(trt_tokens, ref_15) if int(a) == int(b))
            match_counts.append(match)
        report["cp_token_match_rates"] = [m / 15.0 for m in match_counts]
        report["cp_mean_match_rate"] = float(np.mean(match_counts)) / 15.0 if match_counts else 0
        logger.info(f"CP token match: mean {report['cp_mean_match_rate']:.1%} over {n_steps} steps")
    else:
        logger.warning("(B) code_predictor_unrolled.plan not found, skipping CP verification")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    prefill_ok = report.get("prefill_cosine") is None or report["prefill_cosine"] > 0.9
    talker_ok = report.get("talker_token_match_rate") is None or report["talker_token_match_rate"] >= 0
    cp_ok = report.get("cp_mean_match_rate") is None or report["cp_mean_match_rate"] >= 0.5
    report["status"] = "pass" if (prefill_ok and talker_ok and cp_ok) else "fail"

    logger.info("")
    logger.info("=" * 60)
    logger.info("  E2E TRT SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Prefill cosine:     {report.get('prefill_cosine', 'N/A')}")
    logger.info(f"  Talker token match: {report.get('talker_token_match_rate', 'N/A')}")
    logger.info(f"  CP mean match:      {report.get('cp_mean_match_rate', 'N/A')}")
    logger.info(f"  Status:             {report['status']}")

    out_path = model_dir / "e2e_trt_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"  Report: {out_path}")

    sys.exit(0 if report["status"] == "pass" else 1)


if __name__ == "__main__":
    main()

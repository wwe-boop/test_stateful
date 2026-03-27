#!/usr/bin/env python3
"""
Phase 0: ONNX Runtime (FP32) vs PyTorch reference — precision isolation.

Runs talker_unified.onnx with ONNX Runtime in FP32, step-by-step prefill + decode,
and compares logits / codec_token_0 / full_codec against e2e_trt_ref.npz.
If ORT FP32 matches PyTorch, precision issues are in TRT/BF16; if not, issue is in ONNX export.

Usage (host, conda with onnxruntime):
  python scripts/python/verify_precision_ort.py \\
    --onnx workspace/exported/design-1.7b/talker_unified.onnx \\
    --ref workspace/exported/design-1.7b/e2e_trt_ref.npz \\
    [--steps 200] [--model-dir workspace/exported/design-1.7b]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("verify_ort")


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).flatten()
    b = np.asarray(b, dtype=np.float64).flatten()
    if a.size == 0:
        return 1.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _load_talker_dims(model_dir: Path):
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
        description="ORT FP32 vs PyTorch reference (talker_unified.onnx)"
    )
    parser.add_argument("--onnx", required=True, help="Path to talker_unified.onnx")
    parser.add_argument("--ref", required=True, help="Path to e2e_trt_ref.npz")
    parser.add_argument("--model-dir", default=None, help="Exported variant dir (default: parent of --ref)")
    parser.add_argument("--steps", type=int, default=None, help="Max decode steps (default: ref n_steps)")
    parser.add_argument("--provider", default="CPUExecutionProvider", help="ORT provider (e.g. CUDAExecutionProvider)")
    args = parser.parse_args()

    onnx_path = Path(args.onnx)
    ref_file = Path(args.ref)
    if not onnx_path.exists():
        logger.error(f"ONNX not found: {onnx_path}")
        sys.exit(1)
    if not ref_file.exists():
        logger.error(f"Reference not found: {ref_file}. Run verify_e2e_trt_ref.py first.")
        sys.exit(1)

    model_dir = Path(args.model_dir) if args.model_dir else ref_file.parent
    H, num_kv_heads, head_dim, num_layers = _load_talker_dims(model_dir)

    ref = np.load(ref_file)
    inputs_embeds = np.asarray(ref["inputs_embeds"], dtype=np.float32)
    prefill_logits_ref = np.asarray(ref["prefill_logits"], dtype=np.float32)
    pad_embed = np.asarray(ref["pad_embed"], dtype=np.float32) if "pad_embed" in ref else None
    trailing_text_hidden = (
        np.asarray(ref["trailing_text_hidden"], dtype=np.float32)
        if "trailing_text_hidden" in ref
        else None
    )
    if trailing_text_hidden is not None and trailing_text_hidden.ndim == 3:
        trailing_text_hidden = trailing_text_hidden[:, np.newaxis, :, :]
    seq_len = int(ref["seq_len"])
    n_steps_ref = int(ref["n_steps"])
    n_steps = n_steps_ref if args.steps is None else min(n_steps_ref, args.steps)
    all_codec_tokens = ref["all_codec_tokens"]
    step_output_codec_token_0_ref = (
        ref["step_output_codec_token_0"]
        if "step_output_codec_token_0" in ref
        else all_codec_tokens[:, 0]
    )
    step_talker_logits_ref = ref["step_talker_logits"]
    codec_eos_id = int(ref["codec_eos_id"]) if "codec_eos_id" in ref else 4198
    ref_eos_step = int(ref["eos_step"]) if "eos_step" in ref else -1
    if pad_embed is not None and pad_embed.ndim == 2:
        pad_embed = pad_embed.reshape(1, 1, -1)

    if inputs_embeds.shape[1] != seq_len or inputs_embeds.shape[2] != H:
        inputs_embeds = inputs_embeds[:, :seq_len, :].astype(np.float32)
        if inputs_embeds.shape[2] != H:
            logger.error("Ref hidden size does not match config H")
            sys.exit(1)

    import onnxruntime as ort

    sess_options = ort.SessionOptions()
    session = ort.InferenceSession(
        str(onnx_path),
        sess_options,
        providers=[args.provider],
    )

    input_names = [inp.name for inp in session.get_inputs()]
    output_names = [out.name for out in session.get_outputs()]

    # Prefill: match ref (no past): position_ids 0..S-1, past_kv length 0
    # ONNX input position_ids shape: (B, 3, S, 1) for multimodal RoPE export.
    B, S = 1, seq_len
    position_ids_prefill = np.arange(0, S, dtype=np.int64)
    position_ids_prefill = np.broadcast_to(
        position_ids_prefill.reshape(1, 1, -1, 1), (B, 3, S, 1)
    )

    feed = {
        "input_embeds": inputs_embeds,
        "position_ids": position_ids_prefill,
    }
    for i in range(num_layers):
        feed[f"past_kv_{i}_k"] = np.zeros(
            (1, num_kv_heads, 0, head_dim), dtype=np.float32
        )
        feed[f"past_kv_{i}_v"] = np.zeros(
            (1, num_kv_heads, 0, head_dim), dtype=np.float32
        )

    logger.info("(A) ORT prefill ...")
    outs = session.run(output_names, feed)
    out_map = dict(zip(output_names, outs))

    logits = out_map["logits"]
    if logits.ndim == 3:
        logits_last = logits[:, -1, :]
    else:
        logits_last = logits
    prefill_ref = prefill_logits_ref
    if prefill_ref.ndim == 3:
        prefill_ref = prefill_ref[:, -1, :]
    prefill_cos = cosine_sim(logits_last, prefill_ref)
    logger.info(f"  Prefill logits cosine: {prefill_cos:.6f}")

    codec_sum = out_map["codec_sum"]
    full_codec = out_map["full_codec"]
    ref_first = int(all_codec_tokens[0, 0])
    ort_first = int(full_codec[0, 0])
    logger.info(f"  First codec token: ORT={ort_first}, ref={ref_first}, match={ort_first == ref_first}")

    # Decode loop: next input = codec_sum + pad_embed (match ref)
    logger.info(f"(B) ORT decode loop ({n_steps} steps) ...")
    token_matches = 0
    full_codec_matches = 0
    step_cosines = []
    ort_eos_step = -1
    first_divergence = -1

    past_kv = []
    for i in range(num_layers):
        past_kv.append(out_map[f"present_kv_{i}_k"].copy())
        past_kv.append(out_map[f"present_kv_{i}_v"].copy())

    n_trailing = trailing_text_hidden.shape[0] if trailing_text_hidden is not None else 0
    # First decode input: prefill codec_sum + (trailing[0] or pad_embed)
    text_add_0 = (
        trailing_text_hidden[0]
        if (trailing_text_hidden is not None and n_trailing > 0)
        else pad_embed
    )
    if text_add_0 is not None:
        if text_add_0.ndim == 2:
            text_add_0 = text_add_0.reshape(1, 1, -1)
        current_codec_sum = (codec_sum.astype(np.float64) + text_add_0.astype(np.float64)).astype(np.float32)
    else:
        current_codec_sum = codec_sum.copy()
    current_pos = S

    for step in range(n_steps):
        pos_step = np.full((B, 3, 1, 1), current_pos, dtype=np.int64)
        dec_feed = {
            "input_embeds": current_codec_sum,
            "position_ids": pos_step,
        }
        for i in range(num_layers):
            dec_feed[f"past_kv_{i}_k"] = past_kv[2 * i]
            dec_feed[f"past_kv_{i}_v"] = past_kv[2 * i + 1]

        dec_outs = session.run(output_names, dec_feed)
        dec_map = dict(zip(output_names, dec_outs))

        codec_sum_step = dec_map["codec_sum"]
        # Next step input: this step's codec_sum + (trailing[step+1] or pad_embed)
        next_text = (
            trailing_text_hidden[step + 1]
            if (trailing_text_hidden is not None and step + 1 < n_trailing)
            else pad_embed
        )
        if next_text is not None:
            if next_text.ndim == 2:
                next_text = next_text.reshape(1, 1, -1)
            current_codec_sum = (codec_sum_step.astype(np.float64) + next_text.astype(np.float64)).astype(np.float32)
        else:
            current_codec_sum = codec_sum_step
        full_codec_step = dec_map["full_codec"]
        step_logits = dec_map["logits"]
        if step_logits.ndim == 3:
            step_logits_last = step_logits[:, -1, :]
        else:
            step_logits_last = step_logits

        for i in range(num_layers):
            past_kv[2 * i] = dec_map[f"present_kv_{i}_k"].copy()
            past_kv[2 * i + 1] = dec_map[f"present_kv_{i}_v"].copy()

        current_pos += 1

        ref_step_logits = np.asarray(step_talker_logits_ref[step], dtype=np.float64)
        if ref_step_logits.ndim == 3:
            ref_step_logits = ref_step_logits[:, -1, :]
        cos_s = cosine_sim(step_logits_last, ref_step_logits)
        step_cosines.append(cos_s)

        ort_token_0 = int(full_codec_step[0, 0])
        ref_token_0 = int(step_output_codec_token_0_ref[step])
        if ort_token_0 == ref_token_0:
            token_matches += 1
        else:
            if first_divergence < 0:
                first_divergence = step
        ref_full = all_codec_tokens[step]
        ort_full = full_codec_step[0].astype(np.int64)
        n_c = min(ort_full.shape[0], ref_full.shape[0])
        if np.all(ort_full[:n_c] == ref_full[:n_c]):
            full_codec_matches += 1
        if ort_eos_step < 0 and ort_token_0 == codec_eos_id:
            ort_eos_step = step

        if (step + 1) % 50 == 0 or step == 0:
            logger.info(
                f"  step {step}: token_0 ORT={ort_token_0} ref={ref_token_0} match={ort_token_0 == ref_token_0} cos={cos_s:.4f}"
            )

    total_compare = n_steps
    match_rate = token_matches / total_compare if total_compare > 0 else 0
    full_match_rate = full_codec_matches / total_compare if total_compare > 0 else 0
    min_cos = float(min(step_cosines)) if step_cosines else 0.0

    logger.info("")
    logger.info("=" * 60)
    logger.info("  ORT FP32 vs PyTorch REF SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Prefill logits cosine: {prefill_cos:.6f}")
    logger.info(f"  codec_token_0 match: {token_matches}/{total_compare} = {match_rate:.1%}")
    logger.info(f"  full_codec match: {full_codec_matches}/{total_compare} = {full_match_rate:.1%}")
    logger.info(f"  Min step logits cosine: {min_cos:.4f}")
    logger.info(f"  EOS: ref_step={ref_eos_step}, ort_step={ort_eos_step}")
    if first_divergence >= 0:
        logger.info(f"  First token divergence at step: {first_divergence}")

    ok = prefill_cos > 0.9999 and match_rate >= 0.99 and min_cos > 0.999
    status = "pass" if ok else "fail"
    logger.info(f"  Status: {status} (expect cosine>0.9999, token match>=99%%)")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

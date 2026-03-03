#!/usr/bin/env python3
"""
Master export script: exports all Qwen3-TTS components for Triton deployment.

Runs the numbered export scripts (01-06) in order for all (or selected) variants.
Dual-engine mode: pure ONNX Runtime and pure TensorRT (no TRT-LLM).

Export pipeline:
  01. Speech Tokenizer Encoder → ONNX  (shared)
  02. Code2Wav Decoder → ONNX          (shared)
  Per-variant:
    03. Speaker Encoder → ONNX         (base variants only)
    04. Talker Context Fused → ONNX    (prefill + CP + codec_sum)
    05. Talker Decode Fused → ONNX     (decode step + CP + codec_sum)
    06. Embedding weights → .pt        (for Orchestrator prefill; text_embedding in-process)

Usage:
  python scripts/export/export_all.py                        # all variants, bf16, auto GPU
  python scripts/export/export_all.py --variant custom-1.7b  # single variant
  python scripts/export/export_all.py --dtype fp32           # fp32 precision
  python scripts/export/export_all.py --device cpu           # force CPU
"""

import argparse
import logging
import sys
import time
from pathlib import Path

from utils import (
    setup_logging,
    MODEL_VARIANTS,
    has_model_weights,
    resolve_device,
    resolve_dtype,
    DEFAULT_DTYPE,
    DTYPE_NAMES,
)

logger = logging.getLogger("onnx_export")


def main():
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Export all Qwen3-TTS components to ONNX / PyTorch weights"
    )
    parser.add_argument("--variant", type=str, default=None,
                        help=f"Model variant. Available: {list(MODEL_VARIANTS.keys())}. "
                             "Default: export all downloaded variants")
    parser.add_argument("--models-dir", type=str, default=None,
                        help="Models directory (default: workspace/models)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: workspace/exported)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: cpu | cuda | cuda:0 | cuda:1 ... "
                             "(default: auto-select GPU with most free VRAM)")
    parser.add_argument("--dtype", type=str, default="bf16",
                        choices=["bf16", "fp16", "fp32"],
                        help="Target inference precision for embedding weights and TRT engine build "
                             "(default: bf16). ONNX export always uses fp32 internally.")
    parser.add_argument("--skip-tokenizer", action="store_true",
                        help="Skip shared tokenizer exports (steps 01-02)")
    parser.add_argument("--skip-talker", action="store_true",
                        help="Skip Talker Backbone export (step 04)")
    parser.add_argument("--skip-embeddings", action="store_true",
                        help="Skip Embedding weights export (step 06)")
    args = parser.parse_args()

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)
    logger.info(f"Export config: device={device}, target_dtype={DTYPE_NAMES[dtype]}, onnx_export=fp32")

    t_start = time.time()
    results = {}

    # ── Steps 01 & 02: Shared tokenizer components ──
    if not args.skip_tokenizer:
        logger.info("=" * 60)
        logger.info("Step 01/06: Speech Tokenizer Encoder → ONNX")
        logger.info("=" * 60)
        try:
            from export_01_speech_tokenizer_encoder import export_speech_tokenizer_encoder
            path = export_speech_tokenizer_encoder(args.models_dir, args.output_dir, device, dtype)
            results["speech_tokenizer_encoder"] = ("OK", path)
        except Exception as e:
            logger.error(f"Speech Tokenizer Encoder failed: {e}", exc_info=True)
            results["speech_tokenizer_encoder"] = ("FAILED", str(e))

        logger.info("=" * 60)
        logger.info("Step 02/06: Code2Wav Decoder → ONNX")
        logger.info("=" * 60)
        try:
            from export_02_code2wav_decoder import export_code2wav_decoder
            path = export_code2wav_decoder(args.models_dir, args.output_dir, device, dtype)
            results["code2wav_decoder"] = ("OK", path)
        except Exception as e:
            logger.error(f"Code2Wav Decoder failed: {e}", exc_info=True)
            results["code2wav_decoder"] = ("FAILED", str(e))
    else:
        logger.info("Skipping shared tokenizer exports (--skip-tokenizer)")

    # ── Steps 03-06: Per-variant exports ──
    from utils import DEFAULT_MODELS_DIR
    models_base = Path(args.models_dir) if args.models_dir else DEFAULT_MODELS_DIR

    if args.variant:
        variant_dir = models_base / MODEL_VARIANTS[args.variant]
        if not has_model_weights(variant_dir):
            logger.error(
                f"Variant '{args.variant}' has no weight files in {variant_dir}. "
                "Only config found — weights may not have been downloaded."
            )
            sys.exit(1)
        variants = [args.variant]
    else:
        variants = [
            v for v, dirname in MODEL_VARIANTS.items()
            if (models_base / dirname).exists() and has_model_weights(models_base / dirname)
        ]
        skipped = [
            v for v, dirname in MODEL_VARIANTS.items()
            if (models_base / dirname).exists() and not has_model_weights(models_base / dirname)
        ]
        if skipped:
            logger.warning(f"Skipping variants without weights: {skipped}")
        if not variants:
            logger.error(f"No model variants with weights found in {models_base}")
            sys.exit(1)
        logger.info(f"Detected variants with weights: {variants}")

    for variant in variants:
        logger.info("")
        logger.info("#" * 60)
        logger.info(f"  Exporting variant: {variant}")
        logger.info("#" * 60)

        # 03. Speaker Encoder (base variants only)
        if variant.startswith("base-"):
            logger.info("=" * 60)
            logger.info(f"Step 03/06: [{variant}] Speaker Encoder → ONNX")
            logger.info("=" * 60)
            try:
                from export_03_speaker_encoder import export_speaker_encoder
                path = export_speaker_encoder(variant, args.models_dir, args.output_dir, device, dtype)
                results[f"{variant}/speaker_encoder"] = ("OK", path)
            except Exception as e:
                logger.error(f"[{variant}] Speaker Encoder failed: {e}", exc_info=True)
                results[f"{variant}/speaker_encoder"] = ("FAILED", str(e))

        # 04. Talker Unified → ONNX (single engine for prefill + decode; replaces context + decode_fused)
        if not args.skip_talker:
            logger.info("=" * 60)
            logger.info(f"Step 04/06: [{variant}] Talker Unified → ONNX")
            logger.info("=" * 60)
            try:
                from export_04_talker_unified import export_talker_unified
                out = export_talker_unified(
                    variant, args.models_dir, args.output_dir, device,
                )
                path = out.get("onnx", out) if isinstance(out, dict) else out
                results[f"{variant}/talker_unified"] = ("OK", path)
            except Exception as e:
                logger.error(f"[{variant}] Talker Unified failed: {e}", exc_info=True)
                results[f"{variant}/talker_unified"] = ("FAILED", str(e))

        # 06. Embedding weights (.pt) for Orchestrator prefill
        if not args.skip_embeddings:
            logger.info("=" * 60)
            logger.info(f"Step 06/06: [{variant}] Embedding weights → .pt")
            logger.info("=" * 60)
            try:
                from export_06_embeddings import export_embeddings
                path = export_embeddings(variant, args.models_dir, args.output_dir, device, dtype)
                results[f"{variant}/embeddings"] = ("OK", str(path))
            except Exception as e:
                logger.error(f"[{variant}] Embeddings failed: {e}", exc_info=True)
                results[f"{variant}/embeddings"] = ("FAILED", str(e))

    # ── Summary ──
    elapsed = time.time() - t_start
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"  EXPORT SUMMARY  (target_dtype={DTYPE_NAMES[dtype]}, onnx=fp32, device={device})")
    logger.info("=" * 60)

    n_ok = sum(1 for s, _ in results.values() if s == "OK")
    n_fail = sum(1 for s, _ in results.values() if s == "FAILED")

    for name, (status, detail) in results.items():
        icon = "✓" if status == "OK" else "✗"
        logger.info(f"  {icon} {name}: {status}")
        if status == "FAILED":
            logger.info(f"    → {detail}")

    logger.info("")
    logger.info(f"Total: {n_ok} succeeded, {n_fail} failed, {elapsed:.1f}s elapsed")

    if n_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

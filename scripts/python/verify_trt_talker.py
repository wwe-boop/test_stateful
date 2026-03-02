#!/usr/bin/env python3
"""
Verify Pure TRT Talker engines (context + decode_fused).

Loads talker_context.engine and talker_decode_fused.engine from a variant
export dir, runs one context() and N decode_step() calls, and checks
output shapes and basic numerical sanity. No PyTorch reference required.

Usage:
  python scripts/python/verify_trt_talker.py --variant base-0.6b [--steps 5]
  python scripts/python/verify_trt_talker.py --engine-dir workspace/exported/base-0.6b [--steps 5]
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "model_repository" / "tts_orchestrator" / "1"))


def main():
    parser = argparse.ArgumentParser(description="Verify Pure TRT Talker engines")
    parser.add_argument("--variant", default=None, help="Model variant (e.g. base-0.6b)")
    parser.add_argument("--engine-dir", default=None, help="Path to dir with talker_context.engine + talker_decode_fused.engine")
    parser.add_argument("--steps", type=int, default=5, help="Number of decode steps to run")
    args = parser.parse_args()

    if args.engine_dir:
        engine_dir = Path(args.engine_dir)
    elif args.variant:
        engine_dir = REPO_ROOT / "workspace" / "exported" / args.variant
    else:
        parser.error("Provide --variant or --engine-dir")

    if not (engine_dir / "talker_context.engine").exists():
        print(f"ERROR: {engine_dir / 'talker_context.engine'} not found")
        print("  Run: python scripts/export/export_04a_talker_context.py --variant <variant>")
        print("       bash scripts/bash/build_engines.sh --pure-trt --variant <variant>")
        sys.exit(1)
    if not (engine_dir / "talker_decode_fused.engine").exists():
        print(f"ERROR: {engine_dir / 'talker_decode_fused.engine'} not found")
        print("  Run: python scripts/export/export_04b_talker_decode_fused.py --variant <variant>")
        print("       bash scripts/bash/build_engines.sh --pure-trt --variant <variant>")
        sys.exit(1)

    import torch
    from talker_runner import TalkerRunner

    print(f"Loading engines from {engine_dir} ...")
    runner = TalkerRunner(str(engine_dir), device=0)

    B, S, H = 1, 8, runner.hidden_size
    input_embeds = torch.randn(B, S, H, dtype=torch.bfloat16, device=runner.device)
    position_ids = torch.arange(S, device=runner.device, dtype=torch.int64)
    position_ids = position_ids.unsqueeze(0).unsqueeze(0).expand(3, B, S)

    print("Running context() ...")
    hidden, logits = runner.context(input_embeds, position_ids)
    print(f"  last_hidden: {hidden.shape}, last_logits: {logits.shape}")
    assert hidden.shape == (B, 1, H), f"expected (1,1,{H}), got {hidden.shape}"
    assert logits.shape == (B, 1, runner.vocab_size)

    print(f"Running {args.steps} decode_step() ...")
    for step in range(args.steps):
        next_embed = torch.randn(B, 1, H, dtype=torch.bfloat16, device=runner.device)
        position_id = torch.full((3, B, 1), runner.current_seq_len, device=runner.device, dtype=torch.int64)
        codec_sum, full_codec, logits_out = runner.decode_step(next_embed, position_id)
        assert codec_sum.shape == (B, 1, H)
        assert full_codec.shape == (B, 16)
        assert logits_out.shape == (B, 1, runner.vocab_size)
        if step == 0:
            print(f"  codec_sum: {codec_sum.shape}, full_codec: {full_codec.shape}, logits: {logits_out.shape}")

    print("  OK: context + decode_step shapes and execution succeeded.")
    runner.reset()
    print("Verify TRT Talker: PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

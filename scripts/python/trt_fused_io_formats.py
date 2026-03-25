#!/usr/bin/env python3
# English comments only.
"""
Build trtexec --inputIOFormats / --outputIOFormats for talker_code2wav_fused.onnx.

I/O order must match export_09_talker_code2wav_fused.py (input_names / output_names).
Integer tensors use int64:chw; float tensors use {fp32|fp16|bf16}:chw from manifest
triton_io_float_dtype.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _normalize_float_io_token(s: str) -> str:
    x = (s or "fp32").lower().strip()
    if x in ("float32", "float"):
        return "fp32"
    if x in ("bfloat16",):
        return "bf16"
    if x in ("float16",):
        return "fp16"
    if x in ("fp32", "bf16", "fp16"):
        return x
    return "fp32"


def _normalize_engine_dtype(s: str) -> str:
    x = (s or "bf16").lower().strip()
    if x in ("bfloat16",):
        return "bf16"
    if x in ("float16",):
        return "fp16"
    if x in ("float32", "float"):
        return "fp32"
    if x in ("fp8", "float8"):
        return "fp8"
    if x in ("bf16", "fp16", "fp32", "fp8"):
        return x
    return "bf16"


def trtexec_precision_args(engine_dtype: str) -> List[str]:
    """Flags as argv tokens (e.g. ['--bf16']); empty for fp32."""
    ed = _normalize_engine_dtype(engine_dtype)
    if ed == "bf16":
        return ["--bf16"]
    if ed == "fp16":
        return ["--fp16"]
    if ed == "fp8":
        return ["--fp8"]
    return []


def fused_input_output_io_format_strings(manifest: Dict[str, Any]) -> Tuple[str, str]:
    """
    Return (input_io_formats, output_io_formats) comma-separated for trtexec.
    """
    talker = manifest.get("talker") or {}
    nl = int(talker.get("num_layers", 28))
    c2w = manifest.get("code2wav_fused") or {}
    c2w_in = list(c2w.get("c2w_state_input_names") or [])
    c2w_out = list(c2w.get("c2w_state_output_names") or [])
    if not c2w_in or not c2w_out:
        raise ValueError("manifest missing code2wav_fused I/O name lists")

    raw_io = manifest.get("triton_io_float_dtype") or manifest.get("onnx_io_dtype") or "fp32"
    ft = _normalize_float_io_token(str(raw_io))
    fp_spec = f"{ft}:chw"
    i64 = "int64:chw"

    # Inputs: input_embeds, position_ids, attention_bias, past_seq_lens, cache_position, past_kv_* x 2*nl, c2w_*
    in_parts: List[str] = [fp_spec, i64, fp_spec, i64, i64]
    in_parts.extend([fp_spec] * (2 * nl))
    in_parts.extend([fp_spec] * len(c2w_in))

    # Outputs: wav, codec_sum, full_codec, hidden, logits, present_kv x 2*nl, c2w outs
    out_parts: List[str] = [fp_spec, fp_spec, i64, fp_spec, fp_spec]
    out_parts.extend([fp_spec] * (2 * nl))
    out_parts.extend([fp_spec] * len(c2w_out))

    return ",".join(in_parts), ",".join(out_parts)


def load_manifest(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "manifest",
        type=Path,
        help="Path to triton_manifest.json (variant export dir)",
    )
    p.add_argument(
        "--emit",
        choices=("input", "output", "prec", "all"),
        default="all",
        help="Print one value or all three lines (input, output, prec argv)",
    )
    args = p.parse_args()
    m = load_manifest(args.manifest)

    inp, out = fused_input_output_io_format_strings(m)
    prec_list = trtexec_precision_args(str(m.get("engine_dtype", "bf16")))
    prec_str = " ".join(prec_list) if prec_list else ""

    if args.emit == "input":
        print(inp, end="")
    elif args.emit == "output":
        print(out, end="")
    elif args.emit == "prec":
        print(prec_str, end="")
    else:
        print(inp)
        print(out)
        print(prec_str)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Update Triton manifest engine profile metadata.

Phase B owns TensorRT build-time limits.  Export writes an architecture
manifest before engines exist; this helper records the actual TRT profile used
by build_engines.sh so deploy/runtime can validate requested limits early.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _normalize_dtype(value: str) -> str:
    raw = (value or "").strip().lower()
    aliases = {
        "bfloat16": "bf16",
        "float16": "fp16",
        "float32": "fp32",
        "float8": "fp8",
    }
    return aliases.get(raw, raw)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def update_manifest(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))

    engine_dtype = _normalize_dtype(args.engine_dtype)
    triton_io_dtype = _normalize_dtype(args.triton_io_float_dtype or engine_dtype)

    if args.engine_mode:
        data["engine_mode"] = args.engine_mode
    data["engine_dtype"] = engine_dtype
    data["triton_io_float_dtype"] = triton_io_dtype

    architecture = data.setdefault("architecture", {})
    if isinstance(architecture, dict):
        architecture["dtype"] = engine_dtype

    profile = data.setdefault("engine_profile", {})
    if not isinstance(profile, dict):
        profile = {}
        data["engine_profile"] = profile

    profile.update(
        {
            "profile_schema_version": 1,
            "builder": args.builder,
            "engine_mode": args.engine_mode or data.get("engine_mode", "trt"),
            "engine_dtype": engine_dtype,
            "triton_io_float_dtype": triton_io_dtype,
            "max_batch_size": int(args.max_batch_size),
            "max_input_len": int(args.max_input_len),
            "max_seq_len": int(args.max_seq_len),
            "builder_image": args.builder_image or "",
            "target_driver": args.target_driver or "",
        }
    )
    if not args.skip_built_at:
        profile["built_at_utc"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    else:
        profile.pop("built_at_utc", None)

    _write_json(manifest_path, data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Path to triton_manifest.json")
    parser.add_argument("--engine-mode", default="trt", choices=("onnx", "trt"))
    parser.add_argument("--engine-dtype", required=True, help="bf16|fp16|fp32|fp8")
    parser.add_argument("--triton-io-float-dtype", default="", help="Default: same as --engine-dtype")
    parser.add_argument("--max-batch-size", type=_positive_int, required=True)
    parser.add_argument("--max-input-len", type=_positive_int, required=True)
    parser.add_argument("--max-seq-len", type=_positive_int, required=True)
    parser.add_argument("--builder", default="trtexec")
    parser.add_argument("--builder-image", default="")
    parser.add_argument("--target-driver", default="")
    parser.add_argument("--skip-built-at", action="store_true")
    args = parser.parse_args()
    update_manifest(args)


if __name__ == "__main__":
    main()

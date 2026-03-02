#!/usr/bin/env python3
"""
Inspect TRT-LLM engine I/O tensors (for TalkerRunner implementation).

Run inside NGC TRT-LLM container to list all input/output tensor names,
shapes, and dtypes. Used to decide between GenerationSession vs raw TRT.

Usage (in container):
  python3 /mnt/scripts/inspect_trtllm_engine.py --engine-dir /mnt/model/trtllm_engine
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import tensorrt as trt
except ImportError:
    print("ERROR: tensorrt not found. Run inside NGC TRT-LLM container.", file=sys.stderr)
    sys.exit(1)

# Load TRT-LLM plugins so custom ops (Gemm, etc.) can be deserialized
try:
    import tensorrt_llm
    tensorrt_llm.plugin.init_all_plugins()
except Exception:
    try:
        from tensorrt_llm.plugin import _load_plugin_lib
        _load_plugin_lib()
    except Exception:
        print("WARNING: Could not load TRT-LLM plugins", file=sys.stderr)


def inspect_engine(engine_path: str) -> dict:
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        return {"error": "Failed to deserialize engine"}

    result = {
        "num_layers": engine.num_layers,
        "num_io_tensors": engine.num_io_tensors,
        "inputs": [],
        "outputs": [],
    }

    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        shape = tuple(engine.get_tensor_shape(name))
        dtype = engine.get_tensor_dtype(name)
        dtype_str = str(dtype).replace("DataType.", "") if dtype is not None else "unknown"
        entry = {"name": name, "shape": list(shape), "dtype": dtype_str}
        if mode == trt.TensorIOMode.INPUT:
            result["inputs"].append(entry)
        else:
            result["outputs"].append(entry)

    return result


def main():
    parser = argparse.ArgumentParser(description="Inspect TRT-LLM engine I/O")
    parser.add_argument("--engine-dir", default="/mnt/model/trtllm_engine", help="Directory containing rank0.engine")
    parser.add_argument("--json", action="store_true", help="Output JSON only")
    args = parser.parse_args()

    engine_dir = Path(args.engine_dir)
    engine_path = engine_dir / "rank0.engine"
    if not engine_path.exists():
        print(f"ERROR: Engine not found: {engine_path}", file=sys.stderr)
        sys.exit(1)

    data = inspect_engine(str(engine_path))
    if "error" in data:
        print(f"ERROR: {data['error']}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(data, indent=2))
        return

    print("=" * 60)
    print("  TRT-LLM Engine I/O (raw TensorRT bindings)")
    print("=" * 60)
    print(f"  num_layers: {data['num_layers']}")
    print(f"  num_io_tensors: {data['num_io_tensors']}")
    print()
    print("  INPUTS:")
    for t in data["inputs"]:
        print(f"    {t['name']}: shape={t['shape']} dtype={t['dtype']}")
    print()
    print("  OUTPUTS:")
    for t in data["outputs"]:
        print(f"    {t['name']}: shape={t['shape']} dtype={t['dtype']}")
    print("=" * 60)


if __name__ == "__main__":
    main()

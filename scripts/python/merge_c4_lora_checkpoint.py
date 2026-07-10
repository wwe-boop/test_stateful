#!/usr/bin/env python3
"""Merge a PEFT LoRA adapter into a Qwen3-TTS checkpoint.

This intentionally saves by copying the base config/tokenizer files and writing
the merged state dict directly. Some Qwen3-TTS config objects fail in
Transformers' diff-based ``save_pretrained`` path because of custom dtype keys.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from safetensors.torch import save_file

try:
    from qwen_tts import Qwen3TTSModel
except ImportError:
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    qwen_root = repo_root.parent / "Qwen3-TTS"
    if qwen_root.exists():
        sys.path.insert(0, str(qwen_root))
    from qwen_tts import Qwen3TTSModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--note", default="")
    return parser.parse_args()


def torch_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def copy_package_files(base: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for src in base.iterdir():
        if src.name in {"model.safetensors", "pytorch_model.bin"}:
            continue
        dst = out / src.name
        if src.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        elif src.is_file():
            shutil.copy2(src, dst)


def main() -> int:
    args = parse_args()
    if args.out_dir.exists():
        shutil.rmtree(args.out_dir)

    print(f"[merge] load base: {args.base_model}", flush=True)
    tts = Qwen3TTSModel.from_pretrained(
        str(args.base_model),
        device_map=args.device_map,
        dtype=torch_dtype(args.dtype),
    )
    print(f"[merge] load adapter: {args.adapter_dir}", flush=True)
    peft_model = PeftModel.from_pretrained(tts.model, str(args.adapter_dir)).eval()
    print("[merge] merge_and_unload", flush=True)
    merged = peft_model.merge_and_unload()

    print(f"[merge] copy package files: {args.out_dir}", flush=True)
    copy_package_files(args.base_model, args.out_dir)
    print("[merge] save merged model.safetensors", flush=True)
    state = {key: value.detach().cpu().contiguous() for key, value in merged.state_dict().items()}
    save_file(state, str(args.out_dir / "model.safetensors"), metadata={"format": "pt"})

    meta = {
        "base_model": str(args.base_model),
        "adapter_dir": str(args.adapter_dir),
        "dtype": args.dtype,
        "merge": "peft.merge_and_unload",
        "note": args.note,
    }
    (args.out_dir / "c4_merge_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("[merge] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Generate C4 target segments through the official CustomVoice wrapper."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import soundfile as sf
import torch
from peft import PeftModel

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from qwen_tts import Qwen3TTSModel
except ImportError:
    # The official PyTorch package is often a sibling checkout of this Triton repo.
    qwen_root = REPO_ROOT.parent / "Qwen3-TTS"
    if qwen_root.exists():
        sys.path.insert(0, str(qwen_root))
    from qwen_tts import Qwen3TTSModel


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-jsonl", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--target-segment-index", type=int, default=1)
    parser.add_argument("--speaker", default="001")
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--subtalker-dosample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--subtalker-top-k", type=int, default=50)
    parser.add_argument("--subtalker-top-p", type=float, default=1.0)
    parser.add_argument("--subtalker-temperature", type=float, default=0.9)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows = read_jsonl(args.manifest_jsonl)[: args.limit]
    tts = Qwen3TTSModel.from_pretrained(str(args.model_dir), device_map="cuda:0", dtype=torch.bfloat16)
    if args.adapter_dir:
        tts.model = PeftModel.from_pretrained(tts.model, str(args.adapter_dir)).eval()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    items = []
    for index, row in enumerate(rows):
        sample_id = str(row["sample_id"])
        segment = row["segments"][args.target_segment_index]
        wavs, sample_rate = tts.generate_custom_voice(
            text=segment["text"],
            speaker=args.speaker,
            language=args.language,
            do_sample=args.do_sample,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            subtalker_dosample=args.subtalker_dosample,
            subtalker_top_k=args.subtalker_top_k,
            subtalker_top_p=args.subtalker_top_p,
            subtalker_temperature=args.subtalker_temperature,
            max_new_tokens=args.max_new_tokens,
        )
        sample_dir = args.out_dir / f"{index:03d}_{sample_id}"
        sample_dir.mkdir(exist_ok=True)
        wav_path = sample_dir / "official_single.wav"
        sf.write(str(wav_path), wavs[0], sample_rate)
        item = {
            "key": f"{args.variant}:official_single:{sample_id}",
            "variant": args.variant,
            "generation_mode": "official_single",
            "sample_id": sample_id,
            "speaker": args.speaker,
            "language": args.language,
            "seed": args.seed,
            "wav_path": str(wav_path),
            "reference": segment["text"],
            "seconds": len(wavs[0]) / sample_rate,
            "sample_rate": sample_rate,
            "target_segment_index": args.target_segment_index,
            "target_frames": len(segment.get("codes") or []),
        }
        items.append(item)
        print(json.dumps(item, ensure_ascii=False), flush=True)

    payload = {
        "variant": args.variant,
        "model_dir": str(args.model_dir),
        "adapter_dir": str(args.adapter_dir) if args.adapter_dir else None,
        "manifest_jsonl": str(args.manifest_jsonl),
        "items": items,
    }
    (args.out_dir / "asr_manifest.json").write_text(
        json.dumps({"rows": items}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

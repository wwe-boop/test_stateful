#!/usr/bin/env python3
"""Generate test-prosody-mini samples with DashScope or Volcengine Ark."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from eval.data_synth.llm_client import LLMClient, load_llm_config
from eval.data_synth.prompts import (
    DEFAULT_INSTRUCTS,
    DEFAULT_SPEAKERS,
    SCENARIOS,
    SYSTEM_PROMPT,
    build_user_prompt,
)
from eval.data_synth.punct import boundary_punct_classes, count_chinese_chars, full_text, normalize_segments


def _cycle(items: list[str], index: int) -> str:
    return items[index % len(items)]


def _validate_sample(
    payload: dict[str, Any],
    *,
    sample_index: int,
    scenario: str,
    speaker: str,
    instruct: str,
    min_segments: int,
    max_segments: int,
    provider: str,
    model: str,
) -> dict[str, Any]:
    segments = normalize_segments(payload.get("segments", []))
    if not (min_segments <= len(segments) <= max_segments):
        raise ValueError(
            f"segment count {len(segments)} outside [{min_segments}, {max_segments}]"
        )

    chinese_chars = count_chinese_chars(full_text(segments))
    if chinese_chars < 40:
        raise ValueError(f"too few Chinese characters: {chinese_chars}")

    sample_id = str(payload.get("sample_id") or f"prosody_mini_{sample_index:03d}")
    return {
        "sample_id": sample_id,
        "scenario": str(payload.get("scenario") or scenario),
        "language": str(payload.get("language") or "Chinese"),
        "speaker": str(payload.get("speaker") or speaker),
        "instruct": str(payload.get("instruct") or instruct),
        "segments": segments,
        "boundary_punct_classes": boundary_punct_classes(segments),
        "full_text": full_text(segments),
        "review_status": "llm_generated",
        "metadata": {
            "provider": provider,
            "model": model,
            "generated_at_unix": int(time.time()),
        },
    }


def generate_one_sample(
    client: LLMClient,
    *,
    sample_index: int,
    scenario: str,
    speaker: str,
    instruct: str,
    min_segments: int,
    max_segments: int,
    max_retries: int,
    temperature: float,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            user_prompt = build_user_prompt(
                sample_index=sample_index,
                scenario=scenario,
                speaker=speaker,
                instruct=instruct,
                min_segments=min_segments,
                max_segments=max_segments,
            )
            payload = client.generate_json(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=temperature,
            )
            return _validate_sample(
                payload,
                sample_index=sample_index,
                scenario=scenario,
                speaker=speaker,
                instruct=instruct,
                min_segments=min_segments,
                max_segments=max_segments,
                provider=client.config.provider,
                model=client.config.model,
            )
        except Exception as exc:  # noqa: BLE001 - retry on any generation/validation failure
            last_error = exc
            print(
                f"[warn] sample {sample_index:03d} attempt {attempt}/{max_retries} failed: {exc}",
                file=sys.stderr,
            )
            time.sleep(min(attempt, 3))
    assert last_error is not None
    raise last_error


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_manifest(path: Path, rows: list[dict[str, Any]], *, provider: str, model: str) -> None:
    manifest = {
        "dataset": "test-prosody-mini",
        "count": len(rows),
        "provider": provider,
        "model": model,
        "scenarios": list(SCENARIOS.keys()),
        "samples": [
            {
                "sample_id": row["sample_id"],
                "scenario": row["scenario"],
                "speaker": row["speaker"],
                "segment_count": len(row["segments"]),
            }
            for row in rows
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=50, help="Number of samples to generate")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "workspace" / "datasets" / "test-prosody-mini.jsonl")
    parser.add_argument("--manifest", type=Path, default=None, help="Optional manifest JSON path")
    parser.add_argument("--provider", choices=["dashscope", "ark"], default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-segments", type=int, default=6)
    parser.add_argument("--max-segments", type=int, default=12)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--start-index", type=int, default=1, help="First sample index (for resume)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.count <= 0:
        raise SystemExit("--count must be > 0")
    if args.min_segments > args.max_segments:
        raise SystemExit("--min-segments must be <= --max-segments")

    rng = random.Random(args.seed)
    scenario_names = list(SCENARIOS.keys())
    config = load_llm_config(args.provider)
    client = LLMClient(config)

    rows: list[dict[str, Any]] = []
    for offset in range(args.count):
        sample_index = args.start_index + offset
        scenario = _cycle(scenario_names, offset)
        speaker = _cycle(DEFAULT_SPEAKERS, offset)
        instruct = _cycle(DEFAULT_INSTRUCTS, offset + rng.randint(0, 2))

        print(
            f"[info] generating {sample_index:03d} scenario={scenario} speaker={speaker}",
            file=sys.stderr,
        )
        row = generate_one_sample(
            client,
            sample_index=sample_index,
            scenario=scenario,
            speaker=speaker,
            instruct=instruct,
            min_segments=args.min_segments,
            max_segments=args.max_segments,
            max_retries=args.max_retries,
            temperature=args.temperature,
        )
        rows.append(row)

    write_jsonl(args.output, rows)
    manifest_path = args.manifest or args.output.with_suffix(".manifest.json")
    write_manifest(manifest_path, rows, provider=config.provider, model=config.model)

    print(f"[done] wrote {len(rows)} samples -> {args.output}", file=sys.stderr)
    print(f"[done] manifest -> {manifest_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

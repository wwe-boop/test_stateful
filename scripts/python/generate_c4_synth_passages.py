#!/usr/bin/env python3
"""Generate SteadyStream C4 synthetic full-passage text JSONL."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from eval.data_synth.llm_client import LLMClient, load_llm_config  # noqa: E402
from eval.data_synth.prompts_c4 import (  # noqa: E402
    DEFAULT_INSTRUCTS,
    SCENARIOS,
    SYSTEM_PROMPT,
    build_user_prompt,
)
from eval.data_synth.punct import (  # noqa: E402
    boundary_punct_classes,
    count_chinese_chars,
    full_text,
    normalize_segments,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def collect_forbidden_texts(paths: list[Path]) -> set[str]:
    forbidden: set[str] = set()
    for path in paths:
        for row in read_jsonl(path):
            texts = [
                row.get("text"),
                row.get("full_text"),
                row.get("prompt"),
            ]
            segments = row.get("segments") or []
            if isinstance(segments, list):
                texts.append(full_text(normalize_segments(segments)))
                for seg in segments:
                    if isinstance(seg, dict):
                        texts.append(seg.get("text"))
            for text in texts:
                if isinstance(text, str) and text.strip():
                    forbidden.add(text.strip())
    return forbidden


def has_overlap(text: str, segments: list[dict[str, str]], forbidden: set[str]) -> bool:
    if text in forbidden:
        return True
    return any(str(seg.get("text") or "").strip() in forbidden for seg in segments)


def validate_sample(
    payload: dict[str, Any],
    *,
    sample_index: int,
    scenario: str,
    instruct: str,
    min_segments: int,
    max_segments: int,
    min_chars: int,
    max_chars: int,
    forbidden_texts: set[str],
    provider: str,
    model: str,
) -> dict[str, Any]:
    segments = normalize_segments(payload.get("segments", []))
    if not (min_segments <= len(segments) <= max_segments):
        raise ValueError(f"segment count {len(segments)} outside [{min_segments}, {max_segments}]")
    puncts = boundary_punct_classes(segments)
    if len(set(puncts)) < 2:
        raise ValueError("needs at least two punctuation classes")
    text = full_text(segments)
    char_count = count_chinese_chars(text)
    if not (min_chars <= char_count <= max_chars):
        raise ValueError(f"Chinese char count {char_count} outside [{min_chars}, {max_chars}]")
    if has_overlap(text, segments, forbidden_texts):
        raise ValueError("text overlaps with forbidden eval/test source list")

    return {
        "sample_id": f"c4_synth_{sample_index:05d}",
        "speaker": "001",
        "language": str(payload.get("language") or "Chinese"),
        "text": text,
        "segments": segments,
        "boundary_punct_classes": puncts,
        "scenario": str(payload.get("scenario") or scenario),
        "instruct": str(payload.get("instruct") or instruct),
        "metadata": {
            "source": "llm_c4_synth_passage_v1",
            "provider": provider,
            "model": model,
            "generated_at_unix": int(time.time()),
            "chinese_chars": char_count,
        },
    }


def generate_batch(
    client: LLMClient,
    *,
    next_index: int,
    batch_size: int,
    scenario: str,
    instruct: str,
    min_segments: int,
    max_segments: int,
    min_chars: int,
    max_chars: int,
    forbidden_texts: set[str],
    temperature: float,
) -> list[dict[str, Any]]:
    payload = client.generate_json(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=build_user_prompt(
            start_index=next_index,
            batch_size=batch_size,
            scenario=scenario,
            instruct=instruct,
            min_segments=min_segments,
            max_segments=max_segments,
            min_chars=min_chars,
            max_chars=max_chars,
        ),
        temperature=temperature,
    )
    samples = payload.get("samples")
    if not isinstance(samples, list):
        raise ValueError("LLM response missing list field: samples")

    rows: list[dict[str, Any]] = []
    for offset, sample in enumerate(samples):
        if not isinstance(sample, dict):
            print("[warn] skipped non-object sample entry", file=sys.stderr)
            continue
        try:
            rows.append(
                validate_sample(
                    sample,
                    sample_index=next_index + offset,
                    scenario=scenario,
                    instruct=instruct,
                    min_segments=min_segments,
                    max_segments=max_segments,
                    min_chars=min_chars,
                    max_chars=max_chars,
                    forbidden_texts=forbidden_texts,
                    provider=client.config.provider,
                    model=client.config.model,
                )
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] skipped invalid sample entry: {exc}", file=sys.stderr)
    return rows


def write_summary(path: Path, rows: list[dict[str, Any]], args: argparse.Namespace, client: LLMClient) -> None:
    scenario_counts: dict[str, int] = {}
    punct_counts: dict[str, int] = {}
    char_counts: list[int] = []
    segment_counts: list[int] = []
    for row in rows:
        scenario_counts[row["scenario"]] = scenario_counts.get(row["scenario"], 0) + 1
        segment_counts.append(len(row["segments"]))
        char_counts.append(int(row["metadata"]["chinese_chars"]))
        for punct in row["boundary_punct_classes"]:
            punct_counts[punct] = punct_counts.get(punct, 0) + 1
    summary = {
        "output": str(args.output),
        "count": len(rows),
        "speaker": "001",
        "provider": client.config.provider,
        "model": client.config.model,
        "seed": args.seed,
        "scenario_counts": scenario_counts,
        "punct_counts": punct_counts,
        "segment_count_min": min(segment_counts) if segment_counts else None,
        "segment_count_max": max(segment_counts) if segment_counts else None,
        "char_count_min": min(char_counts) if char_counts else None,
        "char_count_max": max(char_counts) if char_counts else None,
        "forbidden_sources": [str(p) for p in args.eval_source_list],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "workspace/datasets/c4_synth_passages_v1.jsonl")
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--provider", choices=["dashscope", "ark"], default=None)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--min-segments", type=int, default=4)
    parser.add_argument("--max-segments", type=int, default=12)
    parser.add_argument("--min-chars", type=int, default=80)
    parser.add_argument("--max-chars", type=int, default=250)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--max-attempts", type=int, default=300)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--eval-source-list",
        type=Path,
        action="append",
        default=[
            REPO_ROOT / "workspace/datasets/test-prosody-mini.jsonl",
            REPO_ROOT
            / "workspace/c4_wenet_premium0_nll20_20260709/c4_wenet_manifest_idx2_min20_limit20_with_codes.jsonl",
        ],
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.count <= 0:
        raise SystemExit("--count must be > 0")
    rng = random.Random(args.seed)
    config = load_llm_config(args.provider)
    client = LLMClient(config)
    forbidden_texts = collect_forbidden_texts(args.eval_source_list)

    rows = read_jsonl(args.output) if args.resume else []
    seen_texts = {str(row.get("text") or "") for row in rows}
    scenario_names = list(SCENARIOS.keys())
    attempts = 0

    while len(rows) < args.count and attempts < args.max_attempts:
        attempts += 1
        next_index = len(rows) + 1
        scenario = scenario_names[(next_index - 1) % len(scenario_names)]
        instruct = DEFAULT_INSTRUCTS[(next_index - 1 + rng.randint(0, 2)) % len(DEFAULT_INSTRUCTS)]
        todo = min(args.batch_size, args.count - len(rows))
        try:
            batch = generate_batch(
                client,
                next_index=next_index,
                batch_size=todo,
                scenario=scenario,
                instruct=instruct,
                min_segments=args.min_segments,
                max_segments=args.max_segments,
                min_chars=args.min_chars,
                max_chars=args.max_chars,
                forbidden_texts=forbidden_texts | seen_texts,
                temperature=args.temperature,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] attempt {attempts}/{args.max_attempts} failed: {exc}", file=sys.stderr)
            time.sleep(min(attempts, 5))
            continue

        accepted = 0
        for row in batch:
            text = row["text"]
            if text in seen_texts:
                print(f"[warn] duplicate skipped: {row['sample_id']}", file=sys.stderr)
                continue
            row["sample_id"] = f"c4_synth_{len(rows) + 1:05d}"
            rows.append(row)
            seen_texts.add(text)
            accepted += 1
        write_jsonl(args.output, rows)
        print(f"[info] accepted {accepted}/{len(batch)}; total={len(rows)}/{args.count}", file=sys.stderr)

    if len(rows) < args.count:
        raise RuntimeError(f"only generated {len(rows)} rows after {attempts} attempts")

    summary_path = args.summary or args.output.with_suffix(".summary.json")
    write_summary(summary_path, rows, args, client)
    print(f"[done] wrote {len(rows)} passages -> {args.output}", file=sys.stderr)
    print(f"[done] summary -> {summary_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

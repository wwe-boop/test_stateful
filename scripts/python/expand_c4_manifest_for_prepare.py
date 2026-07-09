#!/usr/bin/env python3
"""Expand a C4 continuation manifest into segment rows for prepare_data.py."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def normalized_chars(text: str) -> int:
    return sum(1 for ch in str(text or "") if not ch.isspace())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def absolutize(path: str) -> str:
    return path if path.startswith("/") else str(Path(path).resolve())


def select_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in read_jsonl(args.manifest_jsonl):
        segments = row.get("segments") or []
        if len(segments) <= args.target_segment_index:
            continue
        target_text = str(segments[args.target_segment_index].get("text") or "")
        if normalized_chars(target_text) < args.min_target_chars:
            continue
        out = dict(row)
        out["segments"] = segments[: args.target_segment_index + 1]
        meta = dict(out.get("meta") or {})
        meta["selection_target_segment_index"] = args.target_segment_index
        meta["selection_min_target_chars"] = args.min_target_chars
        out["meta"] = meta
        selected.append(out)
        if args.limit and len(selected) >= args.limit:
            break
    if not selected:
        raise ValueError("no rows matched selection filters")
    return selected


def expand_segments(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        if not sample_id:
            raise ValueError("manifest row missing sample_id")
        speaker_name = row.get("speaker_name")
        language = row.get("language", "Chinese")
        segments = row.get("segments") or []
        for segment_index, segment in enumerate(segments):
            audio = segment.get("audio")
            text = segment.get("text")
            if not audio or text is None:
                raise ValueError(f"{sample_id}:{segment_index}: missing audio/text")
            audio_path = absolutize(str(audio))
            expanded.append(
                {
                    "sample_id": sample_id,
                    "segment_index": segment_index,
                    "audio": audio_path,
                    "text": text,
                    "ref_audio": audio_path,
                    "speaker_name": speaker_name,
                    "language": language,
                    "pause_ms": segment.get("pause_ms"),
                    "punct_class": segment.get("punct_class"),
                    "source_utterance_id": segment.get("source_utterance_id"),
                }
            )
    return expanded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-jsonl", type=Path, required=True)
    parser.add_argument("--selected-manifest-jsonl", type=Path, required=True)
    parser.add_argument("--prepare-jsonl", type=Path, required=True)
    parser.add_argument("--target-segment-index", type=int, default=2)
    parser.add_argument("--min-target-chars", type=int, default=20)
    parser.add_argument("--limit", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = select_rows(args)
    expanded = expand_segments(rows)
    write_jsonl(args.selected_manifest_jsonl, rows)
    write_jsonl(args.prepare_jsonl, expanded)
    print(
        json.dumps(
            {
                "selected_samples": len(rows),
                "prepare_rows": len(expanded),
                "target_segment_index": args.target_segment_index,
                "min_target_chars": args.min_target_chars,
                "selected_manifest_jsonl": str(args.selected_manifest_jsonl),
                "prepare_jsonl": str(args.prepare_jsonl),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

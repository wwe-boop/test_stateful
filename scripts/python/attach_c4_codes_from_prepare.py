#!/usr/bin/env python3
"""Attach official prepare_data audio_codes to a C4 continuation manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


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


def build_code_map(prepared_rows: list[dict[str, Any]]) -> dict[tuple[str, int], Any]:
    code_map: dict[tuple[str, int], Any] = {}
    for row in prepared_rows:
        sample_id = str(row.get("sample_id") or "")
        segment_index = row.get("segment_index")
        if not sample_id:
            raise ValueError("prepared row missing sample_id")
        if not isinstance(segment_index, int):
            raise ValueError(f"{sample_id}: prepared row missing integer segment_index")
        codes = row.get("audio_codes")
        if not isinstance(codes, list) or not codes:
            raise ValueError(f"{sample_id}:{segment_index}: prepared row missing audio_codes")
        code_map[(sample_id, segment_index)] = codes
    return code_map


def attach_codes(
    manifest_rows: list[dict[str, Any]],
    prepared_rows: list[dict[str, Any]],
    *,
    codes_source: str,
) -> list[dict[str, Any]]:
    code_map = build_code_map(prepared_rows)
    output_rows: list[dict[str, Any]] = []
    for row in manifest_rows:
        sample_id = str(row.get("sample_id") or "")
        if not sample_id:
            raise ValueError("manifest row missing sample_id")
        segments = row.get("segments")
        if not isinstance(segments, list):
            raise ValueError(f"{sample_id}: manifest row missing segments")
        for index, segment in enumerate(segments):
            key = (sample_id, index)
            if key not in code_map:
                raise ValueError(f"{sample_id}:{index}: no prepared audio_codes")
            if not isinstance(segment, dict):
                raise ValueError(f"{sample_id}:{index}: segment must be an object")
            segment["codes"] = code_map[key]
        meta = row.setdefault("meta", {})
        if not isinstance(meta, dict):
            raise ValueError(f"{sample_id}: meta must be an object when present")
        meta["codes_source"] = codes_source
        meta["training_ready_smoke"] = True
        output_rows.append(row)
    return output_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-jsonl", type=Path, required=True)
    parser.add_argument("--prepared-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = attach_codes(
        read_jsonl(args.manifest_jsonl),
        read_jsonl(args.prepared_jsonl),
        codes_source=str(args.prepared_jsonl),
    )
    write_jsonl(args.output_jsonl, rows)
    print(f"[done] wrote {len(rows)} rows -> {args.output_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

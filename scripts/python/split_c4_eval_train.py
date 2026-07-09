#!/usr/bin/env python3
"""Create a frozen C4 eval split and a disjoint train pool."""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--eval-jsonl", type=Path, required=True)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--eval-size", type=int, default=20)
    parser.add_argument("--target-segment-index", type=int, default=1)
    parser.add_argument("--sample-id-key", default="sample_id")
    args = parser.parse_args()

    rows = read_jsonl(args.input_jsonl)
    unique: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for row in rows:
        sample_id = str(row.get(args.sample_id_key) or "")
        if not sample_id:
            raise ValueError(f"row missing {args.sample_id_key}: {row}")
        unique.setdefault(sample_id, row)

    unique_rows = list(unique.values())
    if len(unique_rows) < args.eval_size:
        raise ValueError(f"not enough unique rows for eval_size={args.eval_size}: {len(unique_rows)}")

    eval_ids = {str(row[args.sample_id_key]) for row in unique_rows[-args.eval_size :]}
    eval_rows = [row for row in unique_rows if str(row[args.sample_id_key]) in eval_ids]
    train_rows = [row for row in unique_rows if str(row[args.sample_id_key]) not in eval_ids]

    write_jsonl(args.eval_jsonl, eval_rows)
    write_jsonl(args.train_jsonl, train_rows)
    summary = {
        "source": str(args.input_jsonl),
        "total_rows": len(rows),
        "unique_sample_ids": len(unique_rows),
        "duplicate_rows_dropped_for_split": len(rows) - len(unique_rows),
        "eval_rows": len(eval_rows),
        "train_pool_rows": len(train_rows),
        "eval_sample_ids": [row[args.sample_id_key] for row in eval_rows],
        "target_segment_index": args.target_segment_index,
        "selection": f"last_{args.eval_size}_unique_sample_ids_frozen_eval_train_pool_excludes_eval_ids",
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

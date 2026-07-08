#!/usr/bin/env python3
"""Generate test-boundary JSONL (programmatic by default, optional LLM)."""

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

from eval.data_synth.boundary_builder import (
    QUOTA_PROFILES,
    SCENARIO_QUOTAS,
    generate_samples,
    iter_quota_plan,
)
from eval.data_synth.prompts_boundary import SYSTEM_PROMPT, build_user_prompt
from eval.data_synth.punct import count_chinese_chars
from eval.segmentation.advantage_set import DATASET_DESCRIPTION, write_dataset_meta
from eval.segmentation.ref_splitter import reference_boundaries
from eval.segmentation.tokenize import text_to_segment_tokens


def _validate_record(item: dict[str, Any], *, min_chars: int) -> dict[str, Any]:
    full_text = str(item.get("full_text", "")).strip()
    if len(full_text) < 10:
        raise ValueError("full_text too short")
    if count_chinese_chars(full_text) < min_chars:
        raise ValueError(f"too few Chinese chars: {count_chinese_chars(full_text)} < {min_chars}")

    sample_id = str(item.get("sample_id", "")).strip()
    scenario = str(item.get("scenario_tags", item.get("scenario", ""))).strip()
    if not sample_id:
        raise ValueError("missing sample_id")
    if not scenario:
        raise ValueError("missing scenario_tags")

    return {
        "sample_id": sample_id,
        "scenario_tags": scenario,
        "full_text": full_text,
        "review_status": item.get("review_status", "generated"),
        "metadata": item.get("metadata", {}),
    }


def _annotate_evaluable(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    annotated: list[dict[str, Any]] = []
    for record in records:
        tokens = text_to_segment_tokens(record["full_text"])
        ref = reference_boundaries(tokens)
        annotated.append(
            {
                **record,
                "n_tokens": len(tokens),
                "n_ref_boundaries": len(ref.boundaries),
                "ref_boundaries": ref.boundaries,
                "evaluable": bool(ref.boundaries),
            }
        )
    return annotated


def generate_programmatic(count: int, *, seed: int, profile: str = "full") -> list[dict[str, Any]]:
    rows = generate_samples(count, seed=seed, profile=profile)
    for row in rows:
        row["review_status"] = "programmatic"
        row["metadata"] = {"generator": "boundary_builder", "seed": seed, "profile": profile}
    return rows


def generate_llm(
    count: int,
    *,
    seed: int,
    max_retries: int,
    temperature: float,
    profile: str = "full",
) -> list[dict[str, Any]]:
    from eval.data_synth.llm_client import LLMClient, load_llm_config

    client = LLMClient(load_llm_config())
    rng = random.Random(seed)
    quotas = QUOTA_PROFILES[profile]
    scenarios = iter_quota_plan(count, seed=seed, quotas=quotas)
    rows: list[dict[str, Any]] = []

    for idx, scenario in enumerate(scenarios, start=1):
        last_error: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                payload = client.generate_json(
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=build_user_prompt(sample_index=idx, scenario=scenario),
                    temperature=temperature,
                )
                record = _validate_record(
                    {
                        "sample_id": payload.get("sample_id", f"boundary_{idx:04d}"),
                        "scenario_tags": payload.get("scenario_tags", scenario),
                        "full_text": payload.get("full_text", ""),
                        "review_status": "llm_generated",
                        "metadata": {
                            "provider": client.provider,
                            "model": client.model,
                            "attempt": attempt,
                            "generated_at_unix": int(time.time()),
                        },
                    },
                    min_chars=40,
                )
                rows.append(record)
                break
            except Exception as exc:  # noqa: BLE001 - retry surface for operator
                last_error = exc
                rng.shuffle(scenarios)
        else:
            raise RuntimeError(f"LLM generation failed for sample {idx}: {last_error}") from last_error
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=200, help="Number of samples (default 200)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--profile",
        choices=tuple(QUOTA_PROFILES.keys()),
        default="full",
        help="full=test-boundary mix; advantage=KV/force stress only",
    )
    parser.add_argument(
        "--mode",
        choices=("programmatic", "llm"),
        default="programmatic",
        help="programmatic=offline templates; llm=DashScope/Volcengine",
    )
    parser.add_argument(
        "--out",
        default=str(REPO_ROOT / "workspace" / "datasets" / "test-boundary.jsonl"),
        help="Output JSONL path",
    )
    parser.add_argument("--annotate", action="store_true", help="Attach ref boundary stats (requires tokenizer)")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.profile == "advantage" and args.count == 200 and "--count" not in sys.argv:
        args.count = sum(QUOTA_PROFILES["advantage"].values())

    if args.mode == "programmatic":
        records = generate_programmatic(args.count, seed=args.seed, profile=args.profile)
    else:
        records = generate_llm(
            args.count,
            seed=args.seed,
            max_retries=args.max_retries,
            temperature=args.temperature,
            profile=args.profile,
        )

    if args.annotate:
        records = _annotate_evaluable(records)

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    evaluable = sum(1 for r in records if r.get("evaluable"))
    quotas = QUOTA_PROFILES[args.profile]
    if args.profile == "advantage":
        write_dataset_meta(
            out_path.parent,
            extra={
                "n_samples": len(records),
                "n_evaluable": evaluable,
                "seed": args.seed,
                "output_jsonl": str(out_path),
            },
        )
    if args.annotate:
        print(f"Wrote {len(records)} samples → {out_path} ({evaluable} evaluable) [profile={args.profile}]")
    else:
        print(f"Wrote {len(records)} samples → {out_path} [profile={args.profile}]")
        print(f"Scenarios: {', '.join(f'{k}={v}' for k, v in quotas.items())}")


if __name__ == "__main__":
    main()

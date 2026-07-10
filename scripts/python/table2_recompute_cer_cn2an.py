#!/usr/bin/env python3
"""Recompute Table 2 CER from saved ASR hypotheses with cn2an normalization."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


def compute_cer(reference: str, hypothesis: str) -> float:
    ref = list(reference)
    hyp = list(hypothesis)
    rows = len(ref) + 1
    cols = len(hyp) + 1
    dp = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        dp[i][0] = i
    for j in range(cols):
        dp[0][j] = j
    for i in range(1, rows):
        for j in range(1, cols):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[-1][-1] / len(ref) if ref else 0.0


def normalize_zh_for_cer(text: str) -> str:
    text = re.sub(r"\s+", "", str(text))
    text = re.sub(
        r"[，。！？、；：“”‘’（）【】《》…,.!?;:'\"()\[\]<>—\-]",
        "",
        text,
    )
    try:
        import cn2an

        text = cn2an.transform(text, "cn2an")
    except Exception:
        pass
    return text.lower()


def mean_std(vals: list[float]) -> tuple[float | None, float | None]:
    if not vals:
        return None, None
    mean = sum(vals) / len(vals)
    var = sum((x - mean) ** 2 for x in vals) / len(vals)
    return mean, math.sqrt(var)


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    seed_rows = []
    by_variant_seed: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        by_variant_seed.setdefault((row["variant"], int(row["seed"])), []).append(row)
    for (variant, seed), items in sorted(by_variant_seed.items()):
        vals = [float(item["cer"]) for item in items]
        mean, std = mean_std(vals)
        seed_rows.append(
            {
                "variant": variant,
                "seed": seed,
                "n_samples": len(items),
                "cer_mean": round(mean, 6) if mean is not None else None,
                "cer_sample_std": round(std, 6) if std is not None else None,
                "cer_max": round(max(vals), 6) if vals else None,
                "max_key": max(items, key=lambda x: float(x["cer"])).get("key") if vals else None,
            }
        )
    by_variant: dict[str, list[dict[str, Any]]] = {}
    for row in seed_rows:
        by_variant.setdefault(row["variant"], []).append(row)
    summary = {}
    for variant, items in sorted(by_variant.items()):
        seed_means = [float(item["cer_mean"]) for item in items if item.get("cer_mean") is not None]
        mean, std = mean_std(seed_means)
        all_rows = [row for row in rows if row["variant"] == variant]
        vals = [float(row["cer"]) for row in all_rows]
        worst = max(all_rows, key=lambda x: float(x["cer"])) if all_rows else None
        summary[variant] = {
            "n_seeds": len(items),
            "n_samples_per_seed": [item["n_samples"] for item in items],
            "cer_mean": round(mean, 6) if mean is not None else None,
            "cer_seed_std": round(std, 6) if std is not None else None,
            "cer_max": round(max(vals), 6) if vals else None,
            "cer_over_25pct_count": sum(1 for x in vals if x > 0.25),
            "worst_key": worst.get("key") if worst else None,
        }
    return {"seed_rows": seed_rows, "summary": summary}


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# E57 CER cn2an Recompute",
        "",
        "CER is recomputed from saved ASR hypotheses with punctuation/space stripping and cn2an numeral normalization.",
        "",
        "| Variant | CER mean | seed std | max CER | >25% samples | worst key |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for variant, item in payload["summary"].items():
        lines.append(
            f"| `{variant}` | {pct(item.get('cer_mean'))} | {pct(item.get('cer_seed_std'))} | "
            f"{pct(item.get('cer_max'))} | {item.get('cer_over_25pct_count')} | `{item.get('worst_key')}` |"
        )
    return "\n".join(lines) + "\n"


def pct(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value) * 100:.2f}%"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-json", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()

    raw = json.loads(Path(args.raw_json).read_text(encoding="utf-8"))
    rows = []
    for item in raw.get("rows", []):
        ref_norm = normalize_zh_for_cer(item.get("reference", ""))
        hyp_norm = normalize_zh_for_cer(item.get("hypothesis_raw", ""))
        cer = compute_cer(ref_norm, hyp_norm)
        rows.append(
            {
                **item,
                "reference_norm": ref_norm,
                "hypothesis_norm": hyp_norm,
                "cer": round(cer, 6),
                "ref_chars": len(ref_norm),
                "hyp_chars": len(hyp_norm),
            }
        )
    payload = {
        "source": args.raw_json,
        "normalization_note": "punct/space strip + cn2an.transform(cn2an) + lower",
        "n_rows": len(rows),
        "rows": rows,
        **aggregate(rows),
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.out_md).write_text(render_md(payload), encoding="utf-8")
    print(Path(args.out_md).read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

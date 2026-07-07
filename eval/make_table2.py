#!/usr/bin/env python3
"""Auto-generate Table 2 from experiment results per steadystream_plan_v2.md §0.1.

Usage:
    python make_table2.py --results-dir workspace/experiments/ --output table2.md

Input structure (per variant):
    results_dir/
      ├── stateless/
      │   ├── seed_42/metrics.json
      │   ├── seed_123/metrics.json
      │   └── seed_456/metrics.json
      ├── stateful_triton/
      │   └── ...
      ├── acoustic_tail_only/
      │   └── ...
      └── ...

Each metrics.json contains:
{
  "excess_f0_mean_st": float,
  "excess_energy_mean_db": float,
  "pause_deviation_mean_ms": float,
  "sim_delta_mean": float,
  "fasl_mean_ms": float,
  "cer_mean": float
}

Output: Markdown table with mean±std across seeds, Wilcoxon test vs baseline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats


VARIANT_ORDER = [
    "stateless",
    "stateful_triton",
    "acoustic_tail_only",
    "kv_token_tail_only",
    "tail_kv_pause_recovery",
    "full_steadystream",
]

VARIANT_LABELS = {
    "stateless": "无状态",
    "stateful_triton": "现有 stateful（Triton）†",
    "acoustic_tail_only": "仅声学尾",
    "kv_token_tail_only": "仅 KV/token 尾",
    "tail_kv_pause_recovery": "尾 +KV+ 暂停恢复",
    "full_steadystream": "完整 SteadyStream",
}

COLUMNS = [
    ("excess_f0_mean_st", "F0 跳变↓"),
    ("excess_energy_mean_db", "能量↓"),
    ("pause_deviation_mean_ms", "停顿↓"),
    ("sim_delta_mean", "SIM Δ↓"),
    ("fasl_mean_ms", "FASL↓"),
    ("cer_mean", "CER↓"),
]


def load_variant_results(variant_dir: Path) -> list[dict[str, float]]:
    """Load all seed results for a variant."""
    results = []
    for seed_dir in sorted(variant_dir.glob("seed_*")):
        metrics_file = seed_dir / "metrics.json"
        if not metrics_file.exists():
            continue
        with open(metrics_file) as f:
            results.append(json.load(f))
    return results


def aggregate_seeds(results: list[dict[str, float]], metric_key: str) -> tuple[float, float]:
    """Compute mean±std across seeds."""
    values = [r[metric_key] for r in results if metric_key in r and r[metric_key] is not None]
    if not values:
        return (float('nan'), float('nan'))
    return (float(np.mean(values)), float(np.std(values)))


def wilcoxon_test(baseline: list[float], treatment: list[float]) -> str:
    """Run Wilcoxon signed-rank test and return significance marker."""
    if len(baseline) != len(treatment) or len(baseline) < 3:
        return ""
    try:
        stat, p_value = stats.wilcoxon(baseline, treatment, alternative='greater')
        if p_value < 0.01:
            return "**"
        elif p_value < 0.05:
            return "*"
    except Exception:
        pass
    return ""


def find_best_in_column(variant_data: dict[str, tuple[float, float]], metric_key: str) -> str | None:
    """Find variant with best (lowest) mean for this metric."""
    valid_variants = {
        v: mean for v, (mean, std) in variant_data.items()
        if not np.isnan(mean)
    }
    if not valid_variants:
        return None
    return min(valid_variants, key=valid_variants.get)


def format_cell(mean: float, std: float, is_best: bool = False, sig_marker: str = "") -> str:
    """Format table cell with mean±std."""
    if np.isnan(mean):
        return "–"

    cell = f"{mean:.2f}±{std:.2f}"
    if is_best:
        cell = f"**{cell}**"
    if sig_marker:
        cell += sig_marker

    return cell


def generate_table2(results_dir: Path) -> str:
    """Generate Table 2 markdown."""
    # Load all variants
    all_data = {}
    for variant_name in VARIANT_ORDER:
        variant_dir = results_dir / variant_name
        if not variant_dir.exists():
            continue
        results = load_variant_results(variant_dir)
        if results:
            all_data[variant_name] = results

    if not all_data:
        return "No results found."

    # Build table
    lines = []
    lines.append("# 表 2：有界继承状态下的跨片段边界稳定性（test-prosody）")
    lines.append("")

    # Header
    header = "| 变体 |"
    separator = "|---|"
    for _, col_label in COLUMNS:
        header += f" {col_label} |"
        separator += "---|"
    lines.append(header)
    lines.append(separator)

    # Get baseline data for significance testing
    baseline_name = "stateless"
    baseline_data = {}
    if baseline_name in all_data:
        for metric_key, _ in COLUMNS:
            baseline_data[metric_key] = [
                r[metric_key] for r in all_data[baseline_name]
                if metric_key in r and r[metric_key] is not None
            ]

    # Data rows
    for variant_name in VARIANT_ORDER:
        if variant_name not in all_data:
            continue

        results = all_data[variant_name]
        row_label = VARIANT_LABELS.get(variant_name, variant_name)
        row = f"| {row_label} |"

        for metric_key, _ in COLUMNS:
            mean, std = aggregate_seeds(results, metric_key)

            # Check if best in column
            column_data = {
                v: aggregate_seeds(all_data[v], metric_key)
                for v in all_data.keys()
            }
            best_variant = find_best_in_column(column_data, metric_key)
            is_best = (best_variant == variant_name)

            # Significance test vs baseline
            sig_marker = ""
            if variant_name != baseline_name and metric_key in baseline_data:
                treatment_values = [
                    r[metric_key] for r in results
                    if metric_key in r and r[metric_key] is not None
                ]
                if len(treatment_values) == len(baseline_data[metric_key]):
                    sig_marker = wilcoxon_test(baseline_data[metric_key], treatment_values)

            cell = format_cell(mean, std, is_best, sig_marker)
            row += f" {cell} |"

        lines.append(row)

    # Footer notes
    lines.append("")
    lines.append("† v2.1 新增行：Triton 服务现有的 stateful clause stream 路径作为"服务端现状"对照。")
    lines.append("")
    lines.append("单位：F0 跳变（semitone）、能量（dB）、停顿（ms 偏差）、SIM Δ（相似度损失，×100）、FASL（ms）、CER（%）。")
    lines.append("所有格 = 3 种子均值±std；每列最优加粗；对"无状态"行做 Wilcoxon 显著性标记（* p<0.05, ** p<0.01）。")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True, help="Directory with variant subdirs")
    parser.add_argument("--output", type=Path, default=None, help="Output markdown file (default: stdout)")
    args = parser.parse_args()

    table_md = generate_table2(args.results_dir)

    if args.output:
        args.output.write_text(table_md, encoding="utf-8")
        print(f"Table 2 written to {args.output}")
    else:
        print(table_md)


if __name__ == "__main__":
    main()

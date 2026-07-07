#!/usr/bin/env python3
"""Auto-generate Table 4 (concurrency stress) from run summaries.

Usage:
    python eval/make_table4.py \\
        --results-dir workspace/table4_runs \\
        --output workspace/table4.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

CONCURRENCY_ORDER = [1, 8, 16, 32, 64, 128]
VARIANT_ORDER = ["pad_baseline", "stateful_triton", "full_steadystream"]
VARIANT_LABELS = {
    "pad_baseline": "PAD 基线 (path iv)",
    "stateful_triton": "现有 stateful（Triton）",
    "full_steadystream": "完整 SteadyStream",
}


def _load_run_summaries(results_dir: Path, variant: str, concurrency: int) -> list[dict[str, Any]]:
    runs = []
    pattern = f"{variant}_c{concurrency}_seed*"
    for run_dir in sorted(results_dir.glob(pattern)):
        summary_path = run_dir / "run_summary.json"
        if not summary_path.exists():
            continue
        with open(summary_path, encoding="utf-8") as f:
            data = json.load(f)
        agg = data.get("aggregate") or {}
        agg["_run_dir"] = str(run_dir)
        runs.append(agg)
    return runs


def _mean_std_across_seeds(runs: list[dict[str, Any]], metric: str, stat: str = "mean") -> tuple[float | None, float | None]:
    vals = []
    for run in runs:
        block = run.get(metric) or {}
        v = block.get(stat)
        if v is not None:
            vals.append(float(v))
    if not vals:
        return None, None
    arr = np.asarray(vals, dtype=np.float64)
    return round(float(np.mean(arr)), 2), round(float(np.std(arr)), 2)


def _format_cell(mean: float | None, std: float | None) -> str:
    if mean is None:
        return "–"
    if std is None or std == 0:
        return f"{mean:.1f}"
    return f"{mean:.1f}±{std:.1f}"


def _p50_p95_cell(runs: list[dict[str, Any]], metric: str) -> str:
    p50_vals, p95_vals = [], []
    for run in runs:
        block = run.get(metric) or {}
        if block.get("p50") is not None:
            p50_vals.append(block["p50"])
        if block.get("p95") is not None:
            p95_vals.append(block["p95"])
    if not p50_vals:
        return "–"
    p50 = np.mean(p50_vals)
    p95 = np.mean(p95_vals) if p95_vals else p50
    return f"{p50:.0f}/{p95:.0f}"


def _batch_xrt_cell(runs: list[dict[str, Any]]) -> str:
    vals = [r.get("batch_x_rt") for r in runs if r.get("batch_x_rt") is not None]
    if not vals:
        xrt_m, xrt_s = _mean_std_across_seeds(runs, "x_rt")
        return _format_cell(xrt_m, xrt_s)
    return f"{np.mean(vals):.2f}"


def generate_table4_markdown(results_dir: Path, variant: str = "stateful_triton") -> str:
    lines = [
        "# 表 4：SteadyStream 并发下用户感知稳定性",
        "",
        f"变体：**{VARIANT_LABELS.get(variant, variant)}** ｜ 数据来源：`{results_dir}`",
        "",
        "口径：FASL = 首段上游文本 → VAD 首有声；TTFT/首包 = session init → 首 PCM（客户端）；"
        "RTF = 单路墙钟/音频时长；×RT = 批次总音频/批次墙钟。",
        "",
        "| 并发 | FASL↓ | Jitter↓ | 卡顿↓(%) | TTFT P50/P95 | RTF↓ | ×RT↑ | 成功 |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]

    for c in CONCURRENCY_ORDER:
        runs = _load_run_summaries(results_dir, variant, c)
        if not runs:
            lines.append(f"| {c} | – | – | – | – | – | – | – |")
            continue
        fasl_m, fasl_s = _mean_std_across_seeds(runs, "fasl_vad_ms")
        jit_m, jit_s = _mean_std_across_seeds(runs, "jitter_p95_ms")
        stu_m, stu_s = _mean_std_across_seeds(runs, "stutter_rate_pct")
        ttft = _p50_p95_cell(runs, "ttft_ms")
        rtf_m, rtf_s = _mean_std_across_seeds(runs, "rtf")
        xrt = _batch_xrt_cell(runs)
        n_ok = int(np.mean([r.get("n_ok", 0) for r in runs]))
        lines.append(
            f"| {c} | {_format_cell(fasl_m, fasl_s)} | {_format_cell(jit_m, jit_s)} | "
            f"{_format_cell(stu_m, stu_s)} | {ttft} | {_format_cell(rtf_m, rtf_s)} | {xrt} | {n_ok}/{c} |"
        )

    lines.extend(["", "## 表 4b：上游停顿时长分层", ""])
    lines.append("| 分层 | 并发 | FASL | 卡顿(%) | vs PAD(pp) |")
    lines.append("|---|---:|---:|---:|---:|")

    for strat in ("normal", "long_pause"):
        label = "全量" if strat == "normal" else "停顿>1s"
        for c in [1, 8, 128]:
            pad_runs = _load_run_summaries(results_dir, "pad_baseline", c)
            cand_runs = _load_run_summaries(results_dir, variant, c)
            if not cand_runs:
                continue
            fasl_m, _ = _mean_std_across_seeds(cand_runs, "fasl_vad_ms")
            stu_m, _ = _mean_std_across_seeds(cand_runs, "stutter_rate_pct")
            pad_stu, _ = _mean_std_across_seeds(pad_runs, "stutter_rate_pct")
            delta = (stu_m - pad_stu) if (stu_m is not None and pad_stu is not None) else None
            delta_s = f"{delta:+.2f}" if delta is not None else "–"
            lines.append(
                f"| {label} | {c} | {_format_cell(fasl_m, None)} | {_format_cell(stu_m, None)} | {delta_s} |"
            )

    lines.append("")
    lines.append("> 3 seeds/档；完整曲线见补充材料。server_ttft_ms 写入 run_summary 供附录对照。")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, help="Root dir containing variant_cN_seedM run folders")
    parser.add_argument("--variant", default="stateful_triton", choices=VARIANT_ORDER)
    parser.add_argument("--output", default="", help="Output markdown path")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    md = generate_table4_markdown(results_dir, variant=args.variant)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        print(f"Table 4 written to {out}")
    else:
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

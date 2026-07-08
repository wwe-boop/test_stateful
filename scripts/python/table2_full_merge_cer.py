#!/usr/bin/env python3
"""Merge Table 2 CER output into the no-CER metric payload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


VARIANTS = {
    "stateless_once": "无状态",
    "stateful_stream": "现有 stateful（engine clause stream）†",
    "acoustic_tail_only": "仅声学尾（C1 prototype）",
    "kv_tail_only": "仅 KV/token 尾（C2 prototype）",
    "tail_kv_pause_recovery": "尾 +KV+ 暂停恢复（C1+C2+C3 prototype）",
    "full_steadystream": "完整 SteadyStream（C1+C2+C3, C4未训练）",
}


def fmt(mean: float | None, std: float | None, digits: int = 2) -> str:
    if mean is None:
        return "—"
    return f"{mean:.{digits}f}±{(std or 0.0):.{digits}f}"


def pct(mean: float | None, std: float | None) -> str:
    if mean is None:
        return "—"
    return f"{mean * 100:.2f}%±{(std or 0.0) * 100:.2f}%"


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# 表 2 当前结果（三种子，test-prosody-mini）",
        "",
        "口径：F0/能量为 raw boundary jump；SIM Δ 为同样本 offline_full 直接音色对比 `(1 - MFCC cosine) × 100`；CER 为 Paraformer-zh + 字符级 Levenshtein。",
        "",
        "| 变体 | F0 raw↓ | 能量 raw↓ | 停顿↓ | SIM Δ↓ | FASL↓ | CER↓ | 覆盖率/备注 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for variant in (
        "stateless_once",
        "stateful_stream",
        "acoustic_tail_only",
        "kv_tail_only",
        "tail_kv_pause_recovery",
        "full_steadystream",
    ):
        s = summary.get(variant, {})
        note = "—"
        if s:
            note = (
                f"F0 cov {s.get('f0_coverage'):.2f}; "
                f"E cov {s.get('energy_coverage'):.2f}; "
                f"pause cov {s.get('pause_coverage'):.2f}"
            )
            if variant != "stateless_once":
                note += (
                    f"; exact events {int(s.get('exact_boundary_count_total') or 0)}; "
                    f"proxy samples {int(s.get('proxy_boundary_samples_total') or 0)}"
                )
        lines.append(
            f"| {VARIANTS[variant]} | "
            f"{fmt(s.get('f0_jump_mean_st'), s.get('f0_jump_mean_st_seed_std'))} st | "
            f"{fmt(s.get('energy_jump_mean_db'), s.get('energy_jump_mean_db_seed_std'))} dB | "
            f"{fmt(s.get('pause_deviation_mean_ms'), s.get('pause_deviation_mean_ms_seed_std'))} ms | "
            f"{fmt(s.get('sim_delta_x100'), s.get('sim_delta_x100_seed_std'))} | "
            f"{fmt(s.get('fasl_mean_ms'), s.get('fasl_mean_ms_seed_std'))} ms | "
            f"{pct(s.get('cer_mean'), s.get('cer_seed_std'))} | {note} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-json", default="workspace/table2_runs/table2_metrics_no_cer.json")
    parser.add_argument("--cer-json", default="workspace/table2_runs/table2_cer.json")
    parser.add_argument("--output-json", default="workspace/table2_runs/table2_current.json")
    parser.add_argument("--output-md", default="workspace/table2_runs/table2_current.md")
    args = parser.parse_args()

    metrics = json.loads(Path(args.metrics_json).read_text(encoding="utf-8"))
    cer = json.loads(Path(args.cer_json).read_text(encoding="utf-8"))
    for variant, csum in cer.get("summary", {}).items():
        if variant not in metrics["summary"]:
            continue
        metrics["summary"][variant]["cer_mean"] = csum.get("cer_mean")
        metrics["summary"][variant]["cer_seed_std"] = csum.get("cer_seed_std")
    metrics["cer"] = {
        "normalization_note": cer.get("normalization_note"),
        "seed_rows": cer.get("seed_rows", []),
        "summary": cer.get("summary", {}),
    }
    metrics["metric_scope"] = "Raw F0/energy, same-voice MFCC SIM, Paraformer CER."
    Path(args.output_json).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.output_md).write_text(render_markdown(metrics), encoding="utf-8")
    print(Path(args.output_md).read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

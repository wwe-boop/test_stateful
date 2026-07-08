#!/usr/bin/env python3
"""Generate the current Table 2 delivery note with C4 guardrails.

This script is intentionally conservative: it reports the measured prototype
rows and records C4 smoke-readiness evidence, but it never turns synthetic C4
smoke artifacts into a Full SteadyStream result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


VARIANT_ORDER = [
    "stateless_once",
    "stateful_stream",
    "acoustic_tail_only",
    "kv_tail_only",
    "tail_kv_pause_recovery",
]

METRICS = [
    ("f0_jump_mean_st", "f0_jump_mean_st_seed_std", "F0 raw↓", " st", 1.0),
    ("energy_jump_mean_db", "energy_jump_mean_db_seed_std", "能量 raw↓", " dB", 1.0),
    ("pause_deviation_mean_ms", "pause_deviation_mean_ms_seed_std", "停顿↓", " ms", 1.0),
    ("sim_delta_x100", "sim_delta_x100_seed_std", "SIM Δ↓", "", 1.0),
    ("fasl_mean_ms", "fasl_mean_ms_seed_std", "FASL↓", " ms", 1.0),
    ("cer_mean", "cer_seed_std", "CER↓", "%", 100.0),
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_metric(row: dict[str, Any], mean_key: str, std_key: str, suffix: str, scale: float) -> str:
    mean = row.get(mean_key)
    std = row.get(std_key)
    if mean is None:
        return "—"
    mean = float(mean) * scale
    std = 0.0 if std is None else float(std) * scale
    return f"{mean:.2f}±{std:.2f}{suffix}"


def coverage_note(row: dict[str, Any]) -> str:
    parts = []
    if "f0_coverage" in row:
        parts.append(f"F0 cov {float(row['f0_coverage']):.2f}")
    if "energy_coverage" in row:
        parts.append(f"E cov {float(row['energy_coverage']):.2f}")
    if "pause_coverage" in row:
        parts.append(f"pause cov {float(row['pause_coverage']):.2f}")
    exact = row.get("exact_boundary_count_total")
    proxy = row.get("proxy_boundary_samples_total")
    if exact:
        parts.append(f"exact events {int(exact)}")
    if proxy:
        parts.append(f"proxy samples {int(proxy)}")
    return "; ".join(parts) if parts else "—"


def loss_span(summary: dict[str, Any]) -> str:
    history = summary.get("loss_history") or []
    if not history:
        return "—"
    first = history[0].get("combined_loss")
    last = history[-1].get("combined_loss")
    if first is None or last is None:
        return "—"
    return f"{float(first):.6f} -> {float(last):.6f}"


def render_table2(table2: dict[str, Any]) -> list[str]:
    summary = table2.get("summary") or {}
    lines = [
        "| 变体 | F0 raw↓ | 能量 raw↓ | 停顿↓ | SIM Δ↓ | FASL↓ | CER↓ | 覆盖率/备注 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for variant in VARIANT_ORDER:
        row = summary.get(variant)
        if not row:
            continue
        cells = [str(row.get("label") or variant)]
        cells.extend(
            fmt_metric(row, metric[0], metric[1], metric[3], metric[4])
            for metric in METRICS
        )
        cells.append(coverage_note(row))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("| 完整 SteadyStream | — | — | — | — | — | — | 未测：没有真实 C4 checkpoint，synthetic smoke 不进正式行 |")
    return lines


def render_delivery(
    *,
    table2: dict[str, Any],
    c4_validation: dict[str, Any] | None,
    c4_lora: dict[str, Any] | None,
) -> str:
    lines: list[str] = [
        "# Table 2 Current Delivery",
        "",
        "Date: 2026-07-08",
        "",
        "Scope: Table 2 only, `test-prosody-mini`, remote repo `/home/zehan/workspace/Qwen3-TTS-Triton`.",
        "",
        "## Delivery Verdict",
        "",
        "- Current measured rows: `5/6`.",
        "- Full SteadyStream C4 row: `not measured`.",
        "- Reason: no real continuation-trained C4 checkpoint has been found or deployed.",
        "- Guardrail: API-synthetic C4 smoke validates plumbing only; it is not final Table 2 evidence.",
        "",
        "## Current Table 2",
        "",
        "口径：F0/能量为 raw boundary jump；SIM Δ 为同样本 offline_full 直接音色对比 `(1 - MFCC cosine) × 100`；CER 为 Paraformer-zh + 字符级 Levenshtein。",
        "",
    ]
    lines.extend(render_table2(table2))
    lines.extend(
        [
            "",
            "## CER Diagnosis",
            "",
            "- C2/C3 prototype CER regression is real early-EOS / missing-text behavior, not an ASR aggregation artifact.",
            "- Likely cause: cropped Talker KV tail lacks a verified sink/prefix/position/EOS contract.",
            "- Pause recovery can reduce pause deviation, but it cannot repair semantic continuation once C2 destabilizes EOS.",
            "",
        ]
    )

    if c4_validation:
        lines.extend(
            [
                "## C4 Smoke Readiness",
                "",
                "| Item | Value |",
                "|---|---:|",
                f"| synthetic samples | {int(c4_validation.get('samples', 0))} |",
                f"| segments | {int(c4_validation.get('segments', 0))} |",
                f"| coded segments | {int(c4_validation.get('coded_segments', 0))} |",
                f"| codec frames | {int(c4_validation.get('code_frames', 0))} |",
                f"| speech seconds from codes | {float(c4_validation.get('speech_seconds_from_codes', 0.0)):.2f} |",
                f"| schema issues | {int(c4_validation.get('issue_count', 0))} |",
                "",
            ]
        )

    if c4_lora:
        loaded = c4_lora.get("loaded_adapter") or {}
        lines.extend(
            [
                "## C4 LoRA Smoke",
                "",
                "| Item | Value |",
                "|---|---:|",
                f"| model | `{c4_lora.get('model_dir', '—')}` |",
                f"| manifest rows | {int(c4_lora.get('manifest_rows', 0))} |",
                f"| steps | {int(c4_lora.get('steps', 0))} |",
                f"| schedule | `epochs={c4_lora.get('epochs')}`, `shuffle={c4_lora.get('shuffle')}`, `seed={c4_lora.get('seed')}` |",
                f"| trainable params | {int(c4_lora.get('trainable_params', 0)):,} / {int(c4_lora.get('total_params', 0)):,} |",
                f"| loaded adapter tensors | {int(loaded.get('loaded_tensor_count', 0))} |",
                f"| loaded adapter params | {int(loaded.get('loaded_param_count', 0)):,} |",
                f"| loss span | `{loss_span(c4_lora)}` |",
                "",
                "This smoke proves that the C4 continuation batch can be optimized, saved, restored, and continued on the 0.6B base model. It does not prove Full SteadyStream quality.",
                "",
            ]
        )

    lines.extend(
        [
            "## Remaining Hard Requirements For A Real Full Result",
            "",
            "1. Provide or build a real long-recording continuation dataset with strict train/eval source isolation.",
            "2. Train a C4 checkpoint/adapter compatible with the deployed CustomVoice model.",
            "3. Deploy that checkpoint in the same C1+C2+C3 inference path.",
            "4. Re-run Table 2 on `test-prosody-mini` and then the frozen `test-prosody` set.",
            "5. Only then fill the Full SteadyStream row.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table2-json", type=Path, default=Path("workspace/table2_runs_impl_20260708/table2_current.json"))
    parser.add_argument("--c4-validation-json", type=Path, default=Path("workspace/c4_synthetic_smoke5_20260708/c4_manifest_validation_summary.json"))
    parser.add_argument("--c4-lora-json", type=Path, default=Path("workspace/c4_synthetic_smoke5_20260708/c4_lora_smoke5_resume2_summary.json"))
    parser.add_argument("--output", type=Path, default=Path("workspace/table2_runs_impl_20260708/table2_current_delivery_20260708.md"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    table2 = read_json(args.table2_json)
    c4_validation = read_json(args.c4_validation_json) if args.c4_validation_json.exists() else None
    c4_lora = read_json(args.c4_lora_json) if args.c4_lora_json.exists() else None
    output = render_delivery(table2=table2, c4_validation=c4_validation, c4_lora=c4_lora)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output + "\n", encoding="utf-8")
    print(f"[done] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

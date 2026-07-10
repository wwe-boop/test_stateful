#!/usr/bin/env python3
"""Aggregate E57 Table 2 stability metrics.

This is the formal stability companion to the E56 F0 gate repair. It compares
each streaming variant to the same offline logical boundaries, uses nearest
voiced F0 windows, and reports C1/ICL mechanism coverage.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.python.table2_f0_paired_calibration import (  # noqa: E402
    attach_nearest_voiced,
    metric_map,
    nearest_f0_map,
    offline_boundaries_from_asr,
    pause_vs_offline,
    read_audio,
    values,
    variant_block_from_wav,
)


VARIANTS = [
    "stateful_stream",
    "c4_icl_prefill_c3",
    "full_steadystream_icl_c3",
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def mean(values_in: list[Any]) -> float | None:
    vals = [safe_float(x) for x in values_in]
    vals = [x for x in vals if x is not None]
    if not vals:
        return None
    return round(float(np.mean(vals)), 6)


def mean_std(values_in: list[Any]) -> tuple[float | None, float | None, int]:
    vals = [safe_float(x) for x in values_in]
    vals = [x for x in vals if x is not None]
    if not vals:
        return None, None, 0
    arr = np.asarray(vals, dtype=np.float64)
    return round(float(arr.mean()), 6), round(float(arr.std()), 6), len(vals)


def event_true_count(block: dict[str, Any], key: str) -> int:
    meta = block.get("event_meta_summary") or {}
    item = meta.get(key) or {}
    return int(item.get("true_count") or 0)


def event_values(block: dict[str, Any], key: str) -> dict[str, int]:
    meta = block.get("event_meta_summary") or {}
    item = meta.get(key) or {}
    vals = item.get("values") or {}
    return {str(k): int(v) for k, v in vals.items()}


def collect_values(blocks: list[dict[str, Any]], item_key: str, metric: str) -> list[float]:
    out: list[float] = []
    for block in blocks:
        for item in block.get(item_key, []) or []:
            val = safe_float(item.get(metric))
            if val is not None:
                out.append(val)
    return out


def paired_diffs(
    offline_block: dict[str, Any],
    variant_block: dict[str, Any],
    *,
    nearest_f0: bool,
) -> tuple[list[float], list[float]]:
    if nearest_f0:
        om = nearest_f0_map(offline_block)
        vm = nearest_f0_map(variant_block)
    else:
        om = metric_map(offline_block, "energy_jump_db")
        vm = metric_map(variant_block, "energy_jump_db")
    diffs: list[float] = []
    for idx in sorted(set(om) & set(vm)):
        diffs.append(vm[idx] - om[idx])
    return diffs, [abs(x) for x in diffs]


def summarize_variant(blocks: list[dict[str, Any]], offline_blocks: list[dict[str, Any]]) -> dict[str, Any]:
    n_boundaries = sum(len(b.get("boundary_metrics", []) or []) for b in blocks)
    raw_f0 = collect_values(blocks, "boundary_metrics", "f0_jump_st")
    nearest_f0 = collect_values(blocks, "nearest_voiced_boundary_metrics", "f0_jump_st")
    energy = collect_values(blocks, "boundary_metrics", "energy_jump_db")
    pause_dev = collect_values(blocks, "pause_metrics", "deviation_ms")
    pause_offline_dev = collect_values(blocks, "pause_offline_items", "deviation_ms")
    flux_vals = [
        safe_float((b.get("boundary_artifact") or {}).get("flux_peak_mean"))
        for b in blocks
    ]
    flux_vals = [x for x in flux_vals if x is not None]
    delta_vals = [
        safe_float((b.get("boundary_artifact") or {}).get("max_sample_delta_mean"))
        for b in blocks
    ]
    delta_vals = [x for x in delta_vals if x is not None]
    fasl_vals = [safe_float((b.get("timing") or {}).get("fasl_first_audio_ms")) for b in blocks]
    total_ms_vals = [safe_float((b.get("timing") or {}).get("total_ms")) for b in blocks]
    audio_sec_vals = [safe_float(b.get("audio_sec")) for b in blocks]

    f0_diffs: list[float] = []
    f0_abs_diffs: list[float] = []
    energy_diffs: list[float] = []
    energy_abs_diffs: list[float] = []
    for off, block in zip(offline_blocks, blocks):
        d, ad = paired_diffs(off, block, nearest_f0=True)
        f0_diffs.extend(d)
        f0_abs_diffs.extend(ad)
        d, ad = paired_diffs(off, block, nearest_f0=False)
        energy_diffs.extend(d)
        energy_abs_diffs.extend(ad)

    exact = sum(int(b.get("exact_boundary_count") or 0) for b in blocks)
    expected = sum(int(b.get("boundary_expected_count") or len(b.get("boundary_metrics", []) or [])) for b in blocks)
    bad_mode = sum(
        1
        for b in blocks
        if b.get("stream_input_mode") != "token" or b.get("stream_group_policy") != "none"
    )
    acoustic_tail = sum(event_true_count(b, "steadystream_acoustic_tail") for b in blocks)
    icl_history = sum(event_true_count(b, "steadystream_token_history_full_current") for b in blocks)
    generation_budget = []
    trimmed = []
    for b in blocks:
        generation_budget.extend(int(k) for k, v in event_values(b, "steadystream_token_history_generation_budget_frames").items() for _ in range(v))
        trimmed.extend(int(k) for k, v in event_values(b, "steadystream_token_history_code_frames_trimmed_for_generation_budget").items() for _ in range(v))

    rtf_vals = []
    for total_ms, audio_sec in zip(total_ms_vals, audio_sec_vals):
        if total_ms is not None and audio_sec is not None and audio_sec > 0:
            rtf_vals.append(total_ms / (audio_sec * 1000.0))

    return {
        "n_samples": len(blocks),
        "n_boundaries": n_boundaries,
        "exact_boundary_count": exact,
        "boundary_expected_count": expected,
        "exact_boundary_ratio": round(exact / expected, 6) if expected else None,
        "bad_input_mode_samples": bad_mode,
        "acoustic_tail_true_count": acoustic_tail,
        "icl_history_true_count": icl_history,
        "acoustic_tail_ratio": round(acoustic_tail / expected, 6) if expected else None,
        "icl_history_ratio": round(icl_history / expected, 6) if expected else None,
        "raw_f0": values(raw_f0),
        "raw_f0_coverage": round(len(raw_f0) / n_boundaries, 6) if n_boundaries else None,
        "nearest_f0": values(nearest_f0),
        "nearest_f0_coverage": round(len(nearest_f0) / n_boundaries, 6) if n_boundaries else None,
        "paired_nearest_f0_signed": values(f0_diffs),
        "paired_nearest_f0_abs": values(f0_abs_diffs),
        "energy": values(energy),
        "energy_coverage": round(len(energy) / n_boundaries, 6) if n_boundaries else None,
        "paired_energy_signed": values(energy_diffs),
        "paired_energy_abs": values(energy_abs_diffs),
        "pause_punctuation": values(pause_dev),
        "pause_offline": values(pause_offline_dev),
        "artifact_flux_peak_mean": mean(flux_vals),
        "artifact_max_sample_delta_mean": mean(delta_vals),
        "fasl_first_audio_ms": values([x for x in fasl_vals if x is not None]),
        "rtf": values(rtf_vals),
        "audio_sec": values([x for x in audio_sec_vals if x is not None]),
        "generation_budget_frames": values(generation_budget),
        "trimmed_for_generation_budget_frames": values(trimmed),
    }


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# E57 Table 2 Stability",
        "",
        "Scope: 3 seeds x eval20, token input mode, group_policy=none, exact logical boundaries only.",
        "",
        "| Variant | exact | C1 tail | ICL | nearest F0 mean | paired F0 signed median | paired F0 abs median | energy mean | pause punct | pause offline | artifact flux | FASL | RTF |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        s = payload["summary"][variant]
        lines.append(
            f"| `{variant}` | {s['exact_boundary_count']}/{s['boundary_expected_count']} | "
            f"{s['acoustic_tail_true_count']}/{s['boundary_expected_count']} | "
            f"{s['icl_history_true_count']}/{s['boundary_expected_count']} | "
            f"{fmt(s['nearest_f0'].get('mean'))} st | "
            f"{fmt(s['paired_nearest_f0_signed'].get('median'))} st | "
            f"{fmt(s['paired_nearest_f0_abs'].get('median'))} st | "
            f"{fmt(s['energy'].get('mean'))} dB | "
            f"{fmt(s['pause_punctuation'].get('mean'))} ms | "
            f"{fmt(s['pause_offline'].get('mean'))} ms | "
            f"{fmt(s.get('artifact_flux_peak_mean'))} | "
            f"{fmt(s['fasl_first_audio_ms'].get('mean'))} ms | "
            f"{fmt(s['rtf'].get('mean'))} |"
        )
    lines.extend(["", "## By Seed", ""])
    lines.append("| seed | variant | nearest F0 mean | paired F0 signed median | energy mean | pause offline | artifact flux |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|")
    for row in payload["seed_rows"]:
        s = row["summary"]
        lines.append(
            f"| {row['seed']} | `{row['variant']}` | "
            f"{fmt(s['nearest_f0'].get('mean'))} | "
            f"{fmt(s['paired_nearest_f0_signed'].get('median'))} | "
            f"{fmt(s['energy'].get('mean'))} | "
            f"{fmt(s['pause_offline'].get('mean'))} | "
            f"{fmt(s.get('artifact_flux_peak_mean'))} |"
        )
    return "\n".join(lines) + "\n"


def fmt(value: Any) -> str:
    val = safe_float(value)
    if val is None:
        return "-"
    return f"{val:.3f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--offline-root", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    offline_root = Path(args.offline_root)
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)

    offline_cache: dict[str, dict[str, Any]] = {}
    offline_alignment: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    seed_variant_blocks: dict[tuple[int, str], list[dict[str, Any]]] = {}
    seed_variant_offline: dict[tuple[int, str], list[dict[str, Any]]] = {}

    for result_path in sorted(runs_root.glob("seed_*/*/results.json")):
        data = load_json(result_path)
        sample_id = str(data["sample_id"])
        seed = int(data["seed"])
        if sample_id not in offline_cache:
            off_dir = offline_root / "seed_42" / sample_id
            off_data = load_json(off_dir / "results.json")
            off_wav = off_dir / "offline_full.wav"
            _, off_sr = read_audio(off_wav)
            boundaries, align = offline_boundaries_from_asr(off_wav, off_data["segments"], off_sr)
            offline_alignment[sample_id] = align
            off_block = variant_block_from_wav(
                off_wav,
                boundaries,
                off_data.get("boundary_punct_classes", []),
                "offline_asr_timestamp",
            )
            off_block["_wav_path"] = str(off_wav)
            offline_cache[sample_id] = off_block
        offline_block = offline_cache[sample_id]
        sample_dir = result_path.parent
        for variant in VARIANTS:
            block = dict(data[variant])
            block["_wav_path"] = str(sample_dir / f"{variant}.wav")
            block = attach_nearest_voiced(block)
            pv = pause_vs_offline(offline_block, block)
            block["pause_offline_summary"] = {k: v for k, v in pv.items() if k != "items"}
            block["pause_offline_items"] = pv.get("items", [])
            key = (seed, variant)
            seed_variant_blocks.setdefault(key, []).append(block)
            seed_variant_offline.setdefault(key, []).append(offline_block)
            rows.append(
                {
                    "seed": seed,
                    "sample_id": sample_id,
                    "variant": variant,
                    "exact_boundary_count": block.get("exact_boundary_count"),
                    "boundary_expected_count": block.get("boundary_expected_count"),
                    "stream_input_mode": block.get("stream_input_mode"),
                    "stream_group_policy": block.get("stream_group_policy"),
                    "nearest_f0_summary": block.get("nearest_voiced_summary"),
                    "pause_offline_summary": block.get("pause_offline_summary"),
                    "artifact": block.get("boundary_artifact"),
                    "timing": block.get("timing"),
                }
            )

    seed_rows = []
    for (seed, variant), blocks in sorted(seed_variant_blocks.items()):
        seed_rows.append(
            {
                "seed": seed,
                "variant": variant,
                "summary": summarize_variant(blocks, seed_variant_offline[(seed, variant)]),
            }
        )

    summary: dict[str, Any] = {}
    for variant in VARIANTS:
        blocks: list[dict[str, Any]] = []
        offs: list[dict[str, Any]] = []
        for seed in sorted({row["seed"] for row in seed_rows}):
            blocks.extend(seed_variant_blocks.get((seed, variant), []))
            offs.extend(seed_variant_offline.get((seed, variant), []))
        summary[variant] = summarize_variant(blocks, offs)

    payload = {
        "runs_root": str(runs_root),
        "offline_root": str(offline_root),
        "variants": VARIANTS,
        "n_rows": len(rows),
        "offline_alignment": offline_alignment,
        "seed_rows": seed_rows,
        "summary": summary,
        "per_sample": rows,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(render_md(payload), encoding="utf-8")
    print(out_md.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Post-process Table 2 timed runs: aggregate metrics and same-voice SIM."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


VARIANTS = {
    "stateless_once": "无状态",
    "stateful_stream": "现有 stateful（engine clause stream）†",
    "acoustic_tail_only": "仅声学尾（C1 prototype）",
    "kv_tail_only": "仅 KV/token 尾（C2 prototype）",
    "tail_kv_pause_recovery": "尾 +KV+ 暂停恢复（C1+C2+C3 prototype）",
    "full_steadystream": "完整 SteadyStream（C1+C2+C3, C4未训练）",
}

WAVS = {
    "stateless_once": "stateless_once.wav",
    "stateful_stream": "stateful_stream.wav",
    "acoustic_tail_only": "acoustic_tail_only.wav",
    "kv_tail_only": "kv_tail_only.wav",
    "tail_kv_pause_recovery": "tail_kv_pause_recovery.wav",
    "full_steadystream": "full_steadystream.wav",
    "offline_full": "offline_full.wav",
}


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    wav, sr = sf.read(str(path), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return wav, sr


def mfcc_cosine(a: np.ndarray, b: np.ndarray, sr: int) -> float:
    def feat(x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return np.zeros(20, dtype=np.float32)
        return librosa.feature.mfcc(y=x.astype(np.float32), sr=sr, n_mfcc=20).mean(axis=1)

    fa, fb = feat(a), feat(b)
    return float(np.dot(fa, fb) / ((np.linalg.norm(fa) * np.linalg.norm(fb)) + 1e-8))


def maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        x = float(value)
    except Exception:
        return None
    if math.isnan(x):
        return None
    return x


def mean_std(values: list[float | None]) -> tuple[float | None, float | None, int]:
    vals = [maybe_float(v) for v in values]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None, 0
    arr = np.asarray(vals, dtype=np.float64)
    return float(arr.mean()), float(arr.std()), len(vals)


def fmt(mean: float | None, std: float | None, digits: int = 2) -> str:
    if mean is None:
        return "—"
    return f"{mean:.{digits}f}±{(std or 0.0):.{digits}f}"


def extract_sample_metrics(sample_dir: Path, variant: str) -> dict[str, Any]:
    data = json.loads((sample_dir / "results.json").read_text(encoding="utf-8"))
    block = data[variant]
    bsum = block.get("boundary_summary") or block.get("boundary_summary_proxy") or {}
    psum = block.get("pause_summary") or {}
    timing = block.get("timing") or {}
    if variant == "stateless_once":
        fasl = timing.get("fasl_mean_ms")
    else:
        fasl = timing.get("fasl_first_audio_ms") or timing.get("first_audio_ms")

    offline_wav, offline_sr = load_wav(sample_dir / WAVS["offline_full"])
    variant_wav, variant_sr = load_wav(sample_dir / WAVS[variant])
    if variant_sr != offline_sr:
        offline_wav = librosa.resample(offline_wav, orig_sr=offline_sr, target_sr=variant_sr)
        offline_sr = variant_sr
    sim = mfcc_cosine(variant_wav, offline_wav, variant_sr)

    return {
        "sample_id": data["sample_id"],
        "seed": data["seed"],
        "variant": variant,
        "f0_jump_mean_st": bsum.get("f0_jump_mean_st"),
        "f0_coverage": bsum.get("f0_coverage"),
        "energy_jump_mean_db": bsum.get("energy_jump_mean_db"),
        "energy_coverage": bsum.get("energy_coverage"),
        "pause_deviation_mean_ms": psum.get("pause_deviation_mean_ms"),
        "pause_coverage": psum.get("pause_coverage"),
        "sim_same_voice": round(sim, 6),
        "sim_delta_x100": round((1.0 - sim) * 100.0, 6),
        "fasl_mean_ms": fasl,
        "audio_sec": block.get("audio_sec"),
        "exact_boundary_count": block.get("exact_boundary_count"),
        "boundary_source_used": block.get("boundary_source_used"),
    }


def aggregate_by_seed(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_variant_seed: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        by_variant_seed.setdefault((row["variant"], int(row["seed"])), []).append(row)

    seed_rows = []
    metric_keys = [
        "f0_jump_mean_st",
        "energy_jump_mean_db",
        "pause_deviation_mean_ms",
        "sim_delta_x100",
        "fasl_mean_ms",
        "f0_coverage",
        "energy_coverage",
        "pause_coverage",
    ]
    for (variant, seed), items in sorted(by_variant_seed.items()):
        out = {"variant": variant, "seed": seed, "n_samples": len(items)}
        for key in metric_keys:
            mean, std, n = mean_std([item.get(key) for item in items])
            out[key] = round(mean, 6) if mean is not None else None
            out[f"{key}_sample_std"] = round(std, 6) if std is not None else None
            out[f"{key}_n"] = n
        exact_total = sum(int(item.get("exact_boundary_count") or 0) for item in items)
        out["exact_boundary_count_total"] = exact_total
        out["proxy_boundary_samples"] = sum(
            1 for item in items if item.get("boundary_source_used") == "proxy_proportional"
        )
        seed_rows.append(out)

    by_variant: dict[str, list[dict[str, Any]]] = {}
    for row in seed_rows:
        by_variant.setdefault(row["variant"], []).append(row)

    summary = {}
    for variant, items in by_variant.items():
        entry = {
            "label": VARIANTS.get(variant, variant),
            "n_seeds": len(items),
            "n_samples_per_seed": [item["n_samples"] for item in items],
        }
        for key in metric_keys:
            mean, std, n = mean_std([item.get(key) for item in items])
            entry[key] = round(mean, 6) if mean is not None else None
            entry[f"{key}_seed_std"] = round(std, 6) if std is not None else None
            entry[f"{key}_seed_n"] = n
        entry["exact_boundary_count_total"] = sum(
            int(item.get("exact_boundary_count_total") or 0) for item in items
        )
        entry["proxy_boundary_samples_total"] = sum(
            int(item.get("proxy_boundary_samples") or 0) for item in items
        )
        summary[variant] = entry

    return {"seed_rows": seed_rows, "summary": summary}


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# 表 2 当前可测行（三种子，test-prosody-mini）",
        "",
        "口径：F0/能量为 raw boundary jump；SIM Δ 为同样本 offline_full 直接音色对比 `(1 - MFCC cosine) × 100`；CER 待 ASR 后处理补入。",
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
        note = (
            f"F0 cov {s.get('f0_coverage'):.2f}; "
            f"E cov {s.get('energy_coverage'):.2f}; "
            f"pause cov {s.get('pause_coverage'):.2f}"
            if s
            else "—"
        )
        if variant != "stateless_once" and s:
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
            f"— | {note} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", default="workspace/table2_runs")
    parser.add_argument("--output-json", default="workspace/table2_runs/table2_metrics_no_cer.json")
    parser.add_argument("--output-md", default="workspace/table2_runs/table2_current_no_cer.md")
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    rows: list[dict[str, Any]] = []
    for result_path in sorted(runs_root.glob("seed_*/*/results.json")):
        sample_dir = result_path.parent
        for variant in VARIANTS:
            rows.append(extract_sample_metrics(sample_dir, variant))

    agg = aggregate_by_seed(rows)
    payload = {
        "runs_root": str(runs_root),
        "n_sample_variant_rows": len(rows),
        "metric_scope": "No CER yet; raw F0/energy; same-voice MFCC SIM.",
        "per_sample": rows,
        **agg,
    }
    Path(args.output_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.output_md).write_text(render_markdown(payload), encoding="utf-8")
    print(Path(args.output_md).read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

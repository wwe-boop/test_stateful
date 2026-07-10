#!/usr/bin/env python3
"""Coverage-aware paired F0 calibration for Table 2.

This script repairs the E50-E54 raw-F0 comparison problem by aligning offline
full-sentence audio to the designed clause boundaries and comparing each variant
only on shared measurable logical boundaries.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from eval.boundary_metrics import measure_boundary_metrics, side_f0_mean, summarize_boundary_metrics
from eval.pause_metrics import measure_pause_metrics, summarize_pause_metrics
from scripts.python.table2_boundary_dsp import boundary_artifact_metrics

_ASR = None


def get_asr():
    global _ASR
    if _ASR is None:
        from funasr import AutoModel

        _ASR = AutoModel(
            model="paraformer-zh",
            model_revision="v2.0.4",
            disable_update=True,
            device="cpu",
        )
    return _ASR


def norm_chars(text: str) -> list[str]:
    text = re.sub(r"\s+", "", str(text)).lower()
    text = re.sub(r"[，。！？、；：“”‘’（）【】《》…,.!?;:'\"()\[\]<>—\-]", "", text)
    text = text.replace("℃", "摄氏度")
    return list(text)


def asr_timestamp_chars(wav_path: Path) -> tuple[list[str], list[tuple[int, int]], str]:
    result = get_asr().generate(input=str(wav_path), batch_size_s=300, return_raw_text=True)
    if not result:
        return [], [], ""
    item = result[0]
    text = str(item.get("text") or "") if isinstance(item, dict) else str(item)
    timestamps = item.get("timestamp") if isinstance(item, dict) else None
    toks = text.split()
    if timestamps is None:
        timestamps = []
    chars: list[str] = []
    times: list[tuple[int, int]] = []
    for tok, ts in zip(toks, timestamps):
        ncs = norm_chars(tok)
        if not ncs:
            continue
        start, end = int(ts[0]), int(ts[1])
        width = max(1, end - start)
        for i, ch in enumerate(ncs):
            a = start + round(width * i / len(ncs))
            b = start + round(width * (i + 1) / len(ncs))
            chars.append(ch)
            times.append((a, b))
    return chars, times, text


def align_ref_to_hyp(ref: list[str], hyp: list[str]) -> list[int | None]:
    # Levenshtein DP with backtrace; returns hyp index per ref char when matched/substituted.
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    bt = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
        bt[i][0] = "del"
    for j in range(1, m + 1):
        dp[0][j] = j
        bt[0][j] = "ins"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub_cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            opts = [
                (dp[i - 1][j - 1] + sub_cost, "diag"),
                (dp[i - 1][j] + 1, "del"),
                (dp[i][j - 1] + 1, "ins"),
            ]
            dp[i][j], bt[i][j] = min(opts, key=lambda x: x[0])
    mapping: list[int | None] = [None] * n
    i, j = n, m
    while i > 0 or j > 0:
        op = bt[i][j]
        if op == "diag":
            mapping[i - 1] = j - 1
            i -= 1
            j -= 1
        elif op == "del":
            i -= 1
        elif op == "ins":
            j -= 1
        else:
            break
    return mapping


def offline_boundaries_from_asr(wav_path: Path, segments: list[str], sample_rate: int) -> tuple[list[int], dict[str, Any]]:
    ref_chars = norm_chars("".join(segments))
    hyp_chars, hyp_times_ms, hyp_text = asr_timestamp_chars(wav_path)
    mapping = align_ref_to_hyp(ref_chars, hyp_chars)
    boundaries: list[int] = []
    details: list[dict[str, Any]] = []
    cursor = 0
    for idx, seg in enumerate(segments[:-1], start=1):
        cursor += len(norm_chars(seg))
        left_ref = cursor - 1
        right_ref = cursor
        left_h = next((mapping[k] for k in range(left_ref, -1, -1) if mapping[k] is not None), None)
        right_h = next((mapping[k] for k in range(right_ref, len(mapping)) if mapping[k] is not None), None)
        if left_h is not None and right_h is not None and left_h < len(hyp_times_ms) and right_h < len(hyp_times_ms):
            t_ms = (hyp_times_ms[left_h][1] + hyp_times_ms[right_h][0]) / 2.0
            source = "asr_timestamp_aligned"
        elif left_h is not None and left_h < len(hyp_times_ms):
            t_ms = hyp_times_ms[left_h][1]
            source = "asr_timestamp_left_only"
        elif right_h is not None and right_h < len(hyp_times_ms):
            t_ms = hyp_times_ms[right_h][0]
            source = "asr_timestamp_right_only"
        else:
            # Last-resort fallback is tagged and can be excluded by consumers.
            t_ms = 0.0
            source = "alignment_failed"
        sample = int(round(t_ms * sample_rate / 1000.0))
        boundaries.append(sample)
        details.append({
            "boundary": idx,
            "ref_cursor": cursor,
            "left_hyp_index": left_h,
            "right_hyp_index": right_h,
            "time_ms": round(t_ms, 1),
            "sample_index": sample,
            "source": source,
        })
    aligned = sum(1 for x in mapping if x is not None)
    return boundaries, {
        "hypothesis_raw": hyp_text,
        "ref_chars": len(ref_chars),
        "hyp_chars": len(hyp_chars),
        "aligned_ref_chars": aligned,
        "aligned_ref_ratio": round(aligned / len(ref_chars), 4) if ref_chars else 0.0,
        "boundaries": details,
    }


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]
    return audio.astype(np.float32), int(sr)


def variant_block_from_wav(wav_path: Path, boundaries: list[int], puncts: list[str], source: str) -> dict[str, Any]:
    audio, sr = read_audio(wav_path)
    bm = measure_boundary_metrics(audio, sr, boundaries, boundary_source=source)
    nv = measure_nearest_voiced_f0(audio, sr, boundaries, boundary_source=f"{source}+nearest_voiced")
    artifact = boundary_artifact_metrics(audio, sr, boundaries)
    pauses = measure_pause_metrics(audio, sr, boundaries, [0.0] * len(boundaries))
    # measured_pause_ms is all we use for offline-derived expectations; deviation vs 0 is ignored.
    return {
        "audio_sec": round(len(audio) / sr, 3),
        "boundary_metrics": bm,
        "nearest_voiced_boundary_metrics": nv,
        "nearest_voiced_summary": summarize_boundary_metrics(nv),
        "boundary_summary": summarize_boundary_metrics(bm),
        "boundary_artifact": {k: v for k, v in artifact.items() if k != "boundaries"},
        "boundary_artifact_items": artifact.get("boundaries", []),
        "pause_measured_ms": [p.get("measured_pause_ms") for p in pauses],
    }


def nearest_voiced_side(
    audio: np.ndarray,
    sample_rate: int,
    boundary: int,
    side: str,
    *,
    win_ms: float = 250.0,
    hop_ms: float = 40.0,
    search_ms: float = 1500.0,
) -> tuple[float | None, int | None, float | None]:
    """Find the nearest voiced analysis window on one side of a boundary."""
    win = int(round(win_ms * sample_rate / 1000.0))
    hop = int(round(hop_ms * sample_rate / 1000.0))
    search = int(round(search_ms * sample_rate / 1000.0))
    if win <= 0 or hop <= 0:
        return None, None, None
    if side == "left":
        starts = range(max(0, boundary - win), max(-1, boundary - search - win), -hop)
    else:
        starts = range(boundary, min(len(audio), boundary + search), hop)
    for start in starts:
        end = start + win
        if start < 0 or end > len(audio):
            continue
        f0 = side_f0_mean(audio[start:end], sample_rate)
        if f0 is not None:
            center = start + win // 2
            distance_ms = abs(center - boundary) * 1000.0 / sample_rate
            return f0, center, distance_ms
    return None, None, None


def measure_nearest_voiced_f0(
    audio: np.ndarray,
    sample_rate: int,
    boundaries: list[int],
    *,
    boundary_source: str,
) -> list[dict[str, Any]]:
    """Measure F0 reset using nearest voiced windows rather than pause-centered windows."""
    items: list[dict[str, Any]] = []
    for idx, boundary in enumerate(boundaries, start=1):
        f0_l, center_l, dist_l = nearest_voiced_side(audio, sample_rate, boundary, "left")
        f0_r, center_r, dist_r = nearest_voiced_side(audio, sample_rate, boundary, "right")
        jump = None
        if f0_l is not None and f0_r is not None:
            jump = abs(12.0 * math.log2(f0_r / f0_l))
        items.append(
            {
                "boundary": idx,
                "sample_index": int(boundary),
                "time_sec": round(boundary / sample_rate, 3),
                "boundary_source": boundary_source,
                "f0_left_hz": round(f0_l, 2) if f0_l is not None else None,
                "f0_right_hz": round(f0_r, 2) if f0_r is not None else None,
                "f0_jump_st": round(float(jump), 3) if jump is not None else None,
                "left_center_sample": center_l,
                "right_center_sample": center_r,
                "left_distance_ms": round(dist_l, 1) if dist_l is not None else None,
                "right_distance_ms": round(dist_r, 1) if dist_r is not None else None,
                "energy_left_db": None,
                "energy_right_db": None,
                "energy_jump_db": None,
            }
        )
    return items


def existing_block(result_path: Path, variant: str) -> dict[str, Any] | None:
    data = load_json(result_path)
    return data.get(variant)


def attach_nearest_voiced(block: dict[str, Any]) -> dict[str, Any]:
    wav_path = block.get("_wav_path")
    if not wav_path:
        return block
    boundaries = [int(x.get("sample_index")) for x in block.get("boundary_metrics", [])]
    audio, sr = read_audio(Path(wav_path))
    nv = measure_nearest_voiced_f0(
        audio,
        sr,
        boundaries,
        boundary_source=f"{block.get('boundary_source_used', 'existing')}+nearest_voiced",
    )
    block["nearest_voiced_boundary_metrics"] = nv
    block["nearest_voiced_summary"] = summarize_boundary_metrics(nv)
    return block


def values(items: list[float]) -> dict[str, Any]:
    if not items:
        return {"n": 0, "mean": None, "median": None, "iqr": None, "min": None, "max": None}
    xs = sorted(float(x) for x in items)
    q1 = float(np.percentile(xs, 25))
    q3 = float(np.percentile(xs, 75))
    return {
        "n": len(xs),
        "mean": round(float(np.mean(xs)), 4),
        "median": round(float(np.median(xs)), 4),
        "iqr": round(q3 - q1, 4),
        "min": round(float(xs[0]), 4),
        "max": round(float(xs[-1]), 4),
    }


def metric_map(block: dict[str, Any], metric: str) -> dict[int, float]:
    out = {}
    for item in block.get("boundary_metrics", []) or []:
        v = item.get(metric)
        if isinstance(v, (int, float)):
            out[int(item["boundary"])] = float(v)
    return out


def nearest_f0_map(block: dict[str, Any]) -> dict[int, float]:
    out = {}
    for item in block.get("nearest_voiced_boundary_metrics", []) or []:
        v = item.get("f0_jump_st")
        if isinstance(v, (int, float)):
            out[int(item["boundary"])] = float(v)
    return out


def paired_against_offline(offline: dict[str, Any], block: dict[str, Any], metric: str) -> dict[str, Any]:
    om = metric_map(offline, metric)
    vm = metric_map(block, metric)
    common = sorted(set(om) & set(vm))
    diffs = [vm[i] - om[i] for i in common]
    abs_diffs = [abs(x) for x in diffs]
    return {
        "overlap": len(common),
        "offline_measured": len(om),
        "variant_measured": len(vm),
        "paired_boundaries": common,
        "diff": values(diffs),
        "abs_diff": values(abs_diffs),
    }


def paired_nearest_f0_against_offline(offline: dict[str, Any], block: dict[str, Any]) -> dict[str, Any]:
    om = nearest_f0_map(offline)
    vm = nearest_f0_map(block)
    common = sorted(set(om) & set(vm))
    diffs = [vm[i] - om[i] for i in common]
    abs_diffs = [abs(x) for x in diffs]
    return {
        "overlap": len(common),
        "offline_measured": len(om),
        "variant_measured": len(vm),
        "paired_boundaries": common,
        "diff": values(diffs),
        "abs_diff": values(abs_diffs),
    }


def pause_vs_offline(offline_block: dict[str, Any], block: dict[str, Any]) -> dict[str, Any]:
    expected = offline_block.get("pause_measured_ms") or []
    boundaries = [int(x.get("sample_index")) for x in block.get("boundary_metrics", [])]
    audio_path = block.get("_wav_path")
    if not audio_path or len(expected) != len(boundaries):
        return {"pause_deviation_mean_ms": None, "pause_coverage": 0.0, "items": []}
    audio, sr = read_audio(Path(audio_path))
    exp = [float(x) if isinstance(x, (int, float)) else math.nan for x in expected]
    items = []
    valid_boundaries = []
    valid_expected = []
    for b, e in zip(boundaries, exp):
        if math.isfinite(e):
            valid_boundaries.append(b)
            valid_expected.append(e)
    if not valid_boundaries:
        return {"pause_deviation_mean_ms": None, "pause_coverage": 0.0, "items": []}
    items = measure_pause_metrics(audio, sr, valid_boundaries, valid_expected)
    return {**summarize_pause_metrics(items), "items": items}


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# E55 F0 Paired Calibration",
        "",
        "F0 is compared only on logical boundaries where both offline and the variant have valid voiced measurements.",
        "",
        "| Variant | boundaries | raw F0 cov | raw F0 mean | nearest F0 cov | nearest F0 mean | paired n vs offline | median abs diff vs offline | energy mean | pause vs offline | artifact flux |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for v, s in payload["summary"].items():
        lines.append(
            f"| `{v}` | {s['n_boundaries']} | {s['f0_coverage']:.3f} | {fmt(s.get('f0_mean'))} | "
            f"{s['nearest_f0_coverage']:.3f} | {fmt(s.get('nearest_f0_mean'))} | "
            f"{s['paired_nearest_f0']['overlap']} | {fmt(s['paired_nearest_f0']['abs_diff'].get('median'))} | "
            f"{fmt(s.get('energy_mean'))} | {fmt(s.get('pause_offline_mean_ms'))} ms | {fmt(s.get('artifact_flux'))} |"
        )
    lines.extend(["", "## Offline Alignment", "", "| sample | aligned ref ratio | failed boundaries |", "|---|---:|---:|"])
    for sid, a in payload.get("offline_alignment", {}).items():
        failed = sum(1 for b in a.get("boundaries", []) if b.get("source") == "alignment_failed")
        lines.append(f"| {sid} | {a.get('aligned_ref_ratio')} | {failed} |")
    return "\n".join(lines) + "\n"


def fmt(x: Any) -> str:
    if x is None:
        return "—"
    if isinstance(x, (int, float)):
        return f"{x:.3f}"
    return str(x)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--e47-root", required=True)
    ap.add_argument("--e54-root", required=True)
    ap.add_argument("--stateless-root", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    e47 = Path(args.e47_root)
    e54 = Path(args.e54_root)
    stateless_root = Path(args.stateless_root)
    variants = {
        "offline_full": {"root": e47, "source": "align_offline"},
        "stateless_once": {"root": stateless_root, "source": "existing"},
        "stateful_stream": {"root": e47, "source": "existing"},
        "c4_icl_prefill": {"root": e47, "source": "existing"},
        "c4_icl_prefill_c3": {"root": e47, "source": "existing"},
        "full_steadystream_icl_c3": {"root": e54, "source": "existing"},
    }

    per_sample: dict[str, dict[str, Any]] = {}
    offline_alignment: dict[str, Any] = {}
    sample_dirs = sorted((e47 / "seed_42").glob("prosody_mini_*"))
    for sd in sample_dirs:
        sid = sd.name
        base = load_json(sd / "results.json")
        segments = base["segments"]
        puncts = base.get("boundary_punct_classes", [])
        offline_wav = sd / "offline_full.wav"
        off_audio, off_sr = read_audio(offline_wav)
        off_boundaries, align = offline_boundaries_from_asr(offline_wav, segments, off_sr)
        offline_alignment[sid] = align
        sample_payload: dict[str, Any] = {}
        offline_block = variant_block_from_wav(offline_wav, off_boundaries, puncts, "offline_asr_timestamp")
        offline_block["_wav_path"] = str(offline_wav)
        sample_payload["offline_full"] = offline_block
        for variant, info in variants.items():
            if variant == "offline_full":
                continue
            rpath = info["root"] / "seed_42" / sid / "results.json"
            block = existing_block(rpath, variant)
            if not block:
                continue
            block = dict(block)
            block["_wav_path"] = str(info["root"] / "seed_42" / sid / f"{variant}.wav")
            block = attach_nearest_voiced(block)
            sample_payload[variant] = block
        per_sample[sid] = sample_payload

    summary: dict[str, Any] = {}
    for variant in variants:
        blocks = [p[variant] for p in per_sample.values() if variant in p]
        n_boundaries = sum(len(b.get("boundary_metrics", []) or []) for b in blocks)
        f0_vals = [m.get("f0_jump_st") for b in blocks for m in b.get("boundary_metrics", []) if isinstance(m.get("f0_jump_st"), (int, float))]
        energy_vals = [m.get("energy_jump_db") for b in blocks for m in b.get("boundary_metrics", []) if isinstance(m.get("energy_jump_db"), (int, float))]
        nearest_f0_vals = [
            m.get("f0_jump_st")
            for b in blocks
            for m in b.get("nearest_voiced_boundary_metrics", [])
            if isinstance(m.get("f0_jump_st"), (int, float))
        ]
        flux_vals = [b.get("boundary_artifact", {}).get("flux_peak_mean") for b in blocks if isinstance(b.get("boundary_artifact", {}).get("flux_peak_mean"), (int, float))]
        paired_f0_parts = []
        paired_energy_parts = []
        pause_vals = []
        for sid, p in per_sample.items():
            if variant not in p:
                continue
            if variant == "offline_full":
                paired_f0_parts.append({"overlap": len(metric_map(p[variant], "f0_jump_st")), "abs_diff": {"n": 0, "median": 0.0}, "diff": {"n": 0, "median": 0.0}, "offline_measured": len(metric_map(p[variant], "f0_jump_st")), "variant_measured": len(metric_map(p[variant], "f0_jump_st"))})
            else:
                paired_f0_parts.append(paired_against_offline(p["offline_full"], p[variant], "f0_jump_st"))
                paired_energy_parts.append(paired_against_offline(p["offline_full"], p[variant], "energy_jump_db"))
                pv = pause_vs_offline(p["offline_full"], p[variant])
                if isinstance(pv.get("pause_deviation_mean_ms"), (int, float)):
                    pause_vals.append(float(pv["pause_deviation_mean_ms"]))
        # Flatten pair diffs by recomputing from maps for exact global medians.
        f0_abs_diffs: list[float] = []
        f0_diffs: list[float] = []
        nearest_abs_diffs: list[float] = []
        nearest_diffs: list[float] = []
        f0_overlap = 0
        nearest_overlap = 0
        for sid, p in per_sample.items():
            if variant not in p or variant == "offline_full":
                continue
            om = metric_map(p["offline_full"], "f0_jump_st")
            vm = metric_map(p[variant], "f0_jump_st")
            common = sorted(set(om) & set(vm))
            f0_overlap += len(common)
            for i in common:
                d = vm[i] - om[i]
                f0_diffs.append(d)
                f0_abs_diffs.append(abs(d))
            om_nv = nearest_f0_map(p["offline_full"])
            vm_nv = nearest_f0_map(p[variant])
            common_nv = sorted(set(om_nv) & set(vm_nv))
            nearest_overlap += len(common_nv)
            for i in common_nv:
                d = vm_nv[i] - om_nv[i]
                nearest_diffs.append(d)
                nearest_abs_diffs.append(abs(d))
        if variant == "offline_full":
            f0_overlap = len(f0_vals)
            nearest_overlap = len(nearest_f0_vals)
        summary[variant] = {
            "n_boundaries": n_boundaries,
            "f0_coverage": round(len(f0_vals) / n_boundaries, 4) if n_boundaries else 0.0,
            "f0_mean": round(float(np.mean(f0_vals)), 4) if f0_vals else None,
            "f0_median": round(float(np.median(f0_vals)), 4) if f0_vals else None,
            "nearest_f0_coverage": round(len(nearest_f0_vals) / n_boundaries, 4) if n_boundaries else 0.0,
            "nearest_f0_mean": round(float(np.mean(nearest_f0_vals)), 4) if nearest_f0_vals else None,
            "nearest_f0_median": round(float(np.median(nearest_f0_vals)), 4) if nearest_f0_vals else None,
            "energy_coverage": round(len(energy_vals) / n_boundaries, 4) if n_boundaries else 0.0,
            "energy_mean": round(float(np.mean(energy_vals)), 4) if energy_vals else None,
            "artifact_flux": round(float(np.mean(flux_vals)), 6) if flux_vals else None,
            "pause_offline_mean_ms": round(float(np.mean(pause_vals)), 3) if pause_vals else None,
            "paired_f0": {"overlap": f0_overlap, "diff": values(f0_diffs), "abs_diff": values(f0_abs_diffs)},
            "paired_nearest_f0": {
                "overlap": nearest_overlap,
                "diff": values(nearest_diffs),
                "abs_diff": values(nearest_abs_diffs),
            },
        }
    payload = {
        "inputs": {"e47_root": args.e47_root, "e54_root": args.e54_root, "stateless_root": args.stateless_root},
        "offline_alignment": offline_alignment,
        "summary": summary,
        "per_sample": per_sample,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.out_md).write_text(render_md(payload), encoding="utf-8")
    print(Path(args.out_md).read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

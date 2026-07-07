#!/usr/bin/env python3
"""Quick validation for Table 4 stress metrics (no Triton required)."""

from __future__ import annotations

import sys
import time

import numpy as np

from eval.boundary_match import forced_segmentation_rate, match_boundaries
from eval.fasl_vad import measure_fasl_vad_from_packets
from eval.jitter_metrics import intervals_from_packet_timestamps, measure_jitter_ms
from eval.stress_metrics import aggregate_stress_runs, compute_session_metrics, summarize_session_stress
from eval.stutter_metrics import simulate_playout_underflows


def _tone(freq: float, duration: float, sr: int = 24000) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_fasl_vad() -> None:
    sr = 24000
    silence = np.zeros(int(0.2 * sr), dtype=np.float32)
    speech = _tone(220.0, 0.15, sr)
    t0 = time.perf_counter()
    packets = [
        {"client_ts": t0 + 0.05, "samples": silence},
        {"client_ts": t0 + 0.25, "samples": speech},
    ]
    out = measure_fasl_vad_from_packets(packets, first_token_ts=t0)
    assert out["fasl_vad_ms"] is not None
    assert 200 <= out["fasl_vad_ms"] <= 350, out
    print("  ✓ FASL VAD")


def test_jitter_stutter() -> None:
    t0 = time.perf_counter()
    ts = [t0 + i * 0.04 for i in range(20)]
    durs = [0.04] * 20
    intervals = intervals_from_packet_timestamps(ts)
    jitter = measure_jitter_ms(intervals, expected_interval_ms=40.0)
    assert jitter["jitter_mean_ms"] is not None
    assert jitter["jitter_mean_ms"] < 1.0
    # Introduce a gap -> underflow
    ts_gap = ts[:10] + [ts[10] + 0.3] + ts[11:]
    stutter = simulate_playout_underflows(ts_gap, durs[:11] + durs[10:])
    assert stutter["stutter_count"] >= 1
    print("  ✓ Jitter + Stutter")


def test_boundary_match() -> None:
    m = match_boundaries([10, 25, 40], [11, 26, 55], tolerance_chars=2)
    assert m["matched_pairs"] == 2
    assert m["boundary_f1"] > 0.5
    forced = forced_segmentation_rate(
        [{"reason": "punct"}, {"reason": "force_length"}, {"reason": "force_length"}]
    )
    assert abs(forced["forced_split_rate"] - 2 / 3) < 0.01
    print("  ✓ Boundary match")


def test_rtf_xrt() -> None:
    m = compute_session_metrics(
        {"total_ms": 1600.0, "total_samples": 24000 * 10, "session_id": "x"}
    )
    assert m["rtf"] == 0.16
    assert m["x_rt"] == 6.25
    print("  ✓ RTF / ×RT")


def test_ttft_anchor() -> None:
    t0 = 1000.0
    m = compute_session_metrics(
        {
            "session_start_ts": t0,
            "first_pcm_ts": t0 + 0.05,
            "first_text_ts": t0 + 0.02,
            "total_ms": 500.0,
            "total_samples": 24000,
        }
    )
    assert m["ttft_ms"] == 50.0
    assert m["ttfb_ms"] == 50.0
    print("  ✓ TTFT = init→首包")


def test_aggregate() -> None:
    sessions = [
        {"fasl_vad_ms": 100, "ttft_ms": 80, "jitter_p95_ms": 5, "stutter_rate_pct": 0.5},
        {"fasl_vad_ms": 110, "ttft_ms": 85, "jitter_p95_ms": 6, "stutter_rate_pct": 0.8},
    ]
    agg = aggregate_stress_runs(sessions)
    assert agg["fasl_vad_ms"]["mean"] == 105.0
    print("  ✓ Aggregate")


def main() -> int:
    print("Table 4 metrics validation...")
    test_fasl_vad()
    test_jitter_stutter()
    test_boundary_match()
    test_rtf_xrt()
    test_ttft_anchor()
    test_aggregate()
    rec = summarize_session_stress(
        {
            "session_id": "s0",
            "first_token_ts": time.perf_counter(),
            "audio_packets": [],
            "trace_pause_after_total_ms": 1500,
        }
    )
    assert rec["stratification"] == "long_pause"
    print("  ✓ Session stratification")
    print("\n✅ All Table 4 metric tests passed!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

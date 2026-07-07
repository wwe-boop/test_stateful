"""Aggregate per-session stress metrics for Table 4."""

from __future__ import annotations

from typing import Any

import numpy as np

from eval.fasl_vad import measure_fasl_vad_from_packets
from eval.jitter_metrics import intervals_from_packet_timestamps, measure_jitter_ms
from eval.stutter_metrics import simulate_playout_underflows


def summarize_session_stress(
    session_record: dict[str, Any],
    *,
    sample_rate: int = 24000,
    expected_packet_ms: float = 40.0,
    prebuffer_ms: float = 200.0,
) -> dict[str, Any]:
    """Compute Table 4 KPIs from a stress client session JSON record."""
    if session_record.get("metrics"):
        return dict(session_record["metrics"])
    packets = session_record.get("audio_packets") or []
    first_token_ts = session_record.get("first_token_ts")
    if first_token_ts is None:
        first_token_ts = session_record.get("session_start_ts")

    ttft_ms = session_record.get("ttft_ms")
    if ttft_ms is None and session_record.get("first_token_ts") and session_record.get("first_pcm_ts"):
        ttft_ms = (session_record["first_pcm_ts"] - session_record["first_token_ts"]) * 1000.0

    fasl = {"fasl_vad_ms": None, "first_packet_ms": None}
    jitter = {"jitter_p95_ms": None}
    stutter = {"stutter_rate_pct": 0.0}

    if packets and first_token_ts is not None:
        fasl = measure_fasl_vad_from_packets(packets, first_token_ts=first_token_ts, sample_rate=sample_rate)
        ts_list = [p["client_ts"] for p in packets]
        dur_list = [len(p["samples"]) / sample_rate for p in packets]
        intervals = intervals_from_packet_timestamps(ts_list)
        jitter = measure_jitter_ms(intervals, expected_interval_ms=expected_packet_ms)
        stutter = simulate_playout_underflows(ts_list, dur_list, prebuffer_ms=prebuffer_ms)

    pause_after_total_ms = float(session_record.get("trace_pause_after_total_ms") or 0.0)
    stratification = "long_pause" if pause_after_total_ms > 1000.0 else "normal"

    return {
        "session_id": session_record.get("session_id"),
        "variant": session_record.get("variant"),
        "concurrency": session_record.get("concurrency"),
        "seed": session_record.get("seed"),
        "fasl_vad_ms": fasl.get("fasl_vad_ms"),
        "first_packet_ms": fasl.get("first_packet_ms"),
        "ttft_ms": round(ttft_ms, 2) if ttft_ms is not None else None,
        "jitter_mean_ms": jitter.get("jitter_mean_ms"),
        "jitter_p95_ms": jitter.get("jitter_p95_ms"),
        "stutter_rate_pct": stutter.get("stutter_rate_pct"),
        "stutter_count": stutter.get("stutter_count"),
        "trace_pause_after_total_ms": pause_after_total_ms,
        "stratification": stratification,
        "error": session_record.get("error"),
    }


def aggregate_stress_runs(sessions: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate session summaries into run-level stats."""
    ok = [s for s in sessions if not s.get("error")]

    def _stats(key: str) -> dict[str, float | None]:
        vals = [s[key] for s in ok if s.get(key) is not None]
        if not vals:
            return {"mean": None, "std": None, "p50": None, "p95": None}
        arr = np.asarray(vals, dtype=np.float64)
        return {
            "mean": round(float(np.mean(arr)), 2),
            "std": round(float(np.std(arr)), 2),
            "p50": round(float(np.percentile(arr, 50)), 2),
            "p95": round(float(np.percentile(arr, 95)), 2),
        }

    return {
        "n_sessions": len(sessions),
        "n_ok": len(ok),
        "n_failed": len(sessions) - len(ok),
        "fasl_vad_ms": _stats("fasl_vad_ms"),
        "ttft_ms": _stats("ttft_ms"),
        "jitter_p95_ms": _stats("jitter_p95_ms"),
        "stutter_rate_pct": _stats("stutter_rate_pct"),
    }

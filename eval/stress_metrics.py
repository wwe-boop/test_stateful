"""Aggregate per-session stress metrics for Table 4."""

from __future__ import annotations

from typing import Any

import numpy as np

from eval.fasl_vad import measure_fasl_vad_from_packets
from eval.jitter_metrics import intervals_from_packet_timestamps, measure_jitter_ms
from eval.stutter_metrics import simulate_playout_underflows

SAMPLE_RATE_DEFAULT = 24000


def compute_session_metrics(
    session_record: dict[str, Any],
    *,
    sample_rate: int = SAMPLE_RATE_DEFAULT,
    expected_packet_ms: float = 40.0,
    prebuffer_ms: float = 200.0,
) -> dict[str, Any]:
    """Compute all Table-4-fillable KPIs for one session.

    Anchors (documented for the table footnote):
      - TTFT / 首包 (ttfb_ms): session init sent -> first PCM at client
      - FASL (fasl_vad_ms): first upstream text chunk -> first VAD speech
      - first_packet_ms: first upstream text chunk -> first PCM (may be silence)
      - RTF: session wall time / audio duration; x_rt = 1 / RTF
    """
    if session_record.get("metrics"):
        return dict(session_record["metrics"])

    session_start = session_record.get("session_start_ts")
    first_text_ts = session_record.get("first_text_ts") or session_record.get("first_token_ts")
    first_pcm_ts = session_record.get("first_pcm_ts")
    total_ms = float(session_record.get("total_ms") or 0.0)
    total_samples = int(session_record.get("total_samples") or 0)
    packets = session_record.get("audio_packets") or []

    # TTFT = init -> first PCM (client); equals 首包 in this harness.
    ttfb_ms = session_record.get("ttfb_ms")
    if ttfb_ms is None and session_start is not None and first_pcm_ts is not None:
        ttfb_ms = (first_pcm_ts - session_start) * 1000.0

    server_ttft_ms = session_record.get("server_ttft_ms")
    if server_ttft_ms is None:
        for ev in session_record.get("events") or []:
            if ev.get("type") != "audio":
                continue
            payload = ev.get("payload") or {}
            raw = payload.get("triton_adapter_ttft_ms") or payload.get("server_ttft_ms")
            if raw is not None:
                try:
                    server_ttft_ms = float(raw)
                except (TypeError, ValueError):
                    pass
                break

    fasl_vad_ms = None
    first_packet_ms = None
    jitter_mean_ms = None
    jitter_p95_ms = None
    stutter_rate_pct = 0.0
    stutter_count = 0

    has_pcm_samples = packets and isinstance(packets[0].get("samples"), np.ndarray)
    if packets and first_text_ts is not None and has_pcm_samples:
        fasl = measure_fasl_vad_from_packets(
            packets, first_token_ts=first_text_ts, sample_rate=sample_rate
        )
        fasl_vad_ms = fasl.get("fasl_vad_ms")
        first_packet_ms = fasl.get("first_packet_ms")
        if fasl_vad_ms is None and first_packet_ms is not None:
            fasl_vad_ms = first_packet_ms
        ts_list = [float(p["client_ts"]) for p in packets]
        if "n_samples" in packets[0] and "samples" not in packets[0]:
            dur_list = [int(p["n_samples"]) / sample_rate for p in packets]
        else:
            dur_list = [len(p["samples"]) / sample_rate for p in packets]
        jitter = measure_jitter_ms(
            intervals_from_packet_timestamps(ts_list),
            expected_interval_ms=expected_packet_ms,
        )
        jitter_mean_ms = jitter.get("jitter_mean_ms")
        jitter_p95_ms = jitter.get("jitter_p95_ms")
        stutter = simulate_playout_underflows(ts_list, dur_list, prebuffer_ms=prebuffer_ms)
        stutter_rate_pct = stutter.get("stutter_rate_pct")
        stutter_count = stutter.get("stutter_count")
    elif session_record.get("fasl_vad_ms") is not None:
        fasl_vad_ms = session_record.get("fasl_vad_ms")
        first_packet_ms = session_record.get("first_packet_ms")
        jitter_p95_ms = session_record.get("jitter_p95_ms")
        stutter_rate_pct = session_record.get("stutter_rate_pct")

    audio_sec = total_samples / sample_rate if total_samples > 0 else 0.0
    rtf = None
    x_rt = None
    if audio_sec > 0 and total_ms > 0:
        rtf = round((total_ms / 1000.0) / audio_sec, 4)
        x_rt = round(audio_sec / (total_ms / 1000.0), 2)

    pause_after_total_ms = float(session_record.get("trace_pause_after_total_ms") or 0.0)

    return {
        "session_id": session_record.get("session_id"),
        "variant": session_record.get("variant"),
        "concurrency": session_record.get("concurrency"),
        "seed": session_record.get("seed"),
        "fasl_vad_ms": fasl_vad_ms,
        "first_packet_ms": first_packet_ms,
        "ttfb_ms": round(ttfb_ms, 2) if ttfb_ms is not None else None,
        "ttft_ms": round(ttfb_ms, 2) if ttfb_ms is not None else None,
        "server_ttft_ms": round(server_ttft_ms, 2) if server_ttft_ms is not None else None,
        "jitter_mean_ms": jitter_mean_ms,
        "jitter_p95_ms": jitter_p95_ms,
        "stutter_rate_pct": stutter_rate_pct,
        "stutter_count": stutter_count,
        "rtf": rtf,
        "x_rt": x_rt,
        "audio_sec": round(audio_sec, 2),
        "trace_pause_after_total_ms": pause_after_total_ms,
        "stratification": "long_pause" if pause_after_total_ms > 1000.0 else "normal",
        "error": session_record.get("error"),
    }


def summarize_session_stress(
    session_record: dict[str, Any],
    **kwargs: Any,
) -> dict[str, Any]:
    """Alias for compute_session_metrics (backward compatible)."""
    return compute_session_metrics(session_record, **kwargs)


def aggregate_stress_runs(
    sessions: list[dict[str, Any]],
    *,
    wall_sec: float | None = None,
) -> dict[str, Any]:
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

    total_audio_sec = sum(float(s.get("audio_sec") or 0.0) for s in ok)
    batch_x_rt = None
    batch_rtf = None
    if wall_sec and wall_sec > 0 and total_audio_sec > 0:
        batch_x_rt = round(total_audio_sec / wall_sec, 2)
        batch_rtf = round(wall_sec / total_audio_sec, 4)

    return {
        "n_sessions": len(sessions),
        "n_ok": len(ok),
        "n_failed": len(sessions) - len(ok),
        "fasl_vad_ms": _stats("fasl_vad_ms"),
        "first_packet_ms": _stats("first_packet_ms"),
        "ttft_ms": _stats("ttft_ms"),
        "ttfb_ms": _stats("ttfb_ms"),
        "server_ttft_ms": _stats("server_ttft_ms"),
        "jitter_p95_ms": _stats("jitter_p95_ms"),
        "stutter_rate_pct": _stats("stutter_rate_pct"),
        "rtf": _stats("rtf"),
        "x_rt": _stats("x_rt"),
        "batch_x_rt": batch_x_rt,
        "batch_rtf": batch_rtf,
        "total_audio_sec": round(total_audio_sec, 2),
    }

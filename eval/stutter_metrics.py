"""Stutter (underflow) rate for a fixed pre-buffer player."""

from __future__ import annotations

from typing import Any

import numpy as np


def simulate_playout_underflows(
    packet_timestamps: list[float],
    packet_durations_sec: list[float],
    *,
    prebuffer_ms: float = 200.0,
) -> dict[str, Any]:
    """Simulate a player with prebuffer_ms initial buffer; count underflow events.

    Each packet becomes playable at its arrival time. The player drains audio
    in real time. Underflow occurs when the next packet has not arrived before
    the current buffer is exhausted.
    """
    if not packet_timestamps or not packet_durations_sec:
        return {
            "stutter_count": 0,
            "stutter_rate": 0.0,
            "total_playout_sec": 0.0,
            "underflow_events": [],
        }

    prebuffer = prebuffer_ms / 1000.0
    underflows: list[dict[str, float]] = []
    # Player starts after prebuffer is filled from first packet arrival
    t0 = packet_timestamps[0]
    playhead = t0 + prebuffer
    buffer_end = t0 + packet_durations_sec[0]

    for i in range(1, len(packet_timestamps)):
        arrival = packet_timestamps[i]
        if arrival > buffer_end and playhead >= buffer_end:
            gap_ms = (arrival - buffer_end) * 1000.0
            underflows.append({"packet_idx": i, "gap_ms": round(gap_ms, 2)})
        buffer_end = max(buffer_end, arrival) + packet_durations_sec[i]
        playhead = max(playhead, arrival)

    total_playout = sum(packet_durations_sec)
    stutter_rate = len(underflows) / max(1, len(packet_timestamps) - 1)
    return {
        "stutter_count": len(underflows),
        "stutter_rate": round(float(stutter_rate), 4),
        "stutter_rate_pct": round(float(stutter_rate * 100.0), 2),
        "total_playout_sec": round(float(total_playout), 3),
        "underflow_events": underflows,
    }


def pause_resume_stutter_delta(
    stutter_rate_baseline: float,
    stutter_rate_candidate: float,
) -> dict[str, float | None]:
    """Relative stutter change vs PAD baseline (path iv)."""
    if stutter_rate_baseline is None or stutter_rate_candidate is None:
        return {"stutter_delta_pct": None, "stutter_ratio": None}
    delta = stutter_rate_candidate - stutter_rate_baseline
    ratio = (
        stutter_rate_candidate / stutter_rate_baseline
        if stutter_rate_baseline > 0
        else None
    )
    return {
        "stutter_delta_pct": round(delta * 100.0, 3),
        "stutter_ratio": round(ratio, 4) if ratio is not None else None,
    }

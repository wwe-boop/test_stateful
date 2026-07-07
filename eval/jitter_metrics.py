"""Packet-interval jitter relative to a 40 ms playback rhythm."""

from __future__ import annotations

from typing import Any

import numpy as np


def measure_jitter_ms(
    packet_intervals_ms: list[float],
    *,
    expected_interval_ms: float = 40.0,
) -> dict[str, Any]:
    """Deviation from fixed packet cadence.

    Args:
        packet_intervals_ms: gaps between consecutive audio packet arrival times
        expected_interval_ms: nominal playback chunk interval (default 40 ms)
    """
    if len(packet_intervals_ms) < 2:
        return {
            "jitter_mean_ms": None,
            "jitter_p95_ms": None,
            "jitter_rms_ms": None,
            "n_intervals": len(packet_intervals_ms),
        }

    arr = np.asarray(packet_intervals_ms, dtype=np.float64)
    deviations = np.abs(arr - expected_interval_ms)
    return {
        "jitter_mean_ms": round(float(np.mean(deviations)), 3),
        "jitter_p95_ms": round(float(np.percentile(deviations, 95)), 3),
        "jitter_rms_ms": round(float(np.sqrt(np.mean(np.square(deviations)))), 3),
        "n_intervals": int(len(arr)),
    }


def intervals_from_packet_timestamps(
    packet_timestamps: list[float],
    *,
    sample_rate: int = 24000,
    packet_samples: list[int] | None = None,
) -> list[float]:
    """Derive inter-arrival intervals in ms from client receive timestamps.

    When packet_samples is provided, also emit synthetic playout boundaries
    based on audio duration (for decode-step sized chunks).
    """
    if len(packet_timestamps) < 2:
        return []
    ts = np.asarray(packet_timestamps, dtype=np.float64)
    if packet_samples and len(packet_samples) == len(ts):
        # Interval at playout boundary: previous packet end -> next packet start
        playout_end = ts[0] + packet_samples[0] / sample_rate
        intervals = []
        for i in range(1, len(ts)):
            intervals.append((ts[i] - playout_end) * 1000.0)
            playout_end = ts[i] + packet_samples[i] / sample_rate
        return intervals
    return ((ts[1:] - ts[:-1]) * 1000.0).tolist()

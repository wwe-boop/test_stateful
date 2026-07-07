from __future__ import annotations

import time
import numpy as np
from typing import Callable, Any


def measure_first_audio_streaming_latency(
    synthesis_func: Callable[[], tuple[np.ndarray, int]],
    *,
    warmup_runs: int = 3,
    measurement_runs: int = 10,
) -> dict[str, Any]:
    """Measure FASL (First Audio Streaming Latency) per §4.5.

    Args:
        synthesis_func: callable that performs synthesis and returns (audio, sample_rate)
        warmup_runs: number of warmup iterations
        measurement_runs: number of measurement iterations

    Returns:
        {
            "fasl_mean_ms": float,
            "fasl_std_ms": float,
            "fasl_min_ms": float,
            "fasl_max_ms": float,
        }
    """
    # Warmup phase
    for _ in range(warmup_runs):
        _ = synthesis_func()

    # Measurement phase
    latencies_ms = []
    for _ in range(measurement_runs):
        start_time = time.perf_counter()
        audio, sample_rate = synthesis_func()
        end_time = time.perf_counter()

        # FASL = time to first audio chunk
        # For batch synthesis, this is total synthesis time
        # For streaming, this would be time to first chunk (needs streaming API)
        latency_ms = (end_time - start_time) * 1000.0
        latencies_ms.append(latency_ms)

    return {
        "fasl_mean_ms": round(float(np.mean(latencies_ms)), 1),
        "fasl_std_ms": round(float(np.std(latencies_ms)), 1),
        "fasl_min_ms": round(float(np.min(latencies_ms)), 1),
        "fasl_max_ms": round(float(np.max(latencies_ms)), 1),
        "measurement_runs": measurement_runs,
    }


def measure_rtf(
    synthesis_func: Callable[[], tuple[np.ndarray, int]],
    *,
    warmup_runs: int = 3,
    measurement_runs: int = 10,
) -> dict[str, Any]:
    """Measure Real-Time Factor (RTF).

    RTF = synthesis_time / audio_duration
    RTF < 1.0 means faster than real-time

    Args:
        synthesis_func: callable that returns (audio, sample_rate)
        warmup_runs: warmup iterations
        measurement_runs: measurement iterations

    Returns:
        RTF statistics
    """
    # Warmup
    for _ in range(warmup_runs):
        _ = synthesis_func()

    # Measurement
    rtf_values = []
    for _ in range(measurement_runs):
        start_time = time.perf_counter()
        audio, sample_rate = synthesis_func()
        end_time = time.perf_counter()

        synthesis_time = end_time - start_time
        audio_duration = len(audio) / sample_rate
        rtf = synthesis_time / audio_duration if audio_duration > 0 else float('inf')
        rtf_values.append(rtf)

    return {
        "rtf_mean": round(float(np.mean(rtf_values)), 4),
        "rtf_std": round(float(np.std(rtf_values)), 4),
        "rtf_min": round(float(np.min(rtf_values)), 4),
        "rtf_max": round(float(np.max(rtf_values)), 4),
    }

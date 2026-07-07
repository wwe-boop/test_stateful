from __future__ import annotations

import librosa
import numpy as np


def detect_boundary_pause(
    audio: np.ndarray,
    sample_rate: int,
    boundary_idx: int,
    *,
    search_radius_ms: float = 500.0,
    energy_threshold_db: float = -40.0,
    min_pause_ms: float = 50.0,
) -> float | None:
    """Detect pause duration around a segment boundary.

    Args:
        audio: waveform
        sample_rate: Hz
        boundary_idx: sample index of the boundary
        search_radius_ms: search window on each side of boundary
        energy_threshold_db: silence threshold
        min_pause_ms: minimum duration to count as pause

    Returns:
        Pause duration in milliseconds, or None if no clear pause detected
    """
    radius_samples = int(search_radius_ms * sample_rate / 1000.0)
    start = max(0, boundary_idx - radius_samples)
    end = min(len(audio), boundary_idx + radius_samples)
    window = audio[start:end]

    if len(window) == 0:
        return None

    # Frame-level energy analysis
    frame_length = 512
    hop_length = 256
    frames = librosa.util.frame(window, frame_length=frame_length, hop_length=hop_length)
    if frames.shape[1] == 0:
        return None

    frame_energies_db = 20 * np.log10(np.sqrt(np.mean(frames**2, axis=0)) + 1e-12)
    silent_frames = frame_energies_db < energy_threshold_db

    # Find longest continuous silent run around the boundary
    boundary_frame_idx = (boundary_idx - start) // hop_length
    if boundary_frame_idx < 0 or boundary_frame_idx >= len(silent_frames):
        return None

    # Expand from boundary to find silent region
    left_idx = boundary_frame_idx
    while left_idx > 0 and silent_frames[left_idx - 1]:
        left_idx -= 1

    right_idx = boundary_frame_idx
    while right_idx < len(silent_frames) - 1 and silent_frames[right_idx + 1]:
        right_idx += 1

    if not silent_frames[boundary_frame_idx]:
        # Boundary itself not silent
        return None

    pause_frames = right_idx - left_idx + 1
    pause_ms = (pause_frames * hop_length / sample_rate) * 1000.0

    if pause_ms < min_pause_ms:
        return None

    return round(pause_ms, 1)


def compute_pause_deviation(
    measured_ms: float | None,
    expected_ms: float,
    *,
    tolerance_ms: float = 80.0,
) -> float | None:
    """Compute pause deviation from expected duration per §4.3.

    Args:
        measured_ms: detected pause duration (None if no pause detected)
        expected_ms: expected natural pause for this punctuation class
        tolerance_ms: acceptable deviation range (±80ms default)

    Returns:
        Absolute deviation in ms if outside tolerance, else 0.0
        Returns None if measurement failed
    """
    if measured_ms is None:
        return None

    deviation = abs(measured_ms - expected_ms)
    if deviation <= tolerance_ms:
        return 0.0

    return round(deviation - tolerance_ms, 1)


def measure_pause_metrics(
    audio: np.ndarray,
    sample_rate: int,
    boundaries: list[int],
    expected_pauses_ms: list[float],
) -> list[dict[str, float | None]]:
    """Measure pause deviations at all boundaries.

    Args:
        audio: waveform
        sample_rate: Hz
        boundaries: list of boundary sample indices
        expected_pauses_ms: expected pause duration per boundary (from punct class)

    Returns:
        List of {
            "boundary": int,
            "measured_pause_ms": float | None,
            "expected_pause_ms": float,
            "deviation_ms": float | None,
        }
    """
    if len(expected_pauses_ms) != len(boundaries):
        raise ValueError(f"expected_pauses_ms length {len(expected_pauses_ms)} != boundary count {len(boundaries)}")

    results = []
    for idx, (boundary, expected) in enumerate(zip(boundaries, expected_pauses_ms), start=1):
        measured = detect_boundary_pause(audio, sample_rate, boundary)
        deviation = compute_pause_deviation(measured, expected)

        results.append({
            "boundary": idx,
            "measured_pause_ms": measured,
            "expected_pause_ms": expected,
            "deviation_ms": deviation,
        })

    return results


def summarize_pause_metrics(items: list[dict[str, float | None]]) -> dict[str, float | None]:
    """Aggregate pause deviation statistics."""
    deviations = [item["deviation_ms"] for item in items if item["deviation_ms"] is not None]

    return {
        "pause_deviation_mean_ms": round(float(np.mean(deviations)), 1) if deviations else None,
        "pause_deviation_max_ms": round(float(np.max(deviations)), 1) if deviations else None,
        "pause_coverage": round(len(deviations) / len(items), 3) if items else 0.0,
    }

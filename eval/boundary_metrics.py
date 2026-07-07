from __future__ import annotations

import math
from typing import Any

import librosa
import numpy as np


def rms_db(samples: np.ndarray) -> float:
    """Compute RMS energy in dB. Returns -120 dB for empty/silent input."""
    if samples.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(samples.astype(np.float64)))) + 1e-12)
    return 20.0 * math.log10(rms + 1e-12)


def vad_gate(
    sig: np.ndarray,
    sample_rate: int,
    *,
    energy_threshold_db: float = -40.0,
    min_speech_ratio: float = 0.3,
) -> bool:
    """VAD gate: returns True if window contains enough speech energy."""
    if len(sig) == 0:
        return False
    energy_db = rms_db(sig)
    if energy_db < energy_threshold_db:
        return False
    # Additional check: at least min_speech_ratio of frames above threshold
    frame_length = 512
    hop_length = 256
    frames = librosa.util.frame(sig, frame_length=frame_length, hop_length=hop_length)
    if frames.shape[1] == 0:
        return False
    frame_energies = np.sqrt(np.mean(frames**2, axis=0))
    speech_frames = np.sum(frame_energies > 10 ** (energy_threshold_db / 20))
    return (speech_frames / frames.shape[1]) >= min_speech_ratio


def side_f0_mean(
    sig: np.ndarray,
    sample_rate: int,
    *,
    frame_length: int = 1024,
    hop_length: int = 256,
    min_voiced_ratio: float = 0.4,
) -> float | None:
    """Compute geometric mean F0 from voiced frames only.

    Returns None if:
    - Window too short
    - VAD gate fails (no speech energy)
    - Insufficient voiced frames (< min_voiced_ratio)
    """
    if len(sig) < frame_length:
        return None

    # VAD gate: reject silent/noise windows
    if not vad_gate(sig, sample_rate):
        return None

    f0, voiced_flag, voiced_probs = librosa.pyin(
        sig.astype(np.float64),
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sample_rate,
        frame_length=frame_length,
        hop_length=hop_length,
    )

    if f0 is None or len(f0) == 0:
        return None

    # Filter: only use voiced frames with valid F0
    voiced_f0 = f0[np.isfinite(f0)]
    if len(voiced_f0) == 0:
        return None

    # Reject if insufficient voiced content
    voiced_ratio = len(voiced_f0) / len(f0)
    if voiced_ratio < min_voiced_ratio:
        return None

    # Geometric mean (log-space average, robust to outliers)
    return float(np.exp(np.mean(np.log(voiced_f0))))


def measure_boundary_metrics(
    audio: np.ndarray,
    sample_rate: int,
    boundaries: list[int],
    *,
    left_ms: float = 250.0,
    right_ms: float = 250.0,
    boundary_source: str = "unknown",
) -> list[dict[str, Any]]:
    """Measure F0 and energy jumps at segment boundaries with VAD gating.

    Per §4.1 & §4.2 calibration requirements:
    - F0: only measure on voiced frames (min 40% voiced ratio)
    - Energy: VAD-gated windows only
    - Returns None for F0/energy when conditions not met (don't force invalid measurements)

    Args:
        audio: waveform (float32)
        sample_rate: Hz
        boundaries: list of sample indices
        left_ms: left window size (default 250ms per §4.1)
        right_ms: right window size (default 250ms per §4.1)
        boundary_source: "exact_event" or "proxy_proportional"

    Returns:
        List of per-boundary metrics. F0/energy may be None if VAD/voiced checks fail.
    """
    metrics: list[dict[str, Any]] = []
    left_samples = int(left_ms * sample_rate / 1000.0)
    right_samples = int(right_ms * sample_rate / 1000.0)

    for idx, boundary in enumerate(boundaries, start=1):
        left_win = audio[max(0, boundary - left_samples):boundary]
        right_win = audio[boundary:min(len(audio), boundary + right_samples)]

        # F0 measurement with voiced frame filtering
        f0_l = side_f0_mean(left_win, sample_rate)
        f0_r = side_f0_mean(right_win, sample_rate)
        f0_jump = None
        if f0_l is not None and f0_r is not None:
            # Semitone jump: 12 * log2(f_r / f_l)
            f0_jump = abs(12.0 * math.log2(f0_r / f0_l))

        # Energy measurement with VAD gating
        energy_l_db = None
        energy_r_db = None
        energy_jump_db = None

        if vad_gate(left_win, sample_rate):
            energy_l_db = rms_db(left_win)
        if vad_gate(right_win, sample_rate):
            energy_r_db = rms_db(right_win)

        if energy_l_db is not None and energy_r_db is not None:
            energy_jump_db = abs(energy_r_db - energy_l_db)

        metrics.append(
            {
                "boundary": idx,
                "sample_index": int(boundary),
                "time_sec": round(boundary / sample_rate, 3),
                "boundary_source": boundary_source,
                "f0_left_hz": round(f0_l, 2) if f0_l is not None else None,
                "f0_right_hz": round(f0_r, 2) if f0_r is not None else None,
                "f0_jump_st": round(float(f0_jump), 3) if f0_jump is not None else None,
                "energy_left_db": round(energy_l_db, 2) if energy_l_db is not None else None,
                "energy_right_db": round(energy_r_db, 2) if energy_r_db is not None else None,
                "energy_jump_db": round(float(energy_jump_db), 3) if energy_jump_db is not None else None,
            }
        )
    return metrics


def summarize_boundary_metrics(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute aggregate statistics from boundary measurements.

    Returns:
        - Mean/max of valid measurements
        - Coverage ratio (how many boundaries passed VAD/voiced checks)
        - Boundary source validation flag
    """
    f0_vals = [item["f0_jump_st"] for item in items if item["f0_jump_st"] is not None]
    energy_vals = [item["energy_jump_db"] for item in items if item["energy_jump_db"] is not None]

    return {
        "n_boundaries": len(items),
        "f0_jump_mean_st": round(float(np.mean(f0_vals)), 3) if f0_vals else None,
        "f0_jump_max_st": round(float(np.max(f0_vals)), 3) if f0_vals else None,
        "f0_jump_std_st": round(float(np.std(f0_vals)), 3) if f0_vals else None,
        "f0_coverage": round(len(f0_vals) / len(items), 3) if items else 0.0,
        "energy_jump_mean_db": round(float(np.mean(energy_vals)), 3) if energy_vals else None,
        "energy_jump_max_db": round(float(np.max(energy_vals)), 3) if energy_vals else None,
        "energy_jump_std_db": round(float(np.std(energy_vals)), 3) if energy_vals else None,
        "energy_coverage": round(len(energy_vals) / len(items), 3) if items else 0.0,
        "all_boundaries_exact": bool(items) and all(item.get("boundary_source") == "exact_event" for item in items),
    }


def boundary_positions_from_concat(parts: list[np.ndarray]) -> list[int]:
    positions: list[int] = []
    cursor = 0
    for idx, wav in enumerate(parts, start=1):
        cursor += len(wav)
        if idx < len(parts):
            positions.append(cursor)
    return positions


def boundary_positions_proportional(audio: np.ndarray, segments: list[str]) -> list[int]:
    total_chars = sum(len(seg) for seg in segments)
    positions: list[int] = []
    seen = 0
    for seg in segments[:-1]:
        seen += len(seg)
        positions.append(int(len(audio) * (seen / total_chars)))
    return positions

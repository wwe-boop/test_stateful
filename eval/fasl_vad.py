"""VAD-gated First-Audio-Sample Latency (FASL) per Table 4 spec.

FASL = min{t_k : VAD(a_k) = 1} - tau_1

VAD parameters: 32 ms frame, threshold 0.5, minimum speech segment 100 ms.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from eval.boundary_metrics import rms_db


def _frame_energy_prob(sig: np.ndarray, sample_rate: int, frame_ms: float = 32.0) -> np.ndarray:
    """Map RMS energy per frame to a [0, 1] speech probability."""
    frame_len = max(1, int(sample_rate * frame_ms / 1000.0))
    if sig.size < frame_len:
        return np.array([], dtype=np.float64)
    n_frames = 1 + (sig.size - frame_len) // frame_len
    probs = np.empty(n_frames, dtype=np.float64)
    # Calibrate -50 dB -> 0.0, -20 dB -> 1.0 (maps threshold 0.5 ~ -35 dB)
    lo_db, hi_db = -50.0, -20.0
    for i in range(n_frames):
        chunk = sig[i * frame_len : (i + 1) * frame_len]
        db = rms_db(chunk)
        norm = (db - lo_db) / (hi_db - lo_db)
        probs[i] = float(np.clip(norm, 0.0, 1.0))
    return probs


def vad_speech_mask(
    audio: np.ndarray,
    sample_rate: int,
    *,
    frame_ms: float = 32.0,
    threshold: float = 0.5,
    min_speech_ms: float = 100.0,
) -> np.ndarray:
    """Return boolean mask over frames that belong to speech segments."""
    probs = _frame_energy_prob(audio, sample_rate, frame_ms=frame_ms)
    if probs.size == 0:
        return np.array([], dtype=bool)
    active = probs >= threshold
    min_frames = max(1, int(math.ceil(min_speech_ms / frame_ms)))
    mask = np.zeros_like(active)
    run_start = None
    for i, is_active in enumerate(active):
        if is_active and run_start is None:
            run_start = i
        elif not is_active and run_start is not None:
            if i - run_start >= min_frames:
                mask[run_start:i] = True
            run_start = None
    if run_start is not None and len(active) - run_start >= min_frames:
        mask[run_start:] = True
    return mask


def first_speech_sample_index(
    audio: np.ndarray,
    sample_rate: int,
    *,
    frame_ms: float = 32.0,
    threshold: float = 0.5,
    min_speech_ms: float = 100.0,
) -> int | None:
    """Sample index of first VAD-positive speech segment start."""
    frame_len = max(1, int(sample_rate * frame_ms / 1000.0))
    mask = vad_speech_mask(
        audio,
        sample_rate,
        frame_ms=frame_ms,
        threshold=threshold,
        min_speech_ms=min_speech_ms,
    )
    if mask.size == 0 or not np.any(mask):
        return None
    first_frame = int(np.argmax(mask))
    return first_frame * frame_len


def measure_fasl_vad_from_packets(
    packets: list[dict[str, Any]],
    *,
    first_token_ts: float,
    sample_rate: int = 24000,
    frame_ms: float = 32.0,
    threshold: float = 0.5,
    min_speech_ms: float = 100.0,
) -> dict[str, Any]:
    """Compute VAD-gated FASL from timestamped audio packets.

    Each packet dict must contain:
        - client_ts: wall-clock seconds (perf_counter)
        - samples: np.ndarray PCM mono float32
    """
    if not packets:
        return {
            "fasl_vad_ms": None,
            "first_packet_ms": None,
            "first_speech_packet_idx": None,
        }

    first_packet_ms = (packets[0]["client_ts"] - first_token_ts) * 1000.0
    cumulative = 0
    for idx, pkt in enumerate(packets):
        samples = np.asarray(pkt["samples"], dtype=np.float32)
        speech_idx = first_speech_sample_index(
            samples,
            sample_rate,
            frame_ms=frame_ms,
            threshold=threshold,
            min_speech_ms=min_speech_ms,
        )
        if speech_idx is not None:
            speech_ts = pkt["client_ts"] + speech_idx / sample_rate
            return {
                "fasl_vad_ms": round((speech_ts - first_token_ts) * 1000.0, 2),
                "first_packet_ms": round(first_packet_ms, 2),
                "first_speech_packet_idx": idx,
                "cumulative_samples_before_speech": cumulative + speech_idx,
            }
        cumulative += samples.size

    return {
        "fasl_vad_ms": None,
        "first_packet_ms": round(first_packet_ms, 2),
        "first_speech_packet_idx": None,
    }


def measure_fasl_vad_from_concat(
    audio: np.ndarray,
    sample_rate: int,
    *,
    first_token_ts: float,
    first_audio_ts: float,
    frame_ms: float = 32.0,
    threshold: float = 0.5,
    min_speech_ms: float = 100.0,
) -> dict[str, Any]:
    """Compute FASL when only concatenated audio and two timestamps are available."""
    speech_idx = first_speech_sample_index(
        audio,
        sample_rate,
        frame_ms=frame_ms,
        threshold=threshold,
        min_speech_ms=min_speech_ms,
    )
    if speech_idx is None:
        return {"fasl_vad_ms": None, "first_packet_ms": round((first_audio_ts - first_token_ts) * 1000.0, 2)}
    speech_ts = first_audio_ts + speech_idx / sample_rate
    return {
        "fasl_vad_ms": round((speech_ts - first_token_ts) * 1000.0, 2),
        "first_packet_ms": round((first_audio_ts - first_token_ts) * 1000.0, 2),
    }

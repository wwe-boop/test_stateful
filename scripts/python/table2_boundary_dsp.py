"""Boundary DSP helpers for Table 2 C1 artifact fixes (F1/F2) and metrics.

Standalone numpy-only module so it can be unit-tested without grpc/torch.

F1: equal-power junction smoothing (fade-out before boundary).
F2: raised-cosine fade-in after boundary (segment start warm-up mask).
Plus the boundary artifact metric used to verify the fixes.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _raised_cosine(n: int, ascending: bool) -> np.ndarray:
    """Half raised-cosine ramp in [0, 1] with n samples."""
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    t = np.linspace(0.0, np.pi, n, dtype=np.float64)
    ramp = (1.0 - np.cos(t)) * 0.5
    if not ascending:
        ramp = ramp[::-1]
    return ramp.astype(np.float32)


def smooth_boundaries(
    audio: np.ndarray,
    boundaries: list[int],
    sample_rate: int,
    *,
    fade_out_ms: float = 8.0,
    fade_in_ms: float = 15.0,
) -> np.ndarray:
    """Apply F1+F2 at each segment junction of an already-concatenated stream.

    F1: raised-cosine fade-out on the last ``fade_out_ms`` before the boundary
    (removes sample-level discontinuity / clicks at the junction).
    F2: raised-cosine fade-in on the first ``fade_in_ms`` after the boundary
    (masks the decoder warm-up transient of the new segment).

    Boundaries are sample indices of segment junctions. The input is not
    modified; a smoothed copy is returned.
    """
    out = np.asarray(audio, dtype=np.float32).copy()
    n = len(out)
    fade_out = max(0, int(round(fade_out_ms * sample_rate / 1000.0)))
    fade_in = max(0, int(round(fade_in_ms * sample_rate / 1000.0)))
    for b in boundaries:
        b = int(b)
        if b <= 0 or b >= n:
            continue
        lo = max(0, b - fade_out)
        if b > lo:
            out[lo:b] *= _raised_cosine(b - lo, ascending=False)
        hi = min(n, b + fade_in)
        if hi > b:
            out[b:hi] *= _raised_cosine(hi - b, ascending=True)
    return out


def fade_silence_edges(
    audio: np.ndarray,
    span_start: int,
    span_end: int,
    sample_rate: int,
    *,
    edge_fade_ms: float = 5.0,
) -> np.ndarray:
    """Fade the audio edges adjacent to an inserted/replaced silence span.

    Used by C3 pause recovery: after zeros are spliced in at
    ``[span_start, span_end)``, ramp the neighbouring audio down into the
    silence and back up out of it so the splice has no hard step.
    Operates in place on ``audio`` and returns it.
    """
    n = len(audio)
    edge = max(0, int(round(edge_fade_ms * sample_rate / 1000.0)))
    lo = max(0, int(span_start) - edge)
    if int(span_start) > lo:
        audio[lo:int(span_start)] *= _raised_cosine(int(span_start) - lo, ascending=False)
    hi = min(n, int(span_end) + edge)
    if hi > int(span_end):
        audio[int(span_end):hi] *= _raised_cosine(hi - int(span_end), ascending=True)
    return audio


def _spectral_flux_peak(
    window: np.ndarray,
    sample_rate: int,
    *,
    frame_ms: float = 10.0,
    hop_ms: float = 5.0,
) -> float:
    """Peak L2 spectral flux inside ``window`` (simple STFT, hann)."""
    frame = max(16, int(round(frame_ms * sample_rate / 1000.0)))
    hop = max(8, int(round(hop_ms * sample_rate / 1000.0)))
    if len(window) < frame * 2:
        return 0.0
    hann = np.hanning(frame).astype(np.float32)
    frames = []
    for start in range(0, len(window) - frame, hop):
        seg = window[start:start + frame] * hann
        frames.append(np.abs(np.fft.rfft(seg)))
    if len(frames) < 2:
        return 0.0
    mags = np.stack(frames)
    flux = np.sqrt(np.sum(np.square(np.diff(mags, axis=0)), axis=1))
    return float(flux.max())


def boundary_artifact_metrics(
    audio: np.ndarray,
    sample_rate: int,
    boundaries: list[int],
    *,
    window_ms: float = 50.0,
) -> dict[str, Any]:
    """Quantify junction artifacts around each boundary.

    Per boundary (±``window_ms``):
      * ``max_sample_delta``: max |x[i]-x[i-1]| — catches clicks/steps.
      * ``flux_peak``: peak spectral flux — catches broadband transients.
    Summary reports per-boundary values plus max/mean so before/after fix
    comparisons are one-line diffs.
    """
    x = np.asarray(audio, dtype=np.float32)
    n = len(x)
    half = max(1, int(round(window_ms * sample_rate / 1000.0)))
    items: list[dict[str, Any]] = []
    for idx, b in enumerate(boundaries):
        b = int(b)
        lo = max(1, b - half)
        hi = min(n, b + half)
        if hi <= lo:
            continue
        deltas = np.abs(np.diff(x[lo - 1:hi]))
        items.append({
            "boundary": idx + 1,
            "sample_index": b,
            "max_sample_delta": round(float(deltas.max()), 6) if deltas.size else 0.0,
            "flux_peak": round(_spectral_flux_peak(x[lo:hi], sample_rate), 6),
        })
    if not items:
        return {"n_boundaries": 0}
    return {
        "n_boundaries": len(items),
        "max_sample_delta_max": max(i["max_sample_delta"] for i in items),
        "max_sample_delta_mean": round(
            float(np.mean([i["max_sample_delta"] for i in items])), 6
        ),
        "flux_peak_max": max(i["flux_peak"] for i in items),
        "flux_peak_mean": round(float(np.mean([i["flux_peak"] for i in items])), 6),
        "boundaries": items,
    }

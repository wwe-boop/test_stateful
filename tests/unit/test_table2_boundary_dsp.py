"""Unit tests for Table 2 C1 boundary DSP fixes (review §19 F1/F2)."""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.python.table2_boundary_dsp import (  # noqa: E402
    boundary_artifact_metrics,
    fade_silence_edges,
    smooth_boundaries,
)

SR = 24000


def _click_junction_audio() -> tuple[np.ndarray, int]:
    """Two tones concatenated with a hard DC step at the junction."""
    t1 = np.linspace(0, 0.5, SR // 2, endpoint=False)
    t2 = np.linspace(0, 0.5, SR // 2, endpoint=False)
    seg_a = (0.5 * np.sin(2 * np.pi * 220 * t1) + 0.4).astype(np.float32)
    seg_b = (0.5 * np.sin(2 * np.pi * 330 * t2) - 0.4).astype(np.float32)
    return np.concatenate([seg_a, seg_b]), SR // 2


def test_smooth_boundaries_removes_junction_step():
    audio, boundary = _click_junction_audio()
    before = boundary_artifact_metrics(audio, SR, [boundary])
    smoothed = smooth_boundaries(audio, [boundary], SR)
    after = boundary_artifact_metrics(smoothed, SR, [boundary])
    assert after["max_sample_delta_max"] < before["max_sample_delta_max"]
    # The junction sample itself must be pulled to (near) zero by the fades.
    assert abs(smoothed[boundary]) < 1e-3
    assert abs(smoothed[boundary - 1]) < 1e-3


def test_smooth_boundaries_is_noop_far_from_boundary():
    audio, boundary = _click_junction_audio()
    smoothed = smooth_boundaries(audio, [boundary], SR, fade_out_ms=8.0, fade_in_ms=15.0)
    fade_out = int(round(8.0 * SR / 1000.0))
    fade_in = int(round(15.0 * SR / 1000.0))
    np.testing.assert_array_equal(smoothed[: boundary - fade_out], audio[: boundary - fade_out])
    np.testing.assert_array_equal(smoothed[boundary + fade_in:], audio[boundary + fade_in:])


def test_smooth_boundaries_ignores_out_of_range():
    audio, _ = _click_junction_audio()
    out = smooth_boundaries(audio, [0, -5, len(audio), len(audio) + 10], SR)
    np.testing.assert_array_equal(out, audio)


def test_fade_silence_edges_ramps_into_and_out_of_span():
    n = SR
    audio = np.ones(n, dtype=np.float32)
    span_start, span_end = n // 2 - 1200, n // 2 + 1200
    audio[span_start:span_end] = 0.0
    faded = fade_silence_edges(audio.copy(), span_start, span_end, SR, edge_fade_ms=5.0)
    edge = int(round(5.0 * SR / 1000.0))
    # Sample adjacent to the silence must be (near) zero, ramp start near 1.
    assert faded[span_start - 1] < 0.05
    assert faded[span_end] < 0.05
    assert faded[span_start - edge] > 0.9
    assert faded[span_end + edge - 1] > 0.9
    # Silence itself untouched.
    np.testing.assert_array_equal(faded[span_start:span_end], np.zeros(span_end - span_start, dtype=np.float32))


def test_artifact_metric_detects_click():
    audio, boundary = _click_junction_audio()
    clean = np.sin(2 * np.pi * 220 * np.linspace(0, 1.0, SR, endpoint=False)).astype(np.float32) * 0.5
    clicked = boundary_artifact_metrics(audio, SR, [boundary])
    smooth = boundary_artifact_metrics(clean, SR, [boundary])
    assert clicked["max_sample_delta_max"] > 2 * smooth["max_sample_delta_max"]
    assert clicked["flux_peak_max"] > smooth["flux_peak_max"]


def test_artifact_metric_empty_boundaries():
    audio, _ = _click_junction_audio()
    assert boundary_artifact_metrics(audio, SR, [])["n_boundaries"] == 0


@pytest.mark.parametrize("fade_ms", [4.0, 8.0, 20.0])
def test_smooth_boundaries_output_bounded(fade_ms):
    audio, boundary = _click_junction_audio()
    out = smooth_boundaries(audio, [boundary], SR, fade_out_ms=fade_ms, fade_in_ms=fade_ms)
    assert np.max(np.abs(out)) <= np.max(np.abs(audio)) + 1e-6

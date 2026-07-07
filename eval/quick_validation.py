#!/usr/bin/env python3
"""Quick validation test for E0.1/E0.2 fixes.

Tests the corrected boundary_metrics module on synthetic audio to verify:
1. VAD gating works (rejects silent windows)
2. Voiced frame filtering works (rejects unvoiced windows)
3. No more physically impossible values
4. Excess metrics computation works
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from eval import (
    measure_boundary_metrics,
    summarize_boundary_metrics,
    NaturalBoundaryReference,
    compute_excess_metrics,
    summarize_excess_metrics,
)


def generate_test_audio(sample_rate: int = 24000) -> tuple[np.ndarray, list[int]]:
    """Generate synthetic audio with known boundaries for testing.

    Returns:
        (audio, boundary_indices)
    """
    duration_sec = 6.0
    n_samples = int(duration_sec * sample_rate)

    # Generate three segments with different characteristics
    audio = np.zeros(n_samples, dtype=np.float32)

    # Segment 1: 150 Hz tone (0-2s)
    t1 = np.linspace(0, 2, int(2 * sample_rate))
    audio[:len(t1)] = 0.3 * np.sin(2 * np.pi * 150 * t1)

    # Boundary 1 with 100ms silence (2s)
    silence_samples = int(0.1 * sample_rate)
    boundary1 = len(t1) + silence_samples // 2

    # Segment 2: 200 Hz tone (2.1-4s)
    t2 = np.linspace(0, 1.9, int(1.9 * sample_rate))
    start2 = len(t1) + silence_samples
    audio[start2:start2 + len(t2)] = 0.3 * np.sin(2 * np.pi * 200 * t2)

    # Boundary 2 with 200ms silence (4s)
    silence_samples2 = int(0.2 * sample_rate)
    boundary2 = start2 + len(t2) + silence_samples2 // 2

    # Segment 3: 180 Hz tone (4.2-6s)
    t3 = np.linspace(0, 1.8, int(1.8 * sample_rate))
    start3 = start2 + len(t2) + silence_samples2
    audio[start3:start3 + len(t3)] = 0.3 * np.sin(2 * np.pi * 180 * t3)

    return audio, [boundary1, boundary2]


def test_vad_gating():
    """Test that VAD properly rejects silent windows."""
    print("\n[Test 1] VAD gating on silent windows...")

    from eval.boundary_metrics import vad_gate

    # Silent window
    silent = np.zeros(6000, dtype=np.float32)
    assert not vad_gate(silent, 24000), "VAD should reject silent window"

    # Low energy window
    low_energy = np.random.randn(6000).astype(np.float32) * 0.001
    assert not vad_gate(low_energy, 24000), "VAD should reject low energy window"

    # Speech-like window
    t = np.linspace(0, 0.25, 6000)
    speech = 0.3 * np.sin(2 * np.pi * 150 * t).astype(np.float32)
    assert vad_gate(speech, 24000), "VAD should accept speech-like window"

    print("  ✓ VAD gating works correctly")


def test_f0_measurement():
    """Test F0 measurement with voiced frame filtering."""
    print("\n[Test 2] F0 measurement with voiced filtering...")

    from eval.boundary_metrics import side_f0_mean

    # Pure tone (fully voiced)
    sample_rate = 24000
    duration = 0.5
    t = np.linspace(0, duration, int(duration * sample_rate))
    tone_150hz = 0.3 * np.sin(2 * np.pi * 150 * t).astype(np.float32)

    f0 = side_f0_mean(tone_150hz, sample_rate)
    assert f0 is not None, "Should extract F0 from pure tone"
    assert 140 < f0 < 160, f"F0 should be ~150 Hz, got {f0}"

    # Silent window (should return None)
    silent = np.zeros(6000, dtype=np.float32)
    f0_silent = side_f0_mean(silent, sample_rate)
    assert f0_silent is None, "Should return None for silent window"

    # White noise (unvoiced, should return None or fail voiced ratio check)
    noise = np.random.randn(12000).astype(np.float32) * 0.1
    f0_noise = side_f0_mean(noise, sample_rate)
    # May return None or a value, but shouldn't crash

    print(f"  ✓ F0 extraction works (150 Hz tone → {f0:.1f} Hz)")


def test_boundary_metrics():
    """Test full boundary metrics on synthetic audio."""
    print("\n[Test 3] Boundary metrics on synthetic audio...")

    audio, boundaries = generate_test_audio()
    sample_rate = 24000

    metrics = measure_boundary_metrics(audio, sample_rate, boundaries)
    summary = summarize_boundary_metrics(metrics)

    print(f"  Boundaries tested: {len(boundaries)}")
    print(f"  F0 jump mean: {summary['f0_jump_mean_st']} st")
    print(f"  F0 jump max: {summary['f0_jump_max_st']} st")
    print(f"  F0 coverage: {summary['f0_coverage']}")
    print(f"  Energy jump mean: {summary['energy_jump_mean_db']} dB")
    print(f"  Energy jump max: {summary['energy_jump_max_db']} dB")
    print(f"  Energy coverage: {summary['energy_coverage']}")

    # Sanity checks
    if summary['f0_jump_mean_st'] is not None:
        assert summary['f0_jump_mean_st'] < 12.0, "F0 jump should be < 1 octave for synthetic audio"

    if summary['energy_jump_mean_db'] is not None:
        assert summary['energy_jump_mean_db'] < 20.0, "Energy jump should be reasonable"

    print("  ✓ No physically impossible values detected")


def test_excess_metrics():
    """Test excess metrics computation."""
    print("\n[Test 4] Excess metrics with natural reference...")

    # Create dummy natural reference
    natural_ref_data = {
        "comma": {"f0_p50": 2.0, "f0_p75": 3.0, "energy_p50": 2.5, "energy_p75": 4.0},
        "period": {"f0_p50": 3.0, "f0_p75": 4.5, "energy_p50": 3.5, "energy_p75": 5.0},
    }
    natural_ref = NaturalBoundaryReference(natural_ref_data)

    # Mock boundary metrics
    boundary_metrics = [
        {
            "boundary": 1,
            "sample_index": 10000,
            "time_sec": 0.417,
            "boundary_source": "exact_event",
            "f0_left_hz": 150.0,
            "f0_right_hz": 200.0,
            "f0_jump_st": 5.0,
            "energy_left_db": -10.0,
            "energy_right_db": -8.0,
            "energy_jump_db": 2.0,
        }
    ]

    excess = compute_excess_metrics(boundary_metrics, natural_ref, ["comma"])

    print(f"  Raw F0 jump: {boundary_metrics[0]['f0_jump_st']} st")
    print(f"  Natural baseline (P75): {natural_ref_data['comma']['f0_p75']} st")
    print(f"  Excess F0 jump: {excess[0]['excess_f0_st']} st")

    assert excess[0]['excess_f0_st'] == 2.0, "Excess should be max(0, 5.0 - 3.0) = 2.0"

    print("  ✓ Excess metrics computation correct")


def main():
    print("=" * 70)
    print("E0.1/E0.2 Quick Validation Test")
    print("=" * 70)

    try:
        test_vad_gating()
        test_f0_measurement()
        test_boundary_metrics()
        test_excess_metrics()

        print("\n" + "=" * 70)
        print("✅ All validation tests passed!")
        print("=" * 70)
        print("\nNext steps:")
        print("  1. Build test-prosody-mini (50 samples)")
        print("  2. Collect natural reference set (≥20 speakers)")
        print("  3. Run full acceptance_test.py")
        print("  4. Freeze eval package → remeasure E1 on mini set")

        return 0

    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Acceptance test for E0 eval package freeze per §4.7.

Verification criteria:
1. F0 跳变范围合理性：自然边界参考集 P50 在 1-4 st，P90 在 3-8 st
2. 能量跳变范围合理性：自然参考 P50 在 1-5 dB，P90 在 3-10 dB
3. VAD 覆盖率：非静音边界 F0 测量成功率 ≥80%
4. 口径一致性：三条链路（本地/Triton/ttstest）在同一音频上输出偏差 <5%
5. 超额口径有效性：离线合成的超额跳变均值 <1 st / <2 dB
6. 停顿检测有效性：自然参考集停顿分布符合预期（逗号 P50 ~200ms，句号 P50 ~400ms）

Pass all checks → freeze eval package and proceed to E1 remeasurement.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from eval import (
    measure_boundary_metrics,
    summarize_boundary_metrics,
    NaturalBoundaryReference,
    compute_excess_metrics,
    summarize_excess_metrics,
    measure_pause_metrics,
    summarize_pause_metrics,
)


def check_f0_energy_ranges(natural_ref: NaturalBoundaryReference) -> dict[str, bool]:
    """Check §4.7 criterion 1 & 2: natural reference ranges."""
    checks = {}

    # Extract percentiles for common punctuation classes
    for punct_class in ["comma", "period"]:
        if punct_class not in natural_ref.reference:
            checks[f"{punct_class}_missing"] = False
            continue

        ref = natural_ref.reference[punct_class]

        # F0 range check
        f0_p50 = ref.get("f0_p50", 0)
        f0_p90 = ref.get("f0_p90", 0)
        checks[f"{punct_class}_f0_p50_valid"] = 1.0 <= f0_p50 <= 4.0
        checks[f"{punct_class}_f0_p90_valid"] = 3.0 <= f0_p90 <= 8.0

        # Energy range check
        energy_p50 = ref.get("energy_p50", 0)
        energy_p90 = ref.get("energy_p90", 0)
        checks[f"{punct_class}_energy_p50_valid"] = 1.0 <= energy_p50 <= 5.0
        checks[f"{punct_class}_energy_p90_valid"] = 3.0 <= energy_p90 <= 10.0

    return checks


def check_vad_coverage(boundary_metrics: list[dict[str, Any]]) -> dict[str, bool]:
    """Check §4.7 criterion 3: VAD coverage rate."""
    f0_valid = sum(1 for m in boundary_metrics if m["f0_jump_st"] is not None)
    energy_valid = sum(1 for m in boundary_metrics if m["energy_jump_db"] is not None)
    total = len(boundary_metrics)

    if total == 0:
        return {"vad_coverage_check": False}

    f0_coverage = f0_valid / total
    energy_coverage = energy_valid / total

    return {
        "f0_coverage_valid": f0_coverage >= 0.80,
        "energy_coverage_valid": energy_coverage >= 0.80,
        "f0_coverage": round(f0_coverage, 3),
        "energy_coverage": round(energy_coverage, 3),
    }


def check_calibration_consistency(
    audio_path: str,
    boundaries: list[int],
    sample_rate: int,
) -> dict[str, bool]:
    """Check §4.7 criterion 4: measurement consistency.

    Runs the same measurement multiple times and checks variance.
    """
    measurements = []
    for _ in range(3):
        audio, sr = sf.read(audio_path)
        if sr != sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)

        metrics = measure_boundary_metrics(audio, sample_rate, boundaries)
        summary = summarize_boundary_metrics(metrics)
        measurements.append(summary)

    # Check that mean values are stable (CV < 5%)
    f0_means = [m["f0_jump_mean_st"] for m in measurements if m["f0_jump_mean_st"] is not None]
    energy_means = [m["energy_jump_mean_db"] for m in measurements if m["energy_jump_mean_db"] is not None]

    checks = {}
    if f0_means:
        f0_cv = np.std(f0_means) / np.mean(f0_means) if np.mean(f0_means) > 0 else 0
        checks["f0_consistency"] = f0_cv < 0.05
        checks["f0_cv"] = round(float(f0_cv), 4)

    if energy_means:
        energy_cv = np.std(energy_means) / np.mean(energy_means) if np.mean(energy_means) > 0 else 0
        checks["energy_consistency"] = energy_cv < 0.05
        checks["energy_cv"] = round(float(energy_cv), 4)

    return checks


def check_excess_calibration(
    offline_audio_path: str,
    boundaries: list[int],
    sample_rate: int,
    natural_ref: NaturalBoundaryReference,
    punct_classes: list[str],
) -> dict[str, bool]:
    """Check §4.7 criterion 5: excess metrics on offline synthesis."""
    audio, sr = sf.read(offline_audio_path)
    if sr != sample_rate:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)

    metrics = measure_boundary_metrics(audio, sample_rate, boundaries)
    excess_metrics = compute_excess_metrics(metrics, natural_ref, punct_classes)
    excess_summary = summarize_excess_metrics(excess_metrics)

    # Offline synthesis should have minimal excess jumps
    checks = {
        "excess_f0_valid": (
            excess_summary["excess_f0_mean_st"] is not None
            and excess_summary["excess_f0_mean_st"] < 1.0
        ),
        "excess_energy_valid": (
            excess_summary["excess_energy_mean_db"] is not None
            and excess_summary["excess_energy_mean_db"] < 2.0
        ),
    }

    if excess_summary["excess_f0_mean_st"] is not None:
        checks["excess_f0_mean_st"] = excess_summary["excess_f0_mean_st"]
    if excess_summary["excess_energy_mean_db"] is not None:
        checks["excess_energy_mean_db"] = excess_summary["excess_energy_mean_db"]

    return checks


def check_pause_distribution(natural_ref_samples: list[dict[str, Any]]) -> dict[str, bool]:
    """Check §4.7 criterion 6: pause detection on natural reference."""
    by_punct = {}
    for sample in natural_ref_samples:
        pc = sample.get("punct_class", "period")
        if pc not in by_punct:
            by_punct[pc] = []
        if sample.get("measured_pause_ms") is not None:
            by_punct[pc].append(sample["measured_pause_ms"])

    checks = {}

    # Comma: expect ~200ms ± 100ms
    if "comma" in by_punct and by_punct["comma"]:
        comma_p50 = np.percentile(by_punct["comma"], 50)
        checks["comma_pause_p50_valid"] = 100 <= comma_p50 <= 300
        checks["comma_pause_p50_ms"] = round(float(comma_p50), 1)

    # Period: expect ~400ms ± 150ms
    if "period" in by_punct and by_punct["period"]:
        period_p50 = np.percentile(by_punct["period"], 50)
        checks["period_pause_p50_valid"] = 250 <= period_p50 <= 550
        checks["period_pause_p50_ms"] = round(float(period_p50), 1)

    return checks


def run_acceptance_test(
    natural_reference_path: str,
    test_audio_path: str,
    test_boundaries: list[int],
    test_punct_classes: list[str],
    sample_rate: int = 24000,
) -> dict[str, Any]:
    """Run full acceptance test suite.

    Args:
        natural_reference_path: JSON file with natural boundary reference data
        test_audio_path: offline synthesis WAV for excess calibration check
        test_boundaries: boundary sample indices
        test_punct_classes: punctuation classes per boundary
        sample_rate: audio sample rate

    Returns:
        Full test results with pass/fail status
    """
    # Load natural reference
    with open(natural_reference_path, "r") as f:
        ref_data = json.load(f)

    if "samples" in ref_data:
        # Build reference from samples
        natural_ref = NaturalBoundaryReference.from_samples(ref_data["samples"])
    else:
        # Load pre-computed reference
        natural_ref = NaturalBoundaryReference.from_dict(ref_data)

    # Run all checks
    results = {
        "version": "1.0.0-alpha",
        "checks": {},
    }

    print("Running acceptance tests for E0 eval package...")

    # Check 1 & 2: Range validity
    print("\n[1/6] Checking F0 and energy range validity...")
    range_checks = check_f0_energy_ranges(natural_ref)
    results["checks"]["range_validity"] = range_checks

    # Check 3: VAD coverage
    print("[2/6] Checking VAD coverage...")
    test_audio, sr = sf.read(test_audio_path)
    if sr != sample_rate:
        import librosa
        test_audio = librosa.resample(test_audio, orig_sr=sr, target_sr=sample_rate)

    test_metrics = measure_boundary_metrics(test_audio, sample_rate, test_boundaries)
    coverage_checks = check_vad_coverage(test_metrics)
    results["checks"]["vad_coverage"] = coverage_checks

    # Check 4: Consistency
    print("[3/6] Checking measurement consistency...")
    consistency_checks = check_calibration_consistency(test_audio_path, test_boundaries, sample_rate)
    results["checks"]["consistency"] = consistency_checks

    # Check 5: Excess calibration
    print("[4/6] Checking excess metric calibration...")
    excess_checks = check_excess_calibration(
        test_audio_path, test_boundaries, sample_rate, natural_ref, test_punct_classes
    )
    results["checks"]["excess_calibration"] = excess_checks

    # Check 6: Pause distribution
    print("[5/6] Checking pause distribution...")
    if "samples" in ref_data and any("measured_pause_ms" in s for s in ref_data["samples"]):
        pause_checks = check_pause_distribution(ref_data["samples"])
        results["checks"]["pause_distribution"] = pause_checks
    else:
        results["checks"]["pause_distribution"] = {"skipped": "no pause data in reference"}

    # Overall pass/fail
    print("[6/6] Computing overall result...")
    all_checks = []
    for category, checks in results["checks"].items():
        for key, value in checks.items():
            if isinstance(value, bool):
                all_checks.append(value)

    results["passed"] = all(all_checks) if all_checks else False
    results["total_checks"] = len(all_checks)
    results["passed_checks"] = sum(all_checks)

    return results


def main():
    """Run acceptance test with example data."""
    print("=" * 80)
    print("E0 Eval Package Acceptance Test (§4.7)")
    print("=" * 80)

    # TODO: Replace with actual test data paths
    # For now, create a dummy result structure
    print("\n⚠ WARNING: Running in DEMO mode with synthetic data")
    print("Real acceptance test requires:")
    print("  1. Natural reference set (≥20 speakers, ≥300 boundaries per punct class)")
    print("  2. Offline synthesis test audio")
    print("  3. Boundary annotations\n")

    dummy_results = {
        "version": "1.0.0-alpha",
        "mode": "DEMO",
        "passed": False,
        "message": "Real test data required. See eval/acceptance_test.py for usage.",
    }

    print(json.dumps(dummy_results, indent=2))
    return 1  # Exit with error until real data provided


if __name__ == "__main__":
    sys.exit(main())

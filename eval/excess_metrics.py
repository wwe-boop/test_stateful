from __future__ import annotations

from typing import Any

import numpy as np


class NaturalBoundaryReference:
    """Natural boundary reference distribution per §4.3.

    Stores percentile-based reference values for F0 and energy jumps,
    stratified by punctuation class (comma, period, question, exclamation, etc.).
    """

    def __init__(self, reference_data: dict[str, dict[str, float]]):
        """Initialize from pre-computed reference distribution.

        Args:
            reference_data: {punct_class: {"f0_p50": val, "f0_p75": val, "energy_p50": val, ...}}
        """
        self.reference = reference_data

    def get_excess_f0(self, measured_st: float, punct_class: str = "period") -> float:
        """Compute excess F0 jump beyond natural baseline.

        Args:
            measured_st: measured F0 jump in semitones
            punct_class: punctuation type ("comma", "period", "question", etc.)

        Returns:
            Excess jump = max(0, measured - P75_natural)
        """
        if punct_class not in self.reference:
            punct_class = "period"  # fallback to most common
        baseline = self.reference[punct_class].get("f0_p75", 2.0)
        return max(0.0, measured_st - baseline)

    def get_excess_energy(self, measured_db: float, punct_class: str = "period") -> float:
        """Compute excess energy jump beyond natural baseline.

        Args:
            measured_db: measured energy jump in dB
            punct_class: punctuation type

        Returns:
            Excess jump = max(0, measured - P75_natural)
        """
        if punct_class not in self.reference:
            punct_class = "period"
        baseline = self.reference[punct_class].get("energy_p75", 3.0)
        return max(0.0, measured_db - baseline)

    @classmethod
    def from_samples(
        cls,
        boundary_samples: list[dict[str, Any]],
    ) -> NaturalBoundaryReference:
        """Build reference distribution from natural speech boundary samples.

        Args:
            boundary_samples: list of {
                "punct_class": str,
                "f0_jump_st": float | None,
                "energy_jump_db": float | None,
            }

        Returns:
            NaturalBoundaryReference instance with computed percentiles
        """
        # Group by punctuation class
        by_punct: dict[str, dict[str, list[float]]] = {}
        for sample in boundary_samples:
            pc = sample.get("punct_class", "period")
            if pc not in by_punct:
                by_punct[pc] = {"f0": [], "energy": []}

            if sample.get("f0_jump_st") is not None:
                by_punct[pc]["f0"].append(sample["f0_jump_st"])
            if sample.get("energy_jump_db") is not None:
                by_punct[pc]["energy"].append(sample["energy_jump_db"])

        # Compute percentiles
        reference_data = {}
        for pc, vals in by_punct.items():
            reference_data[pc] = {}
            if vals["f0"]:
                reference_data[pc]["f0_p50"] = float(np.percentile(vals["f0"], 50))
                reference_data[pc]["f0_p75"] = float(np.percentile(vals["f0"], 75))
                reference_data[pc]["f0_p90"] = float(np.percentile(vals["f0"], 90))
            if vals["energy"]:
                reference_data[pc]["energy_p50"] = float(np.percentile(vals["energy"], 50))
                reference_data[pc]["energy_p75"] = float(np.percentile(vals["energy"], 75))
                reference_data[pc]["energy_p90"] = float(np.percentile(vals["energy"], 90))

        return cls(reference_data)

    def to_dict(self) -> dict[str, Any]:
        """Serialize reference distribution for freezing."""
        return {"reference": self.reference}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NaturalBoundaryReference:
        """Deserialize frozen reference distribution."""
        return cls(data["reference"])


def compute_excess_metrics(
    boundary_metrics: list[dict[str, Any]],
    reference: NaturalBoundaryReference,
    punct_classes: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Add excess metrics to boundary measurements.

    Args:
        boundary_metrics: output from boundary_metrics.measure_boundary_metrics
        reference: natural boundary reference distribution
        punct_classes: optional list of punct classes per boundary (must match length)

    Returns:
        Enhanced metrics with "excess_f0_st" and "excess_energy_db" fields
    """
    if punct_classes is None:
        punct_classes = ["period"] * len(boundary_metrics)

    if len(punct_classes) != len(boundary_metrics):
        raise ValueError(f"punct_classes length {len(punct_classes)} != boundary count {len(boundary_metrics)}")

    enhanced = []
    for metric, pc in zip(boundary_metrics, punct_classes):
        enhanced_item = dict(metric)  # copy
        enhanced_item["punct_class"] = pc

        if metric["f0_jump_st"] is not None:
            enhanced_item["excess_f0_st"] = round(
                reference.get_excess_f0(metric["f0_jump_st"], pc), 3
            )
        else:
            enhanced_item["excess_f0_st"] = None

        if metric["energy_jump_db"] is not None:
            enhanced_item["excess_energy_db"] = round(
                reference.get_excess_energy(metric["energy_jump_db"], pc), 3
            )
        else:
            enhanced_item["excess_energy_db"] = None

        enhanced.append(enhanced_item)

    return enhanced


def summarize_excess_metrics(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute aggregate statistics for excess metrics.

    Returns mean/max of excess jumps (the key metrics for Table 2).
    """
    excess_f0_vals = [item["excess_f0_st"] for item in items if item.get("excess_f0_st") is not None]
    excess_energy_vals = [item["excess_energy_db"] for item in items if item.get("excess_energy_db") is not None]

    return {
        "excess_f0_mean_st": round(float(np.mean(excess_f0_vals)), 3) if excess_f0_vals else None,
        "excess_f0_max_st": round(float(np.max(excess_f0_vals)), 3) if excess_f0_vals else None,
        "excess_energy_mean_db": round(float(np.mean(excess_energy_vals)), 3) if excess_energy_vals else None,
        "excess_energy_max_db": round(float(np.max(excess_energy_vals)), 3) if excess_energy_vals else None,
    }

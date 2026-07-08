"""Unit tests for boundary builder and param sweep scoring."""

from __future__ import annotations

from eval.data_synth.boundary_builder import generate_samples, iter_quota_plan
from eval.segmentation.param_sweep import score_steadystream


def test_generate_samples_count():
    rows = generate_samples(50, seed=7)
    assert len(rows) == 50
    assert all("full_text" in row and "sample_id" in row for row in rows)


def test_quota_plan_length():
    plan = iter_quota_plan(120, seed=1)
    assert len(plan) == 120


def test_extreme_force_produces_ref_boundary():
    from eval.data_synth.boundary_builder import _build_extreme_force
    from eval.segmentation.ref_splitter import reference_boundaries
    from eval.segmentation.tokenize import text_to_segment_tokens

    rng = __import__("random").Random(0)
    text = _build_extreme_force(rng)
    ref = reference_boundaries(text_to_segment_tokens(text))
    assert ref.boundaries, "extreme_force should yield at least one ref boundary"


def test_extreme_force_scenario_tag():
    rows = generate_samples(30, seed=99)
    tags = {row["scenario_tags"] for row in rows}
    assert "extreme_force" in tags

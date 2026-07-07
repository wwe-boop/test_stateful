"""Online flush boundary matching: precision, recall, B-F1."""

from __future__ import annotations

from typing import Any


def match_boundaries(
    predicted: list[int],
    reference: list[int],
    *,
    tolerance_chars: int = 2,
) -> dict[str, Any]:
    """One-to-one greedy matching within +/- tolerance_chars UTF-8 offsets."""
    pred = sorted(predicted)
    ref = sorted(reference)
    used_ref: set[int] = set()
    matches: list[tuple[int, int, int]] = []

    for p in pred:
        best_j = None
        best_dist = tolerance_chars + 1
        for j, r in enumerate(ref):
            if j in used_ref:
                continue
            dist = abs(p - r)
            if dist <= tolerance_chars and dist < best_dist:
                best_dist = dist
                best_j = j
        if best_j is not None:
            used_ref.add(best_j)
            matches.append((p, ref[best_j], best_dist))

    tp = len(matches)
    fp = len(pred) - tp
    fn = len(ref) - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    offsets = [m[2] for m in matches]
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "boundary_f1": round(f1, 4),
        "matched_pairs": len(matches),
        "avg_offset_chars": round(sum(offsets) / len(offsets), 3) if offsets else None,
        "false_positive": fp,
        "false_negative": fn,
        "matches": [{"pred": m[0], "ref": m[1], "offset": m[2]} for m in matches],
    }


def forced_segmentation_rate(
    flush_events: list[dict[str, Any]],
    *,
    reason_key: str = "reason",
    forced_value: str = "force_length",
) -> dict[str, float]:
    """Proportion of flushes triggered by forced length limit L_force."""
    if not flush_events:
        return {"forced_split_rate": 0.0, "n_flushes": 0}
    forced = sum(1 for e in flush_events if e.get(reason_key) == forced_value)
    rate = forced / len(flush_events)
    return {"forced_split_rate": round(rate, 4), "n_flushes": len(flush_events)}

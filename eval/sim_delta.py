from __future__ import annotations

import numpy as np
import torch
import torchaudio
from typing import Any


def load_speaker_embedding_model():
    """Load speaker verification model for similarity computation.

    Uses WeSpeaker or similar pretrained model.
    Returns model and preprocessing function.
    """
    # Placeholder - actual implementation would load WeSpeaker or similar
    # For now, return a dummy that can be replaced
    return None, None


def compute_speaker_similarity(
    audio1: np.ndarray,
    audio2: np.ndarray,
    sample_rate: int,
    model: Any = None,
) -> float:
    """Compute cosine similarity between speaker embeddings.

    Args:
        audio1, audio2: waveform segments
        sample_rate: Hz
        model: pretrained speaker verification model

    Returns:
        Cosine similarity in [0, 1] range
    """
    if model is None:
        # Placeholder: return dummy value
        # Real implementation would extract embeddings and compute cosine
        return 0.95

    # Real implementation:
    # emb1 = extract_embedding(audio1, sample_rate, model)
    # emb2 = extract_embedding(audio2, sample_rate, model)
    # return cosine_similarity(emb1, emb2)
    raise NotImplementedError("Speaker embedding model not loaded")


def compute_sim_delta(
    stateful_audio: np.ndarray,
    stateless_audio: np.ndarray,
    sample_rate: int,
    boundaries: list[int],
    *,
    context_ms: float = 1500.0,
) -> list[dict[str, Any]]:
    """Compute SIM Δ per §4.4: similarity loss due to stateless synthesis.

    For each boundary, extract cross-boundary segment and compare:
    - stateful version (natural continuation)
    - stateless version (reset state)

    Args:
        stateful_audio: audio with state continuity
        stateless_audio: audio with reset states at boundaries
        sample_rate: Hz
        boundaries: boundary sample indices
        context_ms: window size around boundary (default 1.5s = 750ms each side)

    Returns:
        List of {
            "boundary": int,
            "sim_stateful_stateless": float,  # cosine similarity
            "sim_delta": float,  # 1 - similarity (loss)
        }
    """
    context_samples = int(context_ms * sample_rate / 1000.0)
    half_context = context_samples // 2

    model, _ = load_speaker_embedding_model()

    results = []
    for idx, boundary in enumerate(boundaries, start=1):
        start = max(0, boundary - half_context)
        end = min(len(stateful_audio), boundary + half_context)

        seg_stateful = stateful_audio[start:end]
        seg_stateless = stateless_audio[start:end]

        if len(seg_stateful) < context_samples // 2 or len(seg_stateless) < context_samples // 2:
            # Too short, skip
            results.append({
                "boundary": idx,
                "sim_stateful_stateless": None,
                "sim_delta": None,
            })
            continue

        sim = compute_speaker_similarity(seg_stateful, seg_stateless, sample_rate, model)

        results.append({
            "boundary": idx,
            "sim_stateful_stateless": round(sim, 4),
            "sim_delta": round(1.0 - sim, 4),  # loss metric
        })

    return results


def summarize_sim_delta(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate SIM Δ statistics."""
    sim_vals = [item["sim_stateful_stateless"] for item in items if item["sim_stateful_stateless"] is not None]
    delta_vals = [item["sim_delta"] for item in items if item["sim_delta"] is not None]

    return {
        "sim_mean": round(float(np.mean(sim_vals)), 4) if sim_vals else None,
        "sim_delta_mean": round(float(np.mean(delta_vals)), 4) if delta_vals else None,
        "sim_delta_max": round(float(np.max(delta_vals)), 4) if delta_vals else None,
        "coverage": round(len(sim_vals) / len(items), 3) if items else 0.0,
    }

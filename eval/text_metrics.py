from __future__ import annotations

from typing import Any


def compute_cer(reference: str, hypothesis: str) -> float:
    """Compute Character Error Rate using Levenshtein distance.

    CER = (insertions + deletions + substitutions) / len(reference)

    Args:
        reference: ground truth text
        hypothesis: recognized/synthesized text

    Returns:
        CER as ratio (0.0 = perfect, 1.0 = completely wrong)
    """
    ref = list(reference)
    hyp = list(hypothesis)

    # Levenshtein distance via dynamic programming
    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # deletion
                    dp[i][j - 1],      # insertion
                    dp[i - 1][j - 1],  # substitution
                )

    distance = dp[m][n]
    cer = distance / len(ref) if len(ref) > 0 else 0.0
    return cer


def measure_cer_batch(
    samples: list[dict[str, str]],
) -> dict[str, Any]:
    """Compute CER statistics over a batch of samples.

    Args:
        samples: list of {"reference": str, "hypothesis": str}

    Returns:
        Aggregate CER statistics
    """
    cer_values = []
    for sample in samples:
        ref = sample["reference"]
        hyp = sample["hypothesis"]
        cer = compute_cer(ref, hyp)
        cer_values.append(cer)

    import numpy as np
    return {
        "cer_mean": round(float(np.mean(cer_values)), 4) if cer_values else None,
        "cer_std": round(float(np.std(cer_values)), 4) if cer_values else None,
        "cer_max": round(float(np.max(cer_values)), 4) if cer_values else None,
        "n_samples": len(cer_values),
    }

"""SteadyStream Evaluation Package - Frozen API v1.0

This package implements all metrics defined in steadystream_plan_v2.md §4.

Core modules:
- boundary_metrics: F0 and energy jump measurement with VAD gating
- excess_metrics: Excess jumps beyond natural baseline (§4.3)
- pause_metrics: Pause deviation measurement (§4.3)
- sim_delta: Speaker similarity loss (§4.4)
- latency_metrics: FASL and RTF (§4.5)
- text_metrics: CER computation (§4.6)

Status: PROVISIONAL until §4.7 acceptance criteria pass.
"""

__version__ = "1.0.0-alpha"

from .boundary_metrics import (
    measure_boundary_metrics,
    summarize_boundary_metrics,
    boundary_positions_from_concat,
    boundary_positions_proportional,
    vad_gate,
    side_f0_mean,
)

from .excess_metrics import (
    NaturalBoundaryReference,
    compute_excess_metrics,
    summarize_excess_metrics,
)

from .pause_metrics import (
    detect_boundary_pause,
    compute_pause_deviation,
    measure_pause_metrics,
    summarize_pause_metrics,
)

from .sim_delta import (
    compute_sim_delta,
    summarize_sim_delta,
)

from .latency_metrics import (
    measure_first_audio_streaming_latency,
    measure_rtf,
)

from .text_metrics import (
    compute_cer,
    measure_cer_batch,
)

__all__ = [
    # boundary
    "measure_boundary_metrics",
    "summarize_boundary_metrics",
    "boundary_positions_from_concat",
    "boundary_positions_proportional",
    "vad_gate",
    "side_f0_mean",
    # excess
    "NaturalBoundaryReference",
    "compute_excess_metrics",
    "summarize_excess_metrics",
    # pause
    "detect_boundary_pause",
    "compute_pause_deviation",
    "measure_pause_metrics",
    "summarize_pause_metrics",
    # sim_delta
    "compute_sim_delta",
    "summarize_sim_delta",
    # latency
    "measure_first_audio_streaming_latency",
    "measure_rtf",
    # text
    "compute_cer",
    "measure_cer_batch",
]

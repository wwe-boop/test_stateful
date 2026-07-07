"""SteadyStream Evaluation Package - Frozen API v1.0

This package implements metrics from steadystream_plan_v2.md §4 (E0) and
Table 4 concurrency stress helpers (E9).

Status: PROVISIONAL until §4.7 acceptance criteria pass.
"""

__version__ = "1.0.0-alpha"

from .boundary_metrics import (
    boundary_positions_from_concat,
    boundary_positions_proportional,
    measure_boundary_metrics,
    side_f0_mean,
    summarize_boundary_metrics,
    vad_gate,
)
from .boundary_match import forced_segmentation_rate, match_boundaries
from .excess_metrics import (
    NaturalBoundaryReference,
    compute_excess_metrics,
    summarize_excess_metrics,
)
from .fasl_vad import (
    first_speech_sample_index,
    measure_fasl_vad_from_concat,
    measure_fasl_vad_from_packets,
    vad_speech_mask,
)
from .jitter_metrics import intervals_from_packet_timestamps, measure_jitter_ms
from .latency_metrics import measure_first_audio_streaming_latency, measure_rtf
from .pause_metrics import (
    compute_pause_deviation,
    detect_boundary_pause,
    measure_pause_metrics,
    summarize_pause_metrics,
)
from .sim_delta import compute_sim_delta, summarize_sim_delta
from .stress_metrics import aggregate_stress_runs, summarize_session_stress
from .stutter_metrics import pause_resume_stutter_delta, simulate_playout_underflows
from .text_metrics import compute_cer, measure_cer_batch

__all__ = [
    # boundary (E0)
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
    # Table 4 stress
    "match_boundaries",
    "forced_segmentation_rate",
    "measure_fasl_vad_from_packets",
    "measure_fasl_vad_from_concat",
    "first_speech_sample_index",
    "vad_speech_mask",
    "measure_jitter_ms",
    "intervals_from_packet_timestamps",
    "simulate_playout_underflows",
    "pause_resume_stutter_delta",
    "summarize_session_stress",
    "aggregate_stress_runs",
]

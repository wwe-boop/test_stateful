from __future__ import annotations

from dataclasses import asdict, dataclass, field
from statistics import mean
from typing import Any


BACKENDS = (
    "official_pytorch_offline",
    "official_pytorch_streaming",
    "bare_engine_streaming",
    "triton_trt_streaming",
)


@dataclass
class TraceEvent:
    run_id: str
    backend: str
    type: str
    t_ms: float
    server_t_ms: float | None = None
    stream_id: str | None = None
    text: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return {key: value for key, value in data.items() if value is not None}


@dataclass
class RunMetrics:
    first_playable_ms: float | None = None
    total_ms: float | None = None
    official_approx_ttft_ms: float | None = None
    server_ttft_ms: float | None = None
    triton_adapter_ttft_ms: float | None = None
    engine_internal_ttft_ms: float | None = None
    client_ttfb_ms: float | None = None
    first_audible_ms: float | None = None
    full_audio_ready_ms: float | None = None
    audio_duration_ms: float | None = None
    chunks: int = 0
    cache_hit: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return {key: value for key, value in data.items() if value is not None}


@dataclass
class RunResult:
    run_id: str
    backend: str
    label: str
    mode: str
    source: str
    metrics: RunMetrics
    events: list[TraceEvent] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    audio_format: dict[str, Any] = field(default_factory=dict)
    audio: dict[str, Any] = field(default_factory=dict)
    raw_audio: bytes | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "backend": self.backend,
            "label": self.label,
            "mode": self.mode,
            "source": self.source,
            "metrics": self.metrics.to_dict(),
            "events": [event.to_dict() for event in self.events],
            "warnings": list(self.warnings),
            "audio_format": dict(self.audio_format),
            "audio": dict(self.audio),
        }


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(float(value) for value in values)
    rank = (len(ordered) - 1) * pct
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return ordered[low] * (1.0 - frac) + ordered[high] * frac


def summarize_ttft(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "avg_ttft_ms": None,
            "p50_ttft_ms": None,
            "p90_ttft_ms": None,
            "p99_ttft_ms": None,
            "max_ttft_ms": None,
        }
    return {
        "count": len(values),
        "avg_ttft_ms": mean(values),
        "p50_ttft_ms": percentile(values, 0.50),
        "p90_ttft_ms": percentile(values, 0.90),
        "p99_ttft_ms": percentile(values, 0.99),
        "max_ttft_ms": max(values),
    }


def normalize_backend_result(raw: dict[str, Any]) -> dict[str, Any]:
    backend = str(raw.get("backend") or "")
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend in trace result: {backend!r}")
    events = []
    for event in raw.get("events", []) or []:
        if not isinstance(event, dict):
            continue
        event = dict(event)
        event.setdefault("run_id", raw.get("run_id", ""))
        event.setdefault("backend", backend)
        events.append(event)
    normalized = dict(raw)
    normalized["events"] = events
    normalized.setdefault("warnings", [])
    normalized.setdefault("audio_format", {"encoding": "pcm_f32", "sample_rate": 24000, "channels": 1})
    metrics = normalized.setdefault("metrics", {})
    if backend.startswith("official_pytorch"):
        _drop_legacy_official_warnings(normalized)
        _drop_legacy_official_metrics(metrics)
        approx = metrics.get("official_approx_ttft_ms")
        if approx is not None:
            metrics["official_approx_ttft_ms"] = approx
            metrics["first_playable_ms"] = approx
            _set_audio_schedule(normalized, approx)
        else:
            _set_audio_schedule(normalized, metrics.get("first_playable_ms") or metrics.get("full_audio_ready_ms"))
        warning = (
            "Official PyTorch TTFT is approximated as first decode-stage code0 timestamp minus request start; "
            "the public high-level API still returns a complete waveform."
        )
        if warning not in normalized["warnings"]:
            normalized["warnings"].insert(0, warning)
    return normalized


def _set_audio_schedule(result: dict[str, Any], value: Any) -> None:
    audio = result.get("audio")
    if isinstance(audio, dict) and "scheduled_start_ms" in audio and value is not None:
        try:
            audio["scheduled_start_ms"] = float(value)
        except (TypeError, ValueError):
            return


def _drop_legacy_official_metrics(metrics: dict[str, Any]) -> None:
    for key in (
        "official_text_tokenized_ms",
        "official_model_generate_started_ms",
        "official_talker_prefill_done_ms",
        "official_internal_first_codec0_ms",
        "official_decode_first_code0_ms",
        "official_decode_first_code0_done_ms",
        "official_internal_first_codec_group_ms",
        "official_speech_decode_started_ms",
        "official_reference_first_packet_ms",
        "official_reference_lm_ttfp_ms",
        "official_reference_tokenizer_decode_ms",
    ):
        metrics.pop(key, None)


def _drop_legacy_official_warnings(result: dict[str, Any]) -> None:
    legacy_prefixes = (
        "Official code0 playback row is an explicit zero-decode assumption",
        "Official online-text playback is an explicit paper/reference first-packet assumption",
        "Official online-text playback is a paper/reference first-packet replay",
        "Official internal marker timings",
        "This official PyTorch trace was captured before the internal code0 probe existed",
        "This official PyTorch trace was captured with an older code0 probe",
    )
    result["warnings"] = [
        warning
        for warning in result.get("warnings", [])
        if not any(str(warning).startswith(prefix) for prefix in legacy_prefixes)
    ]

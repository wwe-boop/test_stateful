from __future__ import annotations

from dataclasses import asdict, dataclass, field
from statistics import mean
from typing import Any


BACKENDS = (
    "triton_streaming",
    "triton_offline",
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
    server_ttft_ms: float | None = None
    triton_adapter_ttft_ms: float | None = None
    engine_internal_ttft_ms: float | None = None
    client_ttfb_ms: float | None = None
    first_audible_ms: float | None = None
    full_audio_ready_ms: float | None = None
    audio_duration_ms: float | None = None
    simulated_llm_complete_ms: float | None = None
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
    normalized.setdefault("metrics", {})
    return normalized

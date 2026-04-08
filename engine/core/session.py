"""Async-side session: lives entirely in the asyncio event loop.

Holds the per-request Spliter (text segmentation orchestrator), an
asyncio.Queue for receiving results from the engine thread, an
AudioReorder for multi-segment pipelining, and stream-output state.

No GPU tensors, no torch imports.
"""

from __future__ import annotations

import asyncio
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from .types import SessionState, EngineResult

logger = logging.getLogger(__name__)


@dataclass
class SegmentOrderMeta:
    group_idx: int
    local_idx: int
    group_final: bool = True


@dataclass
class Session:
    session_id: str
    speaker_key: Optional[str] = None
    task_type: str = "custom"

    state: SessionState = SessionState.PENDING
    created_at: float = field(default_factory=time.monotonic)
    first_audio_at: Optional[float] = None

    # Spliter (text segmentation orchestrator) — set by Dispatcher
    spliter: Any = None

    # AudioReorder for multi-segment pipelining — set by Dispatcher
    reorder: Any = None

    # Engine thread pushes EngineResult here; asyncio consumer reads them
    result_queue: asyncio.Queue[EngineResult] = field(
        default_factory=lambda: asyncio.Queue(maxsize=256)
    )

    # Accumulated text that hasn't been tokenized yet (streaming buffer)
    _text_buffer: str = ""
    _text_complete: bool = False

    # Segment tracking
    segments_submitted: int = 0
    segments_done: int = 0
    segment_order: dict[int, SegmentOrderMeta] = field(default_factory=dict)
    engine_text_done_sent: bool = False

    # Metrics
    total_steps: int = 0
    total_audio_bytes: int = 0

    def append_text(self, text: str) -> None:
        self._text_buffer += text

    def drain_text(self) -> str:
        t = self._text_buffer
        self._text_buffer = ""
        return t

    @property
    def has_pending_text(self) -> bool:
        return len(self._text_buffer) > 0

    def mark_text_complete(self) -> None:
        self._text_complete = True

    @property
    def text_complete(self) -> bool:
        return self._text_complete

    def record_first_audio(self) -> None:
        if self.first_audio_at is None:
            self.first_audio_at = time.monotonic()

    @property
    def first_audio_latency_ms(self) -> Optional[float]:
        if self.first_audio_at is None:
            return None
        return (self.first_audio_at - self.created_at) * 1000

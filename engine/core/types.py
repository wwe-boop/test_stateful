"""Shared types for asyncio frontend ↔ engine thread communication.

Design principle: these objects cross the thread boundary via thread-safe queues.
They carry only plain data / small CPU tensors — never raw GPU tensors.

Level 2 pipelining: each session may have multiple in-flight segments.
Requests and results are tagged with (session_id, segment_idx) to route
correctly.  The scheduler uses RequestPriority to order prefill work.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------

class SessionState(Enum):
    PENDING   = auto()
    PREFILL   = auto()
    DECODING  = auto()
    SEGMENT   = auto()
    DONE      = auto()


# ---------------------------------------------------------------------------
# Frontend → Engine thread  (via engine_inbox)
# ---------------------------------------------------------------------------

class RequestType(Enum):
    NEW_SESSION      = auto()
    START_SEGMENT    = auto()  # begin a new segment within an existing session
    APPEND_TEXT      = auto()
    TEXT_COMPLETE    = auto()  # per-segment: no more text for this segment
    SESSION_TEXT_DONE = auto() # session-level: upstream has finished all text
    CANCEL_SESSION   = auto()


class RequestPriority(Enum):
    """Lower numeric value = higher urgency."""
    FIRST_SEGMENT = 0   # new session, first segment — TTFB critical
    CONTINUATION  = 1   # next segment while previous is flushing
    PREFETCHED    = 2   # offline pre-split, ahead-of-time preparation


@dataclass
class EngineRequest:
    """A single message from the asyncio world to the engine thread."""
    type: RequestType
    session_id: str
    segment_idx: int = 0
    priority: RequestPriority = RequestPriority.FIRST_SEGMENT
    # NEW_SESSION payload
    speaker_key: Optional[str] = None
    task_type: Optional[str] = None       # "icl" | "custom" | "design"
    ref_audio: Optional[bytes] = None
    # APPEND_TEXT / START_SEGMENT payload
    token_ids: Optional[list[int]] = None
    text: Optional[str] = None
    # back-reference so engine thread can push results to the right queue
    result_queue: Optional[asyncio.Queue] = None


# ---------------------------------------------------------------------------
# Engine thread → Frontend  (via per-session result_queue)
# ---------------------------------------------------------------------------

class ResultType(Enum):
    PREFILL_DONE   = auto()
    AUDIO_CHUNK    = auto()
    SEGMENT_END    = auto()
    SESSION_DONE   = auto()
    RATIO_UPDATE   = auto()  # EMA audio:text ratio feedback
    ERROR          = auto()


@dataclass
class EngineResult:
    type: ResultType
    session_id: str
    segment_idx: int = 0
    audio_bytes: Optional[bytes] = None   # for AUDIO_CHUNK
    error_msg: Optional[str] = None       # for ERROR
    metrics: dict = field(default_factory=dict)  # step_count, rtf, etc.
    ema_ratio: float = 0.0                # for RATIO_UPDATE

"""
Session Manager and Batch Scheduler for continuous batching TTS (Phase 2).

Flow states:
  PENDING  - just created; visible in registry for append_text lookup but
             invisible to engine iteration (get_idle / get_active).
             Only the queue-based path in the engine thread transitions
             PENDING → IDLE or PENDING → ACTIVE.
  IDLE     - engine has acknowledged the session; waiting for text.
  ACTIVE   - decoding.
  DONE     - finished.
"""

from __future__ import annotations

import threading
import time
import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger("tts_orchestrator.session_manager")

# Sentinel: streaming session not yet assigned a GPU batch slot
UNASSIGNED_SLOT = -1


class FlowState(Enum):
    PENDING = "pending"
    IDLE = "idle"
    ACTIVE = "active"
    DONE = "done"


def generate_session_id() -> str:
    return uuid.uuid4().hex[:12]


@dataclass
class TTSSession:
    """Per-request session state for streaming TTS continuous batching."""

    session_id: str
    slot_id: int
    response_sender: Any

    # Request parameters
    req: dict = field(default_factory=dict)
    task_type: Any = None
    language: str = "auto"
    speaker: Optional[str] = None
    instruct: Optional[str] = None
    spk_embedding: Any = None
    ref_codes: Any = None
    ref_codec_sum_vec: Any = None
    ref_text: Optional[str] = None

    # Text segmentation (long-text rollover)
    text_segments: List[str] = field(default_factory=list)
    segment_idx: int = 0
    # Current segment raw text (for dynamic split / punctuation cut)
    current_segment_text: str = ""

    # Decode state (GPU tensors, typed Any)
    kv_tensors: Optional[List[Any]] = None
    c2w_states: Optional[List[Any]] = None
    trailing_text: Optional[List[Any]] = None
    next_embed: Any = None
    text_idx: int = 0
    past_len: int = 0
    frame_idx: int = 0

    # Per-segment ratio accounting (decode steps vs segment start)
    segment_start_past_len: int = 0

    # Optional: last frame logits tail for optional cross-segment acoustic continuity
    last_segment_codec_tail: Optional[Any] = None

    # Saved codec_sum for streaming decode resumption (KV-preserved IDLE → ACTIVE)
    last_codec_sum: Optional[Any] = None

    # Token frequency counts for engine-side repetition penalty [1, V] int64
    token_counts: Optional[Any] = None

    # Whether this session was created as a streaming session (action=init)
    is_streaming: bool = False

    # Whether tts_eos_embed has been injected into trailing for streaming sessions
    _eos_injected: bool = False

    # Pad-phase silence detection: consecutive silent frames during pad phase
    pad_consecutive_silence: int = 0

    # Spurious Codec EOS counter (reset per segment)
    _spurious_eos_count: int = 0

    # Decode FSM + KV checkpoint (post-prefill snapshot for multi-round segments)
    fsm: Any = None
    kv_checkpoint: Any = None
    c2w_checkpoint: Any = None
    checkpoint_past_len: int = 0
    checkpoint_codec_sum: Any = None
    trailing_token_char_offsets: List[int] = field(default_factory=list)
    mlfq_meta: Any = None

    # Flow control
    flow_state: FlowState = FlowState.PENDING
    prefilled: bool = False

    # Greedy streaming tokenizer (set by orchestrator)
    greedy_tokenizer: Any = None
    token_queue: List[Any] = field(default_factory=list)

    # Streaming text buffer (written by execute thread, read by engine thread)
    _text_buffer: List[str] = field(default_factory=list)
    _text_complete: bool = False
    _text_lock: threading.Lock = field(default_factory=threading.Lock)

    # Timing
    created_at: float = field(default_factory=time.monotonic)
    request_start: float = field(default_factory=time.monotonic)
    max_idle_ms: int = 10000

    @property
    def done(self) -> bool:
        return self.flow_state == FlowState.DONE

    @done.setter
    def done(self, value: bool) -> None:
        if value:
            self.flow_state = FlowState.DONE

    def append_text(self, text: str) -> None:
        with self._text_lock:
            self._text_buffer.append(text)
            self.created_at = time.monotonic()

    def mark_text_complete(self) -> None:
        with self._text_lock:
            self._text_complete = True
            self.created_at = time.monotonic()

    @property
    def text_complete(self) -> bool:
        with self._text_lock:
            return self._text_complete

    def drain_text_buffer(self) -> List[str]:
        with self._text_lock:
            buf = list(self._text_buffer)
            self._text_buffer.clear()
            return buf

    def has_pending_text(self) -> bool:
        with self._text_lock:
            return len(self._text_buffer) > 0

    def is_idle_timed_out(self, now: Optional[float] = None) -> bool:
        if self.flow_state != FlowState.IDLE:
            return False
        now = now or time.monotonic()
        return (now - self.created_at) * 1000 > self.max_idle_ms

    @property
    def has_preserved_kv(self) -> bool:
        """True when session went IDLE but kept its KV cache for streaming resumption."""
        return self.kv_tensors is not None and self.last_codec_sum is not None

    def reset_decode_state(self) -> None:
        self.kv_tensors = None
        self.c2w_states = None
        self.next_embed = None
        self.trailing_text = None
        self.text_idx = 0
        self.past_len = 0
        self.frame_idx = 0
        self.segment_start_past_len = 0
        self.last_codec_sum = None
        self.token_counts = None
        self.pad_consecutive_silence = 0
        self.fsm = None
        self.kv_checkpoint = None
        self.c2w_checkpoint = None
        self.checkpoint_past_len = 0
        self.checkpoint_codec_sum = None
        self.trailing_token_char_offsets = []
        self._spurious_eos_count = 0


class SessionManager:
    """Maps session_id -> TTSSession with state queries."""

    def __init__(self) -> None:
        self._sessions: Dict[str, TTSSession] = {}

    def add(self, session: TTSSession) -> None:
        self._sessions[session.session_id] = session

    def remove(self, session_id: str) -> Optional[TTSSession]:
        return self._sessions.pop(session_id, None)

    def get(self, session_id: str) -> Optional[TTSSession]:
        return self._sessions.get(session_id)

    def get_active(self) -> List[TTSSession]:
        return [s for s in self._sessions.values() if s.flow_state == FlowState.ACTIVE]

    def get_idle(self) -> List[TTSSession]:
        return [s for s in self._sessions.values() if s.flow_state == FlowState.IDLE]

    def get_all(self) -> List[TTSSession]:
        return list(self._sessions.values())

    def count_active(self) -> int:
        return sum(1 for s in self._sessions.values() if s.flow_state == FlowState.ACTIVE)

    def has_active(self) -> bool:
        return any(s.flow_state != FlowState.DONE for s in self._sessions.values())


class BatchScheduler:
    """Slot allocation and queue when exhausted."""

    def __init__(
        self,
        max_slots: int = 8,
        max_queue_size: int = 32,
    ) -> None:
        self.max_slots = max_slots
        self.max_queue_size = max_queue_size
        self._slot_to_session: Dict[int, Optional[str]] = {
            i: None for i in range(max_slots)
        }
        self._queue: List[Any] = []

    def try_allocate_slot(self) -> Optional[int]:
        for slot_id, session_id in self._slot_to_session.items():
            if session_id is None:
                return slot_id
        return None

    def release_slot(self, slot_id: int) -> None:
        if 0 <= slot_id < self.max_slots:
            self._slot_to_session[slot_id] = None

    def assign_slot(self, slot_id: int, session_id: str) -> None:
        if 0 <= slot_id < self.max_slots:
            self._slot_to_session[slot_id] = session_id

    def active_count(self) -> int:
        return sum(1 for sid in self._slot_to_session.values() if sid is not None)

    def enqueue(self, item: Any) -> bool:
        if len(self._queue) >= self.max_queue_size:
            return False
        self._queue.append(item)
        return True

    def dequeue(self) -> Optional[Any]:
        return self._queue.pop(0) if self._queue else None

    def queue_size(self) -> int:
        return len(self._queue)

    def find_evictable(
        self,
        session_manager: SessionManager,
    ) -> Optional[TTSSession]:
        """Find oldest IDLE session that has timed out, for forced eviction."""
        now = time.monotonic()
        oldest: Optional[TTSSession] = None
        for session_id in self._slot_to_session.values():
            if session_id is None:
                continue
            session = session_manager.get(session_id)
            if session is None:
                continue
            if session.flow_state == FlowState.IDLE and session.is_idle_timed_out(now):
                if oldest is None or session.created_at < oldest.created_at:
                    oldest = session
        return oldest

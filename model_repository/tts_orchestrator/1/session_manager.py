"""
Session Manager and Batch Scheduler (Phase 3 skeleton).

Per architecture 9.1–9.3 and 10.2:
- SessionManager: holds TTSSession instances, keyed by session_id.
- BatchScheduler: slot allocation, release, queue when slots exhausted, timeout-based slot recovery.

Current orchestrator is single-request and does not use this module yet.
Integrate when moving to multi-session continuous batching (prefill/decode interleave).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import time
import logging

logger = logging.getLogger("tts_orchestrator.session_manager")


class FlowMode(Enum):
    """Flow control mode — adaptive to upstream LLM speed (architecture 11.2)."""
    TOKEN_LEVEL = "token"
    ADAPTIVE = "adaptive"
    SENTENCE_LEVEL = "sentence"


class FlowState(Enum):
    """Flow state — orthogonal to FlowMode (architecture 10.1)."""
    WAITING = "waiting"
    GENERATING = "generating"
    PAUSED = "paused"
    DONE = "done"


@dataclass
class TTSSession:
    """Per-request session state for streaming TTS (architecture 10.1)."""
    session_id: str
    slot_id: int
    flow_state: FlowState = FlowState.WAITING
    flow_mode: FlowMode = FlowMode.TOKEN_LEVEL
    generation_step: int = 0
    prefilled: bool = False
    text_complete: bool = False
    created_at: float = field(default_factory=time.monotonic)
    pause_start_time: Optional[float] = None
    max_pause_ms: int = 3000
    max_idle_ms: int = 10000
    # Optional tensor/request state (typed as Any to avoid torch import at module level)
    spk_embedding: Any = None
    ref_codes: Any = None
    trailing_text_hidden: List[Any] = field(default_factory=list)
    text_consumed_count: int = 0
    codec_buffer: List[Any] = field(default_factory=list)
    kv_tensors: Optional[List[Any]] = None

    @property
    def buffer_available(self) -> int:
        return len(self.trailing_text_hidden) - self.text_consumed_count

    def is_idle_timed_out(self, now: Optional[float] = None) -> bool:
        """True if WAITING and idle longer than max_idle_ms."""
        if self.flow_state != FlowState.WAITING:
            return False
        now = now or time.monotonic()
        return (now - self.created_at) * 1000 > self.max_idle_ms

    def is_pause_timed_out(self, now: Optional[float] = None) -> bool:
        """True if PAUSED and paused longer than max_pause_ms."""
        if self.flow_state != FlowState.PAUSED or self.pause_start_time is None:
            return False
        now = now or time.monotonic()
        return (now - self.pause_start_time) * 1000 > self.max_pause_ms


class SessionManager:
    """Maps session_id -> TTSSession (architecture 10.2)."""

    def __init__(self) -> None:
        self._sessions: Dict[str, TTSSession] = {}

    def add(self, session: TTSSession) -> None:
        self._sessions[session.session_id] = session

    def remove(self, session_id: str) -> Optional[TTSSession]:
        return self._sessions.pop(session_id, None)

    def get(self, session_id: str) -> Optional[TTSSession]:
        return self._sessions.get(session_id)

    def get_active_generating(self) -> List[TTSSession]:
        """Sessions in GENERATING state for batch decode (architecture 9.1)."""
        return [s for s in self._sessions.values() if s.flow_state == FlowState.GENERATING]

    def get_all(self) -> List[TTSSession]:
        return list(self._sessions.values())


class BatchScheduler:
    """
    Slot allocation and queue when exhausted (architecture 9.1, 9.3).

    - allocate_slot(): returns slot_id or blocks until a slot is free (or queue timeout).
    - release_slot(slot_id): frees the slot; if queue non-empty, assign to next waiting request.
    - Slot timeout: PAUSED/WAITING over max_pause_ms/max_idle_ms can be reclaimed by caller.
    """

    def __init__(
        self,
        max_slots: int = 8,
        max_queue_size: int = 32,
    ) -> None:
        self.max_slots = max_slots
        self.max_queue_size = max_queue_size
        self._slot_to_session: Dict[int, Optional[str]] = {i: None for i in range(max_slots)}
        self._queue: List[Any] = []  # Pending (request/session_id) when slots full

    def allocate_slot(self) -> int:
        """
        Return an available slot index (0 .. max_slots-1).
        If all slots are busy, caller should enqueue and retry later (or implement blocking).
        Raises RuntimeError if no free slot and queue would exceed max_queue_size.
        """
        for slot_id, session_id in self._slot_to_session.items():
            if session_id is None:
                return slot_id
        raise RuntimeError("no free slot; caller should enqueue or wait for release_slot")

    def try_allocate_slot(self) -> Optional[int]:
        """Return slot_id if available, else None."""
        for slot_id, session_id in self._slot_to_session.items():
            if session_id is None:
                return slot_id
        return None

    def release_slot(self, slot_id: int) -> None:
        """Free the slot. Optionally pop from queue and assign (caller creates session)."""
        if 0 <= slot_id < self.max_slots:
            self._slot_to_session[slot_id] = None

    def assign_slot(self, slot_id: int, session_id: str) -> None:
        """Mark slot as occupied by session_id."""
        if 0 <= slot_id < self.max_slots:
            self._slot_to_session[slot_id] = session_id

    def enqueue(self, item: Any) -> bool:
        """Add to wait queue. Returns False if queue full."""
        if len(self._queue) >= self.max_queue_size:
            return False
        self._queue.append(item)
        return True

    def dequeue(self) -> Optional[Any]:
        """Pop next waiting item from queue, or None."""
        return self._queue.pop(0) if self._queue else None

    def queue_size(self) -> int:
        return len(self._queue)

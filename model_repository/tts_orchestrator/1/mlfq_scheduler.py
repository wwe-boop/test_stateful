"""
Multi-level feedback queue for step-level batch selection (orchestrator).

English comments only (project convention).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from session_manager import TTSSession

# Level thresholds: steps since last level promotion / session start
_STEP_TO_DOWNGRADE_Q1 = 50
_STEP_TO_DOWNGRADE_Q2 = 200
_AGING_INTERVAL = 100
_STEPS_STARVED = 50


@dataclass
class MLFQMeta:
    level: int = 0  # 0 = highest
    decode_steps: int = 0
    last_global_step: int = 0
    steps_since_schedule: int = 0

    def reset_priority(self) -> None:
        self.level = 0
        self.steps_since_schedule = 0
        self.decode_steps = 0


def _get_meta(session: "TTSSession") -> MLFQMeta:
    if session.mlfq_meta is None:
        session.mlfq_meta = MLFQMeta()
    return session.mlfq_meta


def on_session_created(session: "TTSSession") -> None:
    _get_meta(session).reset_priority()


def on_segment_boundary(session: "TTSSession") -> None:
    _get_meta(session).reset_priority()


def on_decode_step_done(session: "TTSSession") -> None:
    m = _get_meta(session)
    m.decode_steps += 1
    m.steps_since_schedule += 1
    if m.level == 0 and m.decode_steps >= _STEP_TO_DOWNGRADE_Q1:
        m.level = 1
    elif m.level == 1 and m.decode_steps >= _STEP_TO_DOWNGRADE_Q2:
        m.level = 2


def on_scheduled(session: "TTSSession", global_step: int) -> None:
    m = _get_meta(session)
    m.last_global_step = global_step
    m.steps_since_schedule = 0


def global_aging(sessions: List["TTSSession"], global_step: int) -> None:
    if global_step <= 0 or global_step % _AGING_INTERVAL != 0:
        return
    for s in sessions:
        m = _get_meta(s)
        if m.level >= 2 and m.steps_since_schedule >= _STEPS_STARVED:
            m.level = 0
            m.steps_since_schedule = 0


def order_ready_sessions(
    ready: List["TTSSession"],
    global_step: int,
) -> List["TTSSession"]:
    """Order sessions Q0 first, then Q1, then Q2; FIFO within level."""
    buckets: dict[int, list["TTSSession"]] = {0: [], 1: [], 2: []}
    for s in ready:
        m = _get_meta(s)
        lev = min(2, max(0, m.level))
        buckets[lev].append(s)
    out: List["TTSSession"] = []
    for lev in (0, 1, 2):
        out.extend(buckets[lev])
    for s in out:
        on_scheduled(s, global_step)
    return out

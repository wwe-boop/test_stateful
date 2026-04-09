"""Backend-facing dispatcher for frontend session traffic.

This module intentionally stays below the gateway/interface boundary.
It knows how to translate session state + Spliter actions into
``EngineRequest`` messages for the backend thread, but it does not own
external session lifecycle APIs.
"""

from __future__ import annotations

import asyncio
from typing import List

from ..core.session import Session, SegmentOrderMeta
from ..core.types import EngineRequest, InputMode, RequestPriority, RequestType
from .spliter import SegmentAction
from .spliter.driver import ActionType


class Dispatcher:
    """Translate frontend session actions into backend requests."""

    def __init__(self, engine_inbox: asyncio.Queue):
        self._engine_inbox = engine_inbox

    async def submit_new_session(self, session: Session) -> None:
        """Register a new session with the backend engine thread."""
        config = session.config
        await self._engine_inbox.put(EngineRequest(
            type=RequestType.NEW_SESSION,
            session_id=session.session_id,
            session_config=config,
            speaker_key=config.speaker,
            task_type=config.task_type,
            ref_audio=config.ref_audio,
            result_queue=session.result_queue,
        ))

    async def submit_cancel(self, session_id: str) -> None:
        await self._engine_inbox.put(EngineRequest(
            type=RequestType.CANCEL_SESSION,
            session_id=session_id,
        ))

    async def submit_session_text_done(self, session_id: str) -> None:
        await self._engine_inbox.put(EngineRequest(
            type=RequestType.SESSION_TEXT_DONE,
            session_id=session_id,
        ))

    async def dispatch_segment_actions(
        self,
        session: Session,
        actions: List[SegmentAction],
    ) -> None:
        """Translate Spliter actions into EngineRequests and submit them."""
        for sa in actions:
            seg_idx = sa.segment_idx
            action = sa.action
            priority = self._segment_priority(session, sa)
            order_meta = SegmentOrderMeta(
                group_idx=sa.group_idx if sa.group_idx >= 0 else seg_idx,
                local_idx=sa.local_idx if sa.group_idx >= 0 else 0,
                group_final=sa.group_final if sa.group_idx >= 0 else True,
            )

            if action.type == ActionType.PREFILL:
                session.segments_submitted += 1
                session.segment_order[seg_idx] = order_meta
                await self._engine_inbox.put(EngineRequest(
                    type=RequestType.START_SEGMENT,
                    session_id=session.session_id,
                    segment_idx=seg_idx,
                    priority=priority,
                    token_ids=[action.token],
                    result_queue=session.result_queue,
                ))

            elif action.type == ActionType.DECODE:
                await self._engine_inbox.put(EngineRequest(
                    type=RequestType.APPEND_TEXT,
                    session_id=session.session_id,
                    segment_idx=seg_idx,
                    priority=priority,
                    token_ids=[action.token],
                ))

            elif action.type in (ActionType.FLUSH_EOS, ActionType.FLUSH_NOP):
                await self._engine_inbox.put(EngineRequest(
                    type=RequestType.TEXT_COMPLETE,
                    session_id=session.session_id,
                    segment_idx=seg_idx,
                    append_eos=(action.type == ActionType.FLUSH_EOS),
                ))

    async def maybe_send_session_text_done(self, session: Session) -> None:
        """Signal session-level text completion when no more groups remain."""
        if session.engine_text_done_sent or session.spliter is None:
            return
        spliter = session.spliter
        pending_groups = getattr(spliter, "_presplit_groups", None)
        has_presplit_mode = getattr(spliter, "_presplit_thresholds", None) is not None
        if not has_presplit_mode and session.config.input_mode != InputMode.LONG_SEGMENT:
            return
        if pending_groups:
            return
        await self.submit_session_text_done(session.session_id)
        session.engine_text_done_sent = True

    def _segment_priority(
        self,
        session: Session,
        sa: SegmentAction,
    ) -> RequestPriority:
        """Lower numeric value means higher urgency."""
        segment_idx = sa.segment_idx
        if segment_idx == 0 and session.segments_submitted == 0:
            return RequestPriority.FIRST_SEGMENT
        if sa.local_idx > 0:
            return RequestPriority.CONTINUATION
        if segment_idx > 0 and (segment_idx - 1) in (
            session.spliter._flushing if session.spliter else set()
        ):
            return RequestPriority.CONTINUATION
        return RequestPriority.PREFETCHED

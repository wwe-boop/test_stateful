"""Dispatcher: the asyncio-side hub that manages all sessions.

Runs entirely on the asyncio event loop (main thread).  Responsibilities:
  1. Create / destroy sessions on gRPC init / cancel
  2. Feed incoming text chunks into each session's Spliter
  3. Convert Spliter SegmentActions into EngineRequests for the engine thread
  4. Consume EngineResults, route audio through AudioReorder, push to gRPC

Level 2 pipelining:
  - Each session may have multiple in-flight segments decoding in parallel.
  - Audio chunks are reordered per-session before emission.
  - Segment priorities: FIRST_SEGMENT > CONTINUATION > PREFETCHED.

Thread safety:
  - All public methods are coroutines; call them from the asyncio event loop.
  - The only cross-thread interaction is through the engine_inbox (janus queue)
    and per-session result_queues (asyncio.Queue written via call_soon_threadsafe).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable, Dict, List, Optional

from ..core.session import Session
from ..core.types import (
    EngineRequest,
    EngineResult,
    RequestPriority,
    RequestType,
    ResultType,
    SessionState,
)
from .spliter import Spliter, SegmentAction
from .spliter.driver import ActionType
from .spliter.reorder import AudioReorder
from .spliter.tokenizer import LightQwen3TTSTokenizer

logger = logging.getLogger(__name__)


class Dispatcher:
    """Manages all active sessions on the asyncio side."""

    def __init__(
        self,
        engine_inbox: asyncio.Queue,
        tokenizer: LightQwen3TTSTokenizer,
        *,
        max_sessions: int = 128,
        engine_max_decode_len: int = 512,
        prefill_len: int = 12,
        ema_ratio: float = 5.0,
        max_concurrent_segments: int = 2,
        ema_alpha: float = 0.1,
        ema_overflow_alpha: float = 0.5,
        ema_min_ratio: float = 2.0,
        ema_max_ratio: float = 10.0,
        safety_margin: int = 8,
    ):
        self._engine_inbox = engine_inbox
        self._tokenizer = tokenizer
        self._max_sessions = max_sessions
        self._engine_max = engine_max_decode_len
        self._prefill_len = prefill_len
        self._ema_ratio = ema_ratio
        self._max_concurrent = max_concurrent_segments
        self._ema_alpha = ema_alpha
        self._ema_overflow_alpha = ema_overflow_alpha
        self._ema_min_ratio = ema_min_ratio
        self._ema_max_ratio = ema_max_ratio
        self._safety_margin = safety_margin

        self._sessions: Dict[str, Session] = {}
        self._consumer_tasks: Dict[str, asyncio.Task] = {}

    @property
    def active_count(self) -> int:
        return len(self._sessions)

    # ------------------------------------------------------------------
    # Session lifecycle (called by Gateway)
    # ------------------------------------------------------------------

    async def create_session(
        self,
        session_id: str,
        *,
        speaker_key: Optional[str] = None,
        task_type: str = "custom",
        ref_audio: Optional[bytes] = None,
        on_audio: Optional[Callable] = None,
        on_done: Optional[Callable] = None,
    ) -> Session:
        if session_id in self._sessions:
            old_task = self._consumer_tasks.pop(session_id, None)
            if old_task and not old_task.done():
                old_task.cancel()
            self._cleanup_session(session_id)

        if len(self._sessions) >= self._max_sessions:
            raise RuntimeError(f"Max sessions ({self._max_sessions}) reached")

        session = Session(
            session_id=session_id,
            speaker_key=speaker_key,
            task_type=task_type,
        )
        session.spliter = Spliter(
            engine_max_decode_len=self._engine_max,
            prefill_len=self._prefill_len,
            ema_ratio=self._ema_ratio,
            max_concurrent=self._max_concurrent,
            ema_alpha=self._ema_alpha,
            ema_overflow_alpha=self._ema_overflow_alpha,
            ema_min_ratio=self._ema_min_ratio,
            ema_max_ratio=self._ema_max_ratio,
            safety_margin=self._safety_margin,
        )
        session.reorder = AudioReorder()
        self._sessions[session_id] = session

        task = asyncio.create_task(
            self._consume_results(session, on_audio=on_audio, on_done=on_done)
        )
        self._consumer_tasks[session_id] = task

        await self._engine_inbox.put(EngineRequest(
            type=RequestType.NEW_SESSION,
            session_id=session_id,
            speaker_key=speaker_key,
            task_type=task_type,
            ref_audio=ref_audio,
            result_queue=session.result_queue,
        ))

        logger.info("Session %s created (active: %d)", session_id, self.active_count)
        return session

    async def cancel_session(self, session_id: str) -> None:
        session = self._sessions.get(session_id)
        if session is None:
            return
        session.state = SessionState.DONE
        await self._engine_inbox.put(EngineRequest(
            type=RequestType.CANCEL_SESSION,
            session_id=session_id,
        ))

    # ------------------------------------------------------------------
    # Streaming text input (called by Gateway on each text chunk)
    # ------------------------------------------------------------------

    async def feed_text(self, session_id: str, text: str) -> None:
        """Feed a chunk of streaming text from upstream LLM."""
        session = self._sessions.get(session_id)
        if session is None or session.state == SessionState.DONE:
            return

        ids, texts = self._tokenizer.encode_with_text(text, add_special_tokens=False)
        tokens = list(zip(ids, texts))
        if not tokens:
            return

        spliter: Spliter = session.spliter
        seg_actions = spliter.feed_tokens(tokens)
        await self._dispatch_segment_actions(session, seg_actions)

    async def feed_full_text(self, session_id: str, text: str) -> None:
        """Offline mode: set complete text, pre-split, drive all segments."""
        session = self._sessions.get(session_id)
        if session is None or session.state == SessionState.DONE:
            return

        ids, texts = self._tokenizer.encode_with_text(text, add_special_tokens=False)
        tokens = list(zip(ids, texts))

        spliter: Spliter = session.spliter
        seg_actions = spliter.set_full_text(tokens)
        await self._dispatch_segment_actions(session, seg_actions)

        await self._engine_inbox.put(EngineRequest(
            type=RequestType.SESSION_TEXT_DONE,
            session_id=session_id,
        ))

    async def text_complete(self, session_id: str) -> None:
        """Upstream signals no more text will arrive."""
        session = self._sessions.get(session_id)
        if session is None:
            return
        session.mark_text_complete()

        spliter: Spliter = session.spliter
        seg_actions = spliter.text_done()
        await self._dispatch_segment_actions(session, seg_actions)

        await self._engine_inbox.put(EngineRequest(
            type=RequestType.SESSION_TEXT_DONE,
            session_id=session_id,
        ))

    # ------------------------------------------------------------------
    # Translate SegmentActions → EngineRequests
    # ------------------------------------------------------------------

    def _segment_priority(
        self, session: Session, segment_idx: int,
    ) -> RequestPriority:
        """Determine priority for a segment's requests."""
        if segment_idx == 0 and session.segments_submitted == 0:
            return RequestPriority.FIRST_SEGMENT
        if segment_idx > 0 and (segment_idx - 1) in (
                session.spliter._flushing if session.spliter else set()):
            return RequestPriority.CONTINUATION
        return RequestPriority.PREFETCHED

    async def _dispatch_segment_actions(
        self, session: Session, actions: List[SegmentAction],
    ) -> None:
        """Translate SegmentActions into EngineRequests and submit."""
        for sa in actions:
            seg_idx = sa.segment_idx
            action = sa.action
            priority = self._segment_priority(session, seg_idx)

            if action.type == ActionType.PREFILL:
                session.segments_submitted += 1
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
                ))

    # ------------------------------------------------------------------
    # Result consumer: reads from per-session result_queue
    # ------------------------------------------------------------------

    async def _consume_results(
        self,
        session: Session,
        *,
        on_audio: Optional[Callable] = None,
        on_done: Optional[Callable] = None,
    ) -> None:
        """Long-running coroutine: drains the engine's result queue."""
        try:
            while True:
                result: EngineResult = await session.result_queue.get()

                if result.type == ResultType.AUDIO_CHUNK:
                    session.record_first_audio()
                    audio = result.audio_bytes or b""
                    session.total_audio_bytes += len(audio)

                    reorder: AudioReorder = session.reorder
                    ready = reorder.push(result.segment_idx, audio)
                    if ready and on_audio:
                        for chunk in ready:
                            await on_audio(session.session_id, chunk)

                elif result.type == ResultType.SEGMENT_END:
                    seg_idx = result.segment_idx
                    session.segments_done += 1

                    reorder: AudioReorder = session.reorder
                    ready = reorder.mark_done(seg_idx)
                    if ready and on_audio:
                        for chunk in ready:
                            await on_audio(session.session_id, chunk)

                    spliter: Spliter = session.spliter
                    if result.metrics:
                        audio_steps = result.metrics.get("audio_steps", 0)
                        text_tokens = result.metrics.get("text_tokens", 0)
                        overflow = result.metrics.get("overflow", False)
                        if audio_steps > 0 and text_tokens > 0:
                            spliter.update_ratio(
                                audio_steps, text_tokens, overflow=overflow,
                            )

                    new_actions = spliter.on_segment_done(seg_idx)
                    if new_actions:
                        await self._dispatch_segment_actions(session, new_actions)

                elif result.type == ResultType.RATIO_UPDATE:
                    if session.spliter and result.ema_ratio > 0:
                        session.spliter._ema_ratio = result.ema_ratio

                elif result.type == ResultType.SESSION_DONE:
                    session.state = SessionState.DONE
                    if on_done:
                        await on_done(session.session_id, result.metrics)
                    break

                elif result.type == ResultType.ERROR:
                    logger.error("Session %s error: %s",
                                 session.session_id, result.error_msg)
                    session.state = SessionState.DONE
                    if on_done:
                        await on_done(session.session_id, {"error": result.error_msg})
                    break

        except asyncio.CancelledError:
            pass
        finally:
            self._cleanup_session(session.session_id)

    def _cleanup_session(self, session_id: str) -> None:
        session = self._sessions.pop(session_id, None)
        self._consumer_tasks.pop(session_id, None)
        if session:
            latency = session.first_audio_latency_ms
            logger.info(
                "Session %s cleaned up (first_audio=%.1fms, segments=%d/%d)",
                session_id,
                latency or -1,
                session.segments_done,
                session.segments_submitted,
            )

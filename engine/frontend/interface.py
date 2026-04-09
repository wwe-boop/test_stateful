"""Gateway-facing frontend interface.

The interface owns external session lifecycle and input semantics:

- create/cancel sessions for transport adapters
- tokenize and route text according to declared input mode
- consume backend results and surface ordered audio callbacks

It delegates backend request emission to ``frontend.dispatcher.Dispatcher``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable, Dict, Optional

from ..core.session import Session, SegmentOrderMeta
from ..core.types import (
    EngineResult,
    GroupPolicy,
    InputMode,
    ResultType,
    SessionConfig,
    SessionState,
)
from .dispatcher import Dispatcher
from .spliter import Spliter
from .spliter.reorder import AudioReorder
from .spliter.tokenizer import LightQwen3TTSTokenizer

logger = logging.getLogger(__name__)


class FrontendInterface:
    """Gateway-facing session and text interface."""

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
        self._dispatcher = Dispatcher(engine_inbox)
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

    async def create_session(
        self,
        session_id: str,
        *,
        config: Optional[SessionConfig] = None,
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

        if config is None:
            config = SessionConfig(
                task_type=task_type,
                speaker=speaker_key,
                ref_audio=ref_audio,
            )

        session = Session(session_id=session_id, config=config)
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

        await self._dispatcher.submit_new_session(session)
        logger.info("Session %s created (active: %d)", session_id, self.active_count)
        return session

    async def cancel_session(self, session_id: str) -> None:
        session = self._sessions.get(session_id)
        if session is None:
            return
        session.state = SessionState.DONE
        await self._dispatcher.submit_cancel(session_id)

    async def feed_text(self, session_id: str, text: str) -> None:
        """Feed text according to the session's declared input mode."""
        session = self._sessions.get(session_id)
        if session is None or session.state == SessionState.DONE:
            return

        mode = session.config.input_mode
        if mode == InputMode.FULL_TEXT:
            session.append_text(text)
            return

        ids, texts = self._tokenizer.encode_with_text(text, add_special_tokens=False)
        tokens = list(zip(ids, texts))
        if not tokens:
            return

        spliter = session.spliter
        if mode == InputMode.LONG_SEGMENT and session.config.group_policy != GroupPolicy.NONE:
            seg_actions = spliter.push_group_tokens(tokens)
        else:
            seg_actions = spliter.feed_tokens(tokens)
        await self._dispatcher.dispatch_segment_actions(session, seg_actions)

    async def feed_full_text(self, session_id: str, text: str) -> None:
        """Explicit offline mode: set complete text, pre-split, drive all segments."""
        session = self._sessions.get(session_id)
        if session is None or session.state == SessionState.DONE:
            return

        ids, texts = self._tokenizer.encode_with_text(text, add_special_tokens=False)
        tokens = list(zip(ids, texts))
        if not tokens:
            return

        seg_actions = session.spliter.set_full_text(tokens)
        await self._dispatcher.dispatch_segment_actions(session, seg_actions)
        await self._dispatcher.maybe_send_session_text_done(session)

    async def text_complete(self, session_id: str) -> None:
        """Upstream signals no more text will arrive."""
        session = self._sessions.get(session_id)
        if session is None:
            return
        session.mark_text_complete()

        mode = session.config.input_mode
        if mode == InputMode.FULL_TEXT:
            full_text = session.drain_text()
            if full_text.strip():
                await self.feed_full_text(session_id, full_text)
            else:
                await self._dispatcher.submit_session_text_done(session_id)
                session.engine_text_done_sent = True
            return

        if mode == InputMode.LONG_SEGMENT and session.config.group_policy != GroupPolicy.NONE:
            await self._dispatcher.maybe_send_session_text_done(session)
            return

        seg_actions = session.spliter.text_done()
        await self._dispatcher.dispatch_segment_actions(session, seg_actions)
        await self._dispatcher.submit_session_text_done(session_id)
        session.engine_text_done_sent = True

    async def _consume_results(
        self,
        session: Session,
        *,
        on_audio: Optional[Callable] = None,
        on_done: Optional[Callable] = None,
    ) -> None:
        try:
            while True:
                result: EngineResult = await session.result_queue.get()

                if result.type == ResultType.AUDIO_CHUNK:
                    session.record_first_audio()
                    audio = result.audio_bytes or b""
                    session.total_audio_bytes += len(audio)

                    reorder = session.reorder
                    meta = session.segment_order.get(
                        result.segment_idx,
                        SegmentOrderMeta(result.segment_idx, 0, True),
                    )
                    ready = reorder.push(meta.group_idx, meta.local_idx, audio)
                    if ready and on_audio:
                        for chunk in ready:
                            await on_audio(session.session_id, chunk)

                elif result.type == ResultType.SEGMENT_END:
                    seg_idx = result.segment_idx
                    session.segments_done += 1

                    reorder = session.reorder
                    meta = session.segment_order.pop(
                        seg_idx,
                        SegmentOrderMeta(seg_idx, 0, True),
                    )
                    ready = reorder.mark_done(
                        meta.group_idx, meta.local_idx, group_final=meta.group_final,
                    )
                    if ready and on_audio:
                        for chunk in ready:
                            await on_audio(session.session_id, chunk)

                    if result.metrics:
                        audio_steps = result.metrics.get("audio_steps", 0)
                        text_tokens = result.metrics.get("text_tokens", 0)
                        overflow = result.metrics.get("overflow", False)
                        if audio_steps > 0 and text_tokens > 0:
                            session.spliter.update_ratio(
                                audio_steps, text_tokens, overflow=overflow,
                            )

                    new_actions = session.spliter.on_segment_done(seg_idx)
                    if new_actions:
                        await self._dispatcher.dispatch_segment_actions(session, new_actions)
                    await self._dispatcher.maybe_send_session_text_done(session)

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

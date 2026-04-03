"""Engine loop: runs in a dedicated thread, owns all GPU resources.

Pipeline design for high GPU utilization:

    While GPU executes step N,
    CPU simultaneously processes results from step N-1
    and prepares inputs for step N+1.

              time ────────────────────────────────────────────►
    GPU:  ┌─step N─┐           ┌─step N+1─┐           ┌─step N+2─┐
    CPU:  │(idle)  │ process   │(idle)     │ process   │
          └────────┘ N-1+prep  └───────────┘ N+prep    └───────────┘
                     N+1                     N+2

Level 2 pipelining:
    Multiple segments of the same session may decode in parallel.
    Each segment owns its own KV slot.  The engine batches all active
    slots regardless of session, and routes results by (session_id,
    segment_idx).

    Prefill priority: FIRST_SEGMENT > CONTINUATION > PREFETCHED.

Thread safety:
    - Runs entirely in its own thread.
    - Reads from engine_inbox (stdlib queue.Queue, thread-safe).
    - Writes results via asyncio.loop.call_soon_threadsafe().
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from typing import Dict, List, Optional, Tuple

import torch

from ..core.types import (
    EngineRequest,
    EngineResult,
    RequestPriority,
    RequestType,
    ResultType,
)
from .executor import Executor, StepOutput
from .kv_cache_pool import KVCachePool, SlotKVState
from .prefill import PrefillBuilder, PrefillPlan, TaskType, parse_task_type

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-segment tracking (engine-thread side)
# ---------------------------------------------------------------------------

class EngineSegment:
    """Engine thread's view of one segment: owns a KV slot + decode state."""
    __slots__ = (
        "session_id", "segment_idx", "slot", "state", "priority",
        "text_complete", "prefill_plan",
        "trailing_idx", "text_tokens_consumed", "decode_start_frame",
    )

    def __init__(
        self, session_id: str, segment_idx: int,
        priority: RequestPriority = RequestPriority.FIRST_SEGMENT,
    ):
        self.session_id = session_id
        self.segment_idx = segment_idx
        self.slot: Optional[SlotKVState] = None
        self.state: str = "pending_prefill"
        self.priority = priority
        self.text_complete: bool = False
        self.prefill_plan: Optional[PrefillPlan] = None
        self.trailing_idx: int = 0
        self.text_tokens_consumed: int = 0
        self.decode_start_frame: int = 0


class EngineSessionGroup:
    """Groups all segments belonging to one session."""
    __slots__ = (
        "session_id", "request", "result_queue",
        "segments", "text_complete_all",
    )

    def __init__(self, session_id: str, request: EngineRequest):
        self.session_id = session_id
        self.request = request
        self.result_queue: Optional[asyncio.Queue] = request.result_queue
        self.segments: Dict[int, EngineSegment] = {}
        self.text_complete_all: bool = False

    @property
    def active_slot_count(self) -> int:
        return sum(
            1 for seg in self.segments.values()
            if seg.state in ("pending_prefill", "active") and seg.slot is not None
        )


# ---------------------------------------------------------------------------
# Segment key
# ---------------------------------------------------------------------------

def _seg_key(session_id: str, segment_idx: int) -> str:
    return f"{session_id}:{segment_idx}"


# ---------------------------------------------------------------------------
# Engine loop
# ---------------------------------------------------------------------------

class EngineLoop:
    """GPU-owning engine thread with pipelined decode and priority scheduling."""

    MAX_SLOTS_PER_SESSION = 2

    def __init__(
        self,
        engine_inbox: queue.Queue,
        async_loop: asyncio.AbstractEventLoop,
        executor: Executor,
        prefill_builder: Optional[PrefillBuilder] = None,
        *,
        max_batch_size: int = 48,
    ):
        self._inbox = engine_inbox
        self._async_loop = async_loop
        self._executor = executor
        self._prefill_builder = prefill_builder
        self._max_batch = max_batch_size

        self._groups: Dict[str, EngineSessionGroup] = {}
        # Flat index for fast slot→segment lookup during result processing
        self._seg_by_slot: Dict[int, EngineSegment] = {}

        self._running = False
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(
            target=self._run, name="engine-loop", daemon=True,
        )
        self._thread.start()
        logger.info("Engine loop started (max_batch=%d)", self._max_batch)

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Engine loop stopped")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _run(self) -> None:
        prev_output: Optional[StepOutput] = None

        while self._running:
            self._drain_inbox()
            self._try_prefill_one()

            active_slots = self._get_active_slots()
            gpu_future = None
            if active_slots:
                gpu_future = self._executor.launch_decode_step(active_slots)

            if prev_output is not None:
                self._process_step_output(prev_output)

            if gpu_future is not None:
                prev_output = gpu_future.wait()
            else:
                prev_output = None
                if not self._has_work():
                    time.sleep(0.001)

    # ------------------------------------------------------------------
    # Inbox
    # ------------------------------------------------------------------

    def _drain_inbox(self) -> None:
        drained = 0
        while True:
            try:
                req: EngineRequest = self._inbox.get_nowait()
            except queue.Empty:
                break
            self._handle_request(req)
            drained += 1
        if drained > 0:
            logger.debug("Drained %d requests", drained)

    def _handle_request(self, req: EngineRequest) -> None:
        if req.type == RequestType.NEW_SESSION:
            group = EngineSessionGroup(req.session_id, req)
            self._groups[req.session_id] = group
            logger.debug("New session group: %s", req.session_id)

        elif req.type == RequestType.START_SEGMENT:
            group = self._groups.get(req.session_id)
            if group is None:
                logger.warning("START_SEGMENT for unknown session: %s", req.session_id)
                return
            seg = EngineSegment(
                req.session_id, req.segment_idx, req.priority,
            )
            group.segments[req.segment_idx] = seg
            if req.result_queue is not None:
                group.result_queue = req.result_queue
            if req.token_ids:
                pass  # first token handled during prefill
            logger.debug("New segment: %s seg=%d prio=%s",
                         req.session_id, req.segment_idx, req.priority.name)

        elif req.type == RequestType.APPEND_TEXT:
            group = self._groups.get(req.session_id)
            if group is None:
                return
            seg = group.segments.get(req.segment_idx)
            if seg and seg.slot and req.token_ids:
                seg.slot.token_queue.extend(req.token_ids)
                seg.text_tokens_consumed += len(req.token_ids)

        elif req.type == RequestType.TEXT_COMPLETE:
            group = self._groups.get(req.session_id)
            if group is None:
                return
            seg = group.segments.get(req.segment_idx)
            if seg:
                seg.text_complete = True

        elif req.type == RequestType.CANCEL_SESSION:
            self._remove_session(req.session_id)

    # ------------------------------------------------------------------
    # Priority-based prefill
    # ------------------------------------------------------------------

    def _try_prefill_one(self) -> None:
        """Run prefill for the highest-priority pending segment."""
        kv_pool = self._executor.kv_pool
        if kv_pool is None or kv_pool.free_count == 0:
            return

        best: Optional[EngineSegment] = None
        best_group: Optional[EngineSessionGroup] = None

        for group in self._groups.values():
            if group.active_slot_count >= self.MAX_SLOTS_PER_SESSION:
                continue
            for seg in group.segments.values():
                if seg.state != "pending_prefill":
                    continue
                if best is None or seg.priority.value < best.priority.value:
                    best = seg
                    best_group = group

        if best is None or best_group is None:
            return

        slot = kv_pool.allocate(_seg_key(best.session_id, best.segment_idx))
        if slot is None:
            return
        best.slot = slot
        self._seg_by_slot[slot.slot_id] = best

        if self._prefill_builder is not None:
            task_type = parse_task_type(
                best_group.request.task_type or "custom_voice",
            )
            plan = self._prefill_builder.build_plan(
                task_type=task_type,
                text="",
                language="auto",
                speaker=best_group.request.speaker_key,
                include_eos=False,
            )
            best.prefill_plan = plan
            self._executor.prefill(slot, plan.prefill_embeds)
            slot.trailing = plan.trailing
        else:
            self._executor.prefill(slot, torch.zeros(
                1, 1, 1536, device=torch.device("cuda"), dtype=torch.bfloat16,
            ))

        best.state = "active"
        best.decode_start_frame = slot.frame_idx
        self._send_result(best_group, EngineResult(
            type=ResultType.PREFILL_DONE,
            session_id=best.session_id,
            segment_idx=best.segment_idx,
        ))
        logger.debug("Prefill done: %s seg=%d prio=%s (slot=%d, past_len=%d)",
                     best.session_id, best.segment_idx, best.priority.name,
                     slot.slot_id, slot.past_len)

    # ------------------------------------------------------------------
    # Decode batch
    # ------------------------------------------------------------------

    def _get_active_slots(self) -> list[SlotKVState]:
        slots = []
        for group in self._groups.values():
            for seg in group.segments.values():
                if seg.state != "active" or seg.slot is None:
                    continue
                slot = seg.slot

                if slot.next_embed is None:
                    if slot.trailing and slot.text_idx < len(slot.trailing):
                        slot.next_embed = slot.trailing[slot.text_idx]
                        slot.text_idx += 1
                    else:
                        continue

                slots.append(slot)
                if len(slots) >= self._max_batch:
                    return slots
        return slots

    # ------------------------------------------------------------------
    # Result processing (runs while GPU does next step)
    # ------------------------------------------------------------------

    def _process_step_output(self, output: StepOutput) -> None:
        for i, slot in enumerate(output.slots):
            seg = self._seg_by_slot.get(slot.slot_id)
            if seg is None:
                continue
            group = self._groups.get(seg.session_id)
            if group is None:
                continue

            if i < len(output.split_kv) and output.split_kv[i]:
                slot.kv_tensors = output.split_kv[i]
            if i < len(output.split_c2w) and output.split_c2w[i]:
                slot.c2w_states = [t for t in output.split_c2w[i] if t is not None]
            if output.updated_tc is not None:
                slot.token_counts = output.updated_tc[i:i+1].clone()
            slot.past_len += 1
            slot.frame_idx += 1

            if output.codec_sum is not None:
                text_add = torch.zeros_like(output.codec_sum[i:i+1])
                if slot.trailing and slot.text_idx < len(slot.trailing):
                    text_add = slot.trailing[slot.text_idx].to(output.codec_sum.dtype)
                    slot.text_idx += 1
                slot.next_embed = (output.codec_sum[i:i+1] + text_add).to(torch.float32)
            else:
                slot.next_embed = None

            audio = output.audio_chunks[i]
            if audio is not None and len(audio) > 0:
                self._send_result(group, EngineResult(
                    type=ResultType.AUDIO_CHUNK,
                    session_id=seg.session_id,
                    segment_idx=seg.segment_idx,
                    audio_bytes=audio,
                ))

            if output.eos_flags[i]:
                self._handle_segment_eos(group, seg)

    def _handle_segment_eos(
        self, group: EngineSessionGroup, seg: EngineSegment,
    ) -> None:
        """Handle EOS for one segment."""
        audio_steps = 0
        if seg.slot:
            audio_steps = seg.slot.frame_idx - seg.decode_start_frame
        metrics = {
            "audio_steps": audio_steps,
            "text_tokens": seg.text_tokens_consumed,
            "segment_idx": seg.segment_idx,
        }

        seg.state = "done"
        if seg.slot:
            slot_id = seg.slot.slot_id
            self._executor.kv_pool.release(slot_id)
            self._seg_by_slot.pop(slot_id, None)
            seg.slot = None

        self._send_result(group, EngineResult(
            type=ResultType.SEGMENT_END,
            session_id=seg.session_id,
            segment_idx=seg.segment_idx,
            metrics=metrics,
        ))
        logger.info("Segment EOS: %s seg=%d audio_steps=%d text_tokens=%d",
                     seg.session_id, seg.segment_idx,
                     audio_steps, seg.text_tokens_consumed)

        self._check_session_done(group)

    def _check_session_done(self, group: EngineSessionGroup) -> None:
        """Check if ALL segments are done and no more are expected."""
        all_done = all(s.state == "done" for s in group.segments.values())
        if not all_done:
            return

        # More segments may arrive from the frontend; only finish when
        # the frontend has sent TEXT_COMPLETE for the last segment.
        has_pending = any(
            not s.text_complete for s in group.segments.values()
            if s.state != "done"
        )
        if has_pending:
            return

        self._send_result(group, EngineResult(
            type=ResultType.SESSION_DONE,
            session_id=group.session_id,
        ))
        self._remove_session(group.session_id)

    # ------------------------------------------------------------------
    # Session cleanup
    # ------------------------------------------------------------------

    def _remove_session(self, session_id: str) -> None:
        group = self._groups.pop(session_id, None)
        if group is None:
            return
        for seg in group.segments.values():
            if seg.slot:
                self._executor.kv_pool.release(seg.slot.slot_id)
                self._seg_by_slot.pop(seg.slot.slot_id, None)

    # ------------------------------------------------------------------
    # Result delivery (cross-thread)
    # ------------------------------------------------------------------

    def _send_result(
        self, group: EngineSessionGroup, result: EngineResult,
    ) -> None:
        if group.result_queue is None:
            return
        q = group.result_queue
        self._async_loop.call_soon_threadsafe(q.put_nowait, result)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _has_work(self) -> bool:
        return any(
            seg.state in ("pending_prefill", "active")
            for group in self._groups.values()
            for seg in group.segments.values()
        )

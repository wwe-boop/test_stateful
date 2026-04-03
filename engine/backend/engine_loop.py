"""Engine loop: runs in a dedicated thread, owns all GPU resources.

Pipeline design for high GPU utilization:

    ┌─────────────────── Iteration K ───────────────────────────┐
    │                                                           │
    │  Phase 1 (CPU):  Process step K-1 output                  │
    │     • scatter KV to pool                                  │
    │     • compute next_embed from codec_sum                   │
    │     • check EOS, send audio chunks                        │
    │                                                           │
    │  Phase 2 (CPU):  Prefill new sessions (rare, ~10ms each)  │
    │     • runs on separate CUDA stream                        │
    │     • only when new sessions arrive                       │
    │                                                           │
    │  Phase 3 (CPU→GPU):  Build & launch step K                │
    │     • pool gather KV → batched tensor                     │
    │     • launch TRT on compute_stream (non-blocking)         │
    │                                                           │
    │  Phase 4 (CPU ∥ GPU):  Housekeeping while GPU computes    │
    │     • drain inbox                                         │
    │     • evict idle slots                                    │
    │     • check session timeouts                              │
    │                                                           │
    │  Phase 5 (sync):  wait GPU → prev_output for next iter    │
    └───────────────────────────────────────────────────────────┘

    This ordering guarantees autoregressive correctness:
    step K reads KV that includes step K-1's output (Phase 1 runs
    BEFORE Phase 3).  CPU housekeeping overlaps with GPU compute.

Level 2 pipelining:
    Multiple segments of the same session may decode in parallel.
    Each segment owns its own KV slot.  The engine batches all active
    slots regardless of session, and routes results by (session_id,
    segment_idx).

    Prefill priority: FIRST_SEGMENT > CONTINUATION > PREFETCHED.

Scheduling:
    MLFQ (Multi-Level Feedback Queue) dynamically assigns decode
    priority per segment.  Anti-starvation aging boosts long-waiting
    segments to prevent starvation.

Slot management:
    Idle slots past max_idle_sec are evicted.  Backpressure prevents
    new sessions when the queue depth exceeds max_queue_size.

Prefix KV caching:
    System prompt KV tensors are cached across requests with the same
    speaker/task configuration, eliminating redundant prefill GPU work.

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

from ..core.mlfq import MLFQConfig, MLFQMeta, MLFQScheduler
from ..core.types import (
    EngineRequest,
    EngineResult,
    RequestPriority,
    RequestType,
    ResultType,
)
from .executor import Executor, StepOutput
from .kv_cache_pool import KVCachePool, SlotKVState
from .prefix_cache import PrefixKVCache
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
        "mlfq_meta",
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
        self.mlfq_meta: MLFQMeta = MLFQMeta()


class EngineSessionGroup:
    """Groups all segments belonging to one session."""
    __slots__ = (
        "session_id", "request", "result_queue",
        "segments", "text_complete_all", "created_at",
    )

    def __init__(self, session_id: str, request: EngineRequest):
        self.session_id = session_id
        self.request = request
        self.result_queue: Optional[asyncio.Queue] = request.result_queue
        self.segments: Dict[int, EngineSegment] = {}
        self.text_complete_all: bool = False
        self.created_at: float = time.monotonic()

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
        mlfq_config: Optional[MLFQConfig] = None,
        prefix_cache_max_entries: int = 16,
        prefix_cache_max_len: int = 512,
        max_idle_sec: float = 10.0,
        max_queue_size: int = 256,
        session_timeout_sec: float = 300.0,
    ):
        self._inbox = engine_inbox
        self._async_loop = async_loop
        self._executor = executor
        self._prefill_builder = prefill_builder
        self._max_batch = max_batch_size
        self._max_idle_sec = max_idle_sec
        self._max_queue_size = max_queue_size
        self._session_timeout_sec = session_timeout_sec

        self._groups: Dict[str, EngineSessionGroup] = {}
        self._seg_by_slot: Dict[int, EngineSegment] = {}

        self._mlfq = MLFQScheduler(mlfq_config or MLFQConfig())
        self._prefix_cache = PrefixKVCache(
            max_entries=prefix_cache_max_entries,
            max_prefix_len=prefix_cache_max_len,
        )

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_eviction_check: float = 0.0

        self._total_steps: int = 0
        self._total_prefills: int = 0
        self._total_sessions: int = 0
        self._total_eos: int = 0
        self._total_timeouts: int = 0
        self._total_evictions: int = 0

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
            # --- Phase 1: Process previous step output (MUST run before
            # building next inputs to satisfy autoregressive dependency) ---
            if prev_output is not None:
                self._process_step_output(prev_output)
                prev_output = None

            # --- Phase 2: Prefill new sessions (synchronous GPU work,
            # uses dedicated prefill stream so it does not conflict with
            # the decode compute stream) ---
            self._try_prefill_one()

            # --- Phase 3: Build & launch next decode step ---
            active_slots = self._get_active_slots_mlfq()
            gpu_future = None
            if active_slots:
                gpu_future = self._executor.launch_decode_step(active_slots)

            # --- Phase 4: While GPU computes, do CPU housekeeping ---
            self._drain_inbox()
            self._try_evict_idle_slots()
            self._try_timeout_sessions()

            # --- Phase 5: Wait for GPU, store output for next iteration ---
            if gpu_future is not None:
                prev_output = gpu_future.wait()
                self._total_steps += 1
                self._mlfq.tick(self._all_active_mlfq_metas())
            else:
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
            if len(self._groups) >= self._max_queue_size:
                logger.warning(
                    "Backpressure: rejecting session %s (active=%d >= limit=%d)",
                    req.session_id, len(self._groups), self._max_queue_size,
                )
                if req.result_queue is not None:
                    self._async_loop.call_soon_threadsafe(
                        req.result_queue.put_nowait,
                        EngineResult(
                            type=ResultType.ERROR,
                            session_id=req.session_id,
                            error_msg="Server overloaded, please retry later",
                        ),
                    )
                return
            group = EngineSessionGroup(req.session_id, req)
            self._groups[req.session_id] = group
            self._total_sessions += 1
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
        """Run prefill for the highest-priority pending segment.

        Uses prefix KV cache when available to skip redundant GPU work.
        """
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
        self._mlfq.on_segment_created(best.mlfq_meta)

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

            cached = self._prefix_cache.get(plan.prefix_cache_key)
            if cached is not None and plan.request_prefill_embeds is not None:
                slot.talker_kv = cached.talker_kv.clone()
                slot.past_len = cached.prefix_len
                self._executor.prefill(slot, plan.request_prefill_embeds)
                logger.debug(
                    "Prefix cache hit: skipped %d prefix tokens for %s",
                    cached.prefix_len, best.session_id,
                )
            else:
                self._executor.prefill(slot, plan.prefill_embeds)
                if (
                    plan.prefix_cache_key is not None
                    and plan.cacheable_prefix_embeds is not None
                    and slot.talker_kv is not None
                ):
                    prefix_len = int(plan.cacheable_prefix_embeds.shape[1])
                    prefix_kv = slot.talker_kv[:, :, :, :prefix_len, :].clone()
                    self._prefix_cache.put(
                        plan.prefix_cache_key, prefix_kv, prefix_len,
                    )

            slot.trailing = plan.trailing
        else:
            self._executor.prefill(slot, torch.zeros(
                1, 1, 1536, device=torch.device("cuda"), dtype=torch.bfloat16,
            ))

        best.state = "active"
        best.decode_start_frame = slot.frame_idx
        self._total_prefills += 1
        self._send_result(best_group, EngineResult(
            type=ResultType.PREFILL_DONE,
            session_id=best.session_id,
            segment_idx=best.segment_idx,
        ))
        logger.debug("Prefill done: %s seg=%d prio=%s (slot=%d, past_len=%d)",
                     best.session_id, best.segment_idx, best.priority.name,
                     slot.slot_id, slot.past_len)

    # ------------------------------------------------------------------
    # Decode batch (MLFQ-ordered)
    # ------------------------------------------------------------------

    def _get_active_slots_mlfq(self) -> list[SlotKVState]:
        """Build decode batch using MLFQ priority ordering."""
        candidates: list[EngineSegment] = []
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
                candidates.append(seg)

        if not candidates:
            return []

        ordered = self._mlfq.select_batch(
            candidates, self._max_batch,
            get_meta=lambda seg: seg.mlfq_meta,
        )
        return [seg.slot for seg in ordered]

    def _all_active_mlfq_metas(self) -> list[MLFQMeta]:
        """Collect all active segment MLFQ metas for aging."""
        return [
            seg.mlfq_meta
            for group in self._groups.values()
            for seg in group.segments.values()
            if seg.state == "active"
        ]

    # ------------------------------------------------------------------
    # Idle slot eviction
    # ------------------------------------------------------------------

    def _try_evict_idle_slots(self) -> None:
        """Periodically check for idle slots and evict them."""
        now = time.monotonic()
        if now - self._last_eviction_check < 1.0:
            return
        self._last_eviction_check = now

        kv_pool = self._executor.kv_pool
        if kv_pool is None:
            return

        while True:
            candidate = kv_pool.find_eviction_candidate(self._max_idle_sec)
            if candidate is None:
                break
            evicted_session_key = kv_pool.force_evict(candidate.slot_id)
            if evicted_session_key is None:
                break

            seg = self._seg_by_slot.pop(candidate.slot_id, None)
            if seg is None:
                continue

            group = self._groups.get(seg.session_id)
            if group is None:
                continue

            seg.state = "evicted"
            seg.slot = None
            self._send_result(group, EngineResult(
                type=ResultType.ERROR,
                session_id=seg.session_id,
                segment_idx=seg.segment_idx,
                error_msg=f"Slot evicted: idle > {self._max_idle_sec}s",
            ))
            self._total_evictions += 1
            logger.warning(
                "Evicted segment %s:%d due to idle timeout",
                seg.session_id, seg.segment_idx,
            )

    # ------------------------------------------------------------------
    # Result processing (runs while GPU does next step)
    # ------------------------------------------------------------------

    def _process_step_output(self, output: StepOutput) -> None:
        kv_pool = self._executor.kv_pool
        use_pool = kv_pool is not None and kv_pool._preallocate

        # Batch-level KV scatter to pool (single operation, avoids per-slot split)
        if use_pool and output.batch_talker_kv is not None:
            slot_ids = [s.slot_id for s in output.slots]
            kv_pool.scatter_talker_kv(
                slot_ids, output.batch_talker_kv,
                output.original_past_lens, output.padded_past_len, 1,
            )
        if use_pool and output.batch_c2w_kv is not None:
            slot_ids = [s.slot_id for s in output.slots]
            kv_pool.scatter_c2w_kv(slot_ids, output.batch_c2w_kv)

        for i, slot in enumerate(output.slots):
            seg = self._seg_by_slot.get(slot.slot_id)
            if seg is None:
                continue
            group = self._groups.get(seg.session_id)
            if group is None:
                continue

            if not use_pool:
                if output.batch_talker_kv is not None:
                    slot.talker_kv = output.batch_talker_kv[i:i+1]
                if output.batch_c2w_kv is not None:
                    slot.c2w_kv = output.batch_c2w_kv[i:i+1]

            if output.split_c2w_conv[i]:
                slot.c2w_conv_states = [
                    t for t in output.split_c2w_conv[i] if t is not None
                ]
            if output.split_c2w_transconv[i]:
                slot.c2w_transconv_states = [
                    t for t in output.split_c2w_transconv[i] if t is not None
                ]
            if output.updated_tc is not None:
                slot.token_counts = output.updated_tc[i:i+1].clone()
            slot.past_len += 1
            slot.frame_idx += 1
            slot.touch()
            self._mlfq.on_step_done(seg.mlfq_meta)

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
        self._total_eos += 1
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

    # ------------------------------------------------------------------
    # Session timeout
    # ------------------------------------------------------------------

    def _try_timeout_sessions(self) -> None:
        """Cancel sessions that exceed the maximum allowed duration."""
        if self._session_timeout_sec <= 0:
            return
        now = time.monotonic()
        to_cancel: list[str] = []
        for sid, group in self._groups.items():
            if now - group.created_at > self._session_timeout_sec:
                to_cancel.append(sid)
        for sid in to_cancel:
            group = self._groups.get(sid)
            if group is None:
                continue
            self._total_timeouts += 1
            self._send_result(group, EngineResult(
                type=ResultType.ERROR,
                session_id=sid,
                error_msg=f"Session timeout ({self._session_timeout_sec}s exceeded)",
            ))
            self._remove_session(sid)
            logger.warning("Session %s timed out", sid)

    # ------------------------------------------------------------------
    # Health / metrics (thread-safe read)
    # ------------------------------------------------------------------

    def health_stats(self) -> dict:
        """Return a snapshot of engine health metrics.

        Called from the asyncio thread; reads only atomic int/float fields
        so no lock is needed.
        """
        kv_pool = self._executor.kv_pool
        return {
            "running": self._running,
            "active_sessions": len(self._groups),
            "active_slots": kv_pool.used_count if kv_pool else 0,
            "free_slots": kv_pool.free_count if kv_pool else 0,
            "pool_utilization": kv_pool.utilization if kv_pool else 0.0,
            "total_steps": self._total_steps,
            "total_prefills": self._total_prefills,
            "total_sessions": self._total_sessions,
            "total_eos": self._total_eos,
            "total_evictions": self._total_evictions,
            "total_timeouts": self._total_timeouts,
            "prefix_cache_stats": self._prefix_cache.stats,
        }

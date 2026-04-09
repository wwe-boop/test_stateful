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

import numpy as np
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
        "input_complete", "prefill_plan",
        "trailing_idx", "text_tokens_consumed", "decode_start_frame",
        "mlfq_meta",
        "pending_token_ids",
        "eos_trailing_added",
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
        self.input_complete: bool = False
        self.prefill_plan: Optional[PrefillPlan] = None
        self.trailing_idx: int = 0
        self.text_tokens_consumed: int = 0
        self.decode_start_frame: int = 0
        self.mlfq_meta: MLFQMeta = MLFQMeta()
        self.pending_token_ids: list[int] = []
        self.eos_trailing_added: bool = False


class EngineSessionGroup:
    """Groups all segments belonging to one session."""
    __slots__ = (
        "session_id", "request", "result_queue",
        "segments", "input_complete_all", "created_at",
        "overflow_token_ids",
    )

    def __init__(self, session_id: str, request: EngineRequest):
        self.session_id = session_id
        self.request = request
        self.result_queue: Optional[asyncio.Queue] = request.result_queue
        self.segments: Dict[int, EngineSegment] = {}
        self.input_complete_all: bool = False
        self.created_at: float = time.monotonic()
        self.overflow_token_ids: list[int] = []

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
        min_pad_steps: int = 4,
    ):
        self._inbox = engine_inbox
        self._async_loop = async_loop
        self._executor = executor
        self._prefill_builder = prefill_builder
        self._max_batch = max_batch_size
        self._max_idle_sec = max_idle_sec
        self._max_queue_size = max_queue_size
        self._session_timeout_sec = session_timeout_sec
        self._min_pad_steps = min_pad_steps

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

        self._tts_pad_embed = (
            prefill_builder.w.tts_pad_embed.clone()
            if prefill_builder is not None
            else torch.zeros(1, 1, 1536)
        )

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

            # --- Phase 1.5: Drain inbox EARLY so newly arrived requests
            # are immediately available for prefill/decode in this
            # iteration, instead of waiting until the next one. ---
            self._drain_inbox()

            # --- Phase 2: Launch decode for existing active slots FIRST.
            # GPU starts computing on compute_stream while CPU proceeds
            # to prefill on the separate prefill_stream. ---
            active_slots = self._get_active_slots_mlfq()
            gpu_future = None
            if active_slots:
                gpu_future = self._executor.launch_decode_step(active_slots)

            # --- Phase 3: Prefill ALL pending sessions (not just one).
            # Uses the dedicated prefill_stream + separate TRT execution
            # context, so prefill overlaps with the in-flight decode on
            # compute_stream.  First audio chunk is produced during
            # prefill, so first_chunk_latency = time-to-prefill. ---
            try:
                self._try_prefill_pending()
            except Exception:
                logger.exception("Prefill failed unexpectedly")

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

        elif req.type == RequestType.START_TOKENS:
            group = self._groups.get(req.session_id)
            if group is None:
                logger.warning("START_TOKENS for unknown session: %s", req.session_id)
                return
            seg = EngineSegment(
                req.session_id, req.segment_idx, req.priority,
            )
            if group.overflow_token_ids:
                seg.pending_token_ids.extend(group.overflow_token_ids)
                seg.text_tokens_consumed += len(group.overflow_token_ids)
                logger.info(
                    "Prepended %d overflow tokens to %s seg=%d",
                    len(group.overflow_token_ids),
                    req.session_id, req.segment_idx,
                )
                group.overflow_token_ids.clear()
            if req.token_ids:
                seg.pending_token_ids.extend(req.token_ids)
                seg.text_tokens_consumed += len(req.token_ids)
            group.segments[req.segment_idx] = seg
            if req.result_queue is not None:
                group.result_queue = req.result_queue
            logger.debug("New segment: %s seg=%d prio=%s",
                         req.session_id, req.segment_idx, req.priority.name)

        elif req.type == RequestType.APPEND_TOKENS:
            group = self._groups.get(req.session_id)
            if group is None:
                return
            if not req.token_ids:
                return
            seg = group.segments.get(req.segment_idx)
            if seg is None or seg.state == "done":
                group.overflow_token_ids.extend(req.token_ids)
                logger.debug(
                    "Overflow %d tokens for %s seg=%d (seg_state=%s, overflow_total=%d)",
                    len(req.token_ids), req.session_id,
                    req.segment_idx,
                    seg.state if seg else "MISSING",
                    len(group.overflow_token_ids),
                )
                return
            seg.pending_token_ids.extend(req.token_ids)
            seg.text_tokens_consumed += len(req.token_ids)
            if (
                seg.state == "active"
                and seg.slot is not None
                and self._prefill_builder is not None
            ):
                self._append_trailing_tokens(seg.slot, req.token_ids)
                self._resume_streaming_segment_if_ready(seg)
            else:
                logger.debug(
                    "APPEND_TOKENS %d tokens for %s seg=%d state=%s (pre-prefill accumulate)",
                    len(req.token_ids), req.session_id,
                    req.segment_idx, seg.state,
                )

        elif req.type == RequestType.SEGMENT_TOKENS_DONE:
            group = self._groups.get(req.session_id)
            if group is None:
                return
            seg = group.segments.get(req.segment_idx)
            if seg:
                seg.input_complete = True
                if (
                    req.append_eos
                    and
                    seg.state == "active"
                    and seg.slot is not None
                    and not seg.eos_trailing_added
                    and self._prefill_builder is not None
                ):
                    self._append_eos_trailing(seg)
                    self._resume_streaming_segment_if_ready(seg)
                elif req.append_eos:
                    seg.eos_trailing_added = False
                if seg.state == "done":
                    self._check_session_done(group)

        elif req.type == RequestType.SESSION_TOKENS_DONE:
            group = self._groups.get(req.session_id)
            if group is None:
                return
            group.input_complete_all = True
            self._check_session_done(group)

        elif req.type == RequestType.CANCEL_SESSION:
            self._remove_session(req.session_id)

    # ------------------------------------------------------------------
    # Priority-based prefill
    # ------------------------------------------------------------------

    def _try_prefill_pending(self) -> None:
        """Prefill ALL pending segments, not just one.

        Back-to-back prefills eliminate decode-step overhead between
        consecutive new sessions, reducing first-chunk latency when
        multiple requests arrive concurrently.  Capped at max_batch_size
        to bound latency for existing decoding sessions.
        """
        count = 0
        while count < self._max_batch:
            if not self._try_prefill_one():
                break
            count += 1
        if count > 1:
            logger.info("Prefilled %d segments in one pass", count)

    def _try_prefill_one(self) -> bool:
        """Run prefill for the highest-priority pending segment.

        Uses prefix KV cache when available to skip redundant GPU work.
        Returns True if a segment was prefilled, False otherwise.
        """
        kv_pool = self._executor.kv_pool
        if kv_pool is None or kv_pool.free_count == 0:
            return False

        best: Optional[EngineSegment] = None
        best_group: Optional[EngineSessionGroup] = None

        for group in self._groups.values():
            if group.active_slot_count >= self.MAX_SLOTS_PER_SESSION:
                continue
            for seg in group.segments.values():
                if seg.state != "pending_prefill":
                    continue
                if not seg.pending_token_ids:
                    continue
                if best is None or seg.priority.value < best.priority.value:
                    best = seg
                    best_group = group

        if best is None or best_group is None:
            return False

        slot = kv_pool.allocate(_seg_key(best.session_id, best.segment_idx))
        if slot is None:
            return False
        best.slot = slot
        self._seg_by_slot[slot.slot_id] = best
        self._mlfq.on_segment_created(best.mlfq_meta)

        if self._prefill_builder is not None:
            req_cfg = best_group.request.session_config
            task_type_str = (
                req_cfg.task_type
                if req_cfg is not None
                else (best_group.request.task_type or "custom_voice")
            )
            try:
                task_type = parse_task_type(
                    task_type_str,
                    x_vector_only=(req_cfg.x_vector_only if req_cfg is not None else False),
                )
            except ValueError as exc:
                logger.error("Invalid task_type for %s: %s", best.session_id, exc)
                best.state = "error"
                self._seg_by_slot.pop(slot.slot_id, None)
                kv_pool.release(slot.slot_id)
                best.slot = None
                self._send_result(best_group, EngineResult(
                    type=ResultType.ERROR,
                    session_id=best.session_id,
                    segment_idx=best.segment_idx,
                    error_msg=str(exc),
                ))
                return False
            # Check prefix cache BEFORE build_plan to skip the
            # token→text→retokenize round-trip on cache hits.
            cache_key = self._prefill_builder.compute_cache_key(
                task_type,
                req_cfg.language if req_cfg is not None else "auto",
                req_cfg.speaker if req_cfg is not None else best_group.request.speaker_key,
                req_cfg.instruct if req_cfg is not None else None,
                (
                    list(req_cfg.instruct_spec.token_ids)
                    if req_cfg is not None and req_cfg.instruct_spec is not None
                    else None
                ),
            )
            cached = self._prefix_cache.get(cache_key)

            if cached is not None and best.pending_token_ids:
                # ── Cache HIT: embed token IDs directly, skip TRT ──
                req_embeds, trailing = (
                    self._prefill_builder.build_suffix_from_ids(
                        best.pending_token_ids,
                        include_eos=best.input_complete,
                    )
                )
                self._apply_prefix_cache_hit(
                    slot, cached, req_embeds, trailing,
                )
                best.eos_trailing_added = best.input_complete
                prefill_audio = None
                prefill_eos = False
                logger.info(
                    "Prefix cache hit: copied %d KV tokens for %s "
                    "(slot=%d, %d text tokens embedded directly)",
                    cached.prefix_len, best.session_id,
                    slot.slot_id, len(best.pending_token_ids),
                )
            else:
                # ── Cache MISS: full token-native plan + TRT prefill ──
                plan = self._prefill_builder.build_plan_from_ids(
                    task_type=task_type,
                    token_ids=best.pending_token_ids,
                    language=req_cfg.language if req_cfg is not None else "auto",
                    speaker=req_cfg.speaker if req_cfg is not None else best_group.request.speaker_key,
                    instruct=req_cfg.instruct if req_cfg is not None else None,
                    instruct_token_ids=(
                        list(req_cfg.instruct_spec.token_ids)
                        if req_cfg is not None and req_cfg.instruct_spec is not None
                        else None
                    ),
                    ref_text=req_cfg.ref_text if req_cfg is not None else None,
                    ref_text_token_ids=(
                        list(req_cfg.ref_text_spec.token_ids)
                        if req_cfg is not None and req_cfg.ref_text_spec is not None
                        else None
                    ),
                    include_eos=best.input_complete,
                )
                best.prefill_plan = plan

                prefill_audio, prefill_eos = self._executor.prefill(
                    slot, plan.prefill_embeds,
                )
                # Populate cache — read from pool when preallocated
                effective_key = plan.prefix_cache_key or cache_key
                if (
                    effective_key is not None
                    and plan.cacheable_prefix_embeds is not None
                ):
                    prefix_len = int(plan.cacheable_prefix_embeds.shape[1])
                    prefix_kv = self._read_prefix_kv(slot, prefix_len)
                    if prefix_kv is not None:
                        self._prefix_cache.put(
                            effective_key, prefix_kv, prefix_len,
                        )

                slot.trailing = plan.trailing
                best.eos_trailing_added = best.input_complete

                # Combine codec_sum (TRT output) with first trailing text token
                if slot.next_embed is not None and slot.trailing:
                    first_trail = slot.trailing[0].to(slot.next_embed.dtype)
                    slot.next_embed = (
                        slot.next_embed + first_trail
                    ).to(torch.float32)
                    slot.text_idx = 1
        else:
            prefill_audio, prefill_eos = self._executor.prefill(slot, torch.zeros(
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

        if prefill_audio and len(prefill_audio) > 0:
            self._send_result(best_group, EngineResult(
                type=ResultType.AUDIO_CHUNK,
                session_id=best.session_id,
                segment_idx=best.segment_idx,
                audio_bytes=prefill_audio,
            ))
        if prefill_eos:
            self._handle_segment_eos(best_group, best)
            return True
        logger.debug(
            "Prefill done: %s seg=%d prio=%s (slot=%d, past_len=%d, "
            "trailing=%d, input_complete=%s, tokens=%d)",
            best.session_id, best.segment_idx, best.priority.name,
            slot.slot_id, slot.past_len,
            len(slot.trailing), best.input_complete, len(best.pending_token_ids),
        )
        return True

    # ------------------------------------------------------------------
    # Prefix cache helpers
    # ------------------------------------------------------------------

    def _apply_prefix_cache_hit(
        self,
        slot: SlotKVState,
        cached,
        request_prefill_embeds: torch.Tensor,
        trailing: list,
    ) -> None:
        """Set up slot from cached prefix KV without any TRT call.

        Copies the cached talker KV into the KV pool (or slot) and
        prepares the slot for decode.  The suffix token is stored as
        next_embed so the first decode step processes it with full
        attention to the cached KV.
        """
        kv_pool = self._executor.kv_pool
        kv_pool.init_kv_tensors(slot)
        prefix_len = cached.prefix_len

        if kv_pool._preallocate and kv_pool._talker_kv_pool is not None:
            kv_pool._talker_kv_pool[
                slot.slot_id, :, :, :prefix_len, :
            ] = cached.talker_kv[0, :, :, :prefix_len, :]
        else:
            slot.talker_kv = cached.talker_kv.clone()
        slot.past_len = prefix_len
        # Match Executor.prefill(): one fused forward has consumed the prefill
        # chunk; frame_idx feeds cache_position for the vocoder on decode steps.
        slot.frame_idx = 1

        slot.c2w_kv = None
        slot.c2w_conv_states = self._executor.make_zero_conv_states()
        slot.c2w_transconv_states = self._executor.make_zero_transconv_states()
        slot.init_pingpong_buffers()

        slot.next_embed = request_prefill_embeds.to(torch.float32)
        slot.last_codec_sum = None
        slot.trailing = trailing
        slot.text_idx = 0

    def _resume_streaming_segment_if_ready(self, seg: EngineSegment) -> None:
        """Resume a paused streaming segment when new trailing text/EOS arrives."""
        slot = seg.slot
        if slot is None or slot.last_codec_sum is None or slot.next_embed is not None:
            return
        if not slot.trailing or slot.text_idx >= len(slot.trailing):
            return
        text_add = slot.trailing[slot.text_idx].to(slot.last_codec_sum.dtype)
        slot.text_idx += 1
        slot.next_embed = (slot.last_codec_sum + text_add).to(torch.float32)
        slot.last_codec_sum = None
        slot.pad_start_frame = -1
        slot.pad_consecutive_silence = 0
        logger.debug(
            "Resumed paused streaming segment %s:%d (trailing=%d, text_idx=%d)",
            seg.session_id,
            seg.segment_idx,
            len(slot.trailing),
            slot.text_idx,
        )

    def _read_prefix_kv(
        self, slot: SlotKVState, prefix_len: int,
    ) -> Optional[torch.Tensor]:
        """Read prefix KV from pool or slot for cache population."""
        kv_pool = self._executor.kv_pool
        if kv_pool._preallocate and kv_pool._talker_kv_pool is not None:
            return kv_pool._talker_kv_pool[
                slot.slot_id : slot.slot_id + 1,
                :, :, :prefix_len, :,
            ].clone()
        if slot.talker_kv is not None:
            return slot.talker_kv[:, :, :, :prefix_len, :].clone()
        return None

    # ------------------------------------------------------------------
    # Decode batch (MLFQ-ordered)
    # ------------------------------------------------------------------

    def _get_active_slots_mlfq(self) -> list[SlotKVState]:
        """Build decode batch using MLFQ priority ordering."""
        kv_pool = self._executor.kv_pool
        max_seq = kv_pool.max_seq_len if kv_pool else float("inf")
        evict_pairs: list[tuple[EngineSessionGroup, EngineSegment]] = []
        candidates: list[EngineSegment] = []
        for group in self._groups.values():
            for seg in group.segments.values():
                if seg.state != "active" or seg.slot is None:
                    continue
                slot = seg.slot
                if slot.past_len >= max_seq:
                    evict_pairs.append((group, seg))
                    continue
                if slot.next_embed is None:
                    if slot.trailing and slot.text_idx < len(slot.trailing):
                        slot.next_embed = slot.trailing[slot.text_idx]
                        slot.text_idx += 1
                    else:
                        continue
                candidates.append(seg)

        for group, seg in evict_pairs:
            logger.warning("Segment hit max_seq_len (%d): %s seg=%d, forcing EOS (overflow)",
                           max_seq, seg.session_id, seg.segment_idx)
            self._handle_segment_eos(group, seg, overflow=True)

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

        batch_size = len(output.slots)
        for i, slot in enumerate(output.slots):
            seg = self._seg_by_slot.get(slot.slot_id)
            if seg is None:
                continue
            group = self._groups.get(seg.session_id)
            if group is None:
                continue

            if output.batch_c2w_kv is not None:
                kv = output.batch_c2w_kv[i:i+1]
                c2w_max_past = self._executor._config.c2w_sliding_window - 1
                if kv.shape[3] > c2w_max_past:
                    kv = kv[:, :, :, -c2w_max_past:, :]
                slot.c2w_kv = kv.clone()
            if not use_pool:
                if output.batch_talker_kv is not None:
                    slot.talker_kv = output.batch_talker_kv[i:i+1]

            if output.used_pingpong and slot.pingpong_ready:
                # batch=1 zero-copy: TRT wrote directly to write bufs
                slot.flip_c2w_buffers()
            elif slot.pingpong_ready:
                # batch>1 pre-allocated copy: scatter into write bufs
                slot.copy_c2w_and_flip(
                    output.split_c2w_conv[i],
                    output.split_c2w_transconv[i],
                )
            else:
                # Fallback: clone (first step or non-pingpong slot)
                if output.split_c2w_conv[i]:
                    slot.c2w_conv_states = [
                        t.clone() for t in output.split_c2w_conv[i]
                        if t is not None
                    ]
                if output.split_c2w_transconv[i]:
                    slot.c2w_transconv_states = [
                        t.clone() for t in output.split_c2w_transconv[i]
                        if t is not None
                    ]
            if output.updated_tc is not None:
                slot.token_counts = output.updated_tc[i:i+1].clone()
            slot.past_len += 1
            slot.frame_idx += 1
            slot.touch()
            self._mlfq.on_step_done(seg.mlfq_meta)

            # --- Determine text_add and track pad phase ---
            in_pad = False
            if output.codec_sum is not None:
                if slot.trailing and slot.text_idx < len(slot.trailing):
                    text_add = slot.trailing[slot.text_idx].to(output.codec_sum.dtype)
                    slot.text_idx += 1
                    slot.pad_start_frame = -1
                    slot.pad_consecutive_silence = 0
                    slot.last_codec_sum = None
                    slot.next_embed = (output.codec_sum[i:i+1] + text_add).to(torch.float32)
                elif not seg.input_complete:
                    # True streaming pause: preserve the latest codec_sum and
                    # wait for more text instead of injecting pad tokens, which
                    # creates artificial silences and prosody discontinuities.
                    slot.last_codec_sum = output.codec_sum[i:i+1].clone()
                    slot.next_embed = None
                    slot.pad_start_frame = -1
                    slot.pad_consecutive_silence = 0
                    logger.debug(
                        "Paused streaming segment %s:%d awaiting text (frame=%d, past=%d)",
                        seg.session_id,
                        seg.segment_idx,
                        slot.frame_idx,
                        slot.past_len,
                    )
                else:
                    text_add = self._tts_pad_embed.to(output.codec_sum.dtype)
                    in_pad = True
                    slot.last_codec_sum = None
                    if slot.pad_start_frame < 0:
                        slot.pad_start_frame = slot.frame_idx
                    slot.next_embed = (output.codec_sum[i:i+1] + text_add).to(torch.float32)
            else:
                slot.next_embed = None

            # --- Pad phase controls ---
            # Only two safeguards:
            #   1) Dynamic silence abort — stricter as KV budget shrinks
            #   2) KV overflow (past_len >= max_seq_len) — handled by
            #      _get_active_slots_mlfq before the next decode step
            pad_steps = (slot.frame_idx - slot.pad_start_frame) if in_pad and slot.pad_start_frame >= 0 else 0

            if output.eos_flags[i]:
                self._handle_segment_eos(group, seg)
            else:
                audio = output.audio_chunks[i]

                if in_pad:
                    if audio is not None and len(audio) > 0:
                        audio_np = np.frombuffer(audio, dtype=np.float32)
                        if audio_np.size > 0 and np.max(np.abs(audio_np)) < 1e-4:
                            slot.pad_consecutive_silence += 1
                        else:
                            slot.pad_consecutive_silence = 0

                    if pad_steps >= self._min_pad_steps:
                        kv_pool = self._executor.kv_pool
                        max_seq = kv_pool.max_seq_len if kv_pool else 512
                        remaining_kv = max(0, max_seq - slot.past_len)
                        silence_limit = self._dynamic_silence_limit(remaining_kv)
                        if slot.pad_consecutive_silence > silence_limit:
                            logger.info(
                                "Silence abort: %s seg=%d silence=%d limit=%d "
                                "pad=%d remaining_kv=%d",
                                seg.session_id, seg.segment_idx,
                                slot.pad_consecutive_silence,
                                silence_limit, pad_steps, remaining_kv,
                            )
                            self._handle_segment_eos(group, seg)
                            continue

                if audio is not None and len(audio) > 0:
                    self._send_result(group, EngineResult(
                        type=ResultType.AUDIO_CHUNK,
                        session_id=seg.session_id,
                        segment_idx=seg.segment_idx,
                        audio_bytes=audio,
                    ))

    @staticmethod
    def _dynamic_silence_limit(remaining_kv: int) -> int:
        """Silence frame threshold — stricter as KV budget shrinks.

        Mirrors old engine DecodeSessionFSM.dynamic_silence_limit:
        more patience early (remaining > 100 → 12 frames), increasingly
        aggressive as the slot approaches max_seq_len (≤ 20 → 1 frame).
        """
        if remaining_kv > 100:
            return 12
        if remaining_kv > 50:
            return 6
        if remaining_kv > 20:
            return 3
        return 1

    def _handle_segment_eos(
        self, group: EngineSessionGroup, seg: EngineSegment,
        *, overflow: bool = False,
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
            "overflow": overflow,
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
        logger.info("Segment EOS: %s seg=%d audio_steps=%d text_tokens=%d overflow=%s",
                    seg.session_id, seg.segment_idx,
                    audio_steps, seg.text_tokens_consumed, overflow)

        self._check_session_done(group)

    def _check_session_done(self, group: EngineSessionGroup) -> None:
        """Check if ALL segments are done and no more are expected.

        Requires input_complete_all (SESSION_TOKENS_DONE received) so that
        streaming sessions don't conclude before all text has arrived.
        Also waits for overflow_token_ids to be drained into new segments.
        """
        if not group.input_complete_all:
            return

        if group.overflow_token_ids:
            logger.warning(
                "Session %s has %d overflow tokens pending — waiting for new segment",
                group.session_id, len(group.overflow_token_ids),
            )
            return

        all_done = all(s.state == "done" for s in group.segments.values())
        if not all_done:
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
    # Text embedding helpers (for streaming APPEND_TOKENS)
    # ------------------------------------------------------------------

    def _append_trailing_tokens(
        self, slot: SlotKVState, token_ids: list[int],
    ) -> None:
        """Embed new token IDs and append to slot's trailing list.

        Called when APPEND_TOKENS arrives after a segment has already been
        prefilled, so the model can see the new text during decode.
        """
        w = self._prefill_builder.w
        ids_tensor = torch.tensor(
            [token_ids], device=w.device, dtype=torch.int64,
        )
        with torch.no_grad():
            embed = w.text_embed(ids_tensor)
        for i in range(embed.shape[1]):
            slot.trailing.append(embed[:, i : i + 1, :].clone())
        logger.debug(
            "Appended %d trailing tokens (total=%d, text_idx=%d)",
            len(token_ids), len(slot.trailing), slot.text_idx,
        )

    def _append_eos_trailing(self, seg: EngineSegment) -> None:
        """Append tts_eos_embed to trailing when SEGMENT_TOKENS_DONE arrives post-prefill."""
        w = self._prefill_builder.w
        seg.slot.trailing.append(w.tts_eos_embed.clone())
        seg.eos_trailing_added = True
        logger.debug(
            "Appended EOS trailing for seg=%d (total=%d)",
            seg.segment_idx, len(seg.slot.trailing),
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

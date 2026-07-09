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
import math
import queue
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

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


_SS_ACOUSTIC_VARIANTS = {
    "acoustic_tail_only",
    "tail_kv_pause_recovery",
    "full_steadystream",
}
_SS_KV_VARIANTS = {
    "kv_tail_only",
    "tail_kv_pause_recovery",
    "full_steadystream",
}
_SS_VALID_VARIANTS = _SS_ACOUSTIC_VARIANTS | _SS_KV_VARIANTS
_SS_DEFAULT_KV_TAIL_TOKENS = 384


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
        "overflow_token_ids", "steadystream_variant",
        "steadystream_experimental", "steadystream_carry",
    )

    def __init__(self, session_id: str, request: EngineRequest):
        self.session_id = session_id
        self.request = request
        self.result_queue: Optional[asyncio.Queue] = request.result_queue
        self.segments: Dict[int, EngineSegment] = {}
        self.input_complete_all: bool = False
        self.created_at: float = time.monotonic()
        self.overflow_token_ids: list[int] = []
        self.steadystream_experimental: dict[str, str] = {}
        if request.session_config is not None:
            self.steadystream_experimental = dict(
                request.session_config.experimental or {}
            )
        self.steadystream_variant: str = str(
            self.steadystream_experimental.get("steadystream_variant", "")
            or ""
        ).strip()
        if self.steadystream_variant not in _SS_VALID_VARIANTS:
            self.steadystream_variant = ""
        self.steadystream_carry: dict[str, Any] = {}

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
        pad_silence_peak_threshold: float = 5e-4,
        pad_silence_mean_abs_threshold: float = 2e-4,
        max_slots_per_session: int = 2,
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
        self._pad_silence_peak_threshold = float(pad_silence_peak_threshold)
        self._pad_silence_mean_abs_threshold = float(pad_silence_mean_abs_threshold)
        self._max_slots_per_session = max(1, int(max_slots_per_session))

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

        self._embed_device = self._executor._device
        self._embed_dtype = self._executor._config.dtype
        self._hidden_size = self._executor._config.hidden_size
        self._tts_pad_embed = (
            prefill_builder.w.tts_pad_embed.to(
                device=self._embed_device,
                dtype=self._embed_dtype,
            ).clone()
            if prefill_builder is not None
            else torch.zeros(
                1,
                1,
                self._hidden_size,
                device=self._embed_device,
                dtype=self._embed_dtype,
            )
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

            # --- Phase 2: Prefill pending sessions FIRST.
            # Prefill and decode share one TRT execution context, so they
            # must run serially. First audio chunk is still produced during
            # prefill, so first_chunk_latency = time-to-prefill. ---
            try:
                self._try_prefill_pending()
            except Exception:
                logger.exception("Prefill failed unexpectedly")
                self._cleanup_failed_prefills()

            # --- Phase 3: Launch decode for active slots. ---
            active_slots = self._get_active_slots_mlfq()
            gpu_future = None
            if active_slots:
                gpu_future = self._executor.launch_decode_step(active_slots)

            # --- Phase 4: While decode runs, do CPU housekeeping ---
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
            replacing_existing = req.session_id in self._groups
            if not replacing_existing and len(self._groups) >= self._max_queue_size:
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
            if replacing_existing:
                logger.warning(
                    "NEW_SESSION replacing existing backend session: %s",
                    req.session_id,
                )
                self._remove_session(req.session_id)
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
            old_seg = group.segments.get(req.segment_idx)
            if old_seg is not None:
                logger.warning(
                    "START_TOKENS replacing existing segment: %s seg=%d state=%s",
                    req.session_id,
                    req.segment_idx,
                    old_seg.state,
                )
                self._release_segment_slot(old_seg)
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
            group = self._groups.get(req.session_id)
            if group is not None:
                self._send_result(group, EngineResult(
                    type=ResultType.SESSION_DONE,
                    session_id=req.session_id,
                    metrics={"cancelled": True},
                ))
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
            max_slots_for_group = (
                1 if group.steadystream_variant else self._max_slots_per_session
            )
            if group.active_slot_count >= max_slots_for_group:
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
        slot.segment_idx = int(best.segment_idx)
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
                self._remove_session(best.session_id)
                return False
            prefill_metrics = self._prefill_metrics(task_type, req_cfg)
            # Non-ICL tasks can check cache before building the full plan. ICL
            # needs the plan first because its request suffix includes ref
            # codec frames plus target text.
            cache_key = None
            cached = None
            if task_type != TaskType.VOICE_CLONE_ICL:
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
                    spk_embedding=(
                        req_cfg.spk_embedding
                        if req_cfg is not None
                        else None
                    ),
                )
                cached = self._prefix_cache.get(cache_key)

            if (
                self._steadystream_uses_acoustic(best_group)
                and not self._steadystream_uses_kv(best_group)
            ):
                self._restore_steadystream_c2w_carry(
                    best_group, slot, prefill_metrics,
                )
                cached = None
            force_full_prefill = (
                self._steadystream_uses_acoustic(best_group)
                and not self._steadystream_uses_kv(best_group)
            )

            steadystream_prefill = self._try_prefill_from_steadystream_carry(
                best_group, best, slot, task_type, prefill_metrics, cached,
            )
            if steadystream_prefill is not None:
                prefill_audio, prefill_eos = steadystream_prefill
                best.eos_trailing_added = best.input_complete
            elif task_type != TaskType.VOICE_CLONE_ICL and cached is not None and best.pending_token_ids:
                # ── Cache HIT: restore prefix KV and let decode consume first text token ──
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
                prefill_audio, prefill_eos = None, False
                logger.info(
                    "Prefix cache hit: copied %d KV tokens for %s "
                    "(slot=%d, decode will consume first text token in batch)",
                    cached.prefix_len, best.session_id,
                    slot.slot_id,
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
                    spk_embedding=(
                        req_cfg.spk_embedding
                        if req_cfg is not None
                        else None
                    ),
                    ref_text=req_cfg.ref_text if req_cfg is not None else None,
                    ref_text_token_ids=(
                        list(req_cfg.ref_text_spec.token_ids)
                        if req_cfg is not None and req_cfg.ref_text_spec is not None
                        else None
                    ),
                    ref_codec_sum_vec=(
                        req_cfg.ref_codec_sum_vec
                        if req_cfg is not None
                        else None
                    ),
                    ref_audio_sha256=(
                        req_cfg.ref_audio_sha256
                        if req_cfg is not None
                        else None
                    ),
                    ref_feature_cache_key=(
                        req_cfg.ref_feature_cache_key
                        if req_cfg is not None
                        else None
                    ),
                    include_eos=best.input_complete,
                )
                best.prefill_plan = plan
                slot.prefill_source = "full_prefill"

                if req_cfg is not None and req_cfg.ref_warnings:
                    for warning_msg in req_cfg.ref_warnings:
                        self._send_result(best_group, EngineResult(
                            type=ResultType.WARNING,
                            session_id=best.session_id,
                            segment_idx=best.segment_idx,
                            warning_msg=str(warning_msg),
                        ))
                if plan.warnings:
                    for warning_msg in plan.warnings:
                        self._send_result(best_group, EngineResult(
                            type=ResultType.WARNING,
                            session_id=best.session_id,
                            segment_idx=best.segment_idx,
                            warning_msg=str(warning_msg),
                        ))

                # ICL keeps the reference codec path as one complete prefill.
                # Splitting it into a cached reference prefix and request suffix
                # changes the fused Talker/C2W state boundary and can corrupt
                # generation quality.
                split_prefix_prefill = (
                    not force_full_prefill
                    and task_type != TaskType.VOICE_CLONE_ICL
                    and plan.cacheable_prefix_embeds is not None
                    and plan.request_prefill_embeds is not None
                    and int(plan.request_prefill_embeds.shape[1]) == 1
                )
                if split_prefix_prefill:
                    self._executor.prefill_prefix_only(
                        slot, plan.cacheable_prefix_embeds,
                    )
                    prefill_audio, prefill_eos = None, False
                else:
                    if task_type == TaskType.VOICE_CLONE_ICL:
                        self._apply_ref_c2w_warm_state(best_group, best, req_cfg)
                    prefill_audio, prefill_eos = self._executor.prefill(
                        slot, plan.prefill_embeds,
                    )
                if task_type != TaskType.VOICE_CLONE_ICL:
                    # Populate cache — read from pool when preallocated.
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

                    if split_prefix_prefill:
                        self._prime_decode_after_prefix_prefill(
                            slot,
                            plan.request_prefill_embeds,
                            plan.trailing,
                            source="full_prefill_prefix_only",
                        )
                        best.eos_trailing_added = best.input_complete
                    else:
                        self._attach_trailing_after_prefill(slot, plan.trailing)
                        best.eos_trailing_added = best.input_complete
                elif not split_prefix_prefill:
                    self._attach_trailing_after_prefill(slot, plan.trailing)
                    best.eos_trailing_added = best.input_complete
        else:
            prefill_metrics = {}
            prefill_audio, prefill_eos = self._executor.prefill(slot, torch.zeros(
                1,
                1,
                self._hidden_size,
                device=self._embed_device,
                dtype=self._embed_dtype,
            ))

        best.state = "active"
        best.decode_start_frame = slot.frame_idx
        self._total_prefills += 1
        if any(
            key in prefill_metrics
            for key in (
                "ref_source",
                "ref_id",
                "ref_audio_sha256",
                "ref_text_hash",
                "icl_cache_hit",
                "icl_cache_miss",
                "ref_preprocess_runtime",
            )
        ):
            logger.info(
                "Prefill metadata: session=%s segment=%d %s",
                best.session_id,
                best.segment_idx,
                " ".join(
                    f"{key}={value}"
                    for key, value in sorted(prefill_metrics.items())
                ),
            )
        self._send_result(best_group, EngineResult(
            type=ResultType.PREFILL_DONE,
            session_id=best.session_id,
            segment_idx=best.segment_idx,
            metrics=prefill_metrics,
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
        self._restore_prefix_cache(slot, cached)
        self._prime_decode_after_prefix_prefill(
            slot,
            request_prefill_embeds,
            trailing,
            source="prefix_cache_prefix_only",
        )

    def _restore_prefix_cache(self, slot: SlotKVState, cached) -> None:
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
        slot.position_offset = 0

    def _apply_ref_c2w_warm_state(
        self,
        group: EngineSessionGroup,
        seg: EngineSegment,
        req_cfg,
    ) -> None:
        if req_cfg is None or req_cfg.ref_c2w_kv is None or seg.slot is None:
            return
        warmed = self._executor.apply_c2w_warm_state(
            seg.slot,
            req_cfg.ref_c2w_kv,
            req_cfg.ref_c2w_conv_states,
            req_cfg.ref_c2w_transconv_states,
            req_cfg.ref_c2w_frame_idx,
        )
        if warmed:
            logger.info(
                "Applied Code2Wav ref warm state for %s "
                "(slot=%d, frame_idx=%d)",
                seg.session_id,
                seg.slot.slot_id,
                int(req_cfg.ref_c2w_frame_idx),
            )

    def _attach_trailing_after_prefill(
        self,
        slot: SlotKVState,
        trailing: list,
    ) -> None:
        slot.trailing = trailing
        if slot.next_embed is not None and slot.trailing:
            first_trail = slot.trailing[0].to(slot.next_embed.dtype)
            slot.next_embed = (
                slot.next_embed + first_trail
            ).to(torch.float32)
            slot.text_idx = 1

    def _prefill_metrics(self, task_type: TaskType, req_cfg) -> dict:
        metrics: dict[str, str] = {"task_type": task_type.value}
        if req_cfg is None:
            return metrics
        for attr in (
            "ref_source",
            "ref_id",
            "ref_audio_sha256",
            "ref_text_hash",
            "ref_preprocess_runtime",
        ):
            value = getattr(req_cfg, attr, None)
            if value:
                text = str(value)
                if attr in ("ref_audio_sha256", "ref_text_hash"):
                    text = text[:12]
                metrics[attr] = text
        return metrics

    def _prime_decode_after_prefix_prefill(
        self,
        slot: SlotKVState,
        request_prefill_embeds: torch.Tensor,
        trailing: list,
        *,
        source: str,
    ) -> None:
        """Prepare slot so decode step0 consumes the first text token."""
        slot.prefill_source = source
        slot.frame_idx = 0
        slot.pad_start_frame = -1
        slot.pad_consecutive_silence = 0

        slot.c2w_kv = None
        slot.c2w_conv_states = self._executor.make_zero_conv_states()
        slot.c2w_transconv_states = self._executor.make_zero_transconv_states()
        slot.init_pingpong_buffers()

        cfg = self._executor._config
        slot.token_counts = torch.zeros(
            1,
            cfg.codec_vocab_size,
            device=self._embed_device,
            dtype=torch.int64,
        )
        slot.next_embed = self._coerce_embed_tensor(
            request_prefill_embeds,
            dtype=torch.float32,
        )
        slot.last_codec_sum = None
        slot.steadystream_replay_embeds = []
        slot.steadystream_full_codecs = []
        slot.trailing = [self._coerce_embed_tensor(t) for t in trailing]
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

    @staticmethod
    def _steadystream_uses_acoustic(group: EngineSessionGroup) -> bool:
        return group.steadystream_variant in _SS_ACOUSTIC_VARIANTS

    @staticmethod
    def _steadystream_uses_kv(group: EngineSessionGroup) -> bool:
        return group.steadystream_variant in _SS_KV_VARIANTS

    def _steadystream_kv_tail_tokens(self, group: EngineSessionGroup) -> int:
        raw = group.steadystream_experimental.get("kv_tail_tokens", "")
        try:
            value = int(raw)
        except (TypeError, ValueError):
            value = _SS_DEFAULT_KV_TAIL_TOKENS
        return max(16, min(value, self._executor.kv_pool.max_seq_len))

    @staticmethod
    def _steadystream_inherits_token_counts(group: EngineSessionGroup) -> bool:
        raw = str(
            group.steadystream_experimental.get("kv_inherit_token_counts", "")
        ).strip().lower()
        if raw in {"1", "true", "yes", "on"}:
            return True
        if raw in {"", "0", "false", "no", "off"}:
            return False
        return False

    @staticmethod
    def _steadystream_prepends_prefix_to_kv_tail(group: EngineSessionGroup) -> bool:
        raw = str(
            group.steadystream_experimental.get("kv_prepend_prefix", "")
        ).strip().lower()
        return raw in {"1", "true", "yes", "on"}

    @staticmethod
    def _steadystream_uses_reprefill_history(group: EngineSessionGroup) -> bool:
        raw = str(
            group.steadystream_experimental.get("kv_reprefill_history", "")
        ).strip().lower()
        return raw in {"1", "true", "yes", "on"}

    @staticmethod
    def _steadystream_uses_token_history(group: EngineSessionGroup) -> bool:
        raw = str(
            group.steadystream_experimental.get("kv_reprefill_token_history", "")
        ).strip().lower()
        return raw in {"1", "true", "yes", "on"}

    @staticmethod
    def _steadystream_token_history_full_current(group: EngineSessionGroup) -> bool:
        raw = str(
            group.steadystream_experimental.get(
                "kv_reprefill_token_history_full_current", ""
            )
        ).strip().lower()
        return raw in {"1", "true", "yes", "on"}

    def _steadystream_replay_buffer_limit(self, group: EngineSessionGroup) -> int:
        return min(
            max(64, self._steadystream_kv_tail_tokens(group) + 256),
            max(64, self._executor.kv_pool.max_seq_len * 2),
        )

    @staticmethod
    def _steadystream_terminal_drop_mode(group: EngineSessionGroup) -> str:
        raw = str(
            group.steadystream_experimental.get(
                "kv_terminal_drop_mode", "pad_phase"
            )
        ).strip().lower().replace("-", "_")
        if raw in {"eos", "eos_only", "last", "last_token"}:
            return "eos_only"
        if raw in {"silence", "tail_silence"}:
            return "silence"
        return "pad_phase"

    @staticmethod
    def _steadystream_terminal_drop_max_tokens(group: EngineSessionGroup) -> int:
        raw = group.steadystream_experimental.get(
            "kv_terminal_drop_max_tokens", "24"
        )
        try:
            value = int(raw)
        except (TypeError, ValueError):
            value = 24
        return max(1, min(value, 64))

    @staticmethod
    def _steadystream_generation_budget_frames(group: EngineSessionGroup, text_tokens: int) -> int:
        """Reserve decode room for the current segment before keeping history.

        Token-history diagnostics can otherwise spend nearly the whole 512-slot
        window on history and turn a model result into an engineering overflow.
        """
        exp = group.steadystream_experimental
        explicit = str(exp.get("kv_generation_budget_frames", "")).strip()
        if explicit:
            try:
                return max(1, int(explicit))
            except ValueError:
                pass

        def _float_cfg(name: str, default: float) -> float:
            try:
                return float(str(exp.get(name, default)).strip())
            except (TypeError, ValueError):
                return default

        frames_per_token = max(
            0.1,
            _float_cfg("kv_generation_budget_frames_per_text_token", 4.0),
        )
        multiplier = max(
            0.1,
            _float_cfg("kv_generation_budget_multiplier", 1.6),
        )
        min_frames = max(
            1,
            int(_float_cfg("kv_generation_budget_min_frames", 32.0)),
        )
        estimate = math.ceil(max(1, int(text_tokens)) * frames_per_token * multiplier)
        return max(min_frames, int(estimate))

    def _snapshot_talker_kv_tail(
        self,
        slot: SlotKVState,
        *,
        max_tokens: int,
        drop_tail_tokens: int = 0,
    ) -> tuple[Optional[torch.Tensor], int, int]:
        past_len = int(slot.past_len)
        drop_tail_tokens = max(0, int(drop_tail_tokens))
        if drop_tail_tokens > 0:
            past_len = max(0, past_len - drop_tail_tokens)
        if past_len <= 0:
            return None, 0, 0
        keep = min(past_len, int(max_tokens))
        start = past_len - keep
        kv_pool = self._executor.kv_pool
        if kv_pool._preallocate and kv_pool._talker_kv_pool is not None:
            kv = kv_pool._talker_kv_pool[
                slot.slot_id : slot.slot_id + 1,
                :, :, start:past_len, :,
            ].clone()
        elif slot.talker_kv is not None:
            kv = slot.talker_kv[:, :, :, start:past_len, :].clone()
        else:
            return None, 0, 0
        logical_past_len = int(slot.position_offset) + past_len
        return kv.contiguous(), keep, logical_past_len

    @staticmethod
    def _steadystream_terminal_drop_tokens(
        slot: SlotKVState,
        *,
        mode: str = "pad_phase",
        max_tokens: int = 24,
    ) -> int:
        """Drop terminal Talker inputs before carrying KV to a new segment."""
        mode = (mode or "pad_phase").strip().lower().replace("-", "_")
        max_tokens = max(1, int(max_tokens))
        if mode in {"eos", "eos_only", "last", "last_token"}:
            return 1
        if mode in {"silence", "tail_silence"}:
            silence = max(0, int(slot.pad_consecutive_silence))
            return min(max_tokens, max(1, silence + 1))
        if slot.pad_start_frame >= 0 and slot.frame_idx >= slot.pad_start_frame:
            return max(1, int(slot.frame_idx) - int(slot.pad_start_frame) + 1)
        return 1

    def _restore_talker_kv_tail(
        self,
        slot: SlotKVState,
        talker_kv: torch.Tensor,
        compact_len: int,
        logical_past_len: Optional[int] = None,
    ) -> None:
        kv_pool = self._executor.kv_pool
        compact_len = int(compact_len)
        if kv_pool._preallocate and kv_pool._talker_kv_pool is not None:
            kv_pool._talker_kv_pool[
                slot.slot_id, :, :, :compact_len, :
            ] = talker_kv[0, :, :, :compact_len, :].to(
                device=self._embed_device,
                dtype=self._embed_dtype,
            )
        else:
            slot.talker_kv = talker_kv.to(
                device=self._embed_device,
                dtype=self._embed_dtype,
            ).contiguous()
        slot.past_len = compact_len
        if logical_past_len is None:
            logical_past_len = compact_len
        slot.position_offset = max(0, int(logical_past_len) - compact_len)

    def _restore_prefix_plus_talker_kv_tail(
        self,
        slot: SlotKVState,
        cached_prefix: Any,
        talker_kv: torch.Tensor,
        compact_len: int,
        logical_past_len: Optional[int] = None,
    ) -> tuple[int, int]:
        """Restore CustomVoice sink prefix followed by a bounded Talker KV tail."""
        self._restore_prefix_cache(slot, cached_prefix)
        prefix_len = int(cached_prefix.prefix_len)
        max_seq = int(self._executor.kv_pool.max_seq_len)
        tail_len = min(
            int(compact_len),
            int(talker_kv.shape[3]),
            max(0, max_seq - prefix_len),
        )
        if tail_len <= 0:
            return prefix_len, 0

        tail = talker_kv[:, :, :, -tail_len:, :].to(
            device=self._embed_device,
            dtype=self._embed_dtype,
        ).contiguous()
        kv_pool = self._executor.kv_pool
        if kv_pool._preallocate and kv_pool._talker_kv_pool is not None:
            kv_pool._talker_kv_pool[
                slot.slot_id, :, :, prefix_len:prefix_len + tail_len, :
            ] = tail[0]
        else:
            if slot.talker_kv is None:
                slot.talker_kv = cached_prefix.talker_kv.to(
                    device=self._embed_device,
                    dtype=self._embed_dtype,
                ).contiguous()
            slot.talker_kv = torch.cat(
                [slot.talker_kv[:, :, :, :prefix_len, :], tail],
                dim=3,
            ).contiguous()
        slot.past_len = prefix_len + tail_len
        if logical_past_len is None:
            logical_past_len = slot.past_len
        slot.position_offset = max(0, int(logical_past_len) - slot.past_len)
        return prefix_len, tail_len

    def _store_steadystream_carry(
        self,
        group: EngineSessionGroup,
        seg: EngineSegment,
        *,
        drop_talker_tail_tokens: int = 0,
        terminal_drop_mode: str = "pad_phase",
        terminal_source_frames: int = 0,
    ) -> None:
        if not group.steadystream_variant or seg.slot is None:
            return
        slot = seg.slot
        carry: dict[str, Any] = {"from_segment_idx": int(seg.segment_idx)}

        if self._steadystream_uses_kv(group):
            if self._steadystream_uses_reprefill_history(group):
                replay_items = list(slot.steadystream_replay_embeds or [])
                if drop_talker_tail_tokens > 0:
                    replay_items = replay_items[:-int(drop_talker_tail_tokens)]
                keep = min(
                    len(replay_items),
                    self._steadystream_kv_tail_tokens(group),
                )
                if keep > 0:
                    replay = torch.cat(replay_items[-keep:], dim=1).detach().clone()
                    carry["replay_embeds"] = replay.contiguous()
                    carry["replay_len"] = int(keep)
                    carry["replay_dropped_tail_tokens"] = int(drop_talker_tail_tokens)

            if self._steadystream_uses_token_history(group):
                codec_items = list(slot.steadystream_full_codecs or [])
                if drop_talker_tail_tokens > 0:
                    codec_items = codec_items[:-int(drop_talker_tail_tokens)]
                keep = min(
                    len(codec_items),
                    self._steadystream_kv_tail_tokens(group),
                )
                if keep > 0:
                    full_codes = torch.cat(codec_items[-keep:], dim=0)
                    carry["history_text_token_ids"] = list(seg.pending_token_ids)
                    carry["history_text_include_eos"] = bool(seg.input_complete)
                    carry["history_full_codes"] = full_codes.detach().clone().cpu()
                    carry["history_total_codec_frames"] = len(codec_items)
                    carry["history_kept_codec_frames"] = int(keep)
                    carry["history_dropped_tail_tokens"] = int(drop_talker_tail_tokens)

            talker_kv, compact_len, logical_past_len = self._snapshot_talker_kv_tail(
                slot,
                max_tokens=self._steadystream_kv_tail_tokens(group),
                drop_tail_tokens=drop_talker_tail_tokens,
            )
            if talker_kv is not None and compact_len > 0:
                carry["talker_kv"] = talker_kv
                carry["talker_past_len"] = compact_len
                carry["talker_logical_past_len"] = logical_past_len
                carry["talker_dropped_last_token"] = drop_talker_tail_tokens > 0
                carry["talker_dropped_tail_tokens"] = int(drop_talker_tail_tokens)
                carry["talker_terminal_drop_mode"] = str(terminal_drop_mode)
                carry["talker_tail_source_frames"] = int(terminal_source_frames)
                carry["talker_dropped_tail_ratio"] = (
                    round(
                        float(drop_talker_tail_tokens)
                        / max(1.0, float(terminal_source_frames)),
                        6,
                    )
                    if int(terminal_source_frames) > 0
                    else 0.0
                )
                carry["talker_position_offset"] = max(
                    0,
                    int(logical_past_len) - int(compact_len),
                )
                if (
                    self._steadystream_inherits_token_counts(group)
                    and slot.token_counts is not None
                ):
                    carry["token_counts"] = slot.token_counts.clone()

        if self._steadystream_uses_acoustic(group):
            if slot.c2w_kv is not None:
                carry["c2w_kv"] = slot.c2w_kv.clone().contiguous()
            if slot.c2w_conv_states:
                carry["c2w_conv_states"] = [
                    t.clone().contiguous() for t in slot.c2w_conv_states
                ]
            if slot.c2w_transconv_states:
                carry["c2w_transconv_states"] = [
                    t.clone().contiguous() for t in slot.c2w_transconv_states
                ]
            carry["c2w_frame_idx"] = int(slot.frame_idx)

        if len(carry) > 1:
            group.steadystream_carry = carry
            logger.info(
                "Stored SteadyStream carry: session=%s variant=%s seg=%d keys=%s",
                group.session_id,
                group.steadystream_variant,
                seg.segment_idx,
                sorted(k for k in carry.keys() if k != "from_segment_idx"),
            )

    def _restore_steadystream_c2w_carry(
        self,
        group: EngineSessionGroup,
        slot: SlotKVState,
        metrics: dict,
    ) -> bool:
        carry = group.steadystream_carry or {}
        c2w_kv = carry.get("c2w_kv")
        conv_states = carry.get("c2w_conv_states")
        transconv_states = carry.get("c2w_transconv_states")
        if c2w_kv is None or conv_states is None or transconv_states is None:
            return False
        warmed = self._executor.apply_c2w_warm_state(
            slot,
            c2w_kv,
            conv_states,
            transconv_states,
            int(carry.get("c2w_frame_idx") or 0),
        )
        if warmed:
            metrics["steadystream_acoustic_tail"] = "true"
            metrics["steadystream_carry_from_segment"] = str(
                carry.get("from_segment_idx", "")
            )
        return warmed

    def _try_prefill_from_steadystream_carry(
        self,
        group: EngineSessionGroup,
        seg: EngineSegment,
        slot: SlotKVState,
        task_type: TaskType,
        metrics: dict,
        cached_prefix: Any = None,
    ) -> Optional[tuple[Optional[bytes], bool]]:
        if not self._steadystream_uses_kv(group):
            return None
        if task_type == TaskType.VOICE_CLONE_ICL:
            return None
        carry = group.steadystream_carry or {}
        if self._steadystream_uses_token_history(group):
            reprefill = self._try_reprefill_from_steadystream_token_history(
                group,
                seg,
                slot,
                task_type,
                metrics,
            )
            if reprefill is not None:
                return reprefill
        if self._steadystream_uses_reprefill_history(group):
            reprefill = self._try_reprefill_from_steadystream_history(
                group,
                seg,
                slot,
                task_type,
                metrics,
            )
            if reprefill is not None:
                return reprefill
        talker_kv = carry.get("talker_kv")
        compact_len = int(carry.get("talker_past_len") or 0)
        logical_past_len = int(
            carry.get("talker_logical_past_len")
            or (compact_len + int(carry.get("talker_position_offset") or 0))
        )
        if talker_kv is None or compact_len <= 0 or not seg.pending_token_ids:
            return None

        if (
            cached_prefix is not None
            and self._steadystream_prepends_prefix_to_kv_tail(group)
        ):
            prefix_len, tail_len = self._restore_prefix_plus_talker_kv_tail(
                slot,
                cached_prefix,
                talker_kv,
                compact_len,
                logical_past_len,
            )
            metrics["steadystream_kv_prefix"] = "cached"
            metrics["steadystream_kv_prefix_len"] = str(prefix_len)
            metrics["steadystream_kv_tail_effective_len"] = str(tail_len)
        else:
            self._restore_talker_kv_tail(slot, talker_kv, compact_len, logical_past_len)
            metrics["steadystream_kv_prefix"] = "none"
            metrics["steadystream_kv_tail_effective_len"] = str(compact_len)
        req_embeds, trailing = self._prefill_builder.build_suffix_from_ids(
            seg.pending_token_ids,
            include_eos=seg.input_complete,
        )
        self._prime_decode_after_prefix_prefill(
            slot,
            req_embeds,
            trailing,
            source=f"steadystream_{group.steadystream_variant}",
        )
        if carry.get("token_counts") is not None:
            slot.token_counts = carry["token_counts"].clone().to(
                device=self._embed_device,
                dtype=torch.int64,
            )
            metrics["steadystream_token_counts"] = "inherited"
        else:
            metrics["steadystream_token_counts"] = "reset"
        if self._steadystream_uses_acoustic(group):
            self._restore_steadystream_c2w_carry(group, slot, metrics)
        metrics["steadystream_variant"] = group.steadystream_variant
        metrics["steadystream_kv_tail"] = "true"
        metrics["steadystream_kv_past_len"] = str(compact_len)
        metrics["steadystream_kv_logical_past_len"] = str(logical_past_len)
        metrics["steadystream_kv_position_offset"] = str(slot.position_offset)
        metrics["steadystream_kv_dropped_last_token"] = str(
            bool(carry.get("talker_dropped_last_token"))
        ).lower()
        metrics["steadystream_kv_dropped_tail_tokens"] = str(
            int(carry.get("talker_dropped_tail_tokens") or 0)
        )
        metrics["steadystream_kv_terminal_drop_mode"] = str(
            carry.get("talker_terminal_drop_mode", "")
        )
        metrics["steadystream_kv_tail_source_frames"] = str(
            int(carry.get("talker_tail_source_frames") or 0)
        )
        metrics["steadystream_kv_dropped_tail_ratio"] = str(
            carry.get("talker_dropped_tail_ratio", "")
        )
        metrics["steadystream_carry_from_segment"] = str(
            carry.get("from_segment_idx", "")
        )
        logger.info(
            "Applied SteadyStream KV carry: session=%s variant=%s seg=%d "
            "compact_past=%d logical_past=%d position_offset=%d",
            group.session_id,
            group.steadystream_variant,
            seg.segment_idx,
            compact_len,
            logical_past_len,
            slot.position_offset,
        )
        return None, False

    def _steadystream_codec_sum_from_full_codes(
        self,
        full_codes: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if self._prefill_builder is None:
            return None
        w = self._prefill_builder.w
        table = getattr(w, "codec_embeddings_3d", None)
        if table is None:
            return None
        codes = full_codes.to(device=table.device, dtype=torch.int64)
        if codes.dim() == 3 and codes.shape[0] == 1:
            codes = codes[0]
        if codes.dim() != 2 or codes.shape[1] != 16 or codes.shape[0] <= 0:
            return None
        group_ids = torch.arange(
            16,
            device=table.device,
            dtype=torch.int64,
        ).reshape(1, 16).expand(codes.shape[0], 16)
        with torch.no_grad():
            codec_sum = table[group_ids, codes, :].sum(dim=1).unsqueeze(0)
        return codec_sum.to(device=self._embed_device, dtype=self._embed_dtype)

    def _steadystream_text_block_embed(
        self,
        token_ids: list[int],
        *,
        include_eos: bool,
    ) -> Optional[torch.Tensor]:
        if self._prefill_builder is None:
            return None
        w = self._prefill_builder.w
        ids = list(token_ids)
        if include_eos:
            ids.append(int(w.tts_eos_token_id))
        if not ids:
            return torch.zeros(
                1,
                0,
                self._hidden_size,
                device=self._embed_device,
                dtype=self._embed_dtype,
            )
        ids_tensor = torch.tensor([ids], device=w.device, dtype=torch.int64)
        codec_pad_ids = torch.full(
            (1, len(ids)),
            int(w.codec_pad_id),
            device=w.device,
            dtype=torch.int64,
        )
        with torch.no_grad():
            text_embed = w.text_embed(ids_tensor)
            codec_pad = w.codec_embed(codec_pad_ids)
        return (text_embed + codec_pad).to(
            device=self._embed_device,
            dtype=self._embed_dtype,
        )

    def _build_steadystream_token_history_prefill(
        self,
        prefix_embeds: torch.Tensor,
        *,
        generation_budget_frames: int,
        history_text_token_ids: list[int],
        history_text_include_eos: bool,
        history_full_codes: torch.Tensor,
        current_text_token_ids: list[int] | None = None,
        current_text_include_eos: bool = False,
    ) -> Optional[tuple[torch.Tensor, dict[str, int]]]:
        if self._prefill_builder is None:
            return None
        w = self._prefill_builder.w
        history_text = self._steadystream_text_block_embed(
            history_text_token_ids,
            include_eos=history_text_include_eos,
        )
        if history_text is None:
            return None
        current_text = None
        if current_text_token_ids is not None:
            current_text = self._steadystream_text_block_embed(
                current_text_token_ids,
                include_eos=current_text_include_eos,
            )
            if current_text is None:
                return None

        prefix_embeds = prefix_embeds.to(
            device=self._embed_device,
            dtype=self._embed_dtype,
        )
        max_seq = int(self._executor.kv_pool.max_seq_len)
        max_input_len = int(getattr(self._executor, "_max_input_len", 0) or max_seq)
        fixed_prefill_len = (
            int(prefix_embeds.shape[1])
            + int(history_text.shape[1])
            + 1  # history codec BOS
            + 1  # history boundary codec EOS
        )
        if current_text is not None:
            fixed_prefill_len += int(current_text.shape[1]) + 1  # current codec BOS
        generation_budget_frames = max(1, int(generation_budget_frames))
        max_code_frames = min(
            max_input_len - fixed_prefill_len,
            max_seq - fixed_prefill_len - generation_budget_frames,
        )
        if max_code_frames <= 0:
            return None
        total_code_frames = int(history_full_codes.shape[0])
        if total_code_frames <= 0:
            return None
        trimmed_for_generation_budget = max(0, total_code_frames - max_code_frames)
        if total_code_frames > max_code_frames:
            history_full_codes = history_full_codes[-max_code_frames:, :]
        codec_sum = self._steadystream_codec_sum_from_full_codes(
            history_full_codes,
        )
        if codec_sum is None:
            return None

        pad = w.tts_pad_embed.to(device=self._embed_device, dtype=self._embed_dtype)
        codec_bos = w.codec_embed(
            torch.tensor([[w.codec_bos_id]], device=w.device, dtype=torch.int64)
        ).to(device=self._embed_device, dtype=self._embed_dtype)
        codec_eos_id = int(
            getattr(
                w,
                "codec_eos_id",
                getattr(w, "codec_eos_token_id", 0),
            )
        )
        codec_eos = w.codec_embed(
            torch.tensor([[codec_eos_id]], device=w.device, dtype=torch.int64)
        ).to(device=self._embed_device, dtype=self._embed_dtype)

        history_codec_bos = pad + codec_bos
        history_codes = pad.expand(1, codec_sum.shape[1], self._hidden_size) + codec_sum
        history_boundary = pad + codec_eos
        parts = [
            prefix_embeds,
            history_text,
            history_codec_bos,
            history_codes,
            history_boundary,
        ]
        if current_text is not None:
            current_codec_bos = pad + codec_bos
            parts.extend([current_text, current_codec_bos])
        prefill = torch.cat(parts, dim=1).contiguous()
        info = {
            "prefix_len": int(prefix_embeds.shape[1]),
            "history_text_tokens": len(history_text_token_ids),
            "history_code_frames": int(history_full_codes.shape[0]),
            "history_code_frames_total": total_code_frames,
            "history_code_frames_trimmed_for_generation_budget": trimmed_for_generation_budget,
            "current_text_tokens": len(current_text_token_ids or []),
            "current_text_full_prefill": 1 if current_text is not None else 0,
            "prefill_len": int(prefill.shape[1]),
            "max_input_len": max_input_len,
            "max_seq_len": max_seq,
            "generation_budget_frames": generation_budget_frames,
        }
        return prefill, info

    def _try_reprefill_from_steadystream_token_history(
        self,
        group: EngineSessionGroup,
        seg: EngineSegment,
        slot: SlotKVState,
        task_type: TaskType,
        metrics: dict,
    ) -> Optional[tuple[Optional[bytes], bool]]:
        carry = group.steadystream_carry or {}
        history_full_codes = carry.get("history_full_codes")
        history_text_token_ids = carry.get("history_text_token_ids")
        if (
            history_full_codes is None
            or not isinstance(history_text_token_ids, list)
            or not seg.pending_token_ids
        ):
            return None

        req_cfg = group.request.session_config

        def _cfg_attr(name: str, default=None):
            return getattr(req_cfg, name, default) if req_cfg is not None else default

        def _cfg_token_ids(name: str):
            spec = _cfg_attr(name)
            return list(spec.token_ids) if spec is not None else None

        plan = self._prefill_builder.build_plan_from_ids(
            task_type=task_type,
            token_ids=seg.pending_token_ids,
            language=_cfg_attr("language", "auto"),
            speaker=_cfg_attr("speaker", group.request.speaker_key),
            instruct=_cfg_attr("instruct"),
            instruct_token_ids=_cfg_token_ids("instruct_spec"),
            spk_embedding=_cfg_attr("spk_embedding"),
            ref_text=_cfg_attr("ref_text"),
            ref_text_token_ids=_cfg_token_ids("ref_text_spec"),
            ref_codec_sum_vec=_cfg_attr("ref_codec_sum_vec"),
            ref_audio_sha256=_cfg_attr("ref_audio_sha256"),
            ref_feature_cache_key=_cfg_attr("ref_feature_cache_key"),
            include_eos=seg.input_complete,
        )
        if plan.cacheable_prefix_embeds is None:
            return None

        full_current = (
            self._steadystream_token_history_full_current(group)
            and bool(seg.input_complete)
        )
        built = self._build_steadystream_token_history_prefill(
            plan.cacheable_prefix_embeds,
            generation_budget_frames=self._steadystream_generation_budget_frames(
                group,
                len(seg.pending_token_ids),
            ),
            history_text_token_ids=[int(x) for x in history_text_token_ids],
            history_text_include_eos=bool(carry.get("history_text_include_eos", True)),
            history_full_codes=history_full_codes,
            current_text_token_ids=(
                [int(x) for x in seg.pending_token_ids] if full_current else None
            ),
            current_text_include_eos=bool(seg.input_complete),
        )
        if built is None:
            return None
        replay_prefill, info = built
        if not full_current and plan.request_prefill_embeds is None:
            return None

        prefill_audio: Optional[bytes] = None
        prefill_eos = False
        source = f"steadystream_token_history_{group.steadystream_variant}"
        if full_current:
            source = f"steadystream_token_history_full_current_{group.steadystream_variant}"
            if self._steadystream_uses_acoustic(group):
                self._restore_steadystream_c2w_carry(group, slot, metrics)
            prefill_audio, prefill_eos = self._executor.prefill(slot, replay_prefill)
            slot.prefill_source = source
            slot.trailing = []
            slot.text_idx = 0
            slot.steadystream_replay_embeds = []
            slot.steadystream_full_codecs = []
        else:
            self._executor.prefill_prefix_only(slot, replay_prefill)
            self._prime_decode_after_prefix_prefill(
                slot,
                plan.request_prefill_embeds,
                plan.trailing,
                source=source,
            )
            if self._steadystream_uses_acoustic(group):
                self._restore_steadystream_c2w_carry(group, slot, metrics)
        metrics["steadystream_variant"] = group.steadystream_variant
        metrics["steadystream_kv_tail"] = (
            "reprefill_token_history_full_current"
            if full_current
            else "reprefill_token_history"
        )
        metrics["steadystream_kv_prefix"] = "recomputed"
        metrics["steadystream_kv_prefix_len"] = str(info["prefix_len"])
        metrics["steadystream_token_history_prefill_len"] = str(info["prefill_len"])
        metrics["steadystream_token_history_text_tokens"] = str(
            info["history_text_tokens"]
        )
        metrics["steadystream_token_history_code_frames"] = str(
            info["history_code_frames"]
        )
        metrics["steadystream_token_history_code_frames_total"] = str(
            info["history_code_frames_total"]
        )
        metrics["steadystream_token_history_code_frames_trimmed_for_generation_budget"] = str(
            info["history_code_frames_trimmed_for_generation_budget"]
        )
        metrics["steadystream_token_history_current_text_tokens"] = str(
            len(seg.pending_token_ids)
        )
        metrics["steadystream_token_history_current_trailing"] = str(
            0 if full_current else len(plan.trailing)
        )
        metrics["steadystream_token_history_full_current"] = str(
            bool(info["current_text_full_prefill"])
        )
        metrics["steadystream_token_history_max_input_len"] = str(
            info["max_input_len"]
        )
        metrics["steadystream_token_history_max_seq_len"] = str(
            info["max_seq_len"]
        )
        metrics["steadystream_token_history_generation_budget_frames"] = str(
            info["generation_budget_frames"]
        )
        metrics["steadystream_token_history_dropped_tail_tokens"] = str(
            int(carry.get("history_dropped_tail_tokens") or 0)
        )
        metrics["steadystream_kv_terminal_drop_mode"] = str(
            carry.get("talker_terminal_drop_mode", "")
        )
        metrics["steadystream_kv_tail_source_frames"] = str(
            int(carry.get("talker_tail_source_frames") or 0)
        )
        metrics["steadystream_kv_dropped_tail_ratio"] = str(
            carry.get("talker_dropped_tail_ratio", "")
        )
        metrics["steadystream_token_counts"] = "reset"
        metrics["steadystream_carry_from_segment"] = str(
            carry.get("from_segment_idx", "")
        )
        logger.info(
            "Applied SteadyStream token-history prefill: session=%s variant=%s "
            "seg=%d mode=%s prefix=%d hist_text=%d hist_codes=%d/%d current_text=%d",
            group.session_id,
            group.steadystream_variant,
            seg.segment_idx,
            "full_current" if full_current else "streaming_current",
            info["prefix_len"],
            info["history_text_tokens"],
            info["history_code_frames"],
            info["history_code_frames_total"],
            len(seg.pending_token_ids),
        )
        return prefill_audio, prefill_eos

    def _try_reprefill_from_steadystream_history(
        self,
        group: EngineSessionGroup,
        seg: EngineSegment,
        slot: SlotKVState,
        task_type: TaskType,
        metrics: dict,
    ) -> Optional[tuple[Optional[bytes], bool]]:
        carry = group.steadystream_carry or {}
        replay = carry.get("replay_embeds")
        if replay is None or not seg.pending_token_ids:
            return None

        req_cfg = group.request.session_config

        def _cfg_attr(name: str, default=None):
            return getattr(req_cfg, name, default) if req_cfg is not None else default

        def _cfg_token_ids(name: str):
            spec = _cfg_attr(name)
            return list(spec.token_ids) if spec is not None else None

        plan = self._prefill_builder.build_plan_from_ids(
            task_type=task_type,
            token_ids=seg.pending_token_ids,
            language=_cfg_attr("language", "auto"),
            speaker=_cfg_attr("speaker", group.request.speaker_key),
            instruct=_cfg_attr("instruct"),
            instruct_token_ids=_cfg_token_ids("instruct_spec"),
            spk_embedding=_cfg_attr("spk_embedding"),
            ref_text=_cfg_attr("ref_text"),
            ref_text_token_ids=_cfg_token_ids("ref_text_spec"),
            ref_codec_sum_vec=_cfg_attr("ref_codec_sum_vec"),
            ref_audio_sha256=_cfg_attr("ref_audio_sha256"),
            ref_feature_cache_key=_cfg_attr("ref_feature_cache_key"),
            include_eos=seg.input_complete,
        )
        if plan.cacheable_prefix_embeds is None or plan.request_prefill_embeds is None:
            return None

        prefix_embeds = plan.cacheable_prefix_embeds.to(
            device=self._embed_device,
            dtype=self._embed_dtype,
        )
        request_embeds = plan.request_prefill_embeds.to(
            device=self._embed_device,
            dtype=self._embed_dtype,
        )
        replay = replay.to(device=self._embed_device, dtype=self._embed_dtype)
        prefix_len = int(prefix_embeds.shape[1])
        max_history = max(
            0,
            int(self._executor.kv_pool.max_seq_len) - prefix_len - 1,
        )
        if max_history <= 0:
            return None
        original_replay_len = int(replay.shape[1])
        if original_replay_len > max_history:
            replay = replay[:, -max_history:, :]

        replay_prefill = torch.cat([prefix_embeds, replay], dim=1).contiguous()
        self._executor.prefill_prefix_only(slot, replay_prefill)
        self._prime_decode_after_prefix_prefill(
            slot,
            request_embeds,
            plan.trailing,
            source=f"steadystream_reprefill_{group.steadystream_variant}",
        )
        if self._steadystream_uses_acoustic(group):
            self._restore_steadystream_c2w_carry(group, slot, metrics)
        metrics["steadystream_variant"] = group.steadystream_variant
        metrics["steadystream_kv_tail"] = "reprefill_history"
        metrics["steadystream_kv_prefix"] = "recomputed"
        metrics["steadystream_kv_prefix_len"] = str(prefix_len)
        metrics["steadystream_replay_len"] = str(int(replay.shape[1]))
        metrics["steadystream_replay_original_len"] = str(original_replay_len)
        metrics["steadystream_replay_dropped_tail_tokens"] = str(
            int(carry.get("replay_dropped_tail_tokens") or 0)
        )
        metrics["steadystream_kv_terminal_drop_mode"] = str(
            carry.get("talker_terminal_drop_mode", "")
        )
        metrics["steadystream_kv_tail_source_frames"] = str(
            int(carry.get("talker_tail_source_frames") or 0)
        )
        metrics["steadystream_kv_dropped_tail_ratio"] = str(
            carry.get("talker_dropped_tail_ratio", "")
        )
        metrics["steadystream_token_counts"] = "reset"
        metrics["steadystream_carry_from_segment"] = str(
            carry.get("from_segment_idx", "")
        )
        logger.info(
            "Applied SteadyStream replay prefill: session=%s variant=%s seg=%d "
            "prefix=%d replay=%d",
            group.session_id,
            group.steadystream_variant,
            seg.segment_idx,
            prefix_len,
            int(replay.shape[1]),
        )
        return None, False

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
                        slot.next_embed = self._coerce_embed_tensor(
                            slot.trailing[slot.text_idx],
                            dtype=torch.float32,
                        )
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
            self._remove_session(seg.session_id)

    # ------------------------------------------------------------------
    # Result processing (runs while GPU does next step)
    # ------------------------------------------------------------------

    def _process_step_output(self, output: StepOutput) -> None:
        kv_pool = self._executor.kv_pool
        use_pool = kv_pool is not None and kv_pool._preallocate

        # Batch-level KV scatter to pool (single operation, avoids per-slot split)
        if use_pool and output.batch_talker_kv is not None:
            slot_ids = [s.slot_id for s in output.slots]
            kv_pool.scatter_talker_kv_delta(
                slot_ids,
                output.batch_talker_kv,
                output.original_past_lens,
            )

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
                if slot.c2w_kv is None:
                    slot.c2w_kv = kv.clone()
                else:
                    slot.c2w_kv = torch.cat([slot.c2w_kv, kv], dim=3)
                    if slot.c2w_kv.shape[3] > c2w_max_past:
                        slot.c2w_kv = slot.c2w_kv[:, :, :, -c2w_max_past:, :].contiguous()
                    else:
                        slot.c2w_kv = slot.c2w_kv.contiguous()
            if not use_pool:
                if output.batch_talker_kv is not None:
                    kv = output.batch_talker_kv[i:i+1]
                    if slot.talker_kv is None:
                        slot.talker_kv = kv.clone()
                    else:
                        slot.talker_kv = torch.cat([slot.talker_kv, kv], dim=3).contiguous()

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
            if (
                output.step_input_embeds is not None
                and self._steadystream_uses_kv(group)
            ):
                slot.steadystream_replay_embeds.append(
                    output.step_input_embeds[i:i + 1].detach().clone().contiguous()
                )
                replay_limit = self._steadystream_replay_buffer_limit(group)
                if len(slot.steadystream_replay_embeds) > replay_limit:
                    del slot.steadystream_replay_embeds[:-replay_limit]
            if (
                output.full_codec is not None
                and self._steadystream_uses_token_history(group)
            ):
                slot.steadystream_full_codecs.append(
                    output.full_codec[i:i + 1].detach().cpu().clone().contiguous()
                )
                codec_limit = self._steadystream_replay_buffer_limit(group)
                if len(slot.steadystream_full_codecs) > codec_limit:
                    del slot.steadystream_full_codecs[:-codec_limit]
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
                        if self._is_pad_silence(audio):
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

    def _is_pad_silence(self, audio: bytes) -> bool:
        """Detect near-silent pad-phase audio frames.

        The previous peak-only `1e-4` threshold missed pathological pad loops
        that produce almost-flat chunks with tiny residual noise around
        `3e-4 ~ 5e-4`. Use both peak and mean absolute amplitude so we catch
        repeated near-silence without clipping normal quiet speech too
        aggressively.
        """
        audio_np = np.frombuffer(audio, dtype=np.float32)
        if audio_np.size == 0:
            return False
        abs_audio = np.abs(audio_np)
        peak = float(abs_audio.max(initial=0.0))
        mean_abs = float(abs_audio.mean())
        return (
            peak <= self._pad_silence_peak_threshold
            and mean_abs <= self._pad_silence_mean_abs_threshold
        )

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

        terminal_drop_mode = self._steadystream_terminal_drop_mode(group)
        terminal_drop_max = self._steadystream_terminal_drop_max_tokens(group)
        drop_talker_tail_tokens = (
            0
            if overflow or seg.slot is None
            else self._steadystream_terminal_drop_tokens(
                seg.slot,
                mode=terminal_drop_mode,
                max_tokens=terminal_drop_max,
            )
        )
        metrics["steadystream_terminal_drop_mode"] = terminal_drop_mode
        metrics["steadystream_talker_dropped_tail_tokens"] = drop_talker_tail_tokens
        metrics["steadystream_talker_tail_source_frames"] = audio_steps
        metrics["steadystream_talker_dropped_tail_ratio"] = (
            round(float(drop_talker_tail_tokens) / max(1.0, float(audio_steps)), 6)
            if audio_steps > 0
            else 0.0
        )

        self._store_steadystream_carry(
            group,
            seg,
            drop_talker_tail_tokens=drop_talker_tail_tokens,
            terminal_drop_mode=terminal_drop_mode,
            terminal_source_frames=audio_steps,
        )
        seg.state = "done"
        self._release_segment_slot(seg)

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
            logger.debug(
                "Session %s _check_session_done: input_complete_all=False, skipping",
                group.session_id,
            )
            return

        if group.overflow_token_ids:
            logger.warning(
                "Session %s has %d overflow tokens pending — waiting for new segment",
                group.session_id, len(group.overflow_token_ids),
            )
            return

        all_done = all(s.state == "done" for s in group.segments.values())
        if not all_done:
            seg_states = {idx: s.state for idx, s in group.segments.items()}
            logger.debug(
                "Session %s _check_session_done: not all done, seg_states=%s",
                group.session_id, seg_states,
            )
            return

        logger.info(
            "Session %s _check_session_done: ALL DONE, sending SESSION_DONE",
            group.session_id,
        )
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
            self._release_segment_slot(seg)

    def _release_segment_slot(self, seg: EngineSegment) -> None:
        slot = seg.slot
        if slot is None:
            return
        slot_id = slot.slot_id
        kv_pool = self._executor.kv_pool
        if kv_pool is not None:
            kv_pool.release(slot_id)
        self._seg_by_slot.pop(slot_id, None)
        seg.slot = None

    def _cleanup_failed_prefills(self) -> None:
        failed_sessions: list[str] = []
        for group in list(self._groups.values()):
            for seg in group.segments.values():
                if seg.state == "pending_prefill" and seg.slot is not None:
                    failed_sessions.append(group.session_id)
                    self._send_result(group, EngineResult(
                        type=ResultType.ERROR,
                        session_id=seg.session_id,
                        segment_idx=seg.segment_idx,
                        error_msg="Prefill failed unexpectedly",
                    ))
                    logger.warning(
                        "Cleaning failed prefill session %s seg=%d slot=%d",
                        seg.session_id,
                        seg.segment_idx,
                        seg.slot.slot_id,
                    )
                    break
        for session_id in failed_sessions:
            self._remove_session(session_id)

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
        embed = self._coerce_embed_tensor(embed)
        for i in range(embed.shape[1]):
            slot.trailing.append(embed[:, i : i + 1, :].clone())
        logger.debug(
            "Appended %d trailing tokens (total=%d, text_idx=%d)",
            len(token_ids), len(slot.trailing), slot.text_idx,
        )

    def _append_eos_trailing(self, seg: EngineSegment) -> None:
        """Append tts_eos_embed to trailing when SEGMENT_TOKENS_DONE arrives post-prefill."""
        w = self._prefill_builder.w
        seg.slot.trailing.append(self._coerce_embed_tensor(w.tts_eos_embed).clone())
        seg.eos_trailing_added = True
        logger.debug(
            "Appended EOS trailing for seg=%d (total=%d)",
            seg.segment_idx, len(seg.slot.trailing),
        )

    def _coerce_embed_tensor(
        self,
        tensor: torch.Tensor,
        *,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        target_dtype = dtype
        if target_dtype is None and tensor.is_floating_point():
            target_dtype = self._embed_dtype
        return tensor.to(
            device=self._embed_device,
            dtype=target_dtype if target_dtype is not None else tensor.dtype,
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

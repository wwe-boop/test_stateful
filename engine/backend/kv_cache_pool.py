"""KV cache pool: pre-allocated GPU memory for concurrent sessions.

Instead of dynamically allocating KV tensors per session (which causes
fragmentation and OOM under 64-session load), we pre-allocate a fixed
pool and assign slots.

Memory layout per slot (packed tensor format):
    talker_kv:  [1, num_layers*2, kv_heads, cur_len, head_dim]  (single packed tensor)
    c2w_kv:     [1, n_c2w_layers*2, c2w_kv_heads, cur_len, c2w_head_dim]
    c2w_conv:   17 tensors with heterogeneous shapes (static except batch)
    c2w_transconv: 4 tensors with heterogeneous shapes (static except batch)

Total Talker KV per slot (example, 1.7B model):
    28 layers × 2 × 8 heads × 64 dim × 2048 max_seq × 2 bytes (bf16)
    = 28 × 2 × 8 × 64 × 2048 × 2 = ~115 MB/slot
    64 slots = ~7.2 GB (fits on 24GB GPU with model weights)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Model architecture parameters for KV cache sizing."""
    # Talker
    num_layers: int = 28
    kv_heads: int = 8
    head_dim: int = 64
    max_seq_len: int = 2048
    hidden_size: int = 1536
    dtype: torch.dtype = torch.bfloat16
    codec_vocab_size: int = 2176

    # Code2Wav
    n_c2w_layers: int = 8
    c2w_kv_heads: int = 16
    c2w_head_dim: int = 64
    c2w_sliding_window: int = 72
    n_c2w_conv_states: int = 17
    n_c2w_transconv_states: int = 4

    @property
    def num_c2w_states(self) -> int:
        return 2 * self.n_c2w_layers + self.n_c2w_conv_states + self.n_c2w_transconv_states


@dataclass
class SlotKVState:
    """GPU-side state for one session slot.

    Uses packed KV tensors: single [1, L*2, H, S, D] per cache type
    instead of lists of per-layer tensors.  This reduces TRT I/O binding
    count from ~201 to ~61.
    """
    slot_id: int
    session_id: Optional[str] = None
    is_free: bool = True

    # Talker KV: [1, num_layers*2, kv_heads, cur_len, head_dim]
    talker_kv: Optional[torch.Tensor] = None
    past_len: int = 0

    # Code2Wav KV: [1, n_c2w*2, c2w_kv_heads, cur_len, c2w_head_dim]
    c2w_kv: Optional[torch.Tensor] = None
    # Conv/transconv states: heterogeneous shapes, kept as lists
    c2w_conv_states: Optional[list[torch.Tensor]] = None
    c2w_transconv_states: Optional[list[torch.Tensor]] = None
    frame_idx: int = 0

    # Decode tracking
    next_embed: Optional[torch.Tensor] = None
    token_counts: Optional[torch.Tensor] = None

    # Trailing text embeddings for streaming decode
    trailing: list = field(default_factory=list)
    text_idx: int = 0

    @property
    def has_c2w_states(self) -> bool:
        return self.c2w_kv is not None


class KVCachePool:
    """Manages a fixed pool of KV cache slots on GPU.

    Slots are lazily initialized: KV tensors are only allocated when
    a session first does prefill.  On release, tensors are zeroed
    (not freed) to avoid reallocation.

    Pre-allocated pool mode (when enabled):
        A single contiguous tensor per cache type is pre-allocated for all
        slots.  Slots index into this tensor by slot_id, eliminating
        per-step pad+cat overhead during batch assembly.
    """

    def __init__(
        self,
        max_slots: int,
        config: ModelConfig,
        device: torch.device,
        *,
        preallocate: bool = True,
    ):
        self._max_slots = max_slots
        self._config = config
        self._device = device
        self._preallocate = preallocate

        self._slots: list[SlotKVState] = [
            SlotKVState(slot_id=i) for i in range(max_slots)
        ]
        self._free_slots: list[int] = list(range(max_slots))

        self._talker_kv_pool: Optional[torch.Tensor] = None
        self._c2w_kv_pool: Optional[torch.Tensor] = None

        if preallocate:
            self._init_pool_tensors()

        logger.info(
            "KV pool initialized: %d slots, ~%.1f MB/slot (talker KV), "
            "preallocated=%s, device=%s",
            max_slots, self._estimate_slot_mb(), preallocate, device,
        )

    def _init_pool_tensors(self) -> None:
        c = self._config
        self._talker_kv_pool = torch.zeros(
            self._max_slots, c.num_layers * 2, c.kv_heads,
            c.max_seq_len, c.head_dim,
            device=self._device, dtype=c.dtype,
        )
        self._c2w_kv_pool = torch.zeros(
            self._max_slots, c.n_c2w_layers * 2, c.c2w_kv_heads,
            c.c2w_sliding_window, c.c2w_head_dim,
            device=self._device, dtype=c.dtype,
        )

    def _estimate_slot_mb(self) -> float:
        c = self._config
        bytes_per_elem = 2 if c.dtype == torch.bfloat16 else 4
        kv_bytes = (
            c.num_layers * 2 * c.kv_heads * c.head_dim
            * c.max_seq_len * bytes_per_elem
        )
        return kv_bytes / (1024 * 1024)

    @property
    def free_count(self) -> int:
        return len(self._free_slots)

    @property
    def used_count(self) -> int:
        return self._max_slots - len(self._free_slots)

    def allocate(self, session_id: str) -> Optional[SlotKVState]:
        """Allocate a slot for a new session.  Returns None if pool is full."""
        if not self._free_slots:
            return None
        slot_id = self._free_slots.pop()
        slot = self._slots[slot_id]
        slot.session_id = session_id
        slot.is_free = False
        slot.past_len = 0
        slot.frame_idx = 0
        slot.text_idx = 0
        slot.trailing = []
        slot.next_embed = None
        if self._preallocate and self._talker_kv_pool is not None:
            self._talker_kv_pool[slot_id].zero_()
            self._c2w_kv_pool[slot_id].zero_()
        logger.debug("Allocated slot %d for session %s", slot_id, session_id)
        return slot

    def release(self, slot_id: int) -> None:
        """Return a slot to the pool."""
        slot = self._slots[slot_id]
        slot.session_id = None
        slot.is_free = True
        slot.past_len = 0
        slot.frame_idx = 0
        slot.text_idx = 0
        slot.trailing = []
        slot.next_embed = None
        slot.token_counts = None
        slot.talker_kv = None
        slot.c2w_kv = None
        slot.c2w_conv_states = None
        slot.c2w_transconv_states = None
        self._free_slots.append(slot_id)
        logger.debug("Released slot %d (free: %d)", slot_id, len(self._free_slots))

    def get(self, slot_id: int) -> SlotKVState:
        return self._slots[slot_id]

    def get_by_session(self, session_id: str) -> Optional[SlotKVState]:
        for slot in self._slots:
            if not slot.is_free and slot.session_id == session_id:
                return slot
        return None

    def active_slots(self) -> list[SlotKVState]:
        """Return all slots with active sessions."""
        return [s for s in self._slots if not s.is_free]

    def init_kv_tensors(self, slot: SlotKVState) -> None:
        """Lazily initialize KV cache tensors for a slot.

        Called once at first prefill. Creates empty tensors that will be
        populated by the TRT engine's present_kv outputs.
        """
        c = self._config
        slot.token_counts = torch.zeros(
            1, c.codec_vocab_size, device=self._device, dtype=torch.int64,
        )
        logger.debug("Initialized KV tensors for slot %d", slot.slot_id)

    def reset_for_new_segment(self, slot: SlotKVState) -> None:
        """Reset decode state but preserve KV cache (for segment continuation)."""
        slot.text_idx = 0
        slot.trailing = []
        slot.next_embed = None

    # ------------------------------------------------------------------
    # Pre-allocated pool batch helpers
    # ------------------------------------------------------------------

    def gather_talker_kv(
        self, slot_ids: list[int], max_past_len: int,
    ) -> torch.Tensor:
        """Gather talker KV for a batch from the pre-allocated pool.

        Returns [B, L*2, H, max_past_len, D] without any pad+cat.
        """
        if self._talker_kv_pool is None:
            raise RuntimeError("Pool not pre-allocated")
        ids = torch.tensor(slot_ids, device=self._device, dtype=torch.long)
        return self._talker_kv_pool[ids, :, :, :max_past_len, :].contiguous()

    def scatter_talker_kv(
        self,
        slot_ids: list[int],
        present_kv: torch.Tensor,
        original_past_lens: list[int],
        padded_past_len: int,
        seq: int,
    ) -> None:
        """Write decode output KV back to the pre-allocated pool.

        Handles de-padding for heterogeneous past lengths.
        """
        if self._talker_kv_pool is None:
            raise RuntimeError("Pool not pre-allocated")
        uniform = len(set(original_past_lens)) <= 1
        for i, (slot_id, orig_pl) in enumerate(zip(slot_ids, original_past_lens)):
            new_total = orig_pl + seq
            if uniform or orig_pl >= padded_past_len:
                self._talker_kv_pool[slot_id, :, :, :new_total, :] = \
                    present_kv[i, :, :, :new_total, :]
            else:
                self._talker_kv_pool[slot_id, :, :, :orig_pl, :] = \
                    present_kv[i, :, :, :orig_pl, :]
                self._talker_kv_pool[slot_id, :, :, orig_pl:new_total, :] = \
                    present_kv[i, :, :, padded_past_len:padded_past_len + seq, :]

    def gather_c2w_kv(
        self, slot_ids: list[int], max_c2w_len: int,
    ) -> torch.Tensor:
        """Gather C2W KV from the pre-allocated pool."""
        if self._c2w_kv_pool is None:
            raise RuntimeError("Pool not pre-allocated")
        ids = torch.tensor(slot_ids, device=self._device, dtype=torch.long)
        return self._c2w_kv_pool[ids, :, :, :max_c2w_len, :].contiguous()

    def scatter_c2w_kv(
        self, slot_ids: list[int], present_kv: torch.Tensor,
    ) -> None:
        """Write C2W KV back to the pool (sliding window, uniform length)."""
        if self._c2w_kv_pool is None:
            raise RuntimeError("Pool not pre-allocated")
        s_len = present_kv.shape[3]
        for i, slot_id in enumerate(slot_ids):
            self._c2w_kv_pool[slot_id, :, :, :s_len, :] = present_kv[i]

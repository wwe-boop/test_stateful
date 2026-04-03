"""KV cache pool: pre-allocated GPU memory for concurrent sessions.

Instead of dynamically allocating KV tensors per session (which causes
fragmentation and OOM under 64-session load), we pre-allocate a fixed
pool and assign slots.

Memory layout per slot:
    talker:  num_layers × 2 (K+V) × [1, kv_heads, max_seq_len, head_dim]
    c2w:     num_c2w_states × [1, ...]  (variable shapes per state)

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
    num_layers: int = 28
    kv_heads: int = 8
    head_dim: int = 64
    max_seq_len: int = 2048
    hidden_size: int = 1536
    dtype: torch.dtype = torch.bfloat16
    num_c2w_states: int = 37
    code2wav_sliding_window: int = 256
    codec_vocab_size: int = 2176


@dataclass
class SlotKVState:
    """GPU-side state for one session slot."""
    slot_id: int
    session_id: Optional[str] = None
    is_free: bool = True

    # Talker KV: list of 2*num_layers tensors, each [1, kv_heads, cur_len, head_dim]
    kv_tensors: Optional[list[torch.Tensor]] = None
    past_len: int = 0

    # Code2Wav state: list of tensors (shapes vary per state)
    c2w_states: Optional[list[torch.Tensor]] = None
    frame_idx: int = 0

    # Decode tracking
    next_embed: Optional[torch.Tensor] = None
    token_counts: Optional[torch.Tensor] = None

    # Trailing text embeddings for streaming decode
    trailing: list = field(default_factory=list)
    text_idx: int = 0


class KVCachePool:
    """Manages a fixed pool of KV cache slots on GPU.

    Slots are lazily initialized: KV tensors are only allocated when
    a session first does prefill.  On release, tensors are zeroed
    (not freed) to avoid reallocation.
    """

    def __init__(
        self,
        max_slots: int,
        config: ModelConfig,
        device: torch.device,
    ):
        self._max_slots = max_slots
        self._config = config
        self._device = device

        self._slots: list[SlotKVState] = [
            SlotKVState(slot_id=i) for i in range(max_slots)
        ]
        self._free_slots: list[int] = list(range(max_slots))

        logger.info(
            "KV pool initialized: %d slots, ~%.1f MB/slot (talker KV only), device=%s",
            max_slots,
            self._estimate_slot_mb(),
            device,
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
        if slot.kv_tensors is not None:
            return
        c = self._config
        slot.kv_tensors = []
        slot.c2w_states = []
        slot.token_counts = torch.zeros(
            1, c.codec_vocab_size, device=self._device, dtype=torch.int64,
        )
        logger.debug("Initialized KV tensors for slot %d", slot.slot_id)

    def reset_for_new_segment(self, slot: SlotKVState) -> None:
        """Reset decode state but preserve KV cache (for segment continuation)."""
        slot.text_idx = 0
        slot.trailing = []
        slot.next_embed = None

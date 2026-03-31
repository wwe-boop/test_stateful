# English comments only.
"""
Lightweight helpers for multi-session batched fused decode (in-flight batching).

Current runtime may use these helpers incrementally:

- ``zeros_attention_bias`` for single-request / uniform-length fused calls
- ``uniform_past_seq_lens`` for rectangular batches or prefix-cache reuse
- ``pad_talker_past_kv`` + ``padded_attention_bias`` for future heterogeneous
  past-KV batching where each row has a different effective history length
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Generic, List, Sequence, TypeVar

import torch

T = TypeVar("T")


@dataclass
class FusedDecodeTicket(Generic[T]):
    payload: T
    talker_past_len: int
    c2w_past_len: int


def group_by_c2w_past_len(tickets: List[FusedDecodeTicket[T]]) -> Dict[int, List[FusedDecodeTicket[T]]]:
    groups: Dict[int, List[FusedDecodeTicket[T]]] = defaultdict(list)
    for ticket in tickets:
        groups[int(ticket.c2w_past_len)].append(ticket)
    return dict(groups)


def _apply_causal_mask(bias: torch.Tensor, past_len: int, seq: int) -> torch.Tensor:
    """Apply lower-triangular causal mask to the new-token region of an attention bias.

    For seq == 1 (decode) this is a no-op.  For seq > 1 (prefill) token i
    may only attend to past positions and new positions 0..i (not future).
    """
    if seq <= 1:
        return bias
    causal = torch.triu(
        torch.full((seq, seq), float("-inf"), device=bias.device, dtype=bias.dtype),
        diagonal=1,
    )
    bias[:, :, :, past_len:past_len + seq] += causal.unsqueeze(0).unsqueeze(0)
    return bias


def zeros_attention_bias(
    batch: int,
    seq: int,
    past_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    total = past_len + seq
    bias = torch.zeros(batch, 1, seq, total, device=device, dtype=dtype)
    return _apply_causal_mask(bias, past_len, seq)


def padded_attention_bias(
    past_seq_lens: torch.Tensor,
    seq: int,
    padded_past_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    batch = int(past_seq_lens.shape[0])
    total = int(padded_past_len) + int(seq)
    bias = torch.zeros(batch, 1, seq, total, device=device, dtype=dtype)
    if padded_past_len > 0:
        for row, effective_len in enumerate(past_seq_lens.tolist()):
            eff = int(effective_len)
            if eff < padded_past_len:
                bias[row, 0, :, eff:padded_past_len] = float("-inf")
    return _apply_causal_mask(bias, padded_past_len, seq)


def uniform_past_seq_lens(batch: int, past_len: int, device: torch.device) -> torch.Tensor:
    return torch.full((batch,), past_len, device=device, dtype=torch.long)


def pad_talker_past_kv(
    session_kv: Sequence[Sequence[torch.Tensor]],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    if not session_kv:
        return [], torch.empty((0,), device=device, dtype=torch.long)

    num_tensors = len(session_kv[0])
    if any(len(kv_row) != num_tensors for kv_row in session_kv):
        raise ValueError("all session_kv rows must have the same tensor count")

    past_seq_lens = torch.tensor(
        [int(kv_row[0].shape[2]) for kv_row in session_kv],
        device=device,
        dtype=torch.long,
    )
    padded_past_len = int(past_seq_lens.max().item())
    batched: list[torch.Tensor] = []

    for tensor_idx in range(num_tensors):
        rows = []
        for kv_row in session_kv:
            tensor = kv_row[tensor_idx].to(device=device, dtype=dtype).contiguous()
            cur_len = int(tensor.shape[2])
            if cur_len < padded_past_len:
                pad_shape = list(tensor.shape)
                pad_shape[2] = padded_past_len - cur_len
                pad = torch.zeros(*pad_shape, device=device, dtype=dtype)
                tensor = torch.cat([tensor, pad], dim=2)
            rows.append(tensor)
        batched.append(torch.cat(rows, dim=0).contiguous())

    return batched, past_seq_lens

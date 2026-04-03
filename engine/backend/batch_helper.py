"""Batched decode helpers: KV padding, attention bias, KV split.

Migrated from model_repository/tts_orchestrator/1/batch_decode_scheduler.py.
Pure torch — no Triton, no pb_utils.
"""

from __future__ import annotations

from typing import List, Sequence

import torch


# ---------------------------------------------------------------------------
# Attention bias construction
# ---------------------------------------------------------------------------

def _apply_causal_mask(bias: torch.Tensor, past_len: int, seq: int) -> torch.Tensor:
    """Lower-triangular causal mask over the new-token region.

    For seq == 1 (decode) this is a no-op.
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
    """Uniform-past-length attention bias: all positions visible."""
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
    """Attention bias for heterogeneous past lengths.

    Masks out padding positions in the KV cache for each row.
    """
    batch = int(past_seq_lens.shape[0])
    total = int(padded_past_len) + int(seq)
    bias = torch.zeros(batch, 1, seq, total, device=device, dtype=dtype)
    if padded_past_len > 0:
        for row, effective_len in enumerate(past_seq_lens.tolist()):
            eff = int(effective_len)
            if eff < padded_past_len:
                bias[row, 0, :, eff:padded_past_len] = float("-inf")
    return _apply_causal_mask(bias, padded_past_len, seq)


def uniform_past_seq_lens(
    batch: int, past_len: int, device: torch.device,
) -> torch.Tensor:
    return torch.full((batch,), past_len, device=device, dtype=torch.long)


# ---------------------------------------------------------------------------
# KV cache padding and splitting
# ---------------------------------------------------------------------------

def pad_talker_past_kv(
    session_kv: Sequence[Sequence[torch.Tensor]],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Pad heterogeneous KV caches to max length and batch them.

    Args:
        session_kv: list of per-session KV lists.
            Each inner list has 2*num_layers tensors [K0, V0, K1, V1, ...].
            Each tensor shape: [1, kv_heads, seq_len, head_dim].

    Returns:
        (batched_kv, past_seq_lens) where batched_kv tensors have shape
        [B, kv_heads, padded_past_len, head_dim].
    """
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


def split_batched_kv(
    batched_kv_tensors: list[torch.Tensor],
    original_past_lens: list[int],
    padded_past_len: int,
    seq: int,
) -> list[list[torch.Tensor]]:
    """Extract per-session KV from padded batch output.

    After a batch decode step, the TRT engine returns batched KV with
    padded_past_len + seq positions.  We strip the padding to get each
    session's actual KV.
    """
    uniform = len(set(original_past_lens)) <= 1

    if uniform:
        rows: list[list[torch.Tensor]] = []
        for row_idx in range(len(original_past_lens)):
            rows.append([t[row_idx : row_idx + 1] for t in batched_kv_tensors])
        return rows

    rows = []
    for row_idx, orig_pl in enumerate(original_past_lens):
        row: list[torch.Tensor] = []
        needs_depad = orig_pl < padded_past_len
        for tensor in batched_kv_tensors:
            t = tensor[row_idx : row_idx + 1]
            if needs_depad:
                real_past = t[:, :, :orig_pl, :]
                new_part = t[:, :, padded_past_len : padded_past_len + seq, :]
                t = torch.cat([real_past, new_part], dim=2).contiguous()
            else:
                t = t[:, :, : orig_pl + seq, :].clone().contiguous()
            row.append(t)
        rows.append(row)
    return rows

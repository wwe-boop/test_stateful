"""
Greedy streaming tokenizer: buffer incoming text and emit complete BPE tokens only.

Uses longest-prefix stability: tokenization of prefix must match prefix of full token ids.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, List

import numpy as np
import torch

logger = logging.getLogger("tts_orchestrator.greedy_tokenizer")


def _flatten_ids(tokenizer_out: Any) -> np.ndarray:
    payload = tokenizer_out["input_ids"]
    arr = np.asarray(payload, dtype=np.int64)
    return arr.reshape(-1)


class GreedyTokenizer:
    """
    Buffer streaming text; on feed(), return text embedding tensors for stable token prefixes.
    """

    def __init__(
        self,
        tokenizer: Any,
        text_embed_fn: Callable[[torch.Tensor], torch.Tensor],
        device: torch.device,
    ) -> None:
        self.tokenizer = tokenizer
        self.text_embed_fn = text_embed_fn
        self.device = device
        self._pending: str = ""

    def feed(self, text: str) -> List[torch.Tensor]:
        """Append text and return list of [1,1,H] embeddings for newly completed tokens."""
        if not text:
            return []
        self._pending += text
        return self._emit_stable_embeddings()

    def flush(self) -> List[torch.Tensor]:
        """Flush remaining buffer (call on text_complete)."""
        if not self._pending:
            return []
        # Treat entire pending as final
        ids = _flatten_ids(self.tokenizer(self._pending, return_tensors="pt"))
        self._pending = ""
        return self._ids_to_embeddings(ids)

    def pending_len(self) -> int:
        return len(self._pending)

    def _stable_char_prefix_len(self, text: str) -> int:
        """Longest character prefix length such that tokenize(prefix) matches full ids prefix."""
        if not text:
            return 0
        full_ids = _flatten_ids(self.tokenizer(text, return_tensors="pt"))
        if full_ids.size == 0:
            return 0
        lo, hi = 0, len(text)
        best = 0
        while lo <= hi:
            mid = (lo + hi) // 2
            sub = text[:mid]
            sub_ids = _flatten_ids(self.tokenizer(sub, return_tensors="pt"))
            if sub_ids.size == 0:
                lo = mid + 1
                continue
            n = min(len(sub_ids), len(full_ids))
            if n > 0 and np.array_equal(sub_ids[:n], full_ids[:n]):
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        return best

    def _emit_stable_embeddings(self) -> List[torch.Tensor]:
        out: List[torch.Tensor] = []
        while self._pending:
            cut = self._stable_char_prefix_len(self._pending)
            if cut == 0:
                break
            stable_text = self._pending[:cut]
            self._pending = self._pending[cut:]
            ids = _flatten_ids(self.tokenizer(stable_text, return_tensors="pt"))
            out.extend(self._ids_to_embeddings(ids))
        return out

    def _ids_to_embeddings(self, ids: np.ndarray) -> List[torch.Tensor]:
        if ids.size == 0:
            return []
        t = torch.as_tensor(ids, device=self.device, dtype=torch.int64).reshape(1, -1)
        emb = self.text_embed_fn(t)
        # emb [1, S, H] -> list of [1,1,H]
        seq = int(emb.shape[1])
        return [emb[:, i : i + 1, :].clone().contiguous() for i in range(seq)]

"""
Text segmentation helpers for long-form TTS rollover.

This module is intentionally pure Python so it can be unit-tested without Triton
or torch. Segments are selected against the same assistant prompt format used by
PrefillBuilder so token-budget decisions stay aligned with runtime behavior.
"""

from __future__ import annotations

import re
from typing import Any, List

import numpy as np

OFFICIAL_ASSISTANT_FMT = "<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"

_FRAGMENT_BOUNDARY_RE = re.compile(r"(?<=[。！？!?；;：:\n])|(?<=[，,、])|(?<=\s)")


def assistant_token_count(text: str, tokenizer: Any) -> int:
    """Return assistant-format token count for ``text`` using the lightweight tokenizer API."""
    payload = tokenizer(
        OFFICIAL_ASSISTANT_FMT.format(text=text),
        return_tensors="pt",
    )["input_ids"]
    arr = np.asarray(payload, dtype=np.int64)
    if arr.ndim == 1:
        return int(arr.shape[0])
    return int(arr.shape[-1])


def _largest_prefix_within_budget(text: str, tokenizer: Any, max_tokens: int) -> int:
    """Binary-search the longest prefix whose assistant token count fits the budget."""
    lo, hi = 1, len(text)
    best = 1
    while lo <= hi:
        mid = (lo + hi) // 2
        piece = text[:mid].strip() or text[:mid]
        if assistant_token_count(piece, tokenizer) <= max_tokens:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return max(1, best)


def _hard_split_fragment(text: str, tokenizer: Any, max_tokens: int) -> List[str]:
    """Split an oversized fragment by longest fitting prefixes when no punctuation boundary exists."""
    remain = text.strip()
    parts: List[str] = []
    while remain:
        if assistant_token_count(remain, tokenizer) <= max_tokens:
            parts.append(remain)
            break
        cut = _largest_prefix_within_budget(remain, tokenizer, max_tokens)
        piece = remain[:cut].strip()
        if not piece:
            piece = remain[:1]
            cut = 1
        parts.append(piece)
        remain = remain[cut:].lstrip()
    return parts


def split_text_for_token_budget(text: str, tokenizer: Any, max_tokens: int) -> List[str]:
    """
    Greedily pack punctuation-delimited fragments under ``max_tokens``.

    Preference order:
    1. Strong punctuation/newline boundaries already present in the text
    2. Weak punctuation / whitespace boundaries
    3. Hard split by longest prefix within budget
    """
    text = (text or "").strip()
    if not text:
        return []
    if max_tokens <= 0 or assistant_token_count(text, tokenizer) <= max_tokens:
        return [text]

    fragments = [frag for frag in _FRAGMENT_BOUNDARY_RE.split(text) if frag]
    segments: List[str] = []
    current = ""

    for raw_fragment in fragments:
        fragment = raw_fragment if current else raw_fragment.lstrip()
        if not fragment:
            continue

        candidate = (current + fragment).strip()
        if candidate and assistant_token_count(candidate, tokenizer) <= max_tokens:
            current = current + fragment
            continue

        if current.strip():
            segments.append(current.strip())
            current = ""
            fragment = raw_fragment.lstrip()

        if fragment and assistant_token_count(fragment.strip(), tokenizer) <= max_tokens:
            current = fragment
            continue

        segments.extend(_hard_split_fragment(fragment, tokenizer, max_tokens))

    if current.strip():
        segments.append(current.strip())

    return [seg for seg in segments if seg]

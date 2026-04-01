"""
Text segmentation helpers for long-form TTS rollover.

This module is intentionally pure Python so it can be unit-tested without Triton
or torch. Segments are selected against the same assistant prompt format used by
PrefillBuilder so token-budget decisions stay aligned with runtime behavior.
"""

from __future__ import annotations

from typing import Any, List

import numpy as np

OFFICIAL_ASSISTANT_FMT = "<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"

# Phase-A punctuation tiers (L2 includes L1; L3 includes L2 + dash/ellipsis).
PUNCT_LEVEL_1 = set("。！？!?\n")
PUNCT_LEVEL_2 = PUNCT_LEVEL_1 | set("，,；;、：:")
PUNCT_LEVEL_3 = PUNCT_LEVEL_2 | set("—…\u2014\u2026")

_QUOTE_OPEN = set('"\u201c')
_QUOTE_CLOSE = set('"\u201d')

_DEFAULT_QUOTE_LOOKAHEAD_CHARS = 120


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


def find_quote_aware_presplit_cut(
    text: str,
    max_chars: int,
    lookahead: int = _DEFAULT_QUOTE_LOOKAHEAD_CHARS,
) -> int:
    """
    Return exclusive cut index for coarse pre-segmentation.

    Prefers strong punctuation (L1) outside paired quotes; scans up to
    max_chars + lookahead. Falls back to max_chars or first L1 inside quotes.
    """
    if not text:
        return 0
    n = len(text)
    max_chars = min(max(max_chars, 1), n)
    lim = min(n, max_chars + max(16, lookahead))
    in_quote = False
    best_le_max: int = -1
    for i in range(lim):
        ch = text[i]
        if ch in _QUOTE_OPEN:
            in_quote = True
        elif ch in _QUOTE_CLOSE:
            in_quote = False
        if ch in PUNCT_LEVEL_1:
            cut = i + 1
            if cut <= max_chars and not in_quote:
                best_le_max = cut
            elif cut <= max_chars and in_quote and best_le_max < 0:
                best_le_max = cut
    if best_le_max > 0:
        return best_le_max
    in_quote = False
    for i in range(max_chars, lim):
        ch = text[i]
        if ch in _QUOTE_OPEN:
            in_quote = True
        elif ch in _QUOTE_CLOSE:
            in_quote = False
        if not in_quote and ch in PUNCT_LEVEL_1:
            return i + 1
    return max_chars


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
        cut = find_quote_aware_presplit_cut(remain, cut)
        piece = remain[:cut].strip()
        if not piece:
            piece = remain[:1]
            cut = 1
        parts.append(piece)
        remain = remain[cut:].lstrip()
    return parts


def split_text_for_token_budget(text: str, tokenizer: Any, max_tokens: int) -> List[str]:
    """
    Pack text into segments under ``max_tokens`` (assistant format).

    Uses quote-aware coarse cuts near the token-derived char bound, then the
    legacy fragment packing for leftovers.
    """
    text = (text or "").strip()
    if not text:
        return []
    if max_tokens <= 0 or assistant_token_count(text, tokenizer) <= max_tokens:
        return [text]

    segments: List[str] = []
    remain = text
    while remain:
        if assistant_token_count(remain, tokenizer) <= max_tokens:
            segments.append(remain)
            break
        char_hi = _largest_prefix_within_budget(remain, tokenizer, max_tokens)
        cut = find_quote_aware_presplit_cut(remain, char_hi)
        if cut <= 0:
            cut = min(len(remain), char_hi)
        piece = remain[:cut].strip()
        if not piece:
            piece = remain[:1]
            cut = 1
        segments.append(piece)
        remain = remain[cut:].lstrip()

    merged: List[str] = []
    for seg in segments:
        if not seg:
            continue
        if assistant_token_count(seg, tokenizer) <= max_tokens:
            merged.append(seg)
            continue
        merged.extend(_hard_split_fragment(seg, tokenizer, max_tokens))
    return [s for s in merged if s]

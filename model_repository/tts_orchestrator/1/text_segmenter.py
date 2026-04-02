"""
Text segmentation helpers for long-form TTS rollover.

Three-layer architecture:
  Layer 1 (this module) — Pre-segmentation with full text visibility.
    Splits long text into segments that each fit within a token budget,
    preferring sentence-ending punctuation (L1) as cut points.
  Layer 2 (decode_fsm) — Phase-A runtime threshold control.
  Layer 3 (model.py)   — Phase-B fallback guarantee.

Pre-segmentation is pure Python so it can be unit-tested without Triton
or torch.  Segments use the same assistant prompt format as PrefillBuilder
so token-budget decisions stay aligned with runtime behavior.
"""

from __future__ import annotations

from typing import Any, List

import numpy as np

OFFICIAL_ASSISTANT_FMT = "<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"

# Phase-A punctuation tiers (L2 includes L1; L3 includes L2 + dash/ellipsis).
# These are used by the FSM at runtime (decode_fsm.py).
PUNCT_LEVEL_1 = set("。！？!?\n")
PUNCT_LEVEL_2 = PUNCT_LEVEL_1 | set("，,；;、：:")
PUNCT_LEVEL_3 = PUNCT_LEVEL_2 | set("—…\u2014\u2026")

# Pre-segmentation cuts at true sentence-ending punctuation only.
# Bare newlines are formatting separators, not sentence boundaries.
_PRESPLIT_SENTENCE_END = set("。！？!?")
_PRESPLIT_L2 = set("，,；;、：:")

_QUOTE_OPEN = set('"\u201c')
_QUOTE_CLOSE = set('"\u201d')

# Whitespace that is meaningless for speech synthesis.
# Newlines produce dedicated tokens the TTS model never saw during training,
# resulting in garbled or silent audio.  Tabs / carriage-returns are likewise
# irrelevant for speech.
_WHITESPACE_TO_STRIP = str.maketrans({
    '\n': '',
    '\r': '',
    '\t': ' ',
    '\u3000': '',  # fullwidth space (ideographic space)
})


def normalize_tts_text(text: str) -> str:
    """Collapse whitespace that is harmful for TTS tokenization.

    * ``\\n``, ``\\r``, ``\\u3000`` → removed
    * ``\\t`` → single space
    * Consecutive spaces collapsed to one
    * Leading / trailing whitespace stripped
    """
    text = text.translate(_WHITESPACE_TO_STRIP)
    # collapse consecutive spaces
    while '  ' in text:
        text = text.replace('  ', ' ')
    return text.strip()


def assistant_token_count(text: str, tokenizer: Any) -> int:
    """Return assistant-format token count for *text*."""
    payload = tokenizer(
        OFFICIAL_ASSISTANT_FMT.format(text=text),
        return_tensors="pt",
    )["input_ids"]
    arr = np.asarray(payload, dtype=np.int64)
    if arr.ndim == 1:
        return int(arr.shape[0])
    return int(arr.shape[-1])


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _char_budget(text: str, tokenizer: Any, max_tokens: int) -> int:
    """Binary-search the longest char prefix whose token count fits *max_tokens*."""
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


def _find_best_cut(text: str, limit: int) -> int:
    """Find the best cut position within ``text[:limit]``.

    Scans *forward* through ``text[0:limit]`` tracking quote state and
    recording the last (rightmost) punctuation position at each tier.

    Priority:
      1. Last L1 punctuation (。！？!?) outside quotes within limit.
         If L1 immediately follows a closing quote, include the quote mark.
         After a closing-quote + L1 boundary, stop scanning — this prevents
         short post-quote sentences from being swallowed into the current
         segment.
      2. Last L1 inside quotes (fallback when entire range is quoted).
      3. Last L2 punctuation (，；：etc.) outside quotes within limit.
      4. Hard cut at *limit*.

    Returns an exclusive cut index (``text[:cut]`` is the segment).
    """
    if not text or limit <= 0:
        return max(1, limit)

    n = len(text)
    limit = min(limit, n)

    in_quote = False
    just_closed_quote = False
    best_l1: int = -1
    best_l1_in_quote: int = -1
    best_l2: int = -1

    for i in range(limit):
        ch = text[i]

        if ch in _QUOTE_OPEN:
            in_quote = True
            just_closed_quote = False
        elif ch in _QUOTE_CLOSE:
            in_quote = False
            just_closed_quote = True
        elif not ch.isspace():
            just_closed_quote = False

        if ch in _PRESPLIT_SENTENCE_END:
            cut = i + 1
            # Include a trailing close-quote if immediately adjacent
            if cut < n and text[cut] in _QUOTE_CLOSE:
                cut += 1
            if cut <= limit:
                if not in_quote:
                    best_l1 = cut
                    # After closing-quote boundary, lock in this cut —
                    # don't greedily extend past the quote block.
                    if just_closed_quote:
                        return best_l1
                elif best_l1_in_quote < 0:
                    best_l1_in_quote = cut

        elif ch in _PRESPLIT_L2 and not in_quote:
            if (i + 1) <= limit:
                best_l2 = i + 1

    if best_l1 > 0:
        return best_l1
    if best_l1_in_quote > 0:
        return best_l1_in_quote
    if best_l2 > 0:
        return best_l2
    return limit


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def split_text_for_token_budget(
    text: str,
    tokenizer: Any,
    max_tokens: int,
) -> List[str]:
    """Split *text* into segments each fitting within *max_tokens* (assistant fmt).

    The algorithm strictly respects the budget — no lookahead beyond the
    token-derived character limit.  Within that limit it prefers cutting at
    sentence-ending punctuation (L1), falls back to clause punctuation (L2),
    and hard-cuts only as a last resort.

    Returns a list of non-empty segment strings.
    """
    text = normalize_tts_text(text or "")
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

        char_hi = _char_budget(remain, tokenizer, max_tokens)
        cut = _find_best_cut(remain, char_hi)

        piece = remain[:cut].strip()
        if not piece:
            piece = remain[:1]
            cut = 1

        segments.append(piece)
        remain = remain[cut:].lstrip()

    return [s for s in segments if s]

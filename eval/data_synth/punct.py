"""Punctuation class helpers for test-prosody samples."""

from __future__ import annotations

import re
from typing import Any

PUNCT_CLASS_ALIASES = {
    "comma": "comma",
    "period": "period",
    "question": "question",
    "exclamation": "exclamation",
    "semicolon": "semicolon",
    "colon": "colon",
    "pause": "comma",
    "full_stop": "period",
    "question_mark": "question",
    "exclamation_mark": "exclamation",
    "emdash": "comma",
    "dash": "comma",
    "ellipsis": "comma",
}

TRAILING_PUNCT_TO_CLASS = {
    "，": "comma",
    ",": "comma",
    "。": "period",
    ".": "period",
    "？": "question",
    "?": "question",
    "！": "exclamation",
    "!": "exclamation",
    "；": "semicolon",
    ";": "semicolon",
    "：": "colon",
    ":": "colon",
}


def infer_punct_class(text: str, fallback: str = "period") -> str:
    """Map trailing punctuation on a segment to a punct_class bucket."""
    stripped = text.strip()
    if not stripped:
        return fallback

    for ch in reversed(stripped):
        if ch in TRAILING_PUNCT_TO_CLASS:
            return TRAILING_PUNCT_TO_CLASS[ch]
        if not ch.isspace():
            break
    return fallback


def _canonical_punct_class(value: str) -> str:
    key = value.strip().lower().replace("-", "").replace(" ", "_")
    if key in PUNCT_CLASS_ALIASES:
        return PUNCT_CLASS_ALIASES[key]
    raise ValueError(f"Unknown punct_class: {value!r}")


def normalize_segments(raw_segments: list[Any]) -> list[dict[str, str]]:
    """Validate LLM output and attach punct_class when missing."""
    if not raw_segments:
        raise ValueError("segments must not be empty")

    normalized: list[dict[str, str]] = []
    for idx, item in enumerate(raw_segments):
        if isinstance(item, str):
            text = item.strip()
            punct_class = infer_punct_class(text)
        elif isinstance(item, dict):
            text = str(item.get("text", "")).strip()
            if not text:
                raise ValueError(f"segment {idx} has empty text")
            punct_raw = item.get("punct_class")
            punct_class = (
                _canonical_punct_class(str(punct_raw))
                if punct_raw is not None
                else infer_punct_class(text)
            )
        else:
            raise ValueError(f"segment {idx} must be str or dict, got {type(item)!r}")

        if len(text) < 2:
            raise ValueError(f"segment {idx} is too short: {text!r}")

        normalized.append({"text": text, "punct_class": punct_class})

    return normalized


def boundary_punct_classes(segments: list[dict[str, str]]) -> list[str]:
    """Return punct_class for each inter-segment boundary."""
    if len(segments) < 2:
        return []
    return [segments[i]["punct_class"] for i in range(len(segments) - 1)]


def full_text(segments: list[dict[str, str]]) -> str:
    """Join segment texts for CER / synthesis input."""
    return "".join(seg["text"] for seg in segments)


def count_chinese_chars(text: str) -> int:
    return len(re.findall(r"[\u4e00-\u9fff]", text))

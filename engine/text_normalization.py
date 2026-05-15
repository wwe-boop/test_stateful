"""Shared text cleanup helpers for TTS input."""

from __future__ import annotations

_EMOJI_RANGES = (
    (0x1F000, 0x1FAFF),
    (0x2600, 0x27BF),
)
_EMOJI_COMPONENT_RANGES = (
    (0xFE00, 0xFE0F),
    (0xE0020, 0xE007F),
    (0xE0100, 0xE01EF),
)
_EMOJI_COMPONENT_CODEPOINTS = {
    0x200D,  # zero width joiner
    0x20E3,  # combining enclosing keycap
}
_EMOJI_CODEPOINTS = {
    0x00A9,
    0x00AE,
    0x203C,
    0x2049,
    0x2122,
    0x2139,
    0x3030,
    0x303D,
    0x3297,
    0x3299,
}


def _in_ranges(codepoint: int, ranges: tuple[tuple[int, int], ...]) -> bool:
    return any(start <= codepoint <= end for start, end in ranges)


def is_emoji_char(ch: str) -> bool:
    """Return True for emoji code points and emoji sequence components."""
    if not ch:
        return False
    codepoint = ord(ch)
    return (
        codepoint in _EMOJI_CODEPOINTS
        or codepoint in _EMOJI_COMPONENT_CODEPOINTS
        or _in_ranges(codepoint, _EMOJI_RANGES)
        or _in_ranges(codepoint, _EMOJI_COMPONENT_RANGES)
    )


def _is_ascii_word_char(ch: str) -> bool:
    return ch.isascii() and ch.isalnum()


def _keycap_sequence_len(text: str, index: int) -> int:
    ch = text[index]
    if ch not in "0123456789#*":
        return 0
    next_index = index + 1
    if next_index < len(text) and ord(text[next_index]) == 0x20E3:
        return 2
    if (
        next_index + 1 < len(text)
        and ord(text[next_index]) == 0xFE0F
        and ord(text[next_index + 1]) == 0x20E3
    ):
        return 3
    return 0


def _next_non_emoji_char(text: str, start: int) -> str:
    i = start
    while i < len(text):
        keycap_len = _keycap_sequence_len(text, i)
        if keycap_len:
            i += keycap_len
            continue
        if not is_emoji_char(text[i]):
            break
        i += 1
    return text[i] if i < len(text) else ""


def strip_emoji(text: str) -> str:
    """Remove emoji while keeping normal text and punctuation intact."""
    if not text:
        return ""

    out: list[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        keycap_len = _keycap_sequence_len(text, i)
        if keycap_len:
            next_ch = _next_non_emoji_char(text, i + keycap_len)
            if (
                out
                and _is_ascii_word_char(out[-1])
                and _is_ascii_word_char(next_ch)
            ):
                out.append(" ")
            i += keycap_len
            continue

        if not is_emoji_char(ch):
            out.append(ch)
            i += 1
            continue

        next_ch = _next_non_emoji_char(text, i + 1)
        if (
            out
            and _is_ascii_word_char(out[-1])
            and _is_ascii_word_char(next_ch)
        ):
            out.append(" ")

        i += 1
        while i < len(text):
            keycap_len = _keycap_sequence_len(text, i)
            if keycap_len:
                i += keycap_len
                continue
            if not is_emoji_char(text[i]):
                break
            i += 1

    return "".join(out)


__all__ = ("is_emoji_char", "strip_emoji")

"""Per-session audio chunk reorder buffer for Level 2 segment pipelining.

When multiple segments of the same session decode in parallel, audio chunks
may arrive out of segment order.  This buffer ensures chunks are emitted
to the gRPC stream in strict segment order.

Memory overhead: at most one segment's audio buffered (~600 KB typical).
"""

from __future__ import annotations

from collections import defaultdict
from typing import List


class AudioReorder:
    """Reorder buffer that emits audio in segment order.

    Usage::

        reorder = AudioReorder()
        out = reorder.push(1, b"seg1_chunk0")  # buffered (seg 0 not done)
        out = reorder.push(0, b"seg0_chunk0")  # emitted (seg 0 is current)
        out = reorder.mark_done(0)              # drains buffered seg 1
    """

    __slots__ = ("_next_emit", "_buffers", "_done")

    def __init__(self) -> None:
        self._next_emit: int = 0
        self._buffers: dict[int, list[bytes]] = defaultdict(list)
        self._done: set[int] = set()

    @property
    def next_emit_segment(self) -> int:
        return self._next_emit

    def push(self, segment_idx: int, audio: bytes) -> List[bytes]:
        """Push an audio chunk; return chunks ready for emission (in order)."""
        self._buffers[segment_idx].append(audio)
        if segment_idx == self._next_emit:
            return self._try_drain()
        return []

    def mark_done(self, segment_idx: int) -> List[bytes]:
        """Mark a segment as fully complete; drain any contiguous completions."""
        self._done.add(segment_idx)
        return self._try_drain()

    def _try_drain(self) -> List[bytes]:
        out: List[bytes] = []
        while True:
            seg = self._next_emit
            buf = self._buffers.get(seg)
            if buf is None:
                break
            out.extend(buf)
            buf.clear()
            if seg in self._done:
                del self._buffers[seg]
                self._done.discard(seg)
                self._next_emit += 1
            else:
                break
        return out

    def reset(self) -> None:
        self._next_emit = 0
        self._buffers.clear()
        self._done.clear()


__all__ = ("AudioReorder",)

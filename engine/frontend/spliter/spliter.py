"""Text segmentation orchestrator for Qwen3-TTS streaming/offline synthesis.

Responsibilities:
  1. Offline pre-split: find globally optimal L1 punctuation boundaries
     within KV budget so each segment exactly triggers one Driver split.
  2. Text buffering: accumulate upstream text chunks, smooth input rate.
  3. Drive Driver: tokenize buffered text, classify punct level, feed
     events to per-segment StreamingDrivers, collect actions.

Architecture (Level 2 pipelining)::

    Upstream → Spliter → Driver[seg0] → SegmentActions
                       → Driver[seg1] → SegmentActions   (parallel)
                       → ...

The Spliter creates a new Driver when the previous segment produces a
FLUSH action.  In offline mode, all segments can be driven immediately.
In streaming mode, segments are driven as tokens arrive.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .driver import (
    StreamingDriver,
    ActionType,
    ActionResult,
    SplitThresholds,
    compute_thresholds,
)
from .event import SpliterEvent, SpliterEventType as ET
from .defines import LEVEL1_PUNCTIONS, LEVEL2_PUNCTIONS, LEVEL3_PUNCTIONS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output type — actions tagged with segment index
# ---------------------------------------------------------------------------

@dataclass
class SegmentAction:
    """One Driver action tagged with the segment it belongs to."""
    segment_idx: int
    action: ActionResult


# ---------------------------------------------------------------------------
# Spliter
# ---------------------------------------------------------------------------

class Spliter:
    """Orchestrates text splitting and multi-segment Driver management.

    Parameters
    ----------
    engine_max_decode_len : int
        Maximum KV cache steps per segment.
    prefill_len : int
        Estimated prefill length (prompt tokens) for threshold computation.
    ema_ratio : float
        Initial audio:text step ratio (updated via update_ratio).
    max_concurrent : int
        Maximum segments driven in parallel (Level 2).  Set to 1 for
        Level 1 (serial) behavior.
    """

    def __init__(
        self,
        *,
        engine_max_decode_len: int = 512,
        prefill_len: int = 12,
        ema_ratio: float = 5.0,
        safety_margin: int = 8,
        max_concurrent: int = 2,
    ) -> None:
        self._engine_max = engine_max_decode_len
        self._prefill_len = prefill_len
        self._ema_ratio = ema_ratio
        self._safety_margin = safety_margin
        self._max_concurrent = max_concurrent

        # Pre-split token sequences (offline mode)
        self._presplit_segments: List[List[Tuple[int, str, int]]] = []
        self._presplit_cursor: int = 0

        # Streaming token buffer (streaming mode)
        self._token_buffer: List[Tuple[int, str, int]] = []
        self._text_complete: bool = False

        # Per-segment drivers; key = segment_idx
        self._drivers: dict[int, StreamingDriver] = {}
        self._next_segment_idx: int = 0

        # Segments that have entered flush (engine still decoding pad)
        self._flushing: set[int] = set()
        # Segments fully done (engine reported SEGMENT_END)
        self._done: set[int] = set()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def ema_ratio(self) -> float:
        return self._ema_ratio

    @property
    def current_segment_idx(self) -> int:
        """Index that will be assigned to the next segment created."""
        return self._next_segment_idx

    @property
    def active_segment_count(self) -> int:
        return len(self._drivers) - len(self._done)

    # ------------------------------------------------------------------
    # Threshold helpers
    # ------------------------------------------------------------------

    def _make_thresholds(self) -> SplitThresholds:
        remaining_kv = self._engine_max - self._prefill_len
        return compute_thresholds(
            remaining_kv, self._ema_ratio, self._safety_margin,
        )

    def _create_driver(self) -> Tuple[int, StreamingDriver]:
        idx = self._next_segment_idx
        self._next_segment_idx += 1
        driver = StreamingDriver(self._make_thresholds())
        self._drivers[idx] = driver
        return idx, driver

    # ------------------------------------------------------------------
    # Token classification
    # ------------------------------------------------------------------

    @staticmethod
    def classify_punct_level(text: str) -> int:
        """Determine punctuation level from token text.

        Returns 0 (none), 1 (L1), 2 (L2), or 3 (L3).
        Checks the last non-whitespace character.
        """
        stripped = text.rstrip()
        if not stripped:
            return 0
        ch = stripped[-1]
        if ch in LEVEL1_PUNCTIONS:
            return 1
        if ch in LEVEL2_PUNCTIONS:
            return 2
        if ch in LEVEL3_PUNCTIONS:
            return 3
        return 0

    def _make_event(
        self, token_id: int, text: str, punct_level: int,
    ) -> SpliterEvent:
        if punct_level > 0:
            return SpliterEvent(
                type=ET.PUNCTUATION_TOKEN,
                token=token_id,
                text=text,
                punct_level=punct_level,
            )
        return SpliterEvent(
            type=ET.NORMAL_TOKEN,
            token=token_id,
            text=text,
            punct_level=0,
        )

    # ------------------------------------------------------------------
    # Offline pre-split
    # ------------------------------------------------------------------

    def pre_split(
        self, tokens: List[Tuple[int, str]],
    ) -> List[List[Tuple[int, str, int]]]:
        """Split a fully-known token sequence at globally optimal L1 boundaries.

        Returns list of segments, each segment is [(token_id, text, punct_level), ...].

        Algorithm:
          1. Greedy scan; L1 punct (。！？) always triggers a split.
          2. If threshold_d is reached without a qualifying L1 split, look back
             for the LAST L1 punct in the current segment and split there.
             This maximises natural sentence boundaries.
          3. If no L1 punct at all, force-cut at threshold_d.

        L1 always splits to ensure each sub-sentence is fully available
        before prefill, avoiding pad_embed interruptions that degrade
        audio quality.
        """
        if not tokens:
            return []

        th = self._make_thresholds()
        segments: List[List[Tuple[int, str, int]]] = []
        current: List[Tuple[int, str, int]] = []
        last_l1_in_current: int = -1  # 0-indexed position of last L1

        for token_id, text in tokens:
            pl = self.classify_punct_level(text)
            current.append((token_id, text, pl))
            n = len(current)

            if pl == 1:
                last_l1_in_current = n - 1

            if pl == 1:
                segments.append(current)
                current = []
                last_l1_in_current = -1
            elif n >= th.d:
                if last_l1_in_current >= 0:
                    split_at = last_l1_in_current + 1
                    segments.append(current[:split_at])
                    remaining = current[split_at:]
                    current = remaining
                    last_l1_in_current = -1
                    for j, (_, _, p) in enumerate(current):
                        if p == 1:
                            last_l1_in_current = j
                else:
                    segments.append(current)
                    current = []
                    last_l1_in_current = -1

        if current:
            segments.append(current)

        return segments

    # ------------------------------------------------------------------
    # Public API: offline
    # ------------------------------------------------------------------

    def set_full_text(
        self, tokens: List[Tuple[int, str]],
    ) -> List[SegmentAction]:
        """Offline mode: set complete token sequence, pre-split, drive all.

        Returns SegmentActions for up to max_concurrent segments.
        Remaining segments are queued and driven as previous ones flush.
        """
        self._presplit_segments = self.pre_split(tokens)
        self._presplit_cursor = 0
        self._text_complete = True

        return self._drive_presplit_batch()

    def _drive_presplit_batch(self) -> List[SegmentAction]:
        """Drive as many queued pre-split segments as concurrency allows."""
        actions: List[SegmentAction] = []
        while (self._presplit_cursor < len(self._presplit_segments)
               and self.active_segment_count < self._max_concurrent):
            seg_tokens = self._presplit_segments[self._presplit_cursor]
            self._presplit_cursor += 1
            actions.extend(self._drive_segment(seg_tokens, is_final=False))
        # Mark the last driven segment as final if all segments are consumed
        if (self._presplit_cursor >= len(self._presplit_segments)
                and self._text_complete and actions):
            last_idx = actions[-1].segment_idx
            driver = self._drivers.get(last_idx)
            # The Driver's is_final is set via END event in _drive_segment
        return actions

    def _drive_segment(
        self,
        tokens: List[Tuple[int, str, int]],
        *,
        is_final: bool = False,
    ) -> List[SegmentAction]:
        """Create a Driver for one segment and feed all its tokens."""
        idx, driver = self._create_driver()
        actions: List[SegmentAction] = []

        start_evt = SpliterEvent(type=ET.START)
        for r in driver.feed(start_evt):
            actions.append(SegmentAction(idx, r))

        for token_id, text, pl in tokens:
            evt = self._make_event(token_id, text, pl)
            for r in driver.feed(evt):
                actions.append(SegmentAction(idx, r))
                if r.type in (ActionType.FLUSH_EOS, ActionType.FLUSH_NOP):
                    self._flushing.add(idx)

        # Determine if this is the final segment
        is_last = is_final or (
            self._text_complete
            and self._presplit_cursor >= len(self._presplit_segments)
        )

        end_evt = SpliterEvent(type=ET.END)
        for r in driver.feed(end_evt):
            actions.append(SegmentAction(idx, r))

        return actions

    # ------------------------------------------------------------------
    # Public API: streaming
    # ------------------------------------------------------------------

    def feed_tokens(
        self, tokens: List[Tuple[int, str]],
    ) -> List[SegmentAction]:
        """Streaming mode: feed tokenized text, return actions.

        Tokens are classified and fed to the current active Driver.
        If the Driver produces a FLUSH, a new Driver is created for the
        next segment (if concurrency allows).
        """
        classified = [(tid, txt, self.classify_punct_level(txt))
                      for tid, txt in tokens]

        actions: List[SegmentAction] = []

        for token_id, text, pl in classified:
            active_idx = self._get_active_driver_idx()
            if active_idx is None:
                if self.active_segment_count < self._max_concurrent:
                    boot = self._try_start_next()
                    actions.extend(boot)
                    active_idx = self._get_active_driver_idx()
                if active_idx is None:
                    self._token_buffer.append((token_id, text, pl))
                    continue

            driver = self._drivers[active_idx]
            evt = self._make_event(token_id, text, pl)
            results = driver.feed(evt)

            for r in results:
                actions.append(SegmentAction(active_idx, r))
                if r.type in (ActionType.FLUSH_EOS, ActionType.FLUSH_NOP):
                    self._flushing.add(active_idx)
                    actions.extend(self._try_start_next())

        return actions

    def text_done(self) -> List[SegmentAction]:
        """Signal that no more text will arrive (streaming mode)."""
        self._text_complete = True
        actions: List[SegmentAction] = []

        active_idx = self._get_active_driver_idx()
        if active_idx is not None:
            driver = self._drivers[active_idx]
            end_evt = SpliterEvent(type=ET.END)
            for r in driver.feed(end_evt):
                actions.append(SegmentAction(active_idx, r))

        return actions

    def _get_active_driver_idx(self) -> Optional[int]:
        """Find the most recent non-flushing, non-done driver."""
        for idx in range(self._next_segment_idx - 1, -1, -1):
            if idx not in self._flushing and idx not in self._done:
                return idx
        return None

    def _try_start_next(self) -> List[SegmentAction]:
        """After a FLUSH, start a new segment if concurrency allows."""
        if self.active_segment_count >= self._max_concurrent:
            return []

        actions: List[SegmentAction] = []
        idx, driver = self._create_driver()

        start_evt = SpliterEvent(type=ET.START)
        for r in driver.feed(start_evt):
            actions.append(SegmentAction(idx, r))

        # Drain any buffered tokens into the new driver
        remaining: List[Tuple[int, str, int]] = []
        for token_id, text, pl in self._token_buffer:
            evt = self._make_event(token_id, text, pl)
            results = driver.feed(evt)
            for r in results:
                actions.append(SegmentAction(idx, r))
                if r.type in (ActionType.FLUSH_EOS, ActionType.FLUSH_NOP):
                    self._flushing.add(idx)
                    remaining = self._token_buffer[
                        self._token_buffer.index((token_id, text, pl)) + 1:]
                    break
            else:
                continue
            break
        self._token_buffer = remaining

        if self._text_complete and not self._token_buffer:
            end_evt = SpliterEvent(type=ET.END)
            for r in driver.feed(end_evt):
                actions.append(SegmentAction(idx, r))

        return actions

    # ------------------------------------------------------------------
    # Engine feedback
    # ------------------------------------------------------------------

    def on_segment_done(self, segment_idx: int) -> List[SegmentAction]:
        """Called when the engine reports SEGMENT_END for a segment.

        Frees the segment's driver and may start the next queued segment
        (offline pre-split) or drain the token buffer (streaming).
        """
        self._done.add(segment_idx)
        self._flushing.discard(segment_idx)
        self._drivers.pop(segment_idx, None)

        if self._presplit_segments:
            return self._drive_presplit_batch()

        if self._token_buffer or self._text_complete:
            return self._try_start_next()
        return []

    def update_ratio(
        self, actual_audio_steps: int, actual_text_tokens: int,
        *, overflow: bool = False,
    ) -> None:
        """Update EMA audio:text ratio from engine feedback."""
        if actual_text_tokens <= 0:
            return
        actual = actual_audio_steps / actual_text_tokens
        if overflow:
            self._ema_ratio = actual
            logger.warning("EMA ratio overflow reset: %.2f", self._ema_ratio)
        else:
            alpha = 0.3
            self._ema_ratio = alpha * actual + (1 - alpha) * self._ema_ratio
            logger.debug("EMA ratio updated: %.2f", self._ema_ratio)

    # ------------------------------------------------------------------
    # Full reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._presplit_segments.clear()
        self._presplit_cursor = 0
        self._token_buffer.clear()
        self._text_complete = False
        self._drivers.clear()
        self._next_segment_idx = 0
        self._flushing.clear()
        self._done.clear()


__all__ = ("Spliter", "SegmentAction")

"""Text segmentation orchestrator for Qwen3-TTS streaming/offline synthesis.

Responsibilities:
  1. Offline pre-split: split at L1 punctuation (。！？) for optimal
     prosody; forced-cut fallback uses L1 > L2 > L3 priority.
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

from collections import deque
import logging
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

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
    group_idx: int = -1
    local_idx: int = 0
    group_final: bool = True


@dataclass
class PendingGroup:
    """One offline pre-split group with a resumable read cursor."""
    group_idx: int
    tokens: List[Tuple[int, str, int]]
    cursor: int = 0
    next_local_idx: int = 0


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
        ema_alpha: float = 0.1,
        ema_overflow_alpha: float = 0.5,
        ema_min_ratio: float = 2.0,
        ema_max_ratio: float = 10.0,
    ) -> None:
        self._engine_max = engine_max_decode_len
        self._prefill_len = prefill_len
        self._ema_ratio = ema_ratio
        self._safety_margin = safety_margin
        self._max_concurrent = max_concurrent
        self._ema_alpha = ema_alpha
        self._ema_overflow_alpha = ema_overflow_alpha
        self._ema_min_ratio = ema_min_ratio
        self._ema_max_ratio = ema_max_ratio

        # Offline pre-split groups. Each group may still yield multiple
        # backend segments because the Driver remains the final decider.
        self._presplit_groups: Deque[PendingGroup] = deque()
        self._presplit_thresholds: Optional[SplitThresholds] = None

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
        return len(self._drivers)

    # ------------------------------------------------------------------
    # Threshold helpers
    # ------------------------------------------------------------------

    def _make_thresholds(self) -> SplitThresholds:
        remaining_kv = self._engine_max - self._prefill_len
        return compute_thresholds(
            remaining_kv, self._ema_ratio, self._safety_margin,
        )

    def _create_driver(
        self, thresholds: Optional[SplitThresholds] = None,
    ) -> Tuple[int, StreamingDriver]:
        idx = self._next_segment_idx
        self._next_segment_idx += 1
        driver = StreamingDriver(thresholds or self._make_thresholds())
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
        """Split a fully-known token sequence at L1 punctuation boundaries.

        Returns list of segments, each segment is [(token_id, text, punct_level), ...].

        Unlike the streaming Driver (which uses L1/L2/L3 thresholds because it
        lacks global visibility), offline pre-split only cuts at L1 (。！？)
        for optimal prosody and fewer segments.

        Algorithm:
          1. Greedy scan; split at L1 punctuation when token_count >= a.
          2. If threshold_d is reached without an L1 split, look back for
             the best boundary: L1 > L2 > L3 > forced cut.
        """
        if not tokens:
            return []

        th = self._make_thresholds()
        segments: List[List[Tuple[int, str, int]]] = []
        current: List[Tuple[int, str, int]] = []
        last_l1: int = -1
        last_l2: int = -1
        last_l3: int = -1

        def _flush_at(pos: int) -> None:
            nonlocal current, last_l1, last_l2, last_l3
            split_at = pos + 1
            segments.append(current[:split_at])
            remaining = current[split_at:]
            current = remaining
            last_l1 = last_l2 = last_l3 = -1
            for j, (_, _, p) in enumerate(current):
                if p == 1:
                    last_l1 = j
                elif p == 2:
                    last_l2 = j
                elif p == 3:
                    last_l3 = j

        for token_id, text in tokens:
            pl = self.classify_punct_level(text)
            current.append((token_id, text, pl))
            n = len(current)

            if pl == 1:
                last_l1 = n - 1
            elif pl == 2:
                last_l2 = n - 1
            elif pl == 3:
                last_l3 = n - 1

            if pl == 1 and n >= th.a:
                _flush_at(n - 1)
            elif n >= th.d:
                if last_l1 >= 0:
                    _flush_at(last_l1)
                elif last_l2 >= 0:
                    _flush_at(last_l2)
                elif last_l3 >= 0:
                    _flush_at(last_l3)
                else:
                    segments.append(current)
                    current = []
                    last_l1 = last_l2 = last_l3 = -1

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
        Remaining work is queued by group and driven as previous segments flush.
        """
        self._presplit_thresholds = self._make_thresholds()
        self._presplit_groups = deque([
            PendingGroup(i, seg_tokens) for i, seg_tokens in enumerate(self.pre_split(tokens))
        ])
        self._text_complete = True

        return self._drive_presplit_batch()

    def _drive_presplit_batch(self) -> List[SegmentAction]:
        """Drive as many queued offline groups as concurrency allows."""
        actions: List[SegmentAction] = []
        while self._presplit_groups and self.active_segment_count < self._max_concurrent:
            group = self._presplit_groups.popleft()
            group_actions, has_remaining = self._drive_group(group)
            actions.extend(group_actions)
            if has_remaining:
                self._presplit_groups.append(group)
        return actions

    def _drive_group(
        self,
        group: PendingGroup,
    ) -> tuple[List[SegmentAction], bool]:
        """Drive one offline group until it flushes or runs out of tokens.

        Returns ``(actions, has_remaining_tokens)``. A pre-split group may
        still yield multiple backend segments because the Driver keeps the
        final authority to flush inside the group.
        """
        # Recompute thresholds with the latest EMA before starting each new
        # offline segment so long pre-split queues can benefit from ratio
        # learning accumulated by earlier groups.
        thresholds = self._make_thresholds()
        self._presplit_thresholds = thresholds
        idx, driver = self._create_driver(thresholds)
        actions: List[SegmentAction] = []
        flushed = False
        local_idx = group.next_local_idx
        group.next_local_idx += 1

        start_evt = SpliterEvent(type=ET.START)
        for r in driver.feed(start_evt):
            actions.append(SegmentAction(idx, r, group.group_idx, local_idx, False))

        while group.cursor < len(group.tokens):
            token_id, text, pl = group.tokens[group.cursor]
            group.cursor += 1
            evt = self._make_event(token_id, text, pl)
            for r in driver.feed(evt):
                actions.append(SegmentAction(idx, r, group.group_idx, local_idx, False))
                if r.type in (ActionType.FLUSH_EOS, ActionType.FLUSH_NOP):
                    self._flushing.add(idx)
                    flushed = True
                    break
            if flushed:
                break

        if not flushed:
            end_evt = SpliterEvent(type=ET.END)
            for r in driver.feed(end_evt):
                actions.append(SegmentAction(idx, r, group.group_idx, local_idx, False))
                if r.type in (ActionType.FLUSH_EOS, ActionType.FLUSH_NOP):
                    self._flushing.add(idx)

        group_final = group.cursor >= len(group.tokens)
        for sa in actions:
            sa.group_final = group_final

        return actions, group.cursor < len(group.tokens)

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
        for i, (token_id, text, pl) in enumerate(self._token_buffer):
            evt = self._make_event(token_id, text, pl)
            results = driver.feed(evt)
            for r in results:
                actions.append(SegmentAction(idx, r))
                if r.type in (ActionType.FLUSH_EOS, ActionType.FLUSH_NOP):
                    self._flushing.add(idx)
                    remaining = self._token_buffer[i + 1:]
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

        if self._presplit_thresholds is not None:
            return self._drive_presplit_batch()

        if self._token_buffer or self._text_complete:
            return self._try_start_next()
        return []

    def update_ratio(
        self, actual_audio_steps: int, actual_text_tokens: int,
        *, overflow: bool = False,
    ) -> None:
        """Update EMA audio:text ratio from engine feedback.

        After EMA update, refreshes thresholds for all active (non-flushing)
        drivers so subsequent token classification uses the latest ratio.

        Parameters
        ----------
        overflow : bool
            When True, uses ``ema_overflow_alpha`` (default 0.5) for faster
            convergence after a KV cache overflow event.
        """
        if actual_text_tokens <= 0:
            return
        observed = actual_audio_steps / actual_text_tokens
        old_ratio = self._ema_ratio
        if overflow:
            alpha = self._ema_overflow_alpha
            self._ema_ratio = (1.0 - alpha) * self._ema_ratio + alpha * observed
            self._ema_ratio = max(self._ema_min_ratio, min(self._ema_max_ratio, self._ema_ratio))
            logger.warning(
                "EMA overflow update: observed=%.3f ema=%.3f→%.3f (steps=%d tokens=%d)",
                observed, old_ratio, self._ema_ratio, actual_audio_steps, actual_text_tokens,
            )
        else:
            alpha = self._ema_alpha
            self._ema_ratio = (1.0 - alpha) * self._ema_ratio + alpha * observed
            self._ema_ratio = max(self._ema_min_ratio, min(self._ema_max_ratio, self._ema_ratio))
            logger.debug(
                "EMA update: observed=%.3f ema=%.3f→%.3f (steps=%d tokens=%d)",
                observed, old_ratio, self._ema_ratio, actual_audio_steps, actual_text_tokens,
            )

        if abs(self._ema_ratio - old_ratio) > 0.01:
            self._refresh_active_thresholds()

    def _refresh_active_thresholds(self) -> None:
        """Recompute thresholds for all non-flushing active drivers.

        Called after EMA ratio changes so that drivers currently
        accumulating tokens use up-to-date split thresholds.
        """
        new_th = self._make_thresholds()
        refreshed = 0
        for idx, driver in self._drivers.items():
            if idx in self._flushing:
                continue
            driver.thresholds = new_th
            refreshed += 1
        if refreshed:
            logger.debug(
                "Refreshed thresholds for %d active driver(s): a=%d b=%d c=%d d=%d",
                refreshed, new_th.a, new_th.b, new_th.c, new_th.d,
            )

    # ------------------------------------------------------------------
    # Full reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._presplit_groups.clear()
        self._presplit_thresholds = None
        self._token_buffer.clear()
        self._text_complete = False
        self._drivers.clear()
        self._next_segment_idx = 0
        self._flushing.clear()
        self._done.clear()


__all__ = ("Spliter", "SegmentAction")

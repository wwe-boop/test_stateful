"""
EMA tracker for audio decode steps / text token ratio (Phase 2 orchestrator).

Used to estimate how many text tokens fit per segment under KV cache limits.
Supports per-speaker tracking via SpeakerRatioRegistry.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

logger = logging.getLogger("tts_orchestrator.ratio_tracker")


class RatioTracker:
    """EMA tracker for total_decode_steps / text_tokens_consumed."""

    def __init__(
        self,
        initial: float = 5.5,
        alpha: float = 0.1,
        overflow_alpha: float = 0.5,
        min_ratio: float = 2.0,
        max_ratio: float = 10.0,
    ) -> None:
        self.initial = float(initial)
        self.ema = float(initial)
        self.alpha = float(alpha)
        self.overflow_alpha = float(overflow_alpha)
        self.min_ratio = float(min_ratio)
        self.max_ratio = float(max_ratio)

    def _clamp(self, x: float) -> float:
        return max(self.min_ratio, min(self.max_ratio, x))

    def update(self, total_steps: int, text_tokens: int) -> None:
        """Called when a segment finishes normally (EOS)."""
        if text_tokens <= 0:
            return
        observed = float(total_steps) / float(text_tokens)
        self.ema = (1.0 - self.alpha) * self.ema + self.alpha * observed
        self.ema = self._clamp(self.ema)
        logger.debug(
            "RatioTracker.update: observed=%.3f ema=%.3f (steps=%d tokens=%d)",
            observed,
            self.ema,
            total_steps,
            text_tokens,
        )

    def update_overflow(self, total_steps: int, text_tokens: int) -> None:
        """Called when KV cache overflows — higher learning rate."""
        if text_tokens <= 0:
            observed = self.max_ratio
        else:
            observed = float(total_steps) / float(text_tokens)
        self.ema = (1.0 - self.overflow_alpha) * self.ema + self.overflow_alpha * observed
        self.ema = self._clamp(self.ema)
        logger.warning(
            "RatioTracker.update_overflow: observed=%.3f ema=%.3f (steps=%d tokens=%d)",
            observed,
            self.ema,
            total_steps,
            text_tokens,
        )

    def text_budget(self, remaining_kv: int) -> int:
        """How many text tokens can fit given remaining KV capacity (approx)."""
        if remaining_kv <= 0:
            return 1
        return max(1, int(remaining_kv / self.ema))

    def estimate_remaining_steps(
        self,
        remaining_text_tokens: int,
        session_ratio: float = 0.0,
        safety: float = 1.2,
    ) -> int:
        """Estimate total remaining decode steps for remaining_text_tokens."""
        if remaining_text_tokens <= 0:
            return 0
        effective = self.ema
        if session_ratio > 0:
            effective = max(self.ema, session_ratio * safety)
        return int(remaining_text_tokens * effective)


class SpeakerRatioRegistry:
    """Per-speaker EMA ratio tracking.

    Each unique speaker_key (e.g. "Chelsie_zh") gets its own RatioTracker.
    Cold-start speakers inherit the global tracker's current EMA.
    """

    def __init__(self, global_tracker: RatioTracker, max_speakers: int = 64) -> None:
        self._global = global_tracker
        self._max_speakers = max_speakers
        self._trackers: Dict[str, RatioTracker] = {}

    def get(self, speaker_key: Optional[str] = None) -> RatioTracker:
        """Return the tracker for *speaker_key*, creating one if needed."""
        if not speaker_key:
            return self._global
        if speaker_key in self._trackers:
            return self._trackers[speaker_key]
        if len(self._trackers) >= self._max_speakers:
            oldest = next(iter(self._trackers))
            del self._trackers[oldest]
        tracker = RatioTracker(
            initial=self._global.ema,
            alpha=self._global.alpha,
            overflow_alpha=self._global.overflow_alpha,
            min_ratio=self._global.min_ratio,
            max_ratio=self._global.max_ratio,
        )
        self._trackers[speaker_key] = tracker
        logger.info(
            "SpeakerRatioRegistry: created tracker for %s (initial=%.2f)",
            speaker_key,
            tracker.ema,
        )
        return tracker

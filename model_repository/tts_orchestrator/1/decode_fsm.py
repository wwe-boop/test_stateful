"""
Per-session decode finite state machine (Prefill -> Phase A/B -> KV rollback).

English comments only (project convention).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import List, Sequence, Tuple

from text_segmenter import PUNCT_LEVEL_1, PUNCT_LEVEL_2, PUNCT_LEVEL_3

logger = logging.getLogger("tts_orchestrator.decode_fsm")


class DecodePhase(Enum):
    PREFILL = "prefill"
    PHASE_A = "phase_a"
    PHASE_B = "phase_b"
    KV_ROLLBACK = "kv_rollback"
    SEG_DONE = "seg_done"
    DONE = "done"
    IDLE = "idle"
    ERROR = "error"


@dataclass
class PhaseThresholds:
    """Phase-A text-step thresholds (inclusive) for punctuation tiers."""

    a: int
    b: int
    c: int


class DecodeSessionFSM:
    """
    Tracks decode phase and Phase-A/B budgets for one session.
    Engine applies KV restore / segment activation based on phase transitions.
    """

    def __init__(
        self,
        engine_max_decode_len: int,
        rollover_margin: int,
        max_pad_steps: int,
        punct_near_c_margin: int = 8,
        ratio_audio_per_text: float = 5.0,
    ) -> None:
        self.engine_max_decode_len = int(engine_max_decode_len)
        self.rollover_margin = int(rollover_margin)
        self.max_pad_steps = int(max_pad_steps)
        self.punct_near_c_margin = int(punct_near_c_margin)
        self.ratio_audio_per_text = float(ratio_audio_per_text)
        self.phase: DecodePhase = DecodePhase.PREFILL
        self.steps_in_phase_a: int = 0
        self.thresholds: PhaseThresholds = PhaseThresholds(0, 0, 0)
        self.trailing_char_offsets: List[int] = []
        self.segment_text: str = ""
        self._pending_enter_phase_b: bool = False
        self.phase_b_eos_injected: bool = False

    def reset_for_new_segment(self) -> None:
        self.phase = DecodePhase.PREFILL
        self.steps_in_phase_a = 0
        self.thresholds = PhaseThresholds(0, 0, 0)
        self.trailing_char_offsets = []
        self.segment_text = ""
        self._pending_enter_phase_b = False
        self.phase_b_eos_injected = False

    def on_prefill_done(
        self,
        checkpoint_past_len: int,
        trailing_char_offsets: Sequence[int],
        segment_text: str,
        ema_ratio: float,
        trailing_len: int = 0,
    ) -> None:
        """After prefill + first fused step: enter Phase A with budgets."""
        self.segment_text = segment_text or ""
        self.trailing_char_offsets = list(trailing_char_offsets)
        self._pending_enter_phase_b = False
        self.steps_in_phase_a = 0
        self._recompute_thresholds(checkpoint_past_len, ema_ratio, trailing_len)
        self.phase = DecodePhase.PHASE_A
        logger.debug(
            "FSM prefill_done: past=%d thresholds=(%d,%d,%d) trailing=%d offsets=%d",
            checkpoint_past_len,
            self.thresholds.a,
            self.thresholds.b,
            self.thresholds.c,
            trailing_len,
            len(self.trailing_char_offsets),
        )

    def _recompute_thresholds(
        self,
        checkpoint_past_len: int,
        ema_ratio: float,
        trailing_len: int = 0,
    ) -> None:
        remaining = self.engine_max_decode_len - int(checkpoint_past_len)
        remaining = max(1, remaining - max(4, self.rollover_margin // 4))
        r = max(2.0, float(ema_ratio))
        denom = self.ratio_audio_per_text + 1.0
        phase_a_cap = max(8, int(remaining / denom))

        if trailing_len > 0 and phase_a_cap >= trailing_len:
            big = trailing_len + self.engine_max_decode_len
            a = big
            b = big
            c = big
        else:
            a = max(6, int(phase_a_cap * 0.35))
            b = max(a + 4, int(phase_a_cap * 0.65))
            c = max(b + 4, phase_a_cap)
            phase_b_cap = max(remaining - phase_a_cap, 1)
            c = min(c, phase_b_cap + phase_a_cap - 2)
            c = min(c, remaining - 2)
            if c < b + 1:
                c = min(b + 8, remaining - 1)
        self.thresholds = PhaseThresholds(a=a, b=b, c=max(b + 1, c))

    def restart_phase_a_after_extend(
        self,
        checkpoint_past_len: int,
        ema_ratio: float,
        trailing_len: int = 0,
    ) -> None:
        """New trailing arrived during Phase B (streaming); resume text injection."""
        self.phase = DecodePhase.PHASE_A
        self.steps_in_phase_a = 0
        self._pending_enter_phase_b = False
        self.phase_b_eos_injected = False
        self._recompute_thresholds(checkpoint_past_len, ema_ratio, trailing_len)

    def on_kv_rollback_done(
        self,
        checkpoint_past_len: int,
        ema_ratio: float,
        trailing_len: int = 0,
    ) -> None:
        """Restore checkpoint completed; start another Phase-A round."""
        self.steps_in_phase_a = 0
        self._pending_enter_phase_b = False
        self.phase_b_eos_injected = False
        self._recompute_thresholds(checkpoint_past_len, ema_ratio, trailing_len)
        self.phase = DecodePhase.PHASE_A
        logger.debug(
            "FSM rollback_done: thresholds=(%d,%d,%d)",
            self.thresholds.a,
            self.thresholds.b,
            self.thresholds.c,
        )

    def _char_suffix_after_token(self, text_idx: int, trailing_len: int) -> str:
        """Substring of segment_text covered by trailing tokens [text_idx, end)."""
        if not self.segment_text:
            return ""
        if text_idx < 0:
            text_idx = 0
        if not self.trailing_char_offsets:
            return ""
        if text_idx >= len(self.trailing_char_offsets):
            return ""
        start = self.trailing_char_offsets[text_idx]
        if text_idx + 1 < len(self.trailing_char_offsets):
            end = self.trailing_char_offsets[text_idx + 1]
        else:
            end = len(self.segment_text)
        end = min(end, len(self.segment_text))
        start = min(start, len(self.segment_text))
        if start > end:
            return ""
        return self.segment_text[start:end]

    @staticmethod
    def _punct_classes_in_suffix(suffix: str) -> Tuple[bool, bool, bool]:
        """Return (has_l1, has_l2, has_l3) for characters in suffix."""
        has1 = any(ch in PUNCT_LEVEL_1 for ch in suffix)
        has2 = any(ch in PUNCT_LEVEL_2 for ch in suffix)
        has3 = any(ch in PUNCT_LEVEL_3 for ch in suffix)
        return has1, has2, has3

    def should_enter_phase_b_after_consuming_token(
        self,
        text_idx_after_step: int,
        trailing_len: int,
    ) -> bool:
        """
        Called after a Phase-A decode step updated session.text_idx.

        Orchestrator convention: after prefill, text_idx=1 and trailing[0] is in
        first next_embed. When text_idx becomes k>=2, trailing[k-2] was just
        consumed by the completed step.
        """
        if self.phase != DecodePhase.PHASE_A:
            return False
        if text_idx_after_step < 2:
            return False
        last_consumed = text_idx_after_step - 2
        if last_consumed < 0:
            return False
        steps = self.steps_in_phase_a
        a, b, c = self.thresholds.a, self.thresholds.b, self.thresholds.c
        suffix = self._char_suffix_after_token(last_consumed, trailing_len)
        has1, has2, has3 = self._punct_classes_in_suffix(suffix)

        if steps >= c:
            return True
        if steps >= c - self.punct_near_c_margin and has3:
            return True
        if steps >= b and has2:
            return True
        if steps >= a and has1:
            return True
        if text_idx_after_step >= trailing_len:
            return True
        return False

    def note_phase_a_step(self) -> None:
        if self.phase == DecodePhase.PHASE_A:
            self.steps_in_phase_a += 1

    def enter_phase_b(self, reason: str = "") -> None:
        if self.phase == DecodePhase.PHASE_A:
            self.phase = DecodePhase.PHASE_B
            self.phase_b_eos_injected = False
            logger.info(
                "FSM enter Phase B: steps_a=%d thresholds=(%d,%d,%d) reason=%s",
                self.steps_in_phase_a,
                self.thresholds.a, self.thresholds.b, self.thresholds.c,
                reason or "threshold",
            )

    def remaining_kv_steps(self, past_len: int) -> int:
        return max(0, self.engine_max_decode_len - int(past_len))

    def streaming_should_idle(self, text_idx: int, trailing_len: int, text_complete: bool) -> bool:
        if text_complete:
            return False
        return text_idx >= trailing_len

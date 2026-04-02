"""
Per-session decode finite state machine.

States:
  IDLE → Prefill → SA (phase A: accumulate + threshold)
  SA → SB0 (mid-cut: inject EOS) or SA → SB1 (text-complete: direct PAD)
  SB0 → SB1 (pad & eval) → SB2 (overflow) → IDLE
  SAIdle (streaming: awaiting text)

See docs/decode_fsm_design.md for the full state diagram (Mermaid).

The FSM is a pure-logic decision engine.  It does NOT own text_idx or
GPU tensors — those stay in the session / orchestrator.  Each step the
orchestrator feeds an event and receives an action.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Sequence, Tuple

from text_segmenter import PUNCT_LEVEL_1, PUNCT_LEVEL_2, PUNCT_LEVEL_3

logger = logging.getLogger("tts_orchestrator.decode_fsm")


# ---------------------------------------------------------------------------
# States
# ---------------------------------------------------------------------------
class FSMState(Enum):
    S0  = "check_text"
    SP  = "prefill"
    SA  = "phase_a"
    SB0 = "inject_eos"
    SB1 = "pad_eval"
    SB2 = "overflow"
    E0  = "finished"


# ---------------------------------------------------------------------------
# Events fed into the FSM each decode step
# ---------------------------------------------------------------------------
@dataclass
class StepEvent:
    """One decode step's observable signals — provided by orchestrator."""
    text_idx: int = 0
    trailing_len: int = 0
    is_codec_eos: bool = False
    is_silent: bool = False
    past_len: int = 0
    frame_idx: int = 0
    token_punct: Tuple[bool, bool, bool] = (False, False, False)  # (L1, L2, L3)
    text_complete: bool = True


# ---------------------------------------------------------------------------
# Actions returned by the FSM
# ---------------------------------------------------------------------------
class TextAddKind(Enum):
    TRAILING = "trailing"       # inject next trailing text token
    EOS_EMBED = "eos_embed"     # inject tts_eos_embed
    PAD = "pad"                 # inject pad embedding
    NONE = "none"               # no text_add (prefill / finished)


@dataclass
class StepAction:
    """What the orchestrator should do after this FSM step."""
    text_add_kind: TextAddKind = TextAddKind.NONE
    emit_wav: bool = True
    end_segment: bool = False
    need_prefill: bool = False
    overflow: bool = False
    idle: bool = False


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
@dataclass
class PhaseThresholds:
    a: int   # L1 punct gate
    b: int   # L1+L2 punct gate
    c: int   # L1+L2+L3 punct gate
    d: int   # forced cut


# ---------------------------------------------------------------------------
# FSM
# ---------------------------------------------------------------------------
class DecodeSessionFSM:
    """
    Pure-logic state machine for one TTS decode session.

    The orchestrator calls step(event) each decode step.
    text_idx, trailing, GPU state remain in the orchestrator.
    """

    # Mid-cut with ≤ this many remaining tokens produces garbage audio.
    # The KV cache is polluted by Phase B PADs, and the model cannot
    # recover with so few real tokens — the second SB1 phase degrades
    # into hundreds of steps of non-speech noise.
    MIN_TAIL_FOR_CUT: int = 5

    # After this many SB1 pad steps, stop emitting audio.
    # Legitimate speech trailing finishes within ~50-150 steps;
    # anything beyond is likely degraded model output.
    DEFAULT_PAD_EMIT_CUTOFF: int = 200

    def __init__(
        self,
        engine_max_decode_len: int = 512,
        rollover_margin: int = 8,
        min_pad_steps: int = 4,
        max_pad_steps: int = 400,
        ratio_audio_per_text: float = 5.0,
        threshold_a_ratio: float = 0.70,
        threshold_b_ratio: float = 0.80,
        threshold_c_ratio: float = 0.90,
        pad_emit_cutoff: int = 0,
    ) -> None:
        self.engine_max_decode_len = int(engine_max_decode_len)
        self.rollover_margin = int(rollover_margin)
        self.min_pad_steps = int(min_pad_steps)
        self.max_pad_steps = int(max_pad_steps)
        self.ratio_audio_per_text = float(ratio_audio_per_text)
        self.threshold_a_ratio = float(threshold_a_ratio)
        self.threshold_b_ratio = float(threshold_b_ratio)
        self.threshold_c_ratio = float(threshold_c_ratio)
        self.pad_emit_cutoff = int(pad_emit_cutoff) if pad_emit_cutoff > 0 else self.DEFAULT_PAD_EMIT_CUTOFF

        self.state: FSMState = FSMState.S0
        self.steps_in_phase_a: int = 0
        self.thresholds: PhaseThresholds = PhaseThresholds(0, 0, 0, 0)
        self.phase_b_start_frame: int = 0
        self.pad_consecutive_silence: int = 0
        self.phase_b_eos_injected: bool = False
        self.checkpoint_past_len: int = 0

        # Char-offset + segment text for the punct lookup (set by orchestrator)
        self.trailing_char_offsets: List[int] = []
        self.segment_text: str = ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset for a new segment."""
        self.state = FSMState.S0
        self.steps_in_phase_a = 0
        self.thresholds = PhaseThresholds(0, 0, 0, 0)
        self.phase_b_start_frame = 0
        self.pad_consecutive_silence = 0
        self.phase_b_eos_injected = False
        self.trailing_char_offsets = []
        self.segment_text = ""

    def enter_phase_a(self, past_len: int, ema_ratio: float) -> None:
        """Transition to Phase A after prefill completes (SP → SA)."""
        self.compute_thresholds(past_len, ema_ratio)
        self.steps_in_phase_a = 0
        self.state = FSMState.SA

    def step(self, event: StepEvent) -> StepAction:
        """Drive the FSM forward by one decode step."""
        handler = _STATE_HANDLERS.get(self.state)
        if handler is None:
            return StepAction()
        return handler(self, event)

    # ------------------------------------------------------------------
    # Threshold computation
    # ------------------------------------------------------------------
    def compute_thresholds(self, past_len: int, ema_ratio: float) -> None:
        remaining = self.engine_max_decode_len - int(past_len)
        remaining = max(1, remaining - max(4, self.rollover_margin // 4))
        self.ratio_audio_per_text = float(ema_ratio)

        denom = max(1.0, self.ratio_audio_per_text)
        phase_a_cap = max(8, int(remaining / denom))

        a = max(6, int(phase_a_cap * self.threshold_a_ratio))
        b = max(a + 4, int(phase_a_cap * self.threshold_b_ratio))
        c = max(b + 4, int(phase_a_cap * self.threshold_c_ratio))
        d = max(c + 1, phase_a_cap)

        d = min(d, remaining - 2)
        c = min(c, d - 1)
        b = min(b, c - 1)
        a = min(a, b - 1)

        self.thresholds = PhaseThresholds(
            a=max(1, a), b=max(2, b), c=max(3, c), d=max(4, d),
        )
        self.checkpoint_past_len = int(past_len)

        logger.debug(
            "FSM thresholds: a=%d b=%d c=%d d=%d (past=%d remaining=%d cap=%d)",
            self.thresholds.a, self.thresholds.b, self.thresholds.c,
            self.thresholds.d, past_len, remaining, phase_a_cap,
        )

    # ------------------------------------------------------------------
    # Punctuation helpers
    # ------------------------------------------------------------------
    def token_punct_at(self, text_idx: int) -> Tuple[bool, bool, bool]:
        """Return (has_l1, has_l2, has_l3) for the token at text_idx."""
        if text_idx < 0 or not self.segment_text:
            return (False, False, False)
        if text_idx >= len(self.trailing_char_offsets):
            return (False, False, False)
        start = self.trailing_char_offsets[text_idx]
        if text_idx + 1 < len(self.trailing_char_offsets):
            end = self.trailing_char_offsets[text_idx + 1]
        else:
            end = len(self.segment_text)
        start = min(start, len(self.segment_text))
        end = min(end, len(self.segment_text))
        if start > end:
            return (False, False, False)
        suffix = self.segment_text[start:end]
        has1 = any(ch in PUNCT_LEVEL_1 for ch in suffix)
        has2 = any(ch in PUNCT_LEVEL_2 for ch in suffix)
        has3 = any(ch in PUNCT_LEVEL_3 for ch in suffix)
        return (has1, has2, has3)

    # ------------------------------------------------------------------
    # Dynamic silence threshold
    # ------------------------------------------------------------------
    @staticmethod
    def dynamic_silence_limit(remaining_kv: int) -> int:
        if remaining_kv > 100:
            return 12
        if remaining_kv > 50:
            return 6
        if remaining_kv > 20:
            return 3
        return 1


# ---------------------------------------------------------------------------
# State handlers
# ---------------------------------------------------------------------------

def _handle_s0(fsm: DecodeSessionFSM, event: StepEvent) -> StepAction:
    """S0: Check if there is text remaining to synthesize."""
    if event.text_idx < event.trailing_len:
        fsm.state = FSMState.SP
        return StepAction(
            text_add_kind=TextAddKind.NONE,
            emit_wav=False,
            need_prefill=True,
        )
    elif not event.text_complete:
        # Streaming: text not complete yet, go idle
        fsm.state = FSMState.S0
        return StepAction(
            text_add_kind=TextAddKind.NONE,
            emit_wav=False,
            idle=True,
        )
    else:
        fsm.state = FSMState.E0
        return StepAction(
            text_add_kind=TextAddKind.NONE,
            emit_wav=False,
            end_segment=True,
        )


def _handle_sp(fsm: DecodeSessionFSM, event: StepEvent) -> StepAction:
    """SP: Prefill done → compute thresholds and enter Phase A."""
    fsm.enter_phase_a(event.past_len, fsm.ratio_audio_per_text)
    return StepAction(
        text_add_kind=TextAddKind.TRAILING,
        emit_wav=True,
    )


def _handle_sa(fsm: DecodeSessionFSM, event: StepEvent) -> StepAction:
    """SA: Phase A — evaluate threshold after the last consumed token.

    The orchestrator has already consumed a token (text_idx incremented)
    and run one decode step.  We evaluate whether to cut into Phase B
    based on the token that was just *consumed* (text_idx - 1).
    """
    fsm.steps_in_phase_a += 1

    if event.is_codec_eos and event.text_idx < event.trailing_len:
        logger.debug(
            "Spurious codec EOS in Phase A at step %d, text_idx=%d/%d — ignoring",
            fsm.steps_in_phase_a, event.text_idx, event.trailing_len,
        )

    # SA3: threshold check
    should_cut = _should_cut(fsm, event)

    if should_cut:
        mid_cut = event.text_idx < event.trailing_len

        # Streaming: all trailing consumed but EOS not yet received.
        # Go IDLE — orchestrator preserves the clean KV cache and
        # resumes seamlessly when more text (or text_complete) arrives.
        if not mid_cut and not event.text_complete:
            logger.info(
                "Phase A → IDLE: steps=%d text_idx=%d/%d (streaming, awaiting text)",
                fsm.steps_in_phase_a, event.text_idx, event.trailing_len,
            )
            return StepAction(
                text_add_kind=TextAddKind.NONE,
                emit_wav=True,
                idle=True,
            )

        fsm.phase_b_start_frame = event.frame_idx
        fsm.pad_consecutive_silence = 0

        if mid_cut:
            # Mid-cut: text still remains after the cut point.
            # Inject EOS via SB0 so the model gets an end-of-segment
            # signal before the PAD phase begins.
            logger.info(
                "Phase A → SB0: steps=%d thresholds=(%d,%d,%d,%d) "
                "text_idx=%d/%d mid_cut=True",
                fsm.steps_in_phase_a,
                fsm.thresholds.a, fsm.thresholds.b,
                fsm.thresholds.c, fsm.thresholds.d,
                event.text_idx, event.trailing_len,
            )
            fsm.state = FSMState.SB0
            return StepAction(
                text_add_kind=TextAddKind.EOS_EMBED,
                emit_wav=True,
            )

        # Text complete (E already consumed in SA as trailing):
        # skip SB0, go directly to SB1 with PAD.
        logger.info(
            "Phase A → SB1: steps=%d text_idx=%d/%d (E consumed, direct PAD)",
            fsm.steps_in_phase_a, event.text_idx, event.trailing_len,
        )
        fsm.state = FSMState.SB1
        return StepAction(
            text_add_kind=TextAddKind.PAD,
            emit_wav=True,
        )

    # Stay in SA: orchestrator should consume next token
    return StepAction(
        text_add_kind=TextAddKind.TRAILING,
        emit_wav=True,
    )


def _should_cut(fsm: DecodeSessionFSM, event: StepEvent) -> bool:
    """SA3 logic: evaluate if Phase A should transition to Phase B.

    Key invariant: never mid-cut when the remaining tail is too small.
    A tiny tail (≤ MIN_TAIL_FOR_CUT tokens) after mid-cut causes the
    resumed Phase A → SB1 to produce hundreds of steps of garbage audio
    because the KV cache is polluted by Phase B's PAD tokens and the
    model cannot recover with so few real tokens.
    """
    steps = fsm.steps_in_phase_a
    a, b, c, d = (
        fsm.thresholds.a, fsm.thresholds.b,
        fsm.thresholds.c, fsm.thresholds.d,
    )

    if event.text_idx >= event.trailing_len:
        return True

    remaining_tokens = event.trailing_len - event.text_idx

    if steps >= d:
        if remaining_tokens > fsm.MIN_TAIL_FOR_CUT:
            return True
        kv_headroom = fsm.engine_max_decode_len - event.past_len
        tail_cost = remaining_tokens * max(2, int(fsm.ratio_audio_per_text + 1))
        if tail_cost > kv_headroom - 10:
            return True
        logger.debug(
            "Suppressed d-cut: remaining=%d tail_cost=%d headroom=%d",
            remaining_tokens, tail_cost, kv_headroom,
        )
        return False

    if remaining_tokens <= fsm.MIN_TAIL_FOR_CUT:
        has1, has2, has3 = event.token_punct
        would_cut = (
            (steps >= c and (has1 or has2 or has3))
            or (steps >= b and (has1 or has2))
            or (steps >= a and has1)
        )
        if would_cut:
            logger.info(
                "Suppressed tiny-tail cut: steps=%d text_idx=%d/%d "
                "remaining=%d thresholds=(%d,%d,%d,%d)",
                steps, event.text_idx, event.trailing_len,
                remaining_tokens, a, b, c, d,
            )
        return False

    has1, has2, has3 = event.token_punct

    if steps >= c and (has1 or has2 or has3):
        return True
    if steps >= b and (has1 or has2):
        return True
    if steps >= a and has1:
        return True

    return False


def _handle_sb0(fsm: DecodeSessionFSM, event: StepEvent) -> StepAction:
    """SB0: Mid-cut EOS step → SB1.

    The model just processed EOS_EMBED as input (injected by SA→SB0).
    The resulting wav is typically a signal tone, not meaningful speech —
    suppress it.  Begin the PAD phase (SB1) for natural speech trailing.
    """
    fsm.state = FSMState.SB1
    return StepAction(
        text_add_kind=TextAddKind.PAD,
        emit_wav=False,
    )


def _handle_sb1(fsm: DecodeSessionFSM, event: StepEvent) -> StepAction:
    """SB1: Pad & evaluate silence / EOS / KV overflow."""

    if event.is_silent:
        fsm.pad_consecutive_silence += 1
    else:
        fsm.pad_consecutive_silence = 0

    pad_steps = max(0, event.frame_idx - fsm.phase_b_start_frame)
    remaining_kv = max(0, fsm.engine_max_decode_len - event.past_len)

    # KV overflow → SB2
    if event.past_len >= fsm.engine_max_decode_len - 1:
        logger.warning(
            "SB1 → SB2 overflow: past_len=%d pad_steps=%d text_idx=%d/%d",
            event.past_len, pad_steps, event.text_idx, event.trailing_len,
        )
        fsm.state = FSMState.SB2
        return _handle_sb2(fsm, event)

    # Natural EOS → S0 (EOS frame audio is not meaningful speech, skip it)
    if event.is_codec_eos:
        logger.info(
            "SB1 natural EOS: pad_steps=%d text_idx=%d/%d",
            pad_steps, event.text_idx, event.trailing_len,
        )
        fsm.state = FSMState.S0
        return StepAction(
            text_add_kind=TextAddKind.NONE,
            emit_wav=False,
        )

    # Silence threshold → S0 (current frame is silence, don't emit it)
    silence_limit = fsm.dynamic_silence_limit(remaining_kv)
    pad_mature = pad_steps >= fsm.min_pad_steps
    if pad_mature and fsm.pad_consecutive_silence > silence_limit:
        logger.info(
            "SB1 silence abort: silence=%d limit=%d pad_steps=%d text_idx=%d/%d",
            fsm.pad_consecutive_silence, silence_limit,
            pad_steps, event.text_idx, event.trailing_len,
        )
        fsm.state = FSMState.S0
        return StepAction(
            text_add_kind=TextAddKind.NONE,
            emit_wav=False,
        )

    # Pad timeout → S0 (exceeded budget, current frame not useful)
    if pad_steps > fsm.max_pad_steps:
        logger.warning(
            "SB1 pad timeout: pad_steps=%d max=%d text_idx=%d/%d",
            pad_steps, fsm.max_pad_steps, event.text_idx, event.trailing_len,
        )
        fsm.state = FSMState.S0
        return StepAction(
            text_add_kind=TextAddKind.NONE,
            emit_wav=False,
        )

    # Continue padding — suppress audio after cutoff to limit garbage
    emit = pad_steps < fsm.pad_emit_cutoff
    if not emit and pad_steps == fsm.pad_emit_cutoff:
        logger.info(
            "SB1 emit cutoff: pad_steps=%d cutoff=%d text_idx=%d/%d — audio muted",
            pad_steps, fsm.pad_emit_cutoff, event.text_idx, event.trailing_len,
        )
    return StepAction(
        text_add_kind=TextAddKind.PAD,
        emit_wav=emit,
    )


def _handle_sb2(fsm: DecodeSessionFSM, event: StepEvent) -> StepAction:
    """SB2: KV overflow handler."""
    logger.warning(
        "SB2 overflow: past_len=%d text_idx=%d/%d",
        event.past_len, event.text_idx, event.trailing_len,
    )
    fsm.state = FSMState.S0
    return StepAction(
        text_add_kind=TextAddKind.NONE,
        emit_wav=False,
        overflow=True,
    )


def _handle_e0(fsm: DecodeSessionFSM, event: StepEvent) -> StepAction:
    """E0: Finished — no-op."""
    return StepAction(
        text_add_kind=TextAddKind.NONE,
        emit_wav=False,
        end_segment=True,
    )


_STATE_HANDLERS = {
    FSMState.S0:  _handle_s0,
    FSMState.SP:  _handle_sp,
    FSMState.SA:  _handle_sa,
    FSMState.SB0: _handle_sb0,
    FSMState.SB1: _handle_sb1,
    FSMState.SB2: _handle_sb2,
    FSMState.E0:  _handle_e0,
}


# ---------------------------------------------------------------------------
# Legacy aliases for backward compatibility during migration
# ---------------------------------------------------------------------------
class DecodePhase(Enum):
    PREFILL = "prefill"
    PHASE_A = "phase_a"
    PHASE_B = "phase_b"
    KV_ROLLBACK = "kv_rollback"
    SEG_DONE = "seg_done"
    DONE = "done"
    IDLE = "idle"
    ERROR = "error"

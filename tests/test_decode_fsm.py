"""
Offline tests for DecodeSessionFSM.

Drives the FSM with simulated token sequences to verify state transitions
without needing TRT engine or Triton runtime.

Run: python -m pytest tests/test_decode_fsm.py -v
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(__file__), "..", "model_repository", "tts_orchestrator", "1",
))

from decode_fsm import (
    DecodeSessionFSM, FSMState, StepEvent, StepAction, TextAddKind,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_fsm(
    engine_max: int = 512,
    ratio: float = 5.0,
    max_pad: int = 400,
    min_pad: int = 4,
) -> DecodeSessionFSM:
    return DecodeSessionFSM(
        engine_max_decode_len=engine_max,
        ratio_audio_per_text=ratio,
        max_pad_steps=max_pad,
        min_pad_steps=min_pad,
    )


def build_char_offsets(text: str) -> list:
    """1 char = 1 token for simplicity."""
    return list(range(len(text)))


def simulate_full_synthesis(
    fsm: DecodeSessionFSM,
    text: str,
    silence_after_eos: int = 5,
    natural_eos_at_pad_step: int = 10,
    max_total_steps: int = 2000,
) -> dict:
    """Simulate a complete synthesis run.

    The orchestrator loop mirrors what model.py would do:
      1. Ask FSM for action via step(event)
      2. Execute action (consume token / inject eos / pad)
      3. Advance frame/past_len
    """
    offsets = build_char_offsets(text)
    trailing_len = len(text)
    text_idx = 0

    fsm.reset()
    fsm.trailing_char_offsets = offsets
    fsm.segment_text = text

    stats = {
        "tokens_consumed": 0,
        "pad_steps": 0,
        "phase_a_rounds": 0,
        "phase_b_rounds": 0,
        "overflows": 0,
        "states_visited": [],
    }

    frame_idx = 0
    past_len = 12   # simulate prefill length
    phase_b_local_pad = 0
    prev_state = fsm.state

    for _ in range(max_total_steps):
        # Build event with current observables
        token_punct = (False, False, False)
        if text_idx > 0 and fsm.state == FSMState.SA:
            token_punct = fsm.token_punct_at(text_idx - 1)

        is_silent = False
        is_eos = False
        if fsm.state == FSMState.SB1:
            phase_b_local_pad += 1
            if phase_b_local_pad >= natural_eos_at_pad_step:
                is_eos = True
            elif phase_b_local_pad >= natural_eos_at_pad_step - silence_after_eos:
                is_silent = True

        event = StepEvent(
            text_idx=text_idx,
            trailing_len=trailing_len,
            is_codec_eos=is_eos,
            is_silent=is_silent,
            past_len=past_len,
            frame_idx=frame_idx,
            token_punct=token_punct,
            text_complete=True,
        )

        action = fsm.step(event)
        stats["states_visited"].append(fsm.state.value)

        # Track transitions
        if fsm.state == FSMState.SA and prev_state != FSMState.SA:
            stats["phase_a_rounds"] += 1
        if fsm.state == FSMState.SB0:
            stats["phase_b_rounds"] += 1
            phase_b_local_pad = 0

        if action.text_add_kind == TextAddKind.TRAILING:
            text_idx += 1
            stats["tokens_consumed"] += 1
        elif action.text_add_kind == TextAddKind.PAD:
            stats["pad_steps"] += 1

        if action.overflow:
            stats["overflows"] += 1

        if action.end_segment:
            break

        # S0 re-entry: if more text, orchestrator does NOT re-prefill in
        # multi-round mode (KV continues), so directly enter_phase_a.
        if fsm.state == FSMState.S0 and text_idx < trailing_len:
            fsm.enter_phase_a(past_len, fsm.ratio_audio_per_text)
        elif fsm.state == FSMState.S0 and text_idx >= trailing_len:
            # All text consumed, ask FSM to finalize
            final_event = StepEvent(
                text_idx=text_idx, trailing_len=trailing_len,
                past_len=past_len, frame_idx=frame_idx, text_complete=True,
            )
            action = fsm.step(final_event)
            if action.end_segment:
                break

        # SP: orchestrator did prefill, enter phase A
        if fsm.state == FSMState.SP:
            fsm.enter_phase_a(past_len, fsm.ratio_audio_per_text)

        prev_state = fsm.state
        past_len += 1
        frame_idx += 1

    return stats


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFSMInitialState:
    def test_initial_state_is_s0(self):
        fsm = make_fsm()
        fsm.reset()
        assert fsm.state == FSMState.S0

    def test_no_text_goes_to_e0(self):
        fsm = make_fsm()
        fsm.reset()
        action = fsm.step(StepEvent(text_idx=0, trailing_len=0, text_complete=True))
        assert fsm.state == FSMState.E0
        assert action.end_segment

    def test_streaming_idle(self):
        fsm = make_fsm()
        fsm.reset()
        action = fsm.step(StepEvent(text_idx=0, trailing_len=0, text_complete=False))
        assert action.idle
        assert fsm.state == FSMState.S0


class TestS0ToSP:
    def test_has_text_triggers_prefill(self):
        fsm = make_fsm()
        fsm.reset()
        action = fsm.step(StepEvent(text_idx=0, trailing_len=5, text_complete=True))
        assert fsm.state == FSMState.SP
        assert action.need_prefill

    def test_sp_enters_sa(self):
        fsm = make_fsm()
        fsm.reset()
        fsm.step(StepEvent(text_idx=0, trailing_len=5))  # S0 → SP
        fsm.enter_phase_a(past_len=12, ema_ratio=5.0)
        assert fsm.state == FSMState.SA


class TestPhaseAThresholdCuts:
    """Verify that Phase A cuts at the right punctuation level."""

    def test_l1_cut_after_threshold_a(self):
        # engine_max=100, ratio=5.0 → thresholds: a≈9
        # "。" at position 9, remaining after cut = 10 > MIN_TAIL(5) → mid-cut
        text = "一二三四五六七八九。后续文本还有更多的内容"
        fsm = make_fsm(engine_max=100, ratio=5.0)
        offsets = build_char_offsets(text)
        fsm.reset()
        fsm.trailing_char_offsets = offsets
        fsm.segment_text = text

        fsm.step(StepEvent(text_idx=0, trailing_len=len(text)))
        fsm.enter_phase_a(past_len=12, ema_ratio=5.0)

        text_idx = 0
        found_sb0 = False
        for i in range(100):
            token_punct = fsm.token_punct_at(text_idx - 1) if text_idx > 0 else (False, False, False)
            event = StepEvent(
                text_idx=text_idx, trailing_len=len(text),
                past_len=12 + i, frame_idx=i, token_punct=token_punct,
            )
            action = fsm.step(event)
            if action.text_add_kind == TextAddKind.TRAILING:
                text_idx += 1
            if fsm.state == FSMState.SB0:
                found_sb0 = True
                cut_text = text[:text_idx]
                assert "。" in cut_text, f"Should cut at L1 punct, got: {cut_text!r}"
                break

        assert found_sb0, "Should have entered SB0"

    def test_l2_cut_after_threshold_b(self):
        """Text with L2 (，) at a position reachable between threshold b and d, no L1 before it."""
        # engine_max=100, ratio=5.0 → thresholds: a≈9, b≈13
        # "，" at position 13, remaining after cut = 8 > MIN_TAIL(5) → mid-cut
        text = "一二三四五六七八九十壹贰叁，后续文本还有更多内容"
        fsm = make_fsm(engine_max=100, ratio=5.0)
        offsets = build_char_offsets(text)
        fsm.reset()
        fsm.trailing_char_offsets = offsets
        fsm.segment_text = text

        fsm.step(StepEvent(text_idx=0, trailing_len=len(text)))
        fsm.enter_phase_a(past_len=12, ema_ratio=5.0)

        text_idx = 0
        cut_at = None
        for i in range(80):
            token_punct = fsm.token_punct_at(text_idx - 1) if text_idx > 0 else (False, False, False)
            event = StepEvent(
                text_idx=text_idx, trailing_len=len(text),
                past_len=12 + i, frame_idx=i, token_punct=token_punct,
            )
            action = fsm.step(event)
            if action.text_add_kind == TextAddKind.TRAILING:
                text_idx += 1
            if fsm.state == FSMState.SB0:
                cut_at = text_idx
                break

        assert cut_at is not None, "Should have cut"
        cut_text = text[:cut_at]
        assert "，" in cut_text, f"Should include L2 punct, got: {cut_text!r}"

    def test_forced_cut_at_d(self):
        text = "一" * 200
        fsm = make_fsm(engine_max=100, ratio=3.0)
        offsets = build_char_offsets(text)
        fsm.reset()
        fsm.trailing_char_offsets = offsets
        fsm.segment_text = text

        fsm.step(StepEvent(text_idx=0, trailing_len=len(text)))
        fsm.enter_phase_a(past_len=12, ema_ratio=3.0)

        text_idx = 0
        for i in range(80):
            event = StepEvent(
                text_idx=text_idx, trailing_len=len(text),
                past_len=12 + i, frame_idx=i,
            )
            action = fsm.step(event)
            if action.text_add_kind == TextAddKind.TRAILING:
                text_idx += 1
            if fsm.state == FSMState.SB0:
                assert fsm.steps_in_phase_a <= fsm.thresholds.d + 1
                return

        pytest.fail("Should have force-cut at threshold d")


class TestPhaseBNaturalEOS:
    def test_natural_eos_returns_to_s0(self):
        # Short text (3 chars): all consumed naturally → SB1 (direct PAD)
        fsm = make_fsm(engine_max=200, ratio=5.0)
        fsm.reset()
        fsm.trailing_char_offsets = build_char_offsets("你好。")
        fsm.segment_text = "你好。"

        fsm.step(StepEvent(text_idx=0, trailing_len=3))  # S0 → SP
        fsm.enter_phase_a(past_len=12, ema_ratio=5.0)

        text_idx = 0
        frame = 0
        while fsm.state == FSMState.SA:
            token_punct = fsm.token_punct_at(text_idx - 1) if text_idx > 0 else (False, False, False)
            action = fsm.step(StepEvent(
                text_idx=text_idx, trailing_len=3,
                past_len=12 + frame, frame_idx=frame, token_punct=token_punct,
            ))
            if action.text_add_kind == TextAddKind.TRAILING:
                text_idx += 1
            frame += 1

        assert fsm.state == FSMState.SB1, f"Expected SB1, got {fsm.state}"
        assert text_idx == 3, "All text should be consumed"

        # SB1 with codec EOS → S0
        action = fsm.step(StepEvent(
            past_len=12 + frame, frame_idx=frame,
            text_idx=text_idx, trailing_len=3,
            is_codec_eos=True,
        ))
        assert fsm.state == FSMState.S0


class TestPhaseBSilenceAbort:
    def test_silence_triggers_s0(self):
        fsm = make_fsm(engine_max=512, ratio=5.0, min_pad=2)
        fsm.reset()
        fsm.trailing_char_offsets = build_char_offsets("测试。")
        fsm.segment_text = "测试。"

        # Drive to SB1
        fsm.step(StepEvent(text_idx=0, trailing_len=3))
        fsm.enter_phase_a(past_len=12, ema_ratio=5.0)

        text_idx = 0
        frame = 0
        while fsm.state == FSMState.SA:
            token_punct = fsm.token_punct_at(text_idx - 1) if text_idx > 0 else (False, False, False)
            action = fsm.step(StepEvent(
                text_idx=text_idx, trailing_len=3,
                past_len=12 + frame, frame_idx=frame, token_punct=token_punct,
            ))
            if action.text_add_kind == TextAddKind.TRAILING:
                text_idx += 1
            frame += 1

        if fsm.state == FSMState.SB0:
            fsm.step(StepEvent(past_len=12 + frame, frame_idx=frame, text_idx=text_idx, trailing_len=3))
            frame += 1

        assert fsm.state == FSMState.SB1

        for _ in range(200):
            frame += 1
            action = fsm.step(StepEvent(
                past_len=12 + frame, frame_idx=frame,
                text_idx=text_idx, trailing_len=3,
                is_silent=True,
            ))
            if fsm.state == FSMState.S0:
                break

        assert fsm.state == FSMState.S0


class TestKVOverflow:
    def test_overflow_triggers_sb2(self):
        fsm = make_fsm(engine_max=30, ratio=2.0, max_pad=100)
        fsm.reset()
        text = "一" * 100
        fsm.trailing_char_offsets = build_char_offsets(text)
        fsm.segment_text = text

        fsm.step(StepEvent(text_idx=0, trailing_len=100))
        fsm.enter_phase_a(past_len=12, ema_ratio=2.0)

        text_idx = 0
        overflow_seen = False
        for i in range(100):
            past = 12 + i
            event = StepEvent(
                text_idx=text_idx, trailing_len=100,
                past_len=past, frame_idx=i,
            )
            action = fsm.step(event)
            if action.text_add_kind == TextAddKind.TRAILING:
                text_idx += 1
            if action.overflow:
                overflow_seen = True
                break
            if fsm.state == FSMState.E0:
                break

        assert overflow_seen


class TestMultiRoundAB:
    """Verify multiple A→B→S0→SA cycles consume all text."""

    def test_multi_round_consumes_all_text(self):
        text = "这是第一句话。这是第二句话。这是第三句话。"
        stats = simulate_full_synthesis(
            make_fsm(engine_max=512, ratio=5.0), text,
            natural_eos_at_pad_step=8,
        )
        assert stats["tokens_consumed"] == len(text), (
            f"All text should be consumed: got {stats['tokens_consumed']}/{len(text)}"
        )

    def test_no_text_loss_long_story(self):
        # Use engine_max=100 to force mid-cuts on long text
        text = (
            "在一个遥远的王国里，有一位美丽的公主。"
            "她有一双明亮的眼睛和一头乌黑的长发。"
            "传说她每次落下的眼泪会化作一颗颗晶莹剔透的钻石，价值连城。"
        )
        stats = simulate_full_synthesis(
            make_fsm(engine_max=100, ratio=5.0), text,
            natural_eos_at_pad_step=8,
        )
        assert stats["tokens_consumed"] == len(text), (
            f"All text consumed: {stats['tokens_consumed']}/{len(text)}"
        )

    def test_very_short_text_single_round(self):
        text = "你好"
        stats = simulate_full_synthesis(
            make_fsm(engine_max=512, ratio=5.0), text,
            natural_eos_at_pad_step=5,
        )
        assert stats["tokens_consumed"] == len(text)

    def test_only_punctuation(self):
        text = "。！？"
        stats = simulate_full_synthesis(
            make_fsm(engine_max=512, ratio=5.0), text,
            natural_eos_at_pad_step=5,
        )
        assert stats["tokens_consumed"] == len(text)


class TestSpuriousEOSInPhaseA:
    def test_spurious_eos_ignored_in_phase_a(self):
        fsm = make_fsm(engine_max=200, ratio=5.0)
        fsm.reset()
        text = "一二三四五六七八"
        fsm.trailing_char_offsets = build_char_offsets(text)
        fsm.segment_text = text

        fsm.step(StepEvent(text_idx=0, trailing_len=len(text)))
        fsm.enter_phase_a(past_len=12, ema_ratio=5.0)

        action = fsm.step(StepEvent(
            text_idx=0, trailing_len=len(text),
            past_len=13, frame_idx=1, is_codec_eos=True,
        ))
        assert fsm.state == FSMState.SA, "Spurious EOS in Phase A should stay in SA"


class TestFullSegmentCycle:
    def test_complete_cycle(self):
        text = "人工智能正在改变世界。从语音识别到自然语言处理，技术日新月异。"
        stats = simulate_full_synthesis(
            make_fsm(engine_max=512, ratio=5.0), text,
            natural_eos_at_pad_step=6,
        )
        assert stats["tokens_consumed"] == len(text)

    def test_all_punct_levels(self):
        text = "甲。乙，丙——丁！戊；己…庚？"
        stats = simulate_full_synthesis(
            make_fsm(engine_max=512, ratio=5.0), text,
            natural_eos_at_pad_step=6,
        )
        assert stats["tokens_consumed"] == len(text)


class TestThresholdComputation:
    def test_thresholds_ordering(self):
        fsm = make_fsm(engine_max=512, ratio=5.0)
        fsm.compute_thresholds(past_len=12, ema_ratio=5.0)
        t = fsm.thresholds
        assert t.a < t.b < t.c < t.d, f"a={t.a} b={t.b} c={t.c} d={t.d}"

    def test_thresholds_with_small_remaining(self):
        fsm = make_fsm(engine_max=50, ratio=5.0)
        fsm.compute_thresholds(past_len=30, ema_ratio=5.0)
        t = fsm.thresholds
        assert t.a >= 1
        assert t.d <= 50 - 30

    def test_thresholds_with_different_ratios(self):
        for ratio in [2.0, 3.0, 5.0, 8.0, 10.0]:
            fsm = make_fsm(engine_max=512, ratio=ratio)
            fsm.compute_thresholds(past_len=12, ema_ratio=ratio)
            t = fsm.thresholds
            assert t.a < t.b < t.c <= t.d, (
                f"ratio={ratio}: a={t.a} b={t.b} c={t.c} d={t.d}"
            )


class TestDynamicSilenceLimit:
    def test_more_remaining_more_patience(self):
        assert DecodeSessionFSM.dynamic_silence_limit(200) > DecodeSessionFSM.dynamic_silence_limit(10)

    def test_critical_remaining(self):
        assert DecodeSessionFSM.dynamic_silence_limit(5) == 1


class TestTinyTailSuppression:
    """Verify that mid-cuts are suppressed when remaining tokens ≤ MIN_TAIL_FOR_CUT."""

    def test_tiny_tail_suppressed(self):
        # engine_max=100, ratio=5.0 → a≈9.
        # "。" at position 9, but only 2 chars after (remaining=2 ≤ 5) → suppressed
        text = "一二三四五六七八九。后续"
        fsm = make_fsm(engine_max=100, ratio=5.0)
        offsets = build_char_offsets(text)
        fsm.reset()
        fsm.trailing_char_offsets = offsets
        fsm.segment_text = text

        fsm.step(StepEvent(text_idx=0, trailing_len=len(text)))
        fsm.enter_phase_a(past_len=12, ema_ratio=5.0)

        text_idx = 0
        for i in range(50):
            token_punct = fsm.token_punct_at(text_idx - 1) if text_idx > 0 else (False, False, False)
            event = StepEvent(
                text_idx=text_idx, trailing_len=len(text),
                past_len=12 + i, frame_idx=i, token_punct=token_punct,
            )
            action = fsm.step(event)
            if action.text_add_kind == TextAddKind.TRAILING:
                text_idx += 1
            if fsm.state == FSMState.SB0:
                pytest.fail(
                    f"Should NOT mid-cut with remaining={len(text) - text_idx}: "
                    f"tiny tail would produce garbage audio"
                )
            if fsm.state == FSMState.SB1:
                # Direct PAD (natural end) — this is correct
                assert text_idx == len(text), "Should consume all text before SB1"
                return

        pytest.fail("Should have entered SB1 (direct PAD)")

    def test_sufficient_tail_allows_cut(self):
        # Same thresholds, but text has 10 chars after "。" → remaining=10 > 5 → cut allowed
        text = "一二三四五六七八九。后续文本还有更多的内容"
        fsm = make_fsm(engine_max=100, ratio=5.0)
        offsets = build_char_offsets(text)
        fsm.reset()
        fsm.trailing_char_offsets = offsets
        fsm.segment_text = text

        fsm.step(StepEvent(text_idx=0, trailing_len=len(text)))
        fsm.enter_phase_a(past_len=12, ema_ratio=5.0)

        text_idx = 0
        for i in range(50):
            token_punct = fsm.token_punct_at(text_idx - 1) if text_idx > 0 else (False, False, False)
            event = StepEvent(
                text_idx=text_idx, trailing_len=len(text),
                past_len=12 + i, frame_idx=i, token_punct=token_punct,
            )
            action = fsm.step(event)
            if action.text_add_kind == TextAddKind.TRAILING:
                text_idx += 1
            if fsm.state == FSMState.SB0:
                remaining = len(text) - text_idx
                assert remaining > fsm.MIN_TAIL_FOR_CUT, (
                    f"Cut with remaining={remaining} should be > {fsm.MIN_TAIL_FOR_CUT}"
                )
                return

        pytest.fail("Should have entered SB0")


class TestSB1EmitCutoff:
    """Verify that SB1 stops emitting audio after pad_emit_cutoff steps."""

    def test_emit_suppressed_after_cutoff(self):
        fsm = make_fsm(engine_max=512, ratio=5.0, max_pad=400)
        fsm.pad_emit_cutoff = 50
        fsm.reset()
        fsm.state = FSMState.SB1
        fsm.phase_b_start_frame = 0

        # Before cutoff: emit_wav=True
        action = fsm.step(StepEvent(
            past_len=100, frame_idx=30,
            text_idx=10, trailing_len=10,
        ))
        assert action.emit_wav is True, "Before cutoff, should emit audio"

        # At cutoff: emit_wav=False
        action = fsm.step(StepEvent(
            past_len=150, frame_idx=50,
            text_idx=10, trailing_len=10,
        ))
        assert action.emit_wav is False, "At cutoff, should suppress audio"

        # After cutoff: still emit_wav=False
        action = fsm.step(StepEvent(
            past_len=200, frame_idx=100,
            text_idx=10, trailing_len=10,
        ))
        assert action.emit_wav is False, "After cutoff, should suppress audio"

    def test_eos_still_terminates_after_cutoff(self):
        fsm = make_fsm(engine_max=512, ratio=5.0)
        fsm.pad_emit_cutoff = 30
        fsm.reset()
        fsm.state = FSMState.SB1
        fsm.phase_b_start_frame = 0

        action = fsm.step(StepEvent(
            past_len=200, frame_idx=100,
            text_idx=10, trailing_len=10,
            is_codec_eos=True,
        ))
        assert fsm.state == FSMState.S0, "EOS should still terminate even after cutoff"


class TestPadStepsCalculation:
    """Verify pad_steps reset correctly per Phase B round."""

    def test_pad_steps_small_per_round(self):
        """Long text with multiple L1 punctuation ensures multiple Phase B rounds.

        With default thresholds (a=70%, b=80%, c=90%), phase_a_cap≈83 for
        engine_max=512, ratio=5.0, past=12.  a≈58.  Text needs >58 chars
        with L1 punct distributed such that multiple cuts are triggered.
        Use engine_max=200 to shrink phase_a_cap and trigger more rounds.
        """
        text = ("这是一段比较长的测试文本用于验证多轮切分。"
                "第二个句子在这里需要继续合成。"
                "第三个句子还要更长一些确保超过阈值。"
                "第四个句子保证足够多的Phase B轮次。")
        fsm = make_fsm(engine_max=200, ratio=5.0, min_pad=2)
        offsets = build_char_offsets(text)
        fsm.reset()
        fsm.trailing_char_offsets = offsets
        fsm.segment_text = text

        fsm.step(StepEvent(text_idx=0, trailing_len=len(text)))
        fsm.enter_phase_a(past_len=12, ema_ratio=5.0)

        text_idx = 0
        frame = 0
        phase_b_start_frames = []
        for _ in range(500):
            token_punct = fsm.token_punct_at(text_idx - 1) if text_idx > 0 and fsm.state == FSMState.SA else (False, False, False)
            is_eos = False
            is_silent = False

            if fsm.state == FSMState.SB1:
                pad_steps = frame - fsm.phase_b_start_frame
                if pad_steps > 3:
                    is_eos = True

            event = StepEvent(
                text_idx=text_idx, trailing_len=len(text),
                past_len=12 + frame, frame_idx=frame,
                token_punct=token_punct, is_codec_eos=is_eos, is_silent=is_silent,
            )
            action = fsm.step(event)

            if action.text_add_kind == TextAddKind.TRAILING:
                text_idx += 1

            if fsm.state == FSMState.SB0:
                phase_b_start_frames.append(fsm.phase_b_start_frame)

            if fsm.state == FSMState.S0 and text_idx < len(text):
                fsm.enter_phase_a(12 + frame, fsm.ratio_audio_per_text)
            elif fsm.state == FSMState.S0 and text_idx >= len(text):
                action = fsm.step(StepEvent(
                    text_idx=text_idx, trailing_len=len(text),
                    past_len=12 + frame, frame_idx=frame, text_complete=True,
                ))
                if action.end_segment:
                    break

            if fsm.state == FSMState.E0:
                break
            frame += 1

        assert text_idx == len(text), f"All text consumed: {text_idx}/{len(text)}"
        assert len(phase_b_start_frames) >= 2, "Should have multiple Phase B rounds"
        for i, f in enumerate(phase_b_start_frames[1:], 1):
            assert f > phase_b_start_frames[i - 1], (
                f"Phase B start frames should increase: {phase_b_start_frames}"
            )


class TestTokenPunctAt:
    def test_l1_detection(self):
        fsm = make_fsm()
        fsm.segment_text = "你好。世界"
        fsm.trailing_char_offsets = [0, 1, 2, 3, 4]
        h1, h2, h3 = fsm.token_punct_at(2)  # "。"
        assert h1 is True

    def test_l2_detection(self):
        fsm = make_fsm()
        fsm.segment_text = "你好，世界"
        fsm.trailing_char_offsets = [0, 1, 2, 3, 4]
        h1, h2, h3 = fsm.token_punct_at(2)  # "，"
        assert h2 is True
        assert h1 is False

    def test_l3_detection(self):
        fsm = make_fsm()
        fsm.segment_text = "你好—世界"
        fsm.trailing_char_offsets = [0, 1, 2, 3, 4]
        h1, h2, h3 = fsm.token_punct_at(2)  # "—"
        assert h3 is True
        assert h1 is False
        assert h2 is False

    def test_out_of_range(self):
        fsm = make_fsm()
        fsm.segment_text = "你好"
        fsm.trailing_char_offsets = [0, 1]
        assert fsm.token_punct_at(5) == (False, False, False)
        assert fsm.token_punct_at(-1) == (False, False, False)


class TestStreamingEOSInjection:
    """Verify EOS_EMBED injection at SA→SB0 for mid-cuts."""

    def test_streaming_all_consumed_goes_idle(self):
        """Streaming with text_complete=False: after all trailing consumed,
        FSM stays in SA with idle=True (no mid-cut, no SB0)."""
        fsm = make_fsm(engine_max=200, ratio=5.0)
        text = "你好"
        offsets = build_char_offsets(text)
        fsm.reset()
        fsm.trailing_char_offsets = offsets
        fsm.segment_text = text

        fsm.step(StepEvent(text_idx=0, trailing_len=len(text)))
        fsm.enter_phase_a(past_len=12, ema_ratio=5.0)

        text_idx = 0
        idle_reached = False
        for i in range(100):
            token_punct = (
                fsm.token_punct_at(text_idx - 1)
                if text_idx > 0
                else (False, False, False)
            )
            event = StepEvent(
                text_idx=text_idx,
                trailing_len=len(text),
                past_len=12 + i,
                frame_idx=i,
                token_punct=token_punct,
                text_complete=False,
            )
            action = fsm.step(event)
            if action.text_add_kind == TextAddKind.TRAILING:
                text_idx += 1
            if action.idle:
                idle_reached = True
                break

        assert idle_reached, "All text consumed with text_complete=False → IDLE"
        assert text_idx == len(text), "All text should be consumed"

    def test_mid_cut_also_injects_eos_embed(self):
        """Mid-cut (unconsumed trailing remains) should also inject EOS_EMBED."""
        text = "一二三四五六七八九十壹贰叁肆伍。后续文本还有非常多内容需要合成的部分"
        fsm = make_fsm(engine_max=100, ratio=3.0)
        offsets = build_char_offsets(text)
        fsm.reset()
        fsm.trailing_char_offsets = offsets
        fsm.segment_text = text

        fsm.step(StepEvent(text_idx=0, trailing_len=len(text)))
        fsm.enter_phase_a(past_len=12, ema_ratio=3.0)

        text_idx = 0
        for i in range(100):
            token_punct = (
                fsm.token_punct_at(text_idx - 1)
                if text_idx > 0
                else (False, False, False)
            )
            event = StepEvent(
                text_idx=text_idx,
                trailing_len=len(text),
                past_len=12 + i,
                frame_idx=i,
                token_punct=token_punct,
            )
            action = fsm.step(event)
            if action.text_add_kind == TextAddKind.TRAILING:
                text_idx += 1
            if fsm.state == FSMState.SB0:
                assert action.text_add_kind == TextAddKind.EOS_EMBED
                assert text_idx < len(text), "Should be mid-cut"
                return

        pytest.fail("Should have entered SB0")

    def test_streaming_idle_then_text_complete(self):
        """Streaming: consume all → IDLE → text_complete arrives → SB1 (direct PAD)."""
        fsm = make_fsm(engine_max=200, ratio=5.0)
        text = "你好"
        fsm.reset()
        fsm.trailing_char_offsets = build_char_offsets(text)
        fsm.segment_text = text

        fsm.step(StepEvent(text_idx=0, trailing_len=len(text)))
        fsm.enter_phase_a(past_len=12, ema_ratio=5.0)

        text_idx = 0
        frame = 0
        idle_reached = False
        for _ in range(100):
            tp = fsm.token_punct_at(text_idx - 1) if text_idx > 0 else (False, False, False)
            action = fsm.step(StepEvent(
                text_idx=text_idx, trailing_len=len(text),
                past_len=12 + frame, frame_idx=frame,
                token_punct=tp, text_complete=False,
            ))
            if action.text_add_kind == TextAddKind.TRAILING:
                text_idx += 1
            frame += 1
            if action.idle:
                idle_reached = True
                break

        assert idle_reached
        assert text_idx == len(text)

        # Now text_complete arrives — FSM should cut to SB1 (direct PAD)
        action = fsm.step(StepEvent(
            text_idx=text_idx, trailing_len=len(text),
            past_len=12 + frame, frame_idx=frame,
            text_complete=True,
        ))
        assert fsm.state == FSMState.SB1, f"Expected SB1 (direct PAD), got {fsm.state}"

        # SB1: codec EOS → S0
        frame += 1
        action = fsm.step(StepEvent(
            past_len=12 + frame, frame_idx=frame,
            text_idx=text_idx, trailing_len=len(text),
            is_codec_eos=True,
        ))
        assert fsm.state == FSMState.S0

        # S0 with all text consumed and text_complete=True → E0
        action = fsm.step(StepEvent(
            text_idx=text_idx, trailing_len=len(text),
            past_len=12 + frame, frame_idx=frame,
            text_complete=True,
        ))
        assert action.end_segment

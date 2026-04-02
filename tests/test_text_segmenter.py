"""Tests for the three-layer text segmentation (Layer 1: pre-segmentation).

Uses both a fake tokenizer (deterministic, fast) and, when available, the real
lightweight tokenizer for integration tests against story.txt.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ORCH_1 = REPO_ROOT / "model_repository" / "tts_orchestrator" / "1"
if str(ORCH_1) not in sys.path:
    sys.path.insert(0, str(ORCH_1))

from text_segmenter import (
    _PRESPLIT_SENTENCE_END,
    _find_best_cut,
    assistant_token_count,
    split_text_for_token_budget,
)


# ---------------------------------------------------------------------------
# Fake tokenizer: 1 Chinese char ≈ 1.5 tokens (after stripping assistant fmt)
# ---------------------------------------------------------------------------

class _FakeTokenizer:
    """Deterministic tokenizer where ~2 chars → 1 token (for Chinese text).

    The assistant format wrapper adds a fixed overhead (~12 tokens).
    """
    _OVERHEAD = 12

    def __call__(self, text, return_tensors="pt"):
        clean = (
            text.replace("<|im_start|>assistant\n", "")
            .replace("<|im_end|>\n", "")
            .replace("<|im_start|>assistant", "")
            .strip()
        )
        n_tokens = self._OVERHEAD + max(1, (len(clean) + 1) // 2)
        return {"input_ids": list(range(n_tokens))}


# ---------------------------------------------------------------------------
# _find_best_cut unit tests
# ---------------------------------------------------------------------------

class TestFindBestCut:
    def test_l1_preferred(self):
        text = "第一句。第二句，第三句"
        cut = _find_best_cut(text, len(text))
        assert text[:cut].endswith("。")

    def test_l2_fallback(self):
        text = "没有句号只有逗号，继续写"
        cut = _find_best_cut(text, len(text))
        assert text[:cut].endswith("，")

    def test_hard_cut_when_no_punct(self):
        text = "纯文字没有任何标点符号"
        limit = 5
        cut = _find_best_cut(text, limit)
        assert cut == limit

    def test_quote_aware_l1(self):
        """After close-quote + L1, cut immediately (don't greedily extend)."""
        text = '他说："你好！"然后走了。'
        cut = _find_best_cut(text, len(text))
        head = text[:cut]
        assert head.endswith('"') or head.endswith('\u201d'), (
            f"Expected cut right after closing quote, got: {head}"
        )

    def test_closing_quote_included(self):
        """When L1 punct is followed by closing quote, include the quote."""
        text = '她喊道："救命！"随后沉默了。后面还有。'
        cut = _find_best_cut(text, 12)
        assert '"' in text[:cut] or '"' in text[:cut]

    def test_post_quote_sentence_not_merged(self):
        """After closing-quote + L1, short trailing text stays in next segment."""
        text = '"这是引号内容。"一个月后，国王来了。'
        cut = _find_best_cut(text, len(text))
        head = text[:cut]
        assert head.rstrip().endswith('"') or head.rstrip().endswith('"')
        assert "一个月后" not in head

    def test_does_not_exceed_limit(self):
        text = "短句。" + "很长的一段话没有标点" * 20
        limit = 10
        cut = _find_best_cut(text, limit)
        assert cut <= limit


# ---------------------------------------------------------------------------
# split_text_for_token_budget tests
# ---------------------------------------------------------------------------

class TestSplitTextForTokenBudget:
    def test_empty_input(self):
        tok = _FakeTokenizer()
        assert split_text_for_token_budget("", tok, 100) == []
        assert split_text_for_token_budget("  ", tok, 100) == []

    def test_short_text_single_segment(self):
        tok = _FakeTokenizer()
        segs = split_text_for_token_budget("你好。", tok, 100)
        assert segs == ["你好。"]

    def test_no_text_loss(self):
        """Concatenation of all segments reproduces the original (minus whitespace)."""
        tok = _FakeTokenizer()
        text = "第一句。第二句，逗号后继续。第三句！第四句？最后。"
        segs = split_text_for_token_budget(text, tok, 20)
        joined = "".join(segs)
        assert joined == text.strip().replace(" ", "")

    def test_segments_respect_budget(self):
        tok = _FakeTokenizer()
        text = "短句。" * 50
        budget = 20
        segs = split_text_for_token_budget(text, tok, budget)
        for i, seg in enumerate(segs):
            tc = assistant_token_count(seg, tok)
            assert tc <= budget, (
                f"Segment {i} has {tc} tokens > budget {budget}: {seg[:30]}..."
            )

    def test_l1_endings_preferred(self):
        """Most segments (except possibly the last) should end with L1 punct."""
        tok = _FakeTokenizer()
        text = "第一句。第二句！第三句？第四句。第五句！"
        segs = split_text_for_token_budget(text, tok, 22)
        if len(segs) > 1:
            for seg in segs[:-1]:
                last_char = seg.rstrip()[-1]
                assert last_char in _PRESPLIT_SENTENCE_END or last_char in '"\u201d', (
                    f"Segment does not end with L1 punct: ...{seg[-10:]}"
                )

    def test_hard_cut_no_punct(self):
        tok = _FakeTokenizer()
        text = "abcdefghijklmnopqrstuvwxyz" * 3
        segs = split_text_for_token_budget(text, tok, 20)
        assert len(segs) > 1
        joined = "".join(segs)
        assert joined == text

    def test_quoted_proclamation_does_not_swallow_next_sentence(self):
        """After a quoted block closes, post-quote text should not be merged
        into the same segment when budget allows the quote to fit entirely."""
        tok = _FakeTokenizer()
        quoted = '\u201c她们很美丽！\u201d'
        text = quoted + "一个月后，国王来了。还有更多。"
        segs = split_text_for_token_budget(text, tok, 20)
        if len(segs) > 1:
            assert segs[0].rstrip().endswith('\u201d'), (
                f"Expected first segment to end at closing quote, got: {repr(segs[0])}"
            )


# ---------------------------------------------------------------------------
# story.txt integration test (with real tokenizer)
# ---------------------------------------------------------------------------

STORY_PATH = REPO_ROOT / "tests" / "cases" / "story.txt"

# Attempt to locate a real tokenizer directory
_TOKENIZER_CANDIDATES = [
    REPO_ROOT / "workspace" / "models" / "Qwen3-TTS-Tokenizer-12Hz",
    Path.home() / ".cache" / "modelscope" / "hub" / "Qwen" / "Qwen3-TTS-Tokenizer-12Hz",
]


def _try_load_real_tokenizer():
    try:
        sys.path.insert(0, str(ORCH_1))
        from lightweight_tokenizer import load_lightweight_tokenizer
    except ImportError:
        return None
    for p in _TOKENIZER_CANDIDATES:
        if p.is_dir():
            tok = load_lightweight_tokenizer(str(p))
            if tok is not None:
                return tok
    return None


_real_tok = _try_load_real_tokenizer()
_has_real_tok = _real_tok is not None
_has_story = STORY_PATH.is_file()


@pytest.mark.skipif(not _has_real_tok, reason="real tokenizer not found")
@pytest.mark.skipif(not _has_story, reason="story.txt not found")
class TestStoryPresegmentation:
    """Integration tests using the real tokenizer and story.txt."""

    BUDGET = 73  # typical phase_a_cap for ema=5.5

    def _load_story(self) -> str:
        return STORY_PATH.read_text(encoding="utf-8").strip()

    def test_no_text_loss(self):
        text = self._load_story()
        segs = split_text_for_token_budget(text, _real_tok, self.BUDGET)
        original_chars = set(enumerate(text.replace(" ", "").replace("\n", "")))
        joined = "".join(segs).replace(" ", "").replace("\n", "")
        for i, ch in enumerate(joined):
            pass  # basic iteration check
        assert text.replace(" ", "").replace("\n", "") == joined, (
            "Text loss detected after pre-segmentation"
        )

    def test_all_segments_within_budget(self):
        text = self._load_story()
        segs = split_text_for_token_budget(text, _real_tok, self.BUDGET)
        for i, seg in enumerate(segs):
            tc = assistant_token_count(seg, _real_tok)
            assert tc <= self.BUDGET, (
                f"Segment {i} has {tc} tokens > budget {self.BUDGET}: "
                f"{repr(seg[:40])}..."
            )

    def test_segments_end_with_l1_punct(self):
        """All segments except the last should end with L1 or closing quote."""
        text = self._load_story()
        segs = split_text_for_token_budget(text, _real_tok, self.BUDGET)
        allowed_endings = _PRESPLIT_SENTENCE_END | {'"', '\u201d'}
        for seg in segs[:-1]:
            last = seg.rstrip()[-1]
            assert last in allowed_endings, (
                f"Segment does not end with L1/quote: ...{repr(seg[-20:])}"
            )

    def test_quote_integrity(self):
        """Opening and closing quotes should be balanced within each segment,
        or a segment should contain only the opening part (closed in the same seg)."""
        text = self._load_story()
        segs = split_text_for_token_budget(text, _real_tok, self.BUDGET)
        for i, seg in enumerate(segs):
            opens = seg.count('"') + seg.count('\u201c')
            closes = seg.count('"') + seg.count('\u201d')
            assert opens == closes, (
                f"Segment {i} has unbalanced quotes ({opens} opens, {closes} closes): "
                f"{repr(seg[:50])}"
            )

    def test_yigeyuehou_not_lost(self):
        """The phrase '一个月后' (story line 19) must appear in the output."""
        text = self._load_story()
        segs = split_text_for_token_budget(text, _real_tok, self.BUDGET)
        joined = "".join(segs)
        assert "一个月后" in joined, "'一个月后' was lost during segmentation"

    def test_segment_count_reasonable(self):
        text = self._load_story()
        segs = split_text_for_token_budget(text, _real_tok, self.BUDGET)
        assert 5 <= len(segs) <= 40, (
            f"Segment count {len(segs)} seems unreasonable for story.txt"
        )

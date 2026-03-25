import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ORCH_1 = REPO_ROOT / "model_repository" / "tts_orchestrator" / "1"
if str(ORCH_1) not in sys.path:
    sys.path.insert(0, str(ORCH_1))

from text_segmenter import assistant_token_count, split_text_for_token_budget


class _RawTokenizer:
    def __call__(self, text, return_tensors="pt"):
        return {"input_ids": list(range(len(text)))}


class _FakeTokenizer:
    def __call__(self, text, return_tensors="pt"):
        clean = (
            text.replace("<|im_start|>assistant\n", "")
            .replace("<|im_end|>\n", "")
            .replace("<|im_start|>assistant", "")
            .strip()
        )
        n_tokens = max(1, (len(clean) + 1) // 2)
        return {"input_ids": list(range(n_tokens))}


def test_assistant_token_count_uses_assistant_wrapper():
    tokenizer = _RawTokenizer()
    plain = len("你好")
    wrapped = assistant_token_count("你好", tokenizer)
    assert wrapped > plain


def test_split_text_for_token_budget_prefers_punctuation_boundaries():
    tokenizer = _FakeTokenizer()
    text = "第一句。第二句，第三句？第四句"
    segments = split_text_for_token_budget(text, tokenizer, max_tokens=4)
    assert len(segments) >= 2
    assert segments[0].endswith(("。", "，", "？"))


def test_split_text_for_token_budget_hard_splits_without_boundaries():
    tokenizer = _FakeTokenizer()
    text = "abcdefghijklmnopqrstuvwxyz"
    segments = split_text_for_token_budget(text, tokenizer, max_tokens=8)
    assert len(segments) > 1
    assert "".join(segments) == text

import importlib.util
import sys
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "python"
    / "check_c4_runtime_layout_parity.py"
)

SPEC = importlib.util.spec_from_file_location("c4_layout_parity", SCRIPT_PATH)
c4_layout_parity = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = c4_layout_parity
SPEC.loader.exec_module(c4_layout_parity)


class FakeTokenizer:
    def __call__(self, text, return_tensors="pt", padding=True):
        import torch

        body_len = 2 if "first" in text else 3
        ids = [101, 102, 103] + list(range(200, 200 + body_len)) + [901, 902, 903, 904, 905]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}


def _special_ids():
    return c4_layout_parity.load_special_ids.__globals__["C4SpecialIds"](
        tts_pad_token_id=151671,
        tts_bos_token_id=151672,
        tts_eos_token_id=151673,
        codec_pad_id=2148,
        codec_bos_id=2149,
        codec_eos_token_id=2150,
        codec_nothink_id=2155,
        codec_think_bos_id=2156,
        codec_think_eos_id=2157,
    )


def _rows():
    return [
        {
            "sample_id": "s1",
            "segments": [
                {"text": "first", "codes": [[1] * 16, [2] * 16]},
                {"text": "second", "codes": [[3] * 16]},
            ],
        }
    ]


def test_layout_parity_passes_when_prefix_matches():
    summary = c4_layout_parity.check_layout_parity(
        _rows(),
        tokenizer=FakeTokenizer(),
        special_ids=_special_ids(),
        runtime_prefix_len=8,
    )

    assert summary["overall_status"] == "pass"
    check = summary["checks"][0]
    assert check["current_codec_bos"] == check["expected_current_codec_bos"]
    assert check["runtime_full_current_prefill_len_if_same_prefix"] == 20
    assert check["prefix_matches_runtime"] is True


def test_layout_parity_warns_when_runtime_prefix_differs():
    summary = c4_layout_parity.check_layout_parity(
        _rows(),
        tokenizer=FakeTokenizer(),
        special_ids=_special_ids(),
        runtime_prefix_len=21,
    )

    assert summary["overall_status"] == "warn"
    check = summary["checks"][0]
    assert check["status"] == "warn"
    assert check["c4_prefix_len"] == 8
    assert check["runtime_prefix_len"] == 21
    assert check["prefix_matches_runtime"] is False

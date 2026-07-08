import importlib.util
import sys
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "python"
    / "build_c4_continuation_batch.py"
)

SPEC = importlib.util.spec_from_file_location("c4_batch", SCRIPT_PATH)
c4_batch = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = c4_batch
SPEC.loader.exec_module(c4_batch)


class FakeTokenizer:
    def __call__(self, text, return_tensors="pt", padding=True):
        import torch

        # Mimic the assistant template shape: first 3 prefix ids, body ids,
        # then 5 trailing ids that assistant_text_ids drops.
        body_len = 2 if "first" in text else 3
        ids = [101, 102, 103] + list(range(200, 200 + body_len)) + [901, 902, 903, 904, 905]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}


def _special_ids():
    return c4_batch.C4SpecialIds(
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


def test_continuation_batch_masks_loss_to_speech_codes_only():
    rows = [
        {
            "sample_id": "s1",
            "segments": [
                {
                    "text": "first",
                    "pause_ms": 220,
                    "punct_class": "comma",
                    "codes": [[1] * 16, [2] * 16],
                },
                {
                    "text": "second",
                    "pause_ms": 0,
                    "punct_class": "period",
                    "codes": [[3] * 16],
                },
            ],
        }
    ]

    batch, layouts = c4_batch.build_continuation_batch(
        rows,
        tokenizer=FakeTokenizer(),
        special_ids=_special_ids(),
    )
    summary = c4_batch.summarize_batch(batch, layouts)

    assert summary["batch_size"] == 1
    assert summary["codec_mask_true"] == 3
    assert summary["loss_positions"] == 3
    assert layouts[0]["total_codec_frames"] == 3
    assert layouts[0]["total_loss_positions"] == 3

    first = layouts[0]["segments"][0]
    boundary_pos = first["boundary_codec_eos"]
    assert int(batch["input_ids"][0, boundary_pos, 1]) == 2150
    assert int(batch["codec_0_labels"][0, boundary_pos]) == -100


def test_continuation_batch_rejects_bad_code_shape():
    rows = [
        {
            "sample_id": "bad",
            "segments": [
                {
                    "text": "first",
                    "pause_ms": 0,
                    "punct_class": "period",
                    "codes": [[1, 2, 3]],
                }
            ],
        }
    ]

    try:
        c4_batch.build_continuation_batch(
            rows,
            tokenizer=FakeTokenizer(),
            special_ids=_special_ids(),
        )
    except ValueError as exc:
        assert "shape [frames, 16]" in str(exc)
    else:
        raise AssertionError("expected invalid code shape to fail")

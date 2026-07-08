import importlib.util
import sys
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "python"
    / "attach_c4_codes_from_prepare.py"
)

SPEC = importlib.util.spec_from_file_location("attach_codes", SCRIPT_PATH)
attach_codes = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = attach_codes
SPEC.loader.exec_module(attach_codes)


def test_attach_codes_adds_segment_codes_and_metadata():
    manifest_rows = [
        {
            "sample_id": "s1",
            "segments": [
                {"text": "a", "pause_ms": 100, "punct_class": "comma"},
                {"text": "b", "pause_ms": 0, "punct_class": "period"},
            ],
            "meta": {"source": "api_synthetic_c4_smoke"},
        }
    ]
    prepared_rows = [
        {"sample_id": "s1", "segment_index": 0, "audio_codes": [[1] * 16]},
        {"sample_id": "s1", "segment_index": 1, "audio_codes": [[2] * 16]},
    ]

    out = attach_codes.attach_codes(
        manifest_rows,
        prepared_rows,
        codes_source="prepared.jsonl",
    )

    assert out[0]["segments"][0]["codes"] == [[1] * 16]
    assert out[0]["segments"][1]["codes"] == [[2] * 16]
    assert out[0]["meta"]["codes_source"] == "prepared.jsonl"
    assert out[0]["meta"]["training_ready_smoke"] is True


def test_attach_codes_rejects_missing_segment_code():
    manifest_rows = [
        {
            "sample_id": "s1",
            "segments": [
                {"text": "a", "pause_ms": 100, "punct_class": "comma"},
                {"text": "b", "pause_ms": 0, "punct_class": "period"},
            ],
        }
    ]
    prepared_rows = [
        {"sample_id": "s1", "segment_index": 0, "audio_codes": [[1] * 16]},
    ]

    try:
        attach_codes.attach_codes(
            manifest_rows,
            prepared_rows,
            codes_source="prepared.jsonl",
        )
    except ValueError as exc:
        assert "s1:1" in str(exc)
    else:
        raise AssertionError("expected missing segment code to fail")

import importlib.util
import sys
from pathlib import Path

import numpy as np


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "python"
    / "synthesize_c4_smoke_dataset.py"
)

SPEC = importlib.util.spec_from_file_location("c4_smoke", SCRIPT_PATH)
c4_smoke = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = c4_smoke
SPEC.loader.exec_module(c4_smoke)


def test_pause_ms_for_punct_uses_plan_like_defaults():
    assert c4_smoke.pause_ms_for_punct("comma") == 220
    assert c4_smoke.pause_ms_for_punct("period") == 460
    assert c4_smoke.pause_ms_for_punct("question") == 380
    assert c4_smoke.pause_ms_for_punct("period", is_last=True) == 0


def test_append_silence_extends_audio_by_pause_duration():
    audio = np.ones(240, dtype=np.float32)
    out = c4_smoke.append_silence(audio, sample_rate=24000, pause_ms=100)

    assert out.shape[0] == 240 + 2400
    assert np.allclose(out[:240], 1.0)
    assert np.allclose(out[240:], 0.0)


def test_build_manifest_row_marks_synthetic_not_final(tmp_path):
    row = {
        "sample_id": "sample-1",
        "speaker": "Vivian",
        "language": "Chinese",
        "instruct": "calm",
        "scenario": "demo",
    }
    manifest = c4_smoke.build_manifest_row(
        row,
        sample_dir=tmp_path / "sample-1",
        segment_records=[
            {
                "text": "hello",
                "audio": str(tmp_path / "segment.wav"),
                "pause_ms": 0,
                "punct_class": "none",
            }
        ],
        endpoint="127.0.0.1:50051",
    )

    assert manifest["speaker_name"] == "Vivian"
    assert manifest["segments"][0]["text"] == "hello"
    assert manifest["meta"]["synthetic"] is True
    assert manifest["meta"]["not_for_final_table2_c4"] is True

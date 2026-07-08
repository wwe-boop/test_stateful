import importlib.util
import sys
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "python"
    / "run_c4_forward_smoke.py"
)

SPEC = importlib.util.spec_from_file_location("c4_forward", SCRIPT_PATH)
c4_forward = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = c4_forward
SPEC.loader.exec_module(c4_forward)


def test_resolve_audio_path_keeps_absolute_path(tmp_path):
    path = tmp_path / "x.wav"
    assert c4_forward.resolve_audio_path(str(path), repo_root=Path("/repo")) == path


def test_resolve_audio_path_anchors_relative_path():
    assert c4_forward.resolve_audio_path("workspace/x.wav", repo_root=Path("/repo")) == Path(
        "/repo/workspace/x.wav"
    )


def test_first_ref_audio_uses_first_segment_audio():
    row = {
        "sample_id": "s1",
        "segments": [
            {"audio": "workspace/a.wav"},
            {"audio": "workspace/b.wav"},
        ],
    }
    assert c4_forward.first_ref_audio(row, repo_root=Path("/repo")) == Path(
        "/repo/workspace/a.wav"
    )

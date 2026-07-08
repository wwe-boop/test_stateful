import importlib.util
import json
import sys
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "python"
    / "build_c4_continuation_manifest.py"
)

SPEC = importlib.util.spec_from_file_location("c4_manifest", SCRIPT_PATH)
c4_manifest = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = c4_manifest
SPEC.loader.exec_module(c4_manifest)


def _write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def test_validate_training_ready_manifest(tmp_path):
    manifest = tmp_path / "train.jsonl"
    _write_jsonl(
        manifest,
        [
            {
                "sample_id": "sample-1",
                "speaker_name": "speaker-a",
                "language": "Chinese",
                "segments": [
                    {
                        "text": "first",
                        "pause_ms": 180,
                        "punct_class": "comma",
                        "codes": [[1] * 16, [2] * 16],
                    },
                    {
                        "text": "second",
                        "pause_ms": 420,
                        "punct_class": "period",
                        "codes": [[3] * 16],
                    },
                ],
                "meta": {"source": "book-train"},
            }
        ],
    )

    stats, issues = c4_manifest.validate_manifest(manifest, require_codes=True)

    assert issues == []
    assert stats.valid_samples == 1
    assert stats.segments == 2
    assert stats.coded_segments == 2
    assert stats.code_frames == 3
    assert stats.speaker_counts["speaker-a"] == 1


def test_require_codes_and_pause_bounds(tmp_path):
    manifest = tmp_path / "bad.jsonl"
    _write_jsonl(
        manifest,
        [
            {
                "sample_id": "sample-1",
                "speaker_name": "speaker-a",
                "language": "Chinese",
                "segments": [
                    {
                        "text": "first",
                        "pause_ms": 720,
                        "punct_class": "comma",
                    }
                ],
            }
        ],
    )

    _, issues = c4_manifest.validate_manifest(manifest, require_codes=True)
    issue_fields = {issue.field for issue in issues}

    assert "segments" in issue_fields
    assert "segments[0].pause_ms" in issue_fields
    assert "segments[0].codes" in issue_fields


def test_eval_source_overlap_is_reported(tmp_path):
    manifest = tmp_path / "train.jsonl"
    _write_jsonl(
        manifest,
        [
            {
                "sample_id": "sample-1",
                "speaker_name": "speaker-a",
                "language": "Chinese",
                "segments": [
                    {"text": "first", "pause_ms": 0, "punct_class": "none"},
                    {"text": "second", "pause_ms": 100, "punct_class": "period"},
                ],
                "meta": {"book_id": "eval-book"},
            }
        ],
    )

    _, issues = c4_manifest.validate_manifest(
        manifest,
        eval_sources={"eval-book"},
    )

    assert any(issue.field == "meta" for issue in issues)

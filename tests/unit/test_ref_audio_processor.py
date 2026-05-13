from __future__ import annotations

import pytest

from engine.backend.ref_audio_processor import (
    ReferenceAudioProcessor,
    ReferenceAudioSupport,
    _validate_reference_audio_quality,
)


def test_probe_ignores_onnx_for_trt_standalone(tmp_path):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    speaker_engine = runtime / "speaker_encoder.engine"
    speaker_engine.write_bytes(b"speaker-trt")
    (runtime / "speech_tokenizer_codec_fused.onnx").write_bytes(b"onnx")

    processor = ReferenceAudioProcessor(str(runtime), "base-1.7b")

    assert processor.support.speaker_encoder_path == speaker_engine
    assert processor.support.speech_tokenizer_codec_fused_path is None


def test_probe_accepts_model_plan_layout(tmp_path):
    runtime = tmp_path / "runtime"
    (runtime / "speaker_encoder").mkdir(parents=True)
    (runtime / "speech_tokenizer_codec_fused").mkdir(parents=True)
    speaker_plan = runtime / "speaker_encoder" / "model.plan"
    codec_plan = runtime / "speech_tokenizer_codec_fused" / "model.plan"
    speaker_plan.write_bytes(b"speaker-plan")
    codec_plan.write_bytes(b"codec-plan")

    processor = ReferenceAudioProcessor(str(runtime), "icl-1.7b")

    assert processor.support.speaker_encoder_path == speaker_plan
    assert processor.support.speech_tokenizer_codec_fused_path == codec_plan


def test_missing_icl_codec_engine_reports_explicit_error(tmp_path):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    speaker_engine = runtime / "speaker_encoder.engine"
    speaker_engine.write_bytes(b"speaker-trt")
    processor = ReferenceAudioProcessor(str(runtime), "base-1.7b")
    processor._support = ReferenceAudioSupport(
        available=True,
        speaker_encoder_path=speaker_engine,
        speech_tokenizer_codec_fused_path=None,
    )

    with pytest.raises(RuntimeError, match="speech_tokenizer_codec_fused_trt_missing"):
        processor.process(b"not-a-wav", require_ref_codec=True)


def test_feature_cache_key_changes_with_artifact_fingerprint(tmp_path):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    speaker_engine = runtime / "speaker_encoder.engine"
    codec_engine = runtime / "speech_tokenizer_codec_fused.engine"
    speaker_engine.write_bytes(b"speaker-v1")
    codec_engine.write_bytes(b"codec-v1")

    processor = ReferenceAudioProcessor(str(runtime), "base-1.7b")
    processor._support = ReferenceAudioSupport(
        available=True,
        speaker_encoder_path=speaker_engine,
        speech_tokenizer_codec_fused_path=codec_engine,
    )
    key_before = processor._feature_cache_key(
        "audio-sha",
        require_ref_codec=True,
        cache_tag="voice_clone_icl",
    )

    codec_engine.write_bytes(b"codec-v2-has-different-size")
    key_after = processor._feature_cache_key(
        "audio-sha",
        require_ref_codec=True,
        cache_tag="voice_clone_icl",
    )

    assert key_before != key_after


def test_reference_audio_quality_validation_errors_and_warns():
    import numpy as np

    with pytest.raises(ValueError, match="ref_audio_too_short"):
        _validate_reference_audio_quality(np.zeros(12000, dtype=np.float32), 0.5)

    with pytest.raises(ValueError, match="invalid_ref_audio"):
        _validate_reference_audio_quality(np.zeros(24_000 * 21, dtype=np.float32), 21.0)

    warnings = _validate_reference_audio_quality(
        np.zeros(24_000 * 2, dtype=np.float32),
        2.0,
    )
    assert any("outside the recommended 3-10s range" in item for item in warnings)
    assert any("near-silent" in item for item in warnings)

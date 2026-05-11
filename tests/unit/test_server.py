from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from engine.backend.ref_audio_processor import ReferenceAudioSupport
from engine.config import EngineConfig, EngineProfileConfig, ModelArchConfig
from engine.core.types import SessionConfig
from engine.server import TTSEngine


@dataclass
class _StubSupportProbe:
    support: ReferenceAudioSupport


class _StubFrontend:
    def __init__(self):
        self.calls = []

    async def create_session(self, session_id: str, *, config, on_audio=None, on_done=None, on_event=None):
        self.calls.append((session_id, config, on_audio, on_done, on_event))
        return {"session_id": session_id, "config": config}


def test_validate_session_config_rejects_unsupported_task_type():
    engine = TTSEngine(model_arch=ModelArchConfig(
        variant="custom-1.7b",
        tts_model_type="custom_voice",
        supported_task_types=("custom_voice",),
    ))

    with pytest.raises(ValueError, match="has loaded model_type 'custom_voice'"):
        engine._validate_session_config(SessionConfig(task_type="voice_clone", ref_audio=b"x"))


def test_validate_session_config_requires_ref_audio_for_voice_clone():
    engine = TTSEngine(model_arch=ModelArchConfig(
        variant="base-1.7b",
        tts_model_type="base",
        supported_task_types=("base",),
    ))
    engine._ref_audio_processor = _StubSupportProbe(
        ReferenceAudioSupport(available=True)
    )

    with pytest.raises(ValueError, match="ref_audio is required for loaded model_type 'base'"):
        engine._validate_session_config(SessionConfig(task_type="base"))


def test_validate_session_config_reports_unavailable_ref_audio_processor():
    engine = TTSEngine(model_arch=ModelArchConfig(
        variant="base-1.7b",
        tts_model_type="base",
        supported_task_types=("base",),
    ))
    engine._ref_audio_processor = _StubSupportProbe(
        ReferenceAudioSupport(
            available=False,
            reason="onnxruntime is not installed for standalone ref_audio preprocessing",
        )
    )

    with pytest.raises(ValueError, match="voice_clone is not available in standalone mode"):
        engine._validate_session_config(SessionConfig(task_type="base", ref_audio=b"x"))


def test_start_session_validates_before_delegating():
    engine = TTSEngine(model_arch=ModelArchConfig(
        variant="custom-1.7b",
        tts_model_type="custom_voice",
        supported_task_types=("custom_voice",),
    ))
    frontend = _StubFrontend()
    engine._frontend = frontend

    config = SessionConfig(task_type="custom_voice", speaker="Serena")
    result = asyncio.run(engine.start_session("sid-1", config=config))

    assert result["session_id"] == "sid-1"
    assert len(frontend.calls) == 1
    assert frontend.calls[0][1] is config


def test_validate_session_config_binds_empty_task_type_to_loaded_model():
    engine = TTSEngine(model_arch=ModelArchConfig(
        variant="custom-1.7b",
        tts_model_type="custom_voice",
        supported_task_types=("custom_voice",),
    ))

    config = SessionConfig(task_type="", speaker="Serena")
    engine._validate_session_config(config)

    assert config.task_type == "custom_voice"


def test_validate_session_config_maps_base_model_to_internal_voice_clone():
    engine = TTSEngine(model_arch=ModelArchConfig(
        variant="base-1.7b",
        tts_model_type="base",
        supported_task_types=("base",),
    ))
    engine._ref_audio_processor = _StubSupportProbe(
        ReferenceAudioSupport(available=True)
    )

    config = SessionConfig(task_type="", ref_audio=b"wav")
    engine._validate_session_config(config)

    assert config.task_type == "voice_clone"
    assert config.x_vector_only is True


def test_validate_session_config_maps_icl_model_to_internal_voice_clone():
    engine = TTSEngine(model_arch=ModelArchConfig(
        variant="icl-1.7b",
        tts_model_type="icl",
        supported_task_types=("icl",),
    ))
    engine._ref_audio_processor = _StubSupportProbe(
        ReferenceAudioSupport(available=True)
    )

    config = SessionConfig(task_type="", ref_audio=b"wav", ref_text="你好")
    engine._validate_session_config(config)

    assert config.task_type == "voice_clone"
    assert config.x_vector_only is False


def test_validate_session_config_requires_ref_text_for_icl():
    engine = TTSEngine(model_arch=ModelArchConfig(
        variant="icl-1.7b",
        tts_model_type="icl",
        supported_task_types=("icl",),
    ))
    engine._ref_audio_processor = _StubSupportProbe(
        ReferenceAudioSupport(available=True)
    )

    with pytest.raises(ValueError, match="ref_text is required for loaded model_type 'icl'"):
        engine._validate_session_config(SessionConfig(task_type="icl", ref_audio=b"wav"))


def test_validate_session_config_requires_instruct_for_voice_design():
    engine = TTSEngine(model_arch=ModelArchConfig(
        variant="voice-design-1.7b",
        tts_model_type="voice_design",
        supported_task_types=("voice_design",),
    ))

    with pytest.raises(ValueError, match="instruct is required for loaded model_type 'voice_design'"):
        engine._validate_session_config(SessionConfig(task_type="voice_design"))


def test_health_stats_exposes_loaded_model_binding():
    class _StubEngineLoop:
        def health_stats(self):
            return {"running": True}

    engine = TTSEngine(model_arch=ModelArchConfig(
        variant="custom-1.7b",
        tts_model_type="custom_voice",
        supported_task_types=("custom_voice",),
    ))
    engine._engine_loop = _StubEngineLoop()

    stats = engine.health_stats()
    assert stats["variant"] == "custom-1.7b"
    assert stats["loaded_model_type"] == "custom_voice"
    assert stats["declared_supported_task_types"] == ["custom_voice"]


def test_describe_capabilities_reports_loaded_model_contract():
    engine = TTSEngine(model_arch=ModelArchConfig(
        variant="custom-1.7b",
        tts_model_type="custom_voice",
        supported_task_types=("custom_voice",),
    ))
    engine._ref_audio_processor = _StubSupportProbe(
        ReferenceAudioSupport(
            available=False,
            reason="variant 'custom-1.7b' is not a base model; voice_clone is unsupported",
        )
    )

    cap = engine.describe_capabilities()
    assert cap["variant"] == "custom-1.7b"
    assert cap["loaded_model_type"] == "custom_voice"
    assert cap["declared_supported_task_types"] == ["custom_voice"]
    assert cap["supported_input_modes"] == ["token", "clause", "long_segment", "full_text"]
    assert cap["supported_group_policies"] == ["none", "auto"]
    assert cap["ref_audio_available"] is False


def test_runtime_profile_rejects_oversized_batch():
    arch = ModelArchConfig(
        variant="custom-1.7b",
        engine_profile=EngineProfileConfig(max_batch_size=16, max_seq_len=512),
    )

    with pytest.raises(ValueError, match="runtime max_batch_size=32 exceeds engine profile"):
        TTSEngine(config=EngineConfig(), model_arch=arch, max_batch_size=32, max_seq_len=512)


def test_runtime_profile_rejects_oversized_seq_len():
    arch = ModelArchConfig(
        variant="custom-1.7b",
        engine_profile=EngineProfileConfig(max_batch_size=64, max_seq_len=512),
    )

    with pytest.raises(ValueError, match="runtime max_seq_len=1024 exceeds engine profile"):
        TTSEngine(config=EngineConfig(), model_arch=arch, max_batch_size=32, max_seq_len=1024)

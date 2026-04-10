from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from engine.core.types import AudioConfig, AudioEncoding, GroupPolicy, InputMode

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "engine" / "gateway"))

from engine.gateway import tts_pb2
from engine.gateway.grpc_server import (
    TTSServicer,
    _make_capabilities_response,
    _session_config_from_oneshot_request,
    _session_config_from_stream_request,
    _validate_audio_config,
)


def test_stream_request_start_maps_explicit_session_config():
    request = tts_pb2.SynthesizeRequest(
        start=tts_pb2.StartRequest(
            session_id="sid-1",
            config=tts_pb2.SessionConfig(
                task_type="custom_voice",
                speaker="Serena",
                input_mode=tts_pb2.INPUT_MODE_TOKEN,
                group_policy=tts_pb2.GROUP_POLICY_NONE,
                audio=tts_pb2.AudioFormat(
                    encoding=tts_pb2.AUDIO_ENCODING_PCM_S16LE,
                    sample_rate=16000,
                    channels=1,
                ),
            ),
        )
    )

    cfg = _session_config_from_stream_request(request)
    assert cfg.task_type == "custom_voice"
    assert cfg.speaker == "Serena"
    assert cfg.input_mode == InputMode.TOKEN
    assert cfg.group_policy == GroupPolicy.NONE
    assert cfg.audio.encoding == AudioEncoding.PCM_S16LE
    assert cfg.audio.sample_rate == 16000


def test_stream_request_keeps_empty_task_type_when_client_omits_it():
    request = tts_pb2.SynthesizeRequest(
        start=tts_pb2.StartRequest(
            session_id="sid-empty",
            config=tts_pb2.SessionConfig(
                speaker="Serena",
            ),
        )
    )

    cfg = _session_config_from_stream_request(request)
    assert cfg.task_type == ""


def test_make_capabilities_response_maps_contract_fields():
    response = _make_capabilities_response({
        "variant": "custom-1.7b",
        "loaded_model_type": "custom_voice",
        "declared_supported_task_types": ["custom_voice"],
        "supported_input_modes": ["token", "full_text"],
        "supported_group_policies": ["none", "auto"],
        "supported_audio_formats": [
            {"encoding": "pcm_f32", "sample_rate": 24000, "channels": 1},
            {"encoding": "pcm_s16le", "sample_rate": 16000, "channels": 1},
        ],
        "ref_audio_available": False,
        "ref_audio_reason": "voice_clone unavailable",
    })

    assert response.variant == "custom-1.7b"
    assert response.loaded_model_type == "custom_voice"
    assert list(response.declared_supported_task_types) == ["custom_voice"]
    assert list(response.supported_input_modes) == [
        tts_pb2.INPUT_MODE_TOKEN,
        tts_pb2.INPUT_MODE_FULL_TEXT,
    ]
    assert list(response.supported_group_policies) == [
        tts_pb2.GROUP_POLICY_NONE,
        tts_pb2.GROUP_POLICY_AUTO,
    ]
    assert response.supported_audio_formats[1].encoding == tts_pb2.AUDIO_ENCODING_PCM_S16LE
    assert response.supported_audio_formats[1].sample_rate == 16000
    assert response.ref_audio_available is False


@pytest.mark.asyncio
async def test_get_capabilities_returns_engine_contract():
    class _StubEngine:
        def describe_capabilities(self):
            return {
                "variant": "custom-1.7b",
                "loaded_model_type": "custom_voice",
                "declared_supported_task_types": ["custom_voice"],
                "supported_input_modes": ["token", "clause", "long_segment", "full_text"],
                "supported_group_policies": ["none", "auto"],
                "supported_audio_formats": [
                    {"encoding": "pcm_f32", "sample_rate": 24000, "channels": 1},
                ],
                "ref_audio_available": False,
                "ref_audio_reason": "",
            }

    servicer = TTSServicer(_StubEngine())
    response = await servicer.GetCapabilities(tts_pb2.GetCapabilitiesRequest(), context=None)

    assert response.loaded_model_type == "custom_voice"
    assert response.supported_audio_formats[0].sample_rate == 24000


@pytest.mark.asyncio
async def test_streaming_audio_is_not_blocked_by_next_text_chunk():
    class _StubEngine:
        def __init__(self):
            self._on_audio = None
            self._on_done = None
            self._on_event = None

        def describe_capabilities(self):
            return {}

        async def start_session(self, session_id, *, config, on_audio=None, on_done=None, on_event=None):
            self._on_audio = on_audio
            self._on_done = on_done
            self._on_event = on_event
            return session_id

        async def push_text_input(self, session_id, text):
            async def _emit():
                await asyncio.sleep(0.01)
                await self._on_audio(session_id, b"\x00\x00\x00\x00")
            asyncio.create_task(_emit())

        async def mark_input_complete(self, session_id):
            async def _finish():
                await asyncio.sleep(0.01)
                await self._on_done(session_id, {})
            asyncio.create_task(_finish())

        async def cancel(self, session_id):
            return None

    servicer = TTSServicer(_StubEngine())

    async def request_gen():
        yield tts_pb2.SynthesizeRequest(
            start=tts_pb2.StartRequest(
                session_id="sid-stream",
                config=tts_pb2.SessionConfig(
                    task_type="custom_voice",
                    speaker="Serena",
                ),
            )
        )
        yield tts_pb2.SynthesizeRequest(text=tts_pb2.TextChunk(text="你好"))
        await asyncio.sleep(0.2)
        yield tts_pb2.SynthesizeRequest(end=tts_pb2.EndRequest())

    stream = servicer.SynthesizeStream(request_gen(), context=None)
    first_response = await asyncio.wait_for(anext(stream), timeout=0.1)
    second_response = await asyncio.wait_for(anext(stream), timeout=0.1)

    assert first_response.WhichOneof("response") == "event"
    assert first_response.event.type == "start"
    assert first_response.event.audio.sample_rate == 24000
    assert second_response.WhichOneof("response") == "audio"


@pytest.mark.asyncio
async def test_streaming_text_protocol_events_are_forwarded():
    class _StubEngine:
        def __init__(self):
            self._on_audio = None
            self._on_done = None
            self._on_event = None

        def describe_capabilities(self):
            return {}

        async def start_session(self, session_id, *, config, on_audio=None, on_done=None, on_event=None):
            self._on_audio = on_audio
            self._on_done = on_done
            self._on_event = on_event
            return session_id

        async def push_text_input(self, session_id, text):
            await self._on_event(
                session_id,
                {
                    "type": "text_token",
                    "segment_idx": 0,
                    "text": "你",
                    "meta": {"token_idx": "0", "text_complete": "false"},
                },
            )
            await self._on_event(
                session_id,
                {
                    "type": "text_boundary_commit",
                    "segment_idx": 0,
                    "text": "你好。",
                    "meta": {"boundary_reason": "flush_eos", "text_complete": "true"},
                },
            )
            await self._on_audio(session_id, b"\x00\x00\x00\x00")

        async def mark_input_complete(self, session_id):
            await self._on_done(session_id, {})

        async def cancel(self, session_id):
            return None

    servicer = TTSServicer(_StubEngine())

    async def request_gen():
        yield tts_pb2.SynthesizeRequest(
            start=tts_pb2.StartRequest(
                session_id="sid-events",
                config=tts_pb2.SessionConfig(task_type="custom_voice"),
            )
        )
        yield tts_pb2.SynthesizeRequest(text=tts_pb2.TextChunk(text="你好。"))
        yield tts_pb2.SynthesizeRequest(end=tts_pb2.EndRequest())

    responses = []
    async for response in servicer.SynthesizeStream(request_gen(), context=None):
        responses.append(response)

    event_types = [
        response.event.type
        for response in responses
        if response.WhichOneof("response") == "event"
    ]
    assert event_types[:3] == ["start", "text_token", "text_boundary_commit"]
    assert "done" in event_types
    boundary = next(
        response.event for response in responses
        if response.WhichOneof("response") == "event"
        and response.event.type == "text_boundary_commit"
    )
    assert boundary.text == "你好。"
    assert boundary.meta["boundary_reason"] == "flush_eos"


def test_oneshot_request_forces_full_text_semantics():
    request = tts_pb2.SynthesizeOnceRequest(
        session_id="sid-2",
        text="你好",
        config=tts_pb2.SessionConfig(
            task_type="custom_voice",
            input_mode=tts_pb2.INPUT_MODE_TOKEN,
            group_policy=tts_pb2.GROUP_POLICY_NONE,
        ),
    )

    cfg = _session_config_from_oneshot_request(request)
    assert cfg.input_mode == InputMode.FULL_TEXT
    assert cfg.group_policy == GroupPolicy.AUTO


def test_validate_audio_config_rejects_non_mono():
    with pytest.raises(ValueError, match="mono only"):
        _validate_audio_config(AudioConfig(channels=2, sample_rate=24000, encoding=AudioEncoding.PCM_F32))


def test_validate_audio_config_rejects_unsupported_sample_rate():
    with pytest.raises(ValueError, match="expected 16000 or 24000"):
        _validate_audio_config(AudioConfig(channels=1, sample_rate=22050, encoding=AudioEncoding.PCM_F32))

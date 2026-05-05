from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from typing import Any

from aiohttp import ClientSession, WSMsgType

from .schemas import RunMetrics, RunResult, TraceEvent
from .triton_client import TtsRequest


DEFAULT_ENGINE_WS = os.environ.get("QWEN_DEMO_ENGINE_WS", "ws://localhost:50052/v1/ws")
DEFAULT_ENGINE_CAPABILITIES = os.environ.get(
    "QWEN_DEMO_ENGINE_CAPABILITIES",
    "http://localhost:50052/v1/capabilities",
)


class EngineUnavailable(RuntimeError):
    pass


async def probe_ready(capabilities_url: str = DEFAULT_ENGINE_CAPABILITIES, timeout_sec: float = 1.0) -> bool:
    try:
        timeout = aiohttp_timeout(timeout_sec)
        async with ClientSession(timeout=timeout) as session:
            async with session.get(capabilities_url) as response:
                return response.status == 200
    except Exception:
        return False


async def measure_once(
    request: TtsRequest,
    *,
    url: str = DEFAULT_ENGINE_WS,
    timeout_sec: float = 60.0,
) -> RunResult:
    run_id = f"engine-{uuid.uuid4().hex[:10]}"
    started = time.perf_counter()
    first_audio_ms: float | None = None
    total_ms: float | None = None
    chunks = 0
    audio_parts: list[bytes] = []
    audio_format: dict[str, Any] = {
        "encoding": request.audio_encoding,
        "sample_rate": request.sample_rate,
        "channels": 1,
    }
    events = [
        TraceEvent(
            run_id=run_id,
            backend="bare_engine_streaming",
            type="request_started",
            t_ms=0.0,
            meta={"transport": "engine_websocket"},
        )
    ]
    try:
        timeout = aiohttp_timeout(timeout_sec)
        async with ClientSession(timeout=timeout) as session:
            async with session.ws_connect(url, heartbeat=30.0) as ws:
                await ws.send_json(
                    {
                        "type": "oneshot",
                        "session_id": run_id,
                        "text": request.text,
                        "config": {
                            "task_type": request.task_type,
                            "speaker": request.speaker,
                            "language": request.language,
                            "audio": {
                                "encoding": request.audio_encoding,
                                "sample_rate": request.sample_rate,
                                "channels": 1,
                            },
                        },
                    }
                )
                async for message in ws:
                    now_ms = (time.perf_counter() - started) * 1000.0
                    if message.type == WSMsgType.BINARY:
                        if first_audio_ms is None:
                            first_audio_ms = now_ms
                            events.append(
                                TraceEvent(
                                    run_id=run_id,
                                    backend="bare_engine_streaming",
                                    type="first_audio_chunk",
                                    t_ms=now_ms,
                                    meta={"transport": "engine_websocket", "bytes": len(message.data)},
                                )
                            )
                        else:
                            events.append(
                                TraceEvent(
                                    run_id=run_id,
                                    backend="bare_engine_streaming",
                                    type="audio_chunk",
                                    t_ms=now_ms,
                                    meta={"bytes": len(message.data)},
                                )
                            )
                        chunks += 1
                        audio_parts.append(bytes(message.data))
                    elif message.type == WSMsgType.TEXT:
                        payload = json.loads(message.data)
                        event = payload.get("event", {}) if isinstance(payload, dict) else {}
                        event_type = str(event.get("type") or payload.get("type") or "")
                        if event_type == "start":
                            audio_format.update(event.get("audio", {}) or {})
                        if event_type in {"text_token", "text_boundary_commit", "segment_end"}:
                            events.append(
                                TraceEvent(
                                    run_id=run_id,
                                    backend="bare_engine_streaming",
                                    type=event_type,
                                    t_ms=now_ms,
                                    text=str(event.get("text") or ""),
                                    meta=event.get("meta", {}) or {},
                                )
                            )
                        if event_type in {"done", "error"}:
                            total_ms = now_ms
                            if event_type == "error":
                                raise EngineUnavailable(str(event.get("message") or "engine websocket error"))
                            events.append(
                                TraceEvent(
                                    run_id=run_id,
                                    backend="bare_engine_streaming",
                                    type="done",
                                    t_ms=now_ms,
                                    meta=event.get("meta", {}) or {},
                                )
                            )
                            break
                    elif message.type == WSMsgType.ERROR:
                        raise EngineUnavailable(str(ws.exception()))
    except Exception as exc:
        raise EngineUnavailable(str(exc)) from exc

    if first_audio_ms is None:
        raise EngineUnavailable("engine websocket stream completed without audio")
    if total_ms is None:
        total_ms = max(event.t_ms for event in events)

    raw_audio = b"".join(audio_parts)
    audio_duration_ms = None
    if raw_audio and audio_format.get("encoding") == "pcm_f32":
        audio_duration_ms = len(raw_audio) / 4.0 / float(audio_format.get("sample_rate", 24000)) * 1000.0

    return RunResult(
        run_id=run_id,
        backend="bare_engine_streaming",
        label="Bare Engine Streaming",
        mode="standalone TTSEngine WebSocket",
        source="live_engine_websocket",
        metrics=RunMetrics(
            client_ttfb_ms=first_audio_ms,
            first_playable_ms=first_audio_ms,
            first_audible_ms=first_audio_ms + 12.0,
            total_ms=total_ms,
            chunks=chunks,
            audio_duration_ms=audio_duration_ms,
            cache_hit=request.cache_mode == "hit",
        ),
        events=events,
        audio_format=audio_format,
        raw_audio=raw_audio,
    )


def aiohttp_timeout(total: float):
    from aiohttp import ClientTimeout

    return ClientTimeout(total=total)


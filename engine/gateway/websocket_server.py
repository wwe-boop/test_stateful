"""WebSocket streaming gateway for the standalone TTS engine.

Protocol:
  Client text frames:
    {"type":"start","session_id":"...","config":{...}}
    {"type":"text","text":"...","seq_no":1}
    {"type":"end"}
    {"type":"cancel"}
    {"type":"oneshot","session_id":"...","text":"...","config":{...}}
    {"type":"get_capabilities"}

  Server text frames:
    {"type":"event","event":{...}}
    {"type":"capabilities","capabilities":{...}}

  Server binary frames:
    raw PCM audio bytes matching the audio format declared by the ``start`` event.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
import os
import uuid
from typing import TYPE_CHECKING, Any

from ..core.types import (
    AudioConfig,
    AudioEncoding,
    GroupPolicy,
    InputMode,
    SessionConfig,
)
from .grpc_server import _convert_audio_chunk

if TYPE_CHECKING:
    from ..server import TTSEngine

try:
    from aiohttp import WSMsgType, web
except ImportError:  # pragma: no cover - exercised in environments without aiohttp
    WSMsgType = None
    web = None


_WEBSOCKET_AUDIO_QUEUE_MAXSIZE = int(
    os.environ.get("ENGINE_WEBSOCKET_AUDIO_QUEUE_MAXSIZE", "4096") or "4096"
)
_WEBSOCKET_REQUEST_QUEUE_MAXSIZE = int(
    os.environ.get("ENGINE_WEBSOCKET_REQUEST_QUEUE_MAXSIZE", "64") or "64"
)
_CAPABILITIES_PATH = "/v1/capabilities"
_WEBSOCKET_HEARTBEAT_SEC = float(
    os.environ.get("ENGINE_WEBSOCKET_HEARTBEAT_SEC", "30") or "30"
)

logger = logging.getLogger(__name__)


class WebSocketGateway:
    """Bridge a single websocket connection to one TTS engine session."""

    def __init__(self, engine: TTSEngine):
        self._engine = engine

    async def handle_capabilities(self, request):
        return web.json_response(self._engine.describe_capabilities())

    async def handle_websocket(self, request):
        ws = web.WebSocketResponse(heartbeat=_WEBSOCKET_HEARTBEAT_SEC)
        await ws.prepare(request)

        session_id = None
        outbound_queue: asyncio.Queue = asyncio.Queue(maxsize=_WEBSOCKET_AUDIO_QUEUE_MAXSIZE)
        request_queue: asyncio.Queue = asyncio.Queue(maxsize=_WEBSOCKET_REQUEST_QUEUE_MAXSIZE)
        got_cancel = False
        connection_closed = False
        request_task: asyncio.Task | None = None
        outbound_task: asyncio.Task | None = None
        pump_task = asyncio.create_task(self._pump_messages(ws, request_queue))

        try:
            while True:
                if request_task is None and not connection_closed:
                    request_task = asyncio.create_task(request_queue.get())
                if outbound_task is None and session_id and not got_cancel:
                    outbound_task = asyncio.create_task(outbound_queue.get())

                wait_set = {task for task in (request_task, outbound_task) if task is not None}
                if not wait_set:
                    break

                done, _ = await asyncio.wait(wait_set, return_when=asyncio.FIRST_COMPLETED)

                if request_task in done:
                    kind, payload = request_task.result()
                    request_task = None

                    if kind == "error":
                        raise payload

                    if kind == "closed":
                        connection_closed = True
                        got_cancel = True
                    else:
                        message = payload
                        msg_type = str(message.get("type", "") or "").strip().lower()

                        if msg_type == "get_capabilities":
                            await ws.send_json(
                                {
                                    "type": "capabilities",
                                    "capabilities": self._engine.describe_capabilities(),
                                }
                            )
                            continue

                        if msg_type == "start":
                            if session_id is not None:
                                raise ValueError("websocket session has already been started")
                            config = _session_config_from_ws_message(
                                message,
                                default_mode=InputMode.LONG_SEGMENT,
                            )
                            session_id = await self._create_session(
                                message.get("session_id"),
                                config=config,
                                outbound_queue=outbound_queue,
                            )

                        elif msg_type == "oneshot":
                            if session_id is not None:
                                raise ValueError("websocket session has already been started")
                            config = _session_config_from_ws_message(
                                message,
                                default_mode=InputMode.FULL_TEXT,
                            )
                            config.input_mode = InputMode.FULL_TEXT
                            if config.group_policy == GroupPolicy.NONE:
                                config.group_policy = GroupPolicy.AUTO
                            session_id = await self._create_session(
                                message.get("session_id"),
                                config=config,
                                outbound_queue=outbound_queue,
                            )
                            text = str(message.get("text", "") or "")
                            if not text:
                                raise ValueError("oneshot request requires non-empty 'text'")
                            await self._engine.push_text_input(session_id, text)
                            await self._engine.mark_input_complete(session_id)

                        elif msg_type == "text":
                            if not session_id:
                                raise ValueError("received 'text' before 'start'")
                            await self._engine.push_text_input(
                                session_id,
                                str(message.get("text", "") or ""),
                            )

                        elif msg_type == "end":
                            if not session_id:
                                raise ValueError("received 'end' before 'start'")
                            await self._engine.mark_input_complete(session_id)

                        elif msg_type == "cancel":
                            if session_id:
                                await self._engine.cancel(session_id)
                            got_cancel = True
                            connection_closed = True

                        else:
                            raise ValueError(f"unsupported websocket message type: '{msg_type or '<empty>'}'")

                        async for frame in self._drain_available_messages(outbound_queue):
                            await _send_frame(ws, frame)
                            if _is_terminal_frame(frame):
                                return ws

                if outbound_task in done:
                    frame = outbound_task.result()
                    outbound_task = None
                    await _send_frame(ws, frame)
                    if _is_terminal_frame(frame):
                        return ws

                if connection_closed and got_cancel:
                    break

        except asyncio.CancelledError:
            logger.info("WebSocket stream cancelled: %s", session_id)
        except Exception as exc:
            logger.error("WebSocket stream error: %s: %s", session_id, exc)
            if not ws.closed:
                await ws.send_json(
                    _make_event_frame(
                        event_type="error",
                        session_id=session_id or "",
                        message=str(exc),
                    )
                )
        finally:
            for task in (request_task, outbound_task, pump_task):
                if task is not None and not task.done():
                    task.cancel()
            for task in (request_task, outbound_task, pump_task):
                if task is not None:
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            if session_id:
                await self._engine.cancel(session_id)
            if not ws.closed:
                await ws.close()

        return ws

    async def _create_session(
        self,
        session_id: str | None,
        *,
        config: SessionConfig,
        outbound_queue: asyncio.Queue,
    ) -> str:
        session_id = str(session_id or uuid.uuid4())

        async def on_audio(sid: str, data: bytes) -> None:
            converted = _convert_audio_chunk(data, config.audio)
            await outbound_queue.put(
                _make_audio_frame(converted, config.audio)
            )

        async def on_event(sid: str, event: dict) -> None:
            await outbound_queue.put(
                _make_event_frame(
                    event_type=str(event.get("type", "") or ""),
                    session_id=sid,
                    segment_id=int(event.get("segment_idx", -1)),
                    text=str(event.get("text", "") or ""),
                    message=str(event.get("message", "") or ""),
                    audio_format=config.audio if event.get("type") == "start" else None,
                    meta={
                        str(k): str(v)
                        for k, v in (event.get("meta", {}) or {}).items()
                    },
                )
            )

        async def on_done(sid: str, metrics: dict) -> None:
            event_type = "error" if isinstance(metrics, dict) and metrics.get("error") else "done"
            await outbound_queue.put(
                _make_event_frame(
                    event_type=event_type,
                    session_id=sid,
                    message=str(metrics.get("error", "") if isinstance(metrics, dict) else ""),
                    meta={
                        str(k): str(v)
                        for k, v in (metrics or {}).items()
                        if k != "error"
                    } if isinstance(metrics, dict) else {},
                )
            )

        await self._engine.start_session(
            session_id,
            config=config,
            on_audio=on_audio,
            on_done=on_done,
            on_event=on_event,
        )
        await outbound_queue.put(
            _make_event_frame(
                event_type="start",
                session_id=session_id,
                audio_format=config.audio,
                meta={
                    "input_mode": config.input_mode.value,
                    "group_policy": config.group_policy.value,
                    "task_type": config.task_type or "",
                },
            )
        )
        logger.info("WebSocket session started: %s", session_id)
        return session_id

    async def _pump_messages(
        self,
        ws,
        request_queue: asyncio.Queue,
    ) -> None:
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        payload = json.loads(msg.data)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"invalid websocket JSON payload: {exc}") from exc
                    if not isinstance(payload, dict):
                        raise ValueError("websocket payload must be a JSON object")
                    await request_queue.put(("request", payload))
                    continue
                if msg.type == WSMsgType.BINARY:
                    raise ValueError("binary client frames are not supported; send JSON control messages only")
                if msg.type == WSMsgType.ERROR:
                    raise msg.data
        except Exception as exc:
            await request_queue.put(("error", exc))
        finally:
            await request_queue.put(("closed", None))

    async def _drain_available_messages(self, outbound_queue: asyncio.Queue):
        while not outbound_queue.empty():
            yield outbound_queue.get_nowait()


def _make_audio_frame(pcm_bytes: bytes, audio_config: AudioConfig) -> dict[str, Any]:
    return {
        "type": "audio",
        "audio": {
            "pcm_data": pcm_bytes,
            "sample_rate": audio_config.sample_rate,
            "encoding": audio_config.encoding.value,
            "channels": audio_config.channels,
        },
    }


def _make_event_frame(
    *,
    event_type: str,
    session_id: str = "",
    segment_id: int = -1,
    text: str = "",
    message: str = "",
    audio_format: AudioConfig | None = None,
    meta: dict[str, str] | None = None,
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "type": event_type,
        "session_id": session_id,
        "segment_id": segment_id,
        "text": text,
        "message": message,
        "meta": meta or {},
    }
    if audio_format is not None:
        event["audio"] = {
            "encoding": audio_format.encoding.value,
            "sample_rate": audio_format.sample_rate,
            "channels": audio_format.channels,
        }
    return {
        "type": "event",
        "event": event,
    }


async def _send_frame(ws, frame: dict[str, Any]) -> None:
    if frame.get("type") == "audio":
        await ws.send_bytes(frame["audio"]["pcm_data"])
        return
    await ws.send_json(frame)


def _is_terminal_frame(frame: dict[str, Any]) -> bool:
    if frame.get("type") != "event":
        return False
    return frame.get("event", {}).get("type") in {"done", "error"}


async def serve(
    engine: TTSEngine,
    port: int,
    *,
    stop_event: asyncio.Event,
    path: str = "/v1/ws",
) -> None:
    """Start the websocket gateway using aiohttp."""
    if web is None:
        logger.error("aiohttp not installed. Run: pip install aiohttp")
        return

    ws_path = _normalize_ws_path(path)
    gateway = WebSocketGateway(engine)
    app = web.Application()
    app.router.add_get(_CAPABILITIES_PATH, gateway.handle_capabilities)
    app.router.add_get(ws_path, gateway.handle_websocket)

    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    try:
        await site.start()
        logger.info(
            "WebSocket server listening on port %d (ws path %s, capabilities %s)",
            port,
            ws_path,
            _CAPABILITIES_PATH,
        )
        await stop_event.wait()
    finally:
        await runner.cleanup()


def _session_config_from_ws_message(
    message: dict[str, Any],
    *,
    default_mode: InputMode,
) -> SessionConfig:
    raw = _extract_ws_config_payload(message)
    cfg = SessionConfig(
        task_type=str(raw.get("task_type", "") or ""),
        language=str(raw.get("language", "auto") or "auto"),
        speaker=_optional_str(raw.get("speaker")),
        instruct=_optional_str(raw.get("instruct")),
        ref_audio=_decode_optional_base64(raw.get("ref_audio")),
        ref_text=_optional_str(raw.get("ref_text")),
        x_vector_only=_coerce_ws_bool(raw.get("x_vector_only", False)),
        input_mode=_input_mode_from_ws_value(raw.get("input_mode"), default_mode=default_mode),
        group_policy=_group_policy_from_ws_value(raw.get("group_policy")),
        audio=_audio_config_from_ws_value(raw.get("audio")),
    )
    _validate_audio_config(cfg.audio)
    return cfg


def _extract_ws_config_payload(message: dict[str, Any]) -> dict[str, Any]:
    cfg = message.get("config")
    if cfg is None:
        return {
            key: value
            for key, value in message.items()
            if key not in {"type", "text", "seq_no", "session_id"}
        }
    if not isinstance(cfg, dict):
        raise ValueError("websocket 'config' must be an object")
    merged = dict(cfg)
    for field in (
        "task_type",
        "language",
        "speaker",
        "instruct",
        "ref_audio",
        "ref_text",
        "x_vector_only",
        "input_mode",
        "group_policy",
        "audio",
    ):
        if field not in merged and field in message:
            merged[field] = message[field]
    return merged


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


def _decode_optional_base64(value: Any) -> bytes | None:
    if value in (None, ""):
        return None
    if isinstance(value, bytes):
        return value
    try:
        return base64.b64decode(str(value), validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("websocket 'ref_audio' must be valid base64") from exc


def _coerce_ws_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value in (None, "", 0):
        return False
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"false", "0", "no"}:
            return False
        if normalized in {"true", "1", "yes"}:
            return True
    return bool(value)


def _input_mode_from_ws_value(value: Any, *, default_mode: InputMode) -> InputMode:
    if value in (None, "", 0):
        return default_mode
    if isinstance(value, int):
        mapping = {
            1: InputMode.TOKEN,
            2: InputMode.CLAUSE,
            3: InputMode.LONG_SEGMENT,
            4: InputMode.FULL_TEXT,
        }
        if value in mapping:
            return mapping[value]
    normalized = str(value).strip().lower()
    mapping = {
        "token": InputMode.TOKEN,
        "clause": InputMode.CLAUSE,
        "long_segment": InputMode.LONG_SEGMENT,
        "full_text": InputMode.FULL_TEXT,
    }
    if normalized in mapping:
        return mapping[normalized]
    raise ValueError(f"unsupported input_mode: {value!r}")


def _group_policy_from_ws_value(value: Any) -> GroupPolicy:
    if value in (None, "", 0):
        return GroupPolicy.AUTO
    if isinstance(value, int):
        mapping = {
            1: GroupPolicy.NONE,
            2: GroupPolicy.AUTO,
        }
        if value in mapping:
            return mapping[value]
    normalized = str(value).strip().lower()
    mapping = {
        "none": GroupPolicy.NONE,
        "auto": GroupPolicy.AUTO,
    }
    if normalized in mapping:
        return mapping[normalized]
    raise ValueError(f"unsupported group_policy: {value!r}")


def _audio_config_from_ws_value(value: Any) -> AudioConfig:
    if value is None:
        return AudioConfig()
    if not isinstance(value, dict):
        raise ValueError("websocket 'audio' must be an object")
    encoding = _audio_encoding_from_ws_value(value.get("encoding"))
    sample_rate = int(value.get("sample_rate", 24000) or 24000)
    channels = int(value.get("channels", 1) or 1)
    return AudioConfig(
        sample_rate=sample_rate,
        encoding=encoding,
        channels=channels,
    )


def _audio_encoding_from_ws_value(value: Any) -> AudioEncoding:
    if value in (None, "", 0, 1):
        return AudioEncoding.PCM_F32
    if value == 2:
        return AudioEncoding.PCM_S16LE
    normalized = str(value).strip().lower()
    if normalized == "pcm_f32":
        return AudioEncoding.PCM_F32
    if normalized == "pcm_s16le":
        return AudioEncoding.PCM_S16LE
    raise ValueError(f"unsupported audio encoding: {value!r}")


def _validate_audio_config(audio: AudioConfig) -> None:
    if audio.channels != 1:
        raise ValueError(f"Unsupported channel count: {audio.channels} (mono only)")
    if audio.sample_rate not in (16000, 24000):
        raise ValueError(f"Unsupported sample_rate: {audio.sample_rate} (expected 16000 or 24000)")
    if audio.encoding not in (AudioEncoding.PCM_F32, AudioEncoding.PCM_S16LE):
        raise ValueError(f"Unsupported audio encoding: {audio.encoding}")


def _normalize_ws_path(path: str) -> str:
    normalized = str(path or "/v1/ws").strip()
    if not normalized:
        normalized = "/v1/ws"
    if not normalized.startswith("/"):
        normalized = f"/{normalized}"
    return normalized

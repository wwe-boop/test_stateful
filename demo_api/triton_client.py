from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, AsyncIterator

import numpy as np

from .schemas import RunMetrics, RunResult, TraceEvent


DEFAULT_TRITON_GRPC = os.environ.get("QWEN_DEMO_TRITON_GRPC", "localhost:8001")
DEFAULT_TRITON_MODEL = os.environ.get("QWEN_DEMO_TRITON_MODEL", "tts_orchestrator")


class TritonUnavailable(RuntimeError):
    pass


@dataclass
class TtsRequest:
    text: str
    speaker: str = "Serena"
    language: str = "auto"
    cache_mode: str = "hit"
    task_type: str = "custom_voice"
    audio_encoding: str = "pcm_f32"
    sample_rate: int = 24000
    input_mode: str | None = None
    group_policy: str | None = None


def _decode_obj(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _load_triton_client():
    try:
        import tritonclient.grpc as grpcclient
    except Exception as exc:  # pragma: no cover - depends on optional package
        raise TritonUnavailable("tritonclient[grpc] is not installed") from exc
    return grpcclient


def build_payload(request: TtsRequest) -> dict[str, Any]:
    payload = {
        "text": request.text,
        "task_type": request.task_type,
        "speaker": request.speaker,
        "language": request.language,
        "cache_mode": request.cache_mode,
        "audio": {
            "encoding": request.audio_encoding,
            "sample_rate": request.sample_rate,
            "channels": 1,
        },
    }
    if request.input_mode:
        payload["input_mode"] = request.input_mode
    if request.group_policy:
        payload["group_policy"] = request.group_policy
    return payload


def build_action_payload(
    action: str,
    session_id: str,
    *,
    text: str = "",
    request: TtsRequest | None = None,
) -> dict[str, Any]:
    """Build a payload for one tts_orchestrator action.

    `init` / `synthesize` carry the full session config (speaker, language,
    audio format). `append_text` carries `session_id` + `text`.
    `text_complete` and `cancel` only carry `session_id`.
    """
    payload: dict[str, Any] = {"action": action, "session_id": session_id}
    if action in ("init", "start", "synthesize"):
        if request is None:
            raise ValueError(f"action {action!r} requires a TtsRequest")
        payload.update(
            {
                "task_type": request.task_type,
                "speaker": request.speaker,
                "language": request.language,
                "cache_mode": request.cache_mode,
                "audio": {
                    "encoding": request.audio_encoding,
                    "sample_rate": request.sample_rate,
                    "channels": 1,
                },
            }
        )
        if request.input_mode:
            payload["input_mode"] = request.input_mode
        if request.group_policy:
            payload["group_policy"] = request.group_policy
        if action == "synthesize":
            payload["text"] = text or request.text
    elif action in ("append_text", "append"):
        payload["text"] = text
    return payload


def probe_ready(endpoint: str = DEFAULT_TRITON_GRPC, model_name: str = DEFAULT_TRITON_MODEL) -> bool:
    grpcclient = _load_triton_client()
    client = grpcclient.InferenceServerClient(url=endpoint)
    return bool(client.is_server_live() and client.is_server_ready() and client.is_model_ready(model_name))


async def measure_once(
    request: TtsRequest,
    *,
    endpoint: str = DEFAULT_TRITON_GRPC,
    model_name: str = DEFAULT_TRITON_MODEL,
    timeout_sec: float = 60.0,
) -> RunResult:
    run_id = f"trt-{uuid.uuid4().hex[:10]}"
    events: list[TraceEvent] = []
    chunks = 0
    audio_bytes = 0
    audio_parts: list[bytes] = []
    first_audio_ms: float | None = None
    total_ms: float | None = None
    audio_format = {"encoding": request.audio_encoding, "sample_rate": request.sample_rate, "channels": 1}

    async for item in stream_once(
        request,
        run_id=run_id,
        endpoint=endpoint,
        model_name=model_name,
        timeout_sec=timeout_sec,
    ):
        if item["kind"] == "event":
            event = item["event"]
            events.append(event)
            if event.type == "first_audio_chunk":
                first_audio_ms = event.t_ms
            if event.type == "done":
                total_ms = event.t_ms
                if "audio_format" in event.meta:
                    audio_format = dict(event.meta["audio_format"])
        elif item["kind"] == "audio":
            chunks += 1
            audio_bytes += len(item["audio"])
            audio_parts.append(item["audio"])

    if first_audio_ms is None:
        raise TritonUnavailable("Triton stream completed without audio")
    if total_ms is None:
        total_ms = max(event.t_ms for event in events) if events else first_audio_ms

    server_ttft_ms = None
    triton_adapter_ttft_ms = None
    for event in events:
        if event.type == "first_audio_chunk":
            raw = event.meta.get("server_ttft_ms") or event.meta.get("ttft_ms")
            if raw is not None:
                try:
                    server_ttft_ms = float(raw)
                except (TypeError, ValueError):
                    server_ttft_ms = None
            raw_adapter = event.meta.get("triton_adapter_ttft_ms")
            if raw_adapter is not None:
                try:
                    triton_adapter_ttft_ms = float(raw_adapter)
                except (TypeError, ValueError):
                    triton_adapter_ttft_ms = None
            break

    audio_duration_ms = None
    if audio_bytes and audio_format.get("encoding") == "pcm_f32":
        audio_duration_ms = audio_bytes / 4.0 / float(audio_format.get("sample_rate", 24000)) * 1000.0

    return RunResult(
        run_id=run_id,
        backend="triton_trt_streaming",
        label="Triton TRT Streaming",
        mode="TensorRT + graph optimized + token-level streaming",
        source="live_triton",
        metrics=RunMetrics(
            server_ttft_ms=server_ttft_ms,
            triton_adapter_ttft_ms=triton_adapter_ttft_ms,
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
        raw_audio=b"".join(audio_parts) if audio_parts else None,
    )


async def stream_once(
    request: TtsRequest,
    *,
    run_id: str,
    endpoint: str = DEFAULT_TRITON_GRPC,
    model_name: str = DEFAULT_TRITON_MODEL,
    timeout_sec: float = 60.0,
) -> AsyncIterator[dict[str, Any]]:
    grpcclient = _load_triton_client()
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    loop = asyncio.get_running_loop()
    client = grpcclient.InferenceServerClient(url=endpoint)
    req_json = json.dumps(build_payload(request), ensure_ascii=False)

    req_input = grpcclient.InferInput("request", [1], "BYTES")
    req_input.set_data_from_numpy(np.array([req_json], dtype=object))
    outputs = [
        grpcclient.InferRequestedOutput("audio_chunk"),
        grpcclient.InferRequestedOutput("event_type"),
        grpcclient.InferRequestedOutput("event_json"),
        grpcclient.InferRequestedOutput("is_final"),
    ]

    started = time.perf_counter()
    first_audio_seen = False
    audio_format: dict[str, Any] = {
        "encoding": request.audio_encoding,
        "sample_rate": request.sample_rate,
        "channels": 1,
    }

    def emit(item: dict[str, Any]) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, item)

    def callback(result, error) -> None:  # pragma: no cover - exercised against live Triton
        nonlocal first_audio_seen, audio_format
        now_ms = (time.perf_counter() - started) * 1000.0
        if error:
            emit({"kind": "error", "message": str(error), "t_ms": now_ms})
            return
        try:
            event_type_arr = result.as_numpy("event_type")
            event_json_arr = result.as_numpy("event_json")
            audio_arr = result.as_numpy("audio_chunk")
            is_final_arr = result.as_numpy("is_final")
            event_type = (
                _decode_obj(event_type_arr.flatten()[0])
                if event_type_arr is not None and event_type_arr.size
                else ""
            )
            payload: dict[str, Any] = {}
            if event_json_arr is not None and event_json_arr.size:
                raw = _decode_obj(event_json_arr.flatten()[0])
                if raw:
                    parsed = json.loads(raw)
                    if isinstance(parsed, dict):
                        payload = parsed
            if event_type == "start":
                audio_format.update(payload.get("audio_format", {}) or {})
                emit(
                    {
                        "kind": "event",
                        "event": TraceEvent(
                            run_id=run_id,
                            backend="triton_trt_streaming",
                            type="request_started",
                            t_ms=0.0,
                            meta={"audio_format": dict(audio_format), "source_event": "start"},
                        ),
                    }
                )
            elif event_type == "audio" and audio_arr is not None and audio_arr.size:
                raw_audio = audio_arr.flatten()[0]
                if isinstance(raw_audio, str):
                    raw_audio = raw_audio.encode("utf-8")
                event_meta = dict(payload.get("meta", {}) or {})
                trace_type = "audio_chunk"
                if not first_audio_seen:
                    first_audio_seen = True
                    trace_type = "first_audio_chunk"
                event_meta.update({"bytes": len(raw_audio), "audio_format": dict(audio_format)})
                emit(
                    {
                        "kind": "event",
                        "event": TraceEvent(
                            run_id=run_id,
                            backend="triton_trt_streaming",
                            type=trace_type,
                            t_ms=now_ms,
                            meta=event_meta,
                        ),
                    }
                )
                emit({"kind": "audio", "audio": bytes(raw_audio), "t_ms": now_ms})
            elif event_type in {"warning", "text_token", "text_boundary_commit", "segment_end"}:
                emit(
                    {
                        "kind": "event",
                        "event": TraceEvent(
                            run_id=run_id,
                            backend="triton_trt_streaming",
                            type=event_type,
                            t_ms=now_ms,
                            text=str(payload.get("text") or ""),
                            meta=payload,
                        ),
                    }
                )
            elif event_type == "error":
                emit({"kind": "error", "message": str(payload.get("message") or "Triton error"), "t_ms": now_ms})
                return
            if is_final_arr is not None and is_final_arr.size and bool(is_final_arr.flatten()[0]):
                emit(
                    {
                        "kind": "event",
                        "event": TraceEvent(
                            run_id=run_id,
                            backend="triton_trt_streaming",
                            type="done",
                            t_ms=now_ms,
                            meta={"audio_format": dict(audio_format), "source_event": event_type},
                        ),
                    }
                )
                emit({"kind": "complete"})
        except Exception as exc:
            emit({"kind": "error", "message": str(exc), "t_ms": now_ms})

    client.start_stream(callback=callback)
    try:
        client.async_stream_infer(
            model_name=model_name,
            inputs=[req_input],
            outputs=outputs,
        )
        deadline = time.perf_counter() + timeout_sec
        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise TimeoutError(f"Triton stream timed out after {timeout_sec:.1f}s")
            item = await asyncio.wait_for(queue.get(), timeout=remaining)
            if item["kind"] == "complete":
                break
            if item["kind"] == "error":
                raise TritonUnavailable(item["message"])
            yield item
    finally:
        client.stop_stream()

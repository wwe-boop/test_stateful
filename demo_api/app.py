from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import uuid
from typing import Any

from aiohttp import WSMsgType, web

from .audio_assets import attach_audio_to_result
from .audio_store import AudioStore
from . import engine_client
from .jobs import ConcurrencyJobManager
from .official_pytorch import (
    OfficialPyTorchRunner,
    is_official_streaming_info_warning,
)
from .race_capture import RaceCaptureJobManager
from .schemas import TraceEvent
from .trace_store import TraceStore, replace_backend_result
from .triton_client import TtsRequest, measure_once, probe_ready, stream_once


DEFAULT_TEXT = "你好，这是千问3 TTS token级流式语音演示。"
DEFAULT_SPEAKER = os.environ.get("QWEN_DEMO_DEFAULT_SPEAKER", "Serena")
DEFAULT_LANGUAGE = os.environ.get("QWEN_DEMO_DEFAULT_LANGUAGE", "auto")
TRITON_GRPC = os.environ.get("QWEN_DEMO_TRITON_GRPC", "localhost:8001")
TRITON_MODEL = os.environ.get("QWEN_DEMO_TRITON_MODEL", "tts_orchestrator")
TRITON_MAX_BATCH_SLOTS = int(os.environ.get("QWEN_DEMO_TRITON_MAX_BATCH_SLOTS", os.environ.get("TRITON_MAX_BATCH_SLOTS", "64")))
TRITON_MAX_SESSIONS = int(os.environ.get("QWEN_DEMO_TRITON_MAX_SESSIONS", os.environ.get("TRITON_MAX_SESSIONS", "128")))

RELEASE_METADATA = {
    "stage": "engineering_preview",
    "positioning": "工程预览版：展示 Qwen3-TTS TensorRT/token streaming 优化链路，不承诺生产稳定性。",
    "recommended_variant": "custom-1.7b",
    "stable_paths": ["custom_voice"],
    "experimental_paths": ["voice_design"],
    "planned_paths": ["base_voice_clone", "icl_voice_clone"],
}

STREAMING_LIMITATIONS = [
    "当前稳定开源范围优先限定为 custom-1.7b/custom_voice 路径。",
    "流式模式仍可能出现幻觉、重复、漏读、插入未提供内容，长文本更容易触发。",
    "13ms TTFT 只代表特定硬件、warm engine、cache 命中、单路请求下的最低观测值。",
    "WebUI 在 live 后端不可用时只展示 fixture trace/metrics，不再补 synthetic beep 音频。",
    "只有 source 标记为 live_triton、live_engine_websocket 或 live_official_pytorch 且带 audio 的结果才可回放真实合成音频。",
    "Performance PK 中官方 PyTorch TTFT 近似值按 decode 阶段第一个 code0 出现时间减去请求开始时间统计；public API 实际仍是完整 waveform 返回。",
]


@web.middleware
async def cors_middleware(request: web.Request, handler):
    if request.method == "OPTIONS":
        response = web.Response(status=204)
    else:
        response = await handler(request)
    response.headers["Access-Control-Allow-Origin"] = os.environ.get("QWEN_DEMO_CORS_ORIGIN", "*")
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "content-type"
    return response


def create_app() -> web.Application:
    store = TraceStore()
    audio_store = AudioStore()
    jobs = ConcurrencyJobManager(
        enable_live=os.environ.get("QWEN_DEMO_ENABLE_LIVE_CONCURRENCY", "1").lower()
        in {"1", "true", "yes", "on"},
        audio_store=audio_store,
        live_slot_limit=TRITON_MAX_BATCH_SLOTS,
    )
    official_runner = OfficialPyTorchRunner()
    app = web.Application(middlewares=[cors_middleware])
    app["trace_store"] = store
    app["jobs"] = jobs
    app["audio_store"] = audio_store
    app["official_runner"] = official_runner
    app["race_capture_jobs"] = RaceCaptureJobManager(trace_store=store)
    app.router.add_get("/healthz", handle_healthz)
    app.router.add_get("/api/v1/capabilities", handle_capabilities)
    app.router.add_get("/api/v1/audio/{audio_id}", handle_audio)
    app.router.add_post("/api/v1/race", handle_race)
    app.router.add_post("/api/v1/race-capture", handle_race_capture_start)
    app.router.add_get("/api/v1/race-capture/{job_id}", handle_race_capture_ws)
    app.router.add_get("/api/v1/trt-live", handle_trt_live)
    app.router.add_post("/api/v1/concurrency", handle_concurrency_start)
    app.router.add_get("/api/v1/concurrency/{job_id}", handle_concurrency_ws)
    return app


async def handle_healthz(request: web.Request) -> web.Response:
    return web.json_response({"ok": True})


async def handle_capabilities(request: web.Request) -> web.Response:
    triton_ready = False
    engine_ready = False
    triton_error = ""
    try:
        triton_ready = await asyncio.to_thread(probe_ready, TRITON_GRPC, TRITON_MODEL)
    except Exception as exc:
        triton_error = str(exc)
    engine_ready = await engine_client.probe_ready()

    store: TraceStore = request.app["trace_store"]
    jobs: ConcurrencyJobManager = request.app["jobs"]
    race = store.load_default_race()
    return web.json_response(
        {
            "default_request": {
                "text": race.get("default_request", {}).get("text") or DEFAULT_TEXT,
                "speaker": race.get("default_request", {}).get("speaker") or DEFAULT_SPEAKER,
                "language": race.get("default_request", {}).get("language") or DEFAULT_LANGUAGE,
                "cache_mode": race.get("default_request", {}).get("cache_mode") or "hit",
            },
            "backends": [
                {
                    "id": "official_pytorch_offline",
                    "label": "Official PyTorch Offline",
                    "streaming": False,
                    "live_available": request.app["official_runner"].enabled(),
                },
                {
                    "id": "official_pytorch_streaming",
                    "label": "Official PyTorch Online Text",
                    "streaming": True,
                    "live_available": request.app["official_runner"].enabled(),
                },
                {
                    "id": "bare_engine_streaming",
                    "label": "Bare Engine Streaming",
                    "streaming": True,
                    "live_available": engine_ready,
                    "endpoint": engine_client.DEFAULT_ENGINE_WS,
                },
                {
                    "id": "triton_trt_streaming",
                    "label": "Triton TRT Streaming",
                    "streaming": True,
                    "live_available": triton_ready,
                    "endpoint": TRITON_GRPC,
                    "model": TRITON_MODEL,
                    "error": triton_error,
                    "runtime": {
                        "max_batch_slots": TRITON_MAX_BATCH_SLOTS,
                        "max_sessions": TRITON_MAX_SESSIONS,
                    },
                },
            ],
            "concurrency": {
                "live_enabled": jobs.enable_live,
                "triton_active_slot_limit": TRITON_MAX_BATCH_SLOTS,
                "triton_max_sessions": TRITON_MAX_SESSIONS,
            },
            "benchmark_conditions": race.get("benchmark_conditions", {}),
            "release": RELEASE_METADATA,
            "limitations": STREAMING_LIMITATIONS,
            "headline": {
                "single_stream_cache_hit_ttft_ms": 13,
                "concurrent_128_avg_ttft_ms": 180,
            },
        }
    )


async def handle_race(request: web.Request) -> web.Response:
    body = await _read_json(request)
    store: TraceStore = request.app["trace_store"]
    audio_store: AudioStore = request.app["audio_store"]
    official_runner: OfficialPyTorchRunner = request.app["official_runner"]
    race = store.load_default_race()
    warnings: list[str] = list(race.get("warnings", []) or [])
    use_live_triton = body.get("use_live_triton", True)

    if body.get("live_baselines"):
        for backend, streaming_mode in (
            ("official_pytorch_offline", False),
            ("official_pytorch_streaming", True),
        ):
            try:
                official_result = await asyncio.to_thread(
                    official_runner.synthesize,
                    text=str(body.get("text") or DEFAULT_TEXT),
                    speaker=str(body.get("speaker") or DEFAULT_SPEAKER),
                    language=str(body.get("language") or DEFAULT_LANGUAGE),
                    streaming_mode=streaming_mode,
                )
                attach_audio_to_result(official_result, audio_store)
                race = replace_backend_result(race, backend, official_result.to_dict())
                warnings.extend(
                    warning
                    for warning in official_result.warnings
                    if not is_official_streaming_info_warning(warning)
                )
            except Exception as exc:
                warnings.append(f"{backend} live official API unavailable, using fixture trace/metrics without synthetic audio: {exc}")

    if body.get("use_live_engine", True):
        try:
            engine_result = await engine_client.measure_once(
                _tts_request_from_body(body),
                timeout_sec=float(body.get("timeout_sec") or 60.0),
            )
            attach_audio_to_result(engine_result, audio_store)
            race = replace_backend_result(race, "bare_engine_streaming", engine_result.to_dict())
        except Exception as exc:
            warnings.append(f"Bare engine live measurement unavailable, using fixture trace/metrics without synthetic audio: {exc}")

    if use_live_triton:
        try:
            result = await measure_once(
                _tts_request_from_body(body),
                endpoint=TRITON_GRPC,
                model_name=TRITON_MODEL,
                timeout_sec=float(body.get("timeout_sec") or 60.0),
            )
            attach_audio_to_result(result, audio_store)
            race = replace_backend_result(race, "triton_trt_streaming", result.to_dict())
        except Exception as exc:
            warnings.append(f"Triton live measurement unavailable, using fixture trace/metrics without synthetic audio: {exc}")

    response = {
        "type": "race_result",
        "request": {
            "text": body.get("text") or race.get("default_request", {}).get("text") or DEFAULT_TEXT,
            "speaker": body.get("speaker") or race.get("default_request", {}).get("speaker") or DEFAULT_SPEAKER,
            "language": body.get("language") or race.get("default_request", {}).get("language") or DEFAULT_LANGUAGE,
            "cache_mode": body.get("cache_mode") or race.get("default_request", {}).get("cache_mode") or "hit",
        },
        "benchmark_conditions": race.get("benchmark_conditions", {}),
        "release": RELEASE_METADATA,
        "limitations": STREAMING_LIMITATIONS,
        "results": race.get("results", []),
        "warnings": warnings,
        "source_path": race.get("source_path", ""),
    }
    return web.json_response(response)


async def handle_race_capture_start(request: web.Request) -> web.Response:
    body = await _read_json(request)
    manager: RaceCaptureJobManager = request.app["race_capture_jobs"]
    job = await manager.create_job(
        {
            "variant": body.get("variant") or os.environ.get("MODEL_VARIANT") or "custom-1.7b",
            "text": body.get("text") or DEFAULT_TEXT,
            "speaker": body.get("speaker") or DEFAULT_SPEAKER,
            "language": body.get("language") or DEFAULT_LANGUAGE,
            "cache_mode": body.get("cache_mode") or "hit",
            "timeout_sec": float(body.get("timeout_sec") or 120.0),
            "engine_retries": int(body.get("engine_retries") or os.environ.get("QWEN_DEMO_ENGINE_CAPTURE_RETRIES") or 4),
            "official_warmup_rounds": int(body.get("official_warmup_rounds") or os.environ.get("QWEN_DEMO_OFFICIAL_WARMUP_ROUNDS") or 1),
            "official_warmup_text": body.get("official_warmup_text") or os.environ.get("QWEN_DEMO_OFFICIAL_WARMUP_TEXT") or "你好。",
            "triton_slots": int(body.get("triton_slots") or TRITON_MAX_BATCH_SLOTS),
            "strict": bool(body.get("strict")),
            "skip_official": bool(body.get("skip_official")),
            "skip_engine": bool(body.get("skip_engine")),
            "skip_triton": bool(body.get("skip_triton")),
        }
    )
    return web.json_response({"job_id": job.job_id})


async def handle_race_capture_ws(request: web.Request) -> web.StreamResponse:
    manager: RaceCaptureJobManager = request.app["race_capture_jobs"]
    job = manager.get(request.match_info["job_id"])
    if job is None:
        return web.json_response({"error": "unknown job_id"}, status=404)
    ws = web.WebSocketResponse(heartbeat=30.0)
    await ws.prepare(request)
    queue = await manager.subscribe(job)
    try:
        while True:
            try:
                message = await asyncio.wait_for(queue.get(), timeout=30.0)
            except asyncio.TimeoutError:
                await ws.send_json({"type": "heartbeat"})
                continue
            if message.get("type") == "race_capture_done":
                race = message.get("race", {})
                message = {
                    **message,
                    "race": _race_payload_from_trace(
                        race,
                        {
                            "text": job.request.get("text") or DEFAULT_TEXT,
                            "speaker": job.request.get("speaker") or DEFAULT_SPEAKER,
                            "language": job.request.get("language") or DEFAULT_LANGUAGE,
                            "cache_mode": job.request.get("cache_mode") or "hit",
                        },
                    ),
                }
            await ws.send_json(message)
            if message.get("type") in {"race_capture_done", "race_capture_error"}:
                break
    finally:
        manager.unsubscribe(job, queue)
        await ws.close()
    return ws


async def handle_audio(request: web.Request) -> web.StreamResponse:
    audio_store: AudioStore = request.app["audio_store"]
    path = audio_store.path_for(request.match_info["audio_id"])
    if path is None:
        raise web.HTTPNotFound(text="audio not found")
    return web.FileResponse(path, headers={"Cache-Control": "public, max-age=3600"})


async def handle_trt_live(request: web.Request) -> web.StreamResponse:
    ws = web.WebSocketResponse(heartbeat=30.0)
    await ws.prepare(request)
    async for message in ws:
        if message.type == WSMsgType.TEXT:
            try:
                payload = json.loads(message.data)
            except json.JSONDecodeError as exc:
                await ws.send_json({"type": "error", "message": f"invalid JSON: {exc}"})
                continue
            if payload.get("type") != "speak":
                await ws.send_json({"type": "error", "message": "expected message type 'speak'"})
                continue
            await _stream_trt_live(ws, payload)
        elif message.type == WSMsgType.ERROR:
            break
    return ws


async def _stream_trt_live(ws: web.WebSocketResponse, payload: dict[str, Any]) -> None:
    run_id = f"trt-live-{uuid.uuid4().hex[:10]}"
    started = time.perf_counter()
    await ws.send_json(
        {
            "type": "event",
            "event": TraceEvent(
                run_id=run_id,
                backend="triton_trt_streaming",
                type="request_started",
                t_ms=0.0,
                meta={"source": "live_triton"},
            ).to_dict(),
        }
    )
    try:
        async for item in stream_once(
            _tts_request_from_body(payload),
            run_id=run_id,
            endpoint=TRITON_GRPC,
            model_name=TRITON_MODEL,
            timeout_sec=float(payload.get("timeout_sec") or 60.0),
        ):
            if item["kind"] == "event":
                await ws.send_json({"type": "event", "event": item["event"].to_dict()})
            elif item["kind"] == "audio":
                await ws.send_bytes(item["audio"])
        return
    except Exception as exc:
        await ws.send_json(
            {
                "type": "event",
                "event": TraceEvent(
                    run_id=run_id,
                    backend="triton_trt_streaming",
                    type="warning",
                    t_ms=(time.perf_counter() - started) * 1000.0,
                    meta={
                        "source": "live_triton",
                        "message": f"live TRT stream unavailable; no synthetic audio emitted: {exc}",
                    },
                ).to_dict(),
            }
        )
        await ws.send_json({"type": "error", "message": str(exc)})


async def handle_concurrency_start(request: web.Request) -> web.Response:
    body = await _read_json(request)
    jobs: ConcurrencyJobManager = request.app["jobs"]
    job = await jobs.create_job(
        {
            "text": body.get("text") or DEFAULT_TEXT,
            "speaker": body.get("speaker") or DEFAULT_SPEAKER,
            "language": body.get("language") or DEFAULT_LANGUAGE,
            "cache_mode": body.get("cache_mode") or "hit",
            "concurrency": int(body.get("concurrency") or 128),
            "live": bool(body.get("live")),
            "timeout_sec": float(body.get("timeout_sec") or 60.0),
        }
    )
    return web.json_response({"job_id": job.job_id})


async def handle_concurrency_ws(request: web.Request) -> web.StreamResponse:
    jobs: ConcurrencyJobManager = request.app["jobs"]
    job = jobs.get(request.match_info["job_id"])
    if job is None:
        return web.json_response({"error": "unknown job_id"}, status=404)
    ws = web.WebSocketResponse(heartbeat=30.0)
    await ws.prepare(request)
    queue = await jobs.subscribe(job)
    try:
        while True:
            try:
                message = await asyncio.wait_for(queue.get(), timeout=30.0)
            except asyncio.TimeoutError:
                await ws.send_json({"type": "heartbeat"})
                continue
            await ws.send_json(message)
            if message.get("type") == "summary":
                break
    finally:
        jobs.unsubscribe(job, queue)
        await ws.close()
    return ws


async def _read_json(request: web.Request) -> dict[str, Any]:
    if request.content_length == 0:
        return {}
    try:
        payload = await request.json()
    except json.JSONDecodeError as exc:
        raise web.HTTPBadRequest(text=f"invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise web.HTTPBadRequest(text="JSON body must be an object")
    return payload


def _tts_request_from_body(body: dict[str, Any]) -> TtsRequest:
    return TtsRequest(
        text=str(body.get("text") or DEFAULT_TEXT),
        speaker=str(body.get("speaker") or DEFAULT_SPEAKER),
        language=str(body.get("language") or DEFAULT_LANGUAGE),
        cache_mode=str(body.get("cache_mode") or "hit"),
    )


def _race_payload_from_trace(race: dict[str, Any], request_data: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "race_result",
        "request": {
            "text": request_data.get("text") or race.get("default_request", {}).get("text") or DEFAULT_TEXT,
            "speaker": request_data.get("speaker") or race.get("default_request", {}).get("speaker") or DEFAULT_SPEAKER,
            "language": request_data.get("language") or race.get("default_request", {}).get("language") or DEFAULT_LANGUAGE,
            "cache_mode": request_data.get("cache_mode") or race.get("default_request", {}).get("cache_mode") or "hit",
        },
        "benchmark_conditions": race.get("benchmark_conditions", {}),
        "release": RELEASE_METADATA,
        "limitations": STREAMING_LIMITATIONS,
        "results": race.get("results", []),
        "warnings": list(race.get("warnings", []) or []),
        "source_path": race.get("source_path", ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3-TTS Triton WebUI demo API")
    parser.add_argument("--host", default=os.environ.get("QWEN_DEMO_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("QWEN_DEMO_PORT", "7860")))
    args = parser.parse_args()
    web.run_app(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()

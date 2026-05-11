"""TTS Engine server: wires asyncio frontend + GPU engine thread.

Architecture:

    ┌────────── asyncio event loop (main thread) ──────────┐
    │                                                       │
    │  gRPC / WebSocket  ─►  FrontendInterface ─► Dispatcher │
    │                                       │         │      │
    │       ▲                                    │          │
    │       │ audio chunks                       │          │
    │       │ (call_soon_threadsafe)              │          │
    │  session.result_queue  ◄───────────────┐   │          │
    └───────────────────────────────────────┼───┼──────────┘
                                            │   │
                    ┌── Engine Thread ───────┼───┼──────────┐
                    │                       │   ▼           │
                    │  engine_inbox.get() → EngineLoop      │
                    │                      ┌────────────┐   │
                    │                      │ Executor    │   │
                    │                      │ KVCachePool │   │
                    │                      │ PrefillBld  │   │
                    │                      │ PrefixCache │   │
                    │                      └────────────┘   │
                    └───────────────────────────────────────┘

Usage:
    python -m engine.server --config engine.yaml
    python -m engine.server --model-package-dir /models/tts_orchestrator/1
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import queue
import signal
from typing import AsyncIterator, Optional

import torch

from .config import (
    EngineConfig,
    ModelArchConfig,
    apply_model_package_paths,
    load_config,
    load_model_manifest,
    resolve_model_package_paths,
    to_model_config,
)
from .core.mlfq import MLFQConfig
from .core.types import SessionConfig
from .frontend.interface import FrontendInterface
from .frontend.spliter.tokenizer import LightQwen3TTSTokenizer
from .backend.engine_loop import EngineLoop
from .backend.executor import Executor
from .backend.prefill import EmbeddingWeights, PrefillBuilder
from .backend.ref_audio_processor import ReferenceAudioProcessor

logger = logging.getLogger(__name__)


_EXTERNAL_TO_INTERNAL_TASK_TYPE = {
    "base": "voice_clone",
    "icl": "voice_clone",
    "voice_clone": "voice_clone",
    "custom_voice": "custom_voice",
    "voice_design": "voice_design",
    "instruct": "voice_design",
}


def _is_cuda_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    msg = str(exc).lower()
    return "cuda out of memory" in msg or "out of memory" in msg and "cuda" in msg


class TTSEngine:
    """Top-level engine that owns both the asyncio world and the GPU thread."""

    def __init__(
        self,
        config: Optional[EngineConfig] = None,
        model_arch: Optional[ModelArchConfig] = None,
        *,
        tokenizer_dir: str = "",
        weights_dir: str = "",
        engine_dir: str = "",
        device_id: int = 0,
        max_batch_size: int = 48,
        max_sessions: int = 128,
        max_seq_len: int = 512,
    ):
        self._cfg = config or EngineConfig()
        self._model_arch = model_arch or ModelArchConfig()

        self._loop: Optional[asyncio.AbstractEventLoop] = None

        self._engine_inbox = queue.Queue(maxsize=4096)
        self._async_inbox: Optional[asyncio.Queue] = None

        self._tokenizer_dir = tokenizer_dir or self._cfg.paths.tokenizer_dir
        self._weights_dir = weights_dir or self._cfg.paths.weights_dir
        self._engine_dir = engine_dir or self._cfg.paths.engine_dir
        self._device_id = device_id
        self._max_batch = max_batch_size if max_batch_size != 48 else self._cfg.scheduler.max_batch_size
        self._max_sessions = max_sessions if max_sessions != 128 else self._cfg.server.max_sessions
        self._max_seq_len = max_seq_len if max_seq_len != 512 else self._cfg.scheduler.max_seq_len
        self._validate_runtime_profile_bounds()
        self._cfg.scheduler.max_batch_size = self._max_batch
        self._cfg.scheduler.max_seq_len = self._max_seq_len
        self._cfg.server.max_sessions = self._max_sessions

        self._tokenizer: Optional[LightQwen3TTSTokenizer] = None
        self._frontend: Optional[FrontendInterface] = None
        self._executor: Optional[Executor] = None
        self._engine_loop: Optional[EngineLoop] = None
        self._relay_task: Optional[asyncio.Task] = None
        self._ref_audio_processor: Optional[ReferenceAudioProcessor] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _validate_runtime_profile_bounds(self) -> None:
        """Fail early if requested runtime limits exceed the built TRT profile."""
        profile = self._model_arch.engine_profile
        variant = self._model_arch.variant or "unknown"
        if profile.max_batch_size > 0 and self._max_batch > profile.max_batch_size:
            raise ValueError(
                f"runtime max_batch_size={self._max_batch} exceeds engine profile "
                f"max_batch_size={profile.max_batch_size} for variant '{variant}'. "
                "Lower --max-batch / ENGINE_SCHEDULER_MAX_BATCH_SIZE, or rebuild Phase B with "
                f"`bash scripts/bash/build_engines.sh --variant {variant} "
                f"--max-batch-size {self._max_batch}`."
            )
        if profile.max_seq_len > 0 and self._max_seq_len > profile.max_seq_len:
            raise ValueError(
                f"runtime max_seq_len={self._max_seq_len} exceeds engine profile "
                f"max_seq_len={profile.max_seq_len} for variant '{variant}'. "
                "Lower --max-seq-len / ENGINE_SCHEDULER_MAX_SEQ_LEN, or rebuild Phase B with "
                f"`bash scripts/bash/build_engines.sh --variant {variant} "
                f"--max-seq-len {self._max_seq_len}`."
            )

    async def start(self) -> None:
        """Initialize all components, warm up, and start the engine thread."""
        self._loop = asyncio.get_event_loop()
        self._async_inbox = asyncio.Queue(maxsize=4096)

        self._tokenizer = LightQwen3TTSTokenizer(self._tokenizer_dir)

        model_config = to_model_config(self._model_arch, self._cfg)
        sampling = self._cfg.sampling
        package_paths = (
            resolve_model_package_paths(self._cfg.paths.model_package_dir)
            if self._cfg.paths.model_package_dir
            else None
        )
        runtime_artifact = package_paths.runtime_artifact_path if package_paths else ""
        if runtime_artifact:
            logger.info(
                "Resolved model package: package=%s mode=%s runtime_artifact=%s",
                package_paths.package_dir,
                package_paths.engine_mode,
                runtime_artifact,
            )
        self._executor = Executor(
            engine_dir=self._engine_dir,
            weights_dir=self._weights_dir,
            device_id=self._device_id,
            max_batch_size=self._max_batch,
            max_seq_len=self._max_seq_len,
            model_config=model_config,
            do_sample=sampling.do_sample,
            temperature=sampling.temperature,
            repetition_penalty=sampling.repetition_penalty,
            random_seed=sampling.random_seed,
        )
        self._executor.load()
        self._max_batch = self._executor.max_batch_size
        self._max_seq_len = self._executor.max_seq_len

        sc = self._cfg.spliter
        self._frontend = FrontendInterface(
            engine_inbox=self._async_inbox,
            tokenizer=self._tokenizer,
            max_sessions=self._max_sessions,
            engine_max_decode_len=self._max_seq_len,
            prefill_len=sc.prefill_len,
            ema_ratio=sc.ema_ratio_initial,
            max_concurrent_segments=sc.max_concurrent_segments,
            ema_alpha=sc.ema_alpha,
            ema_overflow_alpha=sc.ema_overflow_alpha,
            ema_min_ratio=sc.ema_min_ratio,
            ema_max_ratio=sc.ema_max_ratio,
            safety_margin=sc.safety_margin,
            l1_split_cap_ratio=sc.l1_split_cap_ratio,
            l2_split_cap_ratio=sc.l2_split_cap_ratio,
            l3_split_cap_ratio=sc.l3_split_cap_ratio,
        )
        self._ref_audio_processor = ReferenceAudioProcessor(
            self._engine_dir,
            self._model_arch.variant,
        )

        prefill_builder = None
        if self._weights_dir:
            try:
                pf = self._cfg.prefill
                try:
                    emb_weights = EmbeddingWeights(
                        self._weights_dir,
                        self._device_id,
                        default_speaker=pf.default_speaker,
                        fallback_speaker=pf.fallback_speaker,
                    )
                except Exception as exc:
                    if not _is_cuda_oom(exc):
                        raise
                    logger.warning(
                        "Loading embedding weights on CUDA failed with OOM; retrying on CPU: %s",
                        exc,
                    )
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    emb_weights = EmbeddingWeights(
                        self._weights_dir,
                        self._device_id,
                        device="cpu",
                        default_speaker=pf.default_speaker,
                        fallback_speaker=pf.fallback_speaker,
                    )
                self._executor.set_embedding_weights(emb_weights)
                prefill_builder = PrefillBuilder(
                    emb_weights,
                    self._tokenizer,
                    output_device=self._executor._device,
                )
                logger.info(
                    "PrefillBuilder loaded from %s (weights_device=%s, output_device=%s)",
                    self._weights_dir,
                    emb_weights.device,
                    self._executor._device,
                )
            except Exception as e:
                logger.warning("Could not load embedding weights: %s", e)

        if self._cfg.server.warmup_rounds > 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._executor.warmup(n_rounds=self._cfg.server.warmup_rounds)

        sched = self._cfg.scheduler
        pc = self._cfg.prefix_cache
        mlfq_cfg = MLFQConfig(
            q1_threshold=sched.mlfq_q1_threshold,
            q2_threshold=sched.mlfq_q2_threshold,
            aging_interval=sched.mlfq_aging_interval,
            starvation_limit=sched.mlfq_starvation_limit,
        )

        self._engine_loop = EngineLoop(
            engine_inbox=self._engine_inbox,
            async_loop=self._loop,
            executor=self._executor,
            prefill_builder=prefill_builder,
            max_batch_size=self._max_batch,
            mlfq_config=mlfq_cfg,
            prefix_cache_max_entries=pc.max_entries if pc.enabled else 0,
            prefix_cache_max_len=pc.max_prefix_len,
            max_idle_sec=sched.max_idle_sec,
            max_queue_size=sched.max_queue_size,
            session_timeout_sec=sched.session_timeout_sec,
            min_pad_steps=sched.min_pad_steps,
            pad_silence_peak_threshold=sched.pad_silence_peak_threshold,
            pad_silence_mean_abs_threshold=sched.pad_silence_mean_abs_threshold,
            max_slots_per_session=self._cfg.spliter.max_concurrent_segments,
        )
        self._engine_loop.start()

        self._relay_task = asyncio.create_task(self._relay_inbox())
        logger.info("TTS Engine started (max_batch=%d, max_sessions=%d, mlfq=%s, prefix_cache=%s)",
                     self._max_batch, self._max_sessions,
                     "enabled", "enabled" if pc.enabled else "disabled")

    async def stop(self) -> None:
        if self._relay_task:
            self._relay_task.cancel()
        if self._engine_loop:
            self._engine_loop.stop()
        if self._executor:
            self._executor.shutdown()
        logger.info("TTS Engine stopped")

    # ------------------------------------------------------------------
    # Public API (called by Gateway / gRPC handlers)
    # ------------------------------------------------------------------

    async def synthesize_stream(
        self,
        session_id: str,
        *,
        speaker_key: Optional[str] = None,
        task_type: Optional[str] = None,
        ref_audio: Optional[bytes] = None,
    ) -> AsyncIterator[bytes]:
        """Create a session and yield audio chunks as they are produced."""
        audio_queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue()

        async def on_audio(sid: str, data: bytes) -> None:
            await audio_queue.put(data)

        async def on_done(sid: str, metrics: dict) -> None:
            await audio_queue.put(None)

        await self.start_session(
            session_id,
            config=SessionConfig(
                task_type=task_type or "",
                speaker=speaker_key,
                ref_audio=ref_audio,
            ),
            on_audio=on_audio,
            on_done=on_done,
        )

        while True:
            chunk = await audio_queue.get()
            if chunk is None:
                break
            yield chunk

    async def start_session(
        self,
        session_id: str,
        *,
        config: SessionConfig,
        on_audio=None,
        on_done=None,
        on_event=None,
    ):
        """Create a configured session through the frontend interface.

        This is the public session-entry API for gateways/adapters. It keeps the
        transport layer from reaching into frontend internals directly.
        """
        self._validate_session_config(config)
        return await self._frontend.create_session(
            session_id,
            config=config,
            on_audio=on_audio,
            on_done=on_done,
            on_event=on_event,
        )

    async def push_text_input(self, session_id: str, text: str) -> None:
        """Transport-facing text ingress; frontend converts it to tokens."""
        await self._frontend.push_text_input(session_id, text)

    async def feed_full_text(self, session_id: str, text: str) -> None:
        """Offline mode: set complete text, pre-split, drive all segments."""
        await self._frontend.feed_full_text(session_id, text)

    async def mark_input_complete(self, session_id: str) -> None:
        """Signal that the transport has finished sending text input."""
        await self._frontend.mark_input_complete(session_id)

    async def cancel(self, session_id: str) -> None:
        await self._frontend.cancel_session(session_id)

    def health_stats(self) -> dict:
        """Return engine health metrics (safe to call from asyncio thread)."""
        if self._engine_loop is None:
            return {"running": False}
        stats = self._engine_loop.health_stats()
        stats["variant"] = self._model_arch.variant
        stats["loaded_model_type"] = self._loaded_model_type()
        if self._model_arch.supported_task_types:
            stats["declared_supported_task_types"] = list(self._model_arch.supported_task_types)
        profile = self._model_arch.engine_profile
        if profile.max_batch_size or profile.max_seq_len:
            stats["engine_profile"] = {
                "max_batch_size": profile.max_batch_size,
                "max_input_len": profile.max_input_len,
                "max_seq_len": profile.max_seq_len,
                "engine_dtype": profile.engine_dtype,
                "triton_io_float_dtype": profile.triton_io_float_dtype,
            }
        if self._ref_audio_processor is not None:
            support = self._ref_audio_processor.support
            stats["ref_audio_available"] = support.available
            if support.reason:
                stats["ref_audio_reason"] = support.reason
        return stats

    def describe_capabilities(self) -> dict:
        """Return static standalone capability metadata for clients."""
        ref_audio_available = False
        ref_audio_reason = ""
        if self._ref_audio_processor is not None:
            support = self._ref_audio_processor.support
            ref_audio_available = support.available
            ref_audio_reason = support.reason or ""

        return {
            "variant": self._model_arch.variant,
            "loaded_model_type": self._loaded_model_type(),
            "declared_supported_task_types": list(self._model_arch.supported_task_types or ()),
            "supported_input_modes": ["token", "clause", "long_segment", "full_text"],
            "supported_group_policies": ["none", "auto"],
            "supported_audio_formats": [
                {"encoding": "pcm_f32", "sample_rate": 24000, "channels": 1},
                {"encoding": "pcm_f32", "sample_rate": 16000, "channels": 1},
                {"encoding": "pcm_s16le", "sample_rate": 24000, "channels": 1},
                {"encoding": "pcm_s16le", "sample_rate": 16000, "channels": 1},
            ],
            "ref_audio_available": ref_audio_available,
            "ref_audio_reason": ref_audio_reason,
            "engine_profile": {
                "max_batch_size": self._model_arch.engine_profile.max_batch_size,
                "max_input_len": self._model_arch.engine_profile.max_input_len,
                "max_seq_len": self._model_arch.engine_profile.max_seq_len,
                "engine_dtype": self._model_arch.engine_profile.engine_dtype,
                "triton_io_float_dtype": self._model_arch.engine_profile.triton_io_float_dtype,
            },
        }

    def _validate_session_config(self, config: SessionConfig) -> None:
        loaded_model_type = self._loaded_model_type()
        requested = (config.task_type or "").strip()

        if loaded_model_type != "unknown":
            if requested and requested != loaded_model_type:
                raise ValueError(
                    f"standalone engine has loaded model_type '{loaded_model_type}', "
                    f"but client requested task_type '{requested}'. "
                    "Omit task_type or send the same loaded model type."
                )
            model_type = loaded_model_type
        elif not requested:
            raise ValueError(
                "task_type is required because the loaded engine manifest does not declare tts_model_type"
            )
        else:
            model_type = requested

        self._validate_model_specific_fields(model_type, config)
        config.task_type = self._internal_task_type_for_model(model_type)

        task_type = (config.task_type or "").strip()
        if task_type == "voice_clone":
            if not config.ref_audio:
                raise ValueError("ref_audio is required for task_type 'voice_clone'")
            if self._ref_audio_processor is None or not self._ref_audio_processor.support.available:
                reason = (
                    self._ref_audio_processor.support.reason
                    if self._ref_audio_processor is not None
                    else "reference-audio processor unavailable"
                )
                raise ValueError(f"voice_clone is not available in standalone mode: {reason}")

    def _loaded_model_type(self) -> str:
        model_type = (self._model_arch.tts_model_type or "").strip()
        if model_type and model_type != "unknown":
            return model_type
        supported = tuple(t.strip() for t in self._model_arch.supported_task_types or () if t and t.strip())
        if len(supported) == 1:
            return supported[0]
        return "unknown"

    def _internal_task_type_for_model(self, model_type: str) -> str:
        normalized = (model_type or "").strip()
        internal = _EXTERNAL_TO_INTERNAL_TASK_TYPE.get(normalized)
        if internal:
            return internal
        raise ValueError(f"Unknown loaded model type: '{normalized}'")

    def _validate_model_specific_fields(self, model_type: str, config: SessionConfig) -> None:
        normalized = (model_type or "").strip()
        if normalized == "base":
            if not config.ref_audio:
                raise ValueError("ref_audio is required for loaded model_type 'base'")
            if config.ref_text:
                raise ValueError("ref_text is not used for loaded model_type 'base'")
            if config.instruct:
                raise ValueError("instruct is not supported for loaded model_type 'base'")
            if config.speaker:
                raise ValueError("speaker is not supported for loaded model_type 'base'")
            config.x_vector_only = True
            return

        if normalized in ("icl",):
            if not config.ref_audio:
                raise ValueError("ref_audio is required for loaded model_type 'icl'")
            if not (config.ref_text or "").strip():
                raise ValueError("ref_text is required for loaded model_type 'icl'")
            if config.instruct:
                raise ValueError("instruct is not supported for loaded model_type 'icl'")
            if config.speaker:
                raise ValueError("speaker is not supported for loaded model_type 'icl'")
            config.x_vector_only = False
            return

        if normalized in ("voice_design", "instruct"):
            if not (config.instruct or "").strip():
                raise ValueError("instruct is required for loaded model_type 'voice_design'")
            if config.speaker:
                raise ValueError("speaker is not supported for loaded model_type 'voice_design'")
            if config.ref_audio:
                raise ValueError("ref_audio is not supported for loaded model_type 'voice_design'")
            if config.ref_text:
                raise ValueError("ref_text is not supported for loaded model_type 'voice_design'")
            if config.x_vector_only:
                raise ValueError("x_vector_only is not supported for loaded model_type 'voice_design'")
            return

        if normalized == "custom_voice":
            if config.ref_audio:
                raise ValueError("ref_audio is not supported for loaded model_type 'custom_voice'")
            if config.ref_text:
                raise ValueError("ref_text is not supported for loaded model_type 'custom_voice'")
            if config.x_vector_only:
                raise ValueError("x_vector_only is not supported for loaded model_type 'custom_voice'")
            return

        if normalized == "voice_clone":
            return

        raise ValueError(f"Unknown loaded model type: '{normalized}'")

    # ------------------------------------------------------------------
    # Internal: relay asyncio.Queue → stdlib queue.Queue
    # ------------------------------------------------------------------

    async def _relay_inbox(self) -> None:
        try:
            while True:
                req = await self._async_inbox.get()
                self._engine_inbox.put_nowait(req)
        except asyncio.CancelledError:
            pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(description="TTS Engine Server")
    parser.add_argument("--config", default="engine.yaml",
                        help="Path to engine.yaml config file (default: engine.yaml)")
    parser.add_argument("--model-package-dir", default="",
                        help="Path to shared model package (tts_orchestrator/1)")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--max-batch", type=int, default=0,
                        help="Override scheduler.max_batch_size")
    parser.add_argument("--max-seq-len", type=int, default=0,
                        help="Override scheduler.max_seq_len")
    parser.add_argument("--max-sessions", type=int, default=0,
                        help="Override server.max_sessions")
    parser.add_argument("--port", type=int, default=0,
                        help="Override server.port")
    parser.add_argument("--ws-port", type=int, default=-1,
                        help="Override server.websocket_port (-1 keeps config)")
    parser.add_argument("--ws-path", default="",
                        help="Override server.websocket_path")
    args = parser.parse_args()

    cli_overrides: dict = {}
    if args.model_package_dir:
        cli_overrides.setdefault("paths", {})["model_package_dir"] = args.model_package_dir
    if args.max_batch > 0:
        cli_overrides.setdefault("scheduler", {})["max_batch_size"] = args.max_batch
    if args.max_seq_len > 0:
        cli_overrides.setdefault("scheduler", {})["max_seq_len"] = args.max_seq_len
    if args.max_sessions > 0:
        cli_overrides.setdefault("server", {})["max_sessions"] = args.max_sessions
    if args.port > 0:
        cli_overrides.setdefault("server", {})["port"] = args.port
    if args.ws_port >= 0:
        cli_overrides.setdefault("server", {})["websocket_port"] = args.ws_port
    if args.ws_path:
        cli_overrides.setdefault("server", {})["websocket_path"] = args.ws_path

    cfg = load_config(args.config, cli_overrides=cli_overrides)
    if cfg.paths.model_package_dir:
        apply_model_package_paths(
            cfg,
            resolve_model_package_paths(cfg.paths.model_package_dir),
        )

    engine_dir = cfg.paths.engine_dir
    tokenizer_dir = cfg.paths.tokenizer_dir
    model_arch = load_model_manifest(engine_dir, cfg, tokenizer_dir=tokenizer_dir)

    async def run():
        engine = TTSEngine(
            config=cfg,
            model_arch=model_arch,
            device_id=args.device,
        )
        await engine.start()

        stop_event = asyncio.Event()
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stop_event.set)

        port = cfg.server.port
        websocket_port = cfg.server.websocket_port
        websocket_path = cfg.server.websocket_path
        health_port = cfg.server.health_port

        grpc_task = None
        websocket_task = None
        health_task = None

        try:
            from .gateway.grpc_server import serve as grpc_serve
            grpc_task = asyncio.create_task(
                grpc_serve(engine, port, stop_event=stop_event),
            )
            logger.info("gRPC server launched on port %d", port)
        except Exception as e:
            logger.warning("gRPC server not started: %s", e)

        if websocket_port > 0:
            try:
                from .gateway.websocket_server import serve as websocket_serve
                websocket_task = asyncio.create_task(
                    websocket_serve(
                        engine,
                        websocket_port,
                        stop_event=stop_event,
                        path=websocket_path,
                    ),
                )
                logger.info(
                    "WebSocket server launched on port %d path %s",
                    websocket_port,
                    websocket_path,
                )
            except Exception as e:
                logger.warning("WebSocket server not started: %s", e)

        if health_port > 0:
            health_task = asyncio.create_task(
                _run_health_server(engine, health_port, stop_event),
            )

        logger.info("Engine ready, press Ctrl+C to stop")
        await stop_event.wait()

        if grpc_task:
            try:
                await grpc_task
            except Exception as e:
                logger.warning("gRPC server shutdown: %s", e)
        if websocket_task:
            try:
                await websocket_task
            except Exception as e:
                logger.warning("WebSocket server shutdown: %s", e)
        if health_task:
            try:
                await health_task
            except Exception as e:
                logger.warning("Health server shutdown: %s", e)
        await engine.stop()

    asyncio.run(run())


async def _run_health_server(
    engine: TTSEngine,
    port: int,
    stop_event: asyncio.Event,
) -> None:
    """Minimal HTTP health / metrics endpoint.

    Uses aiohttp if available; otherwise falls back to a simple
    asyncio.start_server implementation.
    """
    import json as _json

    try:
        from aiohttp import web  # type: ignore[import-untyped]

        async def handle_health(request):
            stats = engine.health_stats()
            return web.json_response(stats)

        app = web.Application()
        app.router.add_get("/health", handle_health)
        app.router.add_get("/metrics", handle_health)

        runner = web.AppRunner(app, access_log=None)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", port)
        try:
            await site.start()
            logger.info("Health/metrics HTTP server on port %d (aiohttp)", port)
            await stop_event.wait()
        finally:
            await runner.cleanup()
        return
    except ImportError:
        pass

    async def _handle_connection(reader, writer):
        await reader.read(4096)
        stats = engine.health_stats()
        body = _json.dumps(stats).encode()
        writer.write(
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
            b"Content-Length: %d\r\n\r\n" % len(body) + body
        )
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(_handle_connection, "0.0.0.0", port)
    logger.info("Health/metrics HTTP server on port %d (asyncio)", port)
    async with server:
        serve_task = asyncio.create_task(server.serve_forever())
        try:
            await stop_event.wait()
        finally:
            serve_task.cancel()
            try:
                await serve_task
            except asyncio.CancelledError:
                pass


if __name__ == "__main__":
    main()

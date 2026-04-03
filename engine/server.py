"""TTS Engine server: wires asyncio frontend + GPU engine thread.

Architecture:

    ┌────────── asyncio event loop (main thread) ──────────┐
    │                                                       │
    │  gRPC aio server  ──►  Dispatcher  ──►  engine_inbox  │
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
    python -m engine.server --tokenizer-dir /path/to/tokenizer [--weights-dir ...]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import queue
import signal
from typing import AsyncIterator, Optional

from .config import EngineConfig, load_config, to_model_config
from .core.mlfq import MLFQConfig
from .core.types import EngineResult, ResultType
from .frontend.dispatcher import Dispatcher
from .frontend.spliter.tokenizer import LightQwen3TTSTokenizer
from .backend.engine_loop import EngineLoop
from .backend.executor import Executor
from .backend.prefill import EmbeddingWeights, PrefillBuilder

logger = logging.getLogger(__name__)


class TTSEngine:
    """Top-level engine that owns both the asyncio world and the GPU thread."""

    def __init__(
        self,
        config: Optional[EngineConfig] = None,
        *,
        tokenizer_dir: str = "",
        weights_dir: str = "",
        engine_dir: str = "",
        device_id: int = 0,
        max_batch_size: int = 48,
        max_sessions: int = 128,
        max_seq_len: int = 2048,
    ):
        self._cfg = config or EngineConfig()

        self._loop: Optional[asyncio.AbstractEventLoop] = None

        self._engine_inbox = queue.Queue(maxsize=4096)
        self._async_inbox: Optional[asyncio.Queue] = None

        self._tokenizer_dir = tokenizer_dir or self._cfg.paths.tokenizer_dir
        self._weights_dir = weights_dir or self._cfg.paths.weights_dir
        self._engine_dir = engine_dir or self._cfg.paths.engine_dir
        self._device_id = device_id
        self._max_batch = max_batch_size if max_batch_size != 48 else self._cfg.scheduler.max_batch_size
        self._max_sessions = max_sessions if max_sessions != 128 else self._cfg.server.max_sessions
        self._max_seq_len = max_seq_len if max_seq_len != 2048 else self._cfg.scheduler.max_seq_len

        self._tokenizer: Optional[LightQwen3TTSTokenizer] = None
        self._dispatcher: Optional[Dispatcher] = None
        self._executor: Optional[Executor] = None
        self._engine_loop: Optional[EngineLoop] = None
        self._relay_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Initialize all components, warm up, and start the engine thread."""
        self._loop = asyncio.get_event_loop()
        self._async_inbox = asyncio.Queue(maxsize=4096)

        self._tokenizer = LightQwen3TTSTokenizer(self._tokenizer_dir)

        sc = self._cfg.spliter
        self._dispatcher = Dispatcher(
            engine_inbox=self._async_inbox,
            tokenizer=self._tokenizer,
            max_sessions=self._max_sessions,
            engine_max_decode_len=self._max_seq_len,
            prefill_len=sc.prefill_len,
            ema_ratio=sc.ema_ratio_initial,
            max_concurrent_segments=sc.max_concurrent_segments,
        )

        model_config = to_model_config(self._cfg)
        self._executor = Executor(
            engine_dir=self._engine_dir,
            weights_dir=self._weights_dir,
            device_id=self._device_id,
            max_batch_size=self._max_batch,
            max_seq_len=self._max_seq_len,
            model_config=model_config,
        )
        self._executor.load()

        self._executor.warmup(n_rounds=self._cfg.server.warmup_rounds)

        prefill_builder = None
        if self._weights_dir:
            try:
                emb_weights = EmbeddingWeights(self._weights_dir, self._device_id)
                self._executor.set_embedding_weights(emb_weights)
                prefill_builder = PrefillBuilder(emb_weights, self._tokenizer.tokenizer)
                logger.info("PrefillBuilder loaded from %s", self._weights_dir)
            except Exception as e:
                logger.warning("Could not load embedding weights: %s", e)

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
        task_type: str = "custom",
        ref_audio: Optional[bytes] = None,
    ) -> AsyncIterator[bytes]:
        """Create a session and yield audio chunks as they are produced."""
        audio_queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue()

        async def on_audio(sid: str, data: bytes) -> None:
            await audio_queue.put(data)

        async def on_done(sid: str, metrics: dict) -> None:
            await audio_queue.put(None)

        await self._dispatcher.create_session(
            session_id,
            speaker_key=speaker_key,
            task_type=task_type,
            ref_audio=ref_audio,
            on_audio=on_audio,
            on_done=on_done,
        )

        while True:
            chunk = await audio_queue.get()
            if chunk is None:
                break
            yield chunk

    async def feed_text(self, session_id: str, text: str) -> None:
        await self._dispatcher.feed_text(session_id, text)

    async def text_complete(self, session_id: str) -> None:
        await self._dispatcher.text_complete(session_id)

    async def cancel(self, session_id: str) -> None:
        await self._dispatcher.cancel_session(session_id)

    def health_stats(self) -> dict:
        """Return engine health metrics (safe to call from asyncio thread)."""
        if self._engine_loop is None:
            return {"running": False}
        return self._engine_loop.health_stats()

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
    parser.add_argument("--config", default=None,
                        help="Path to engine.yaml config file")
    parser.add_argument("--tokenizer-dir", default="",
                        help="Path to tokenizer directory (tokenizer.json / vocab.json)")
    parser.add_argument("--weights-dir", default="",
                        help="Path to embedding weights (.pt files)")
    parser.add_argument("--engine-dir", default="",
                        help="Path to TRT engine directory (model.plan)")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--max-batch", type=int, default=0,
                        help="Override scheduler.max_batch_size")
    parser.add_argument("--max-sessions", type=int, default=0,
                        help="Override server.max_sessions")
    parser.add_argument("--port", type=int, default=0,
                        help="Override server.port")
    args = parser.parse_args()

    cli_overrides: dict = {}
    if args.tokenizer_dir:
        cli_overrides.setdefault("paths", {})["tokenizer_dir"] = args.tokenizer_dir
    if args.weights_dir:
        cli_overrides.setdefault("paths", {})["weights_dir"] = args.weights_dir
    if args.engine_dir:
        cli_overrides.setdefault("paths", {})["engine_dir"] = args.engine_dir
    if args.max_batch > 0:
        cli_overrides.setdefault("scheduler", {})["max_batch_size"] = args.max_batch
    if args.max_sessions > 0:
        cli_overrides.setdefault("server", {})["max_sessions"] = args.max_sessions
    if args.port > 0:
        cli_overrides.setdefault("server", {})["port"] = args.port

    cfg = load_config(args.config, cli_overrides=cli_overrides)

    async def run():
        engine = TTSEngine(
            config=cfg,
            device_id=args.device,
        )
        await engine.start()

        stop_event = asyncio.Event()
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stop_event.set)

        port = cfg.server.port
        health_port = cfg.server.health_port

        grpc_task = None
        health_task = None

        try:
            from .gateway.grpc_server import serve as grpc_serve
            grpc_task = asyncio.create_task(grpc_serve(engine, port))
            logger.info("gRPC server launched on port %d", port)
        except Exception as e:
            logger.warning("gRPC server not started: %s", e)

        if health_port > 0:
            health_task = asyncio.create_task(
                _run_health_server(engine, health_port),
            )

        logger.info("Engine ready, press Ctrl+C to stop")
        await stop_event.wait()

        if grpc_task:
            grpc_task.cancel()
        if health_task:
            health_task.cancel()
        await engine.stop()

    asyncio.run(run())


async def _run_health_server(engine: TTSEngine, port: int) -> None:
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
            await asyncio.Event().wait()
        except asyncio.CancelledError:
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
    try:
        async with server:
            await server.serve_forever()
    except asyncio.CancelledError:
        server.close()


if __name__ == "__main__":
    main()

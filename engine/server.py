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
                    │                      └────────────┘   │
                    └───────────────────────────────────────┘

Usage:
    python -m engine.server --tokenizer-dir /path/to/tokenizer [--weights-dir ...]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import queue
import signal
from typing import AsyncIterator, Optional

from .core.types import EngineResult, ResultType
from .frontend.dispatcher import Dispatcher
from .frontend.spliter.tokenizer import LightQwen3TTSTokenizer
from .backend.engine_loop import EngineLoop
from .backend.executor import Executor
from .backend.kv_cache_pool import ModelConfig
from .backend.prefill import EmbeddingWeights, PrefillBuilder

logger = logging.getLogger(__name__)


class TTSEngine:
    """Top-level engine that owns both the asyncio world and the GPU thread."""

    def __init__(
        self,
        tokenizer_dir: str,
        *,
        weights_dir: str = "",
        engine_dir: str = "",
        device_id: int = 0,
        max_batch_size: int = 48,
        max_sessions: int = 128,
        max_seq_len: int = 2048,
    ):
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Bridge: asyncio → engine thread (stdlib queue)
        self._engine_inbox = queue.Queue(maxsize=4096)
        self._async_inbox: Optional[asyncio.Queue] = None

        self._tokenizer_dir = tokenizer_dir
        self._weights_dir = weights_dir
        self._engine_dir = engine_dir
        self._device_id = device_id
        self._max_batch = max_batch_size
        self._max_sessions = max_sessions
        self._max_seq_len = max_seq_len

        self._tokenizer: Optional[LightQwen3TTSTokenizer] = None
        self._dispatcher: Optional[Dispatcher] = None
        self._executor: Optional[Executor] = None
        self._engine_loop: Optional[EngineLoop] = None
        self._relay_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Initialize all components and start the engine thread."""
        self._loop = asyncio.get_event_loop()
        self._async_inbox = asyncio.Queue(maxsize=4096)

        self._tokenizer = LightQwen3TTSTokenizer(self._tokenizer_dir)

        self._dispatcher = Dispatcher(
            engine_inbox=self._async_inbox,
            tokenizer=self._tokenizer,
            max_sessions=self._max_sessions,
            engine_max_decode_len=self._max_seq_len,
        )

        model_config = ModelConfig()
        self._executor = Executor(
            engine_dir=self._engine_dir,
            weights_dir=self._weights_dir,
            device_id=self._device_id,
            max_batch_size=self._max_batch,
            max_seq_len=self._max_seq_len,
            model_config=model_config,
        )
        self._executor.load()

        prefill_builder = None
        if self._weights_dir:
            try:
                emb_weights = EmbeddingWeights(self._weights_dir, self._device_id)
                self._executor.set_embedding_weights(emb_weights)
                prefill_builder = PrefillBuilder(emb_weights, self._tokenizer.tokenizer)
                logger.info("PrefillBuilder loaded from %s", self._weights_dir)
            except Exception as e:
                logger.warning("Could not load embedding weights: %s", e)

        self._engine_loop = EngineLoop(
            engine_inbox=self._engine_inbox,
            async_loop=self._loop,
            executor=self._executor,
            prefill_builder=prefill_builder,
            max_batch_size=self._max_batch,
        )
        self._engine_loop.start()

        self._relay_task = asyncio.create_task(self._relay_inbox())
        logger.info("TTS Engine started (max_batch=%d, max_sessions=%d)",
                     self._max_batch, self._max_sessions)

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
    parser.add_argument("--tokenizer-dir", required=True,
                        help="Path to tokenizer directory (tokenizer.json / vocab.json)")
    parser.add_argument("--weights-dir", default="",
                        help="Path to embedding weights (.pt files)")
    parser.add_argument("--engine-dir", default="",
                        help="Path to TRT engine directory (model.plan)")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--max-batch", type=int, default=48)
    parser.add_argument("--max-sessions", type=int, default=128)
    parser.add_argument("--port", type=int, default=50051,
                        help="gRPC port")
    args = parser.parse_args()

    async def run():
        engine = TTSEngine(
            tokenizer_dir=args.tokenizer_dir,
            weights_dir=args.weights_dir,
            engine_dir=args.engine_dir,
            device_id=args.device,
            max_batch_size=args.max_batch,
            max_sessions=args.max_sessions,
        )
        await engine.start()

        stop_event = asyncio.Event()
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stop_event.set)

        logger.info("Engine ready on port %d, press Ctrl+C to stop", args.port)

        # TODO: start gRPC server (see engine/gateway/)
        # from .gateway.grpc_server import serve
        # await serve(engine, args.port)

        await stop_event.wait()
        await engine.stop()

    asyncio.run(run())


if __name__ == "__main__":
    main()

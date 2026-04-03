"""gRPC streaming gateway for the TTS engine.

Handles bidirectional streaming: client sends text chunks, server sends audio.

Protocol:
  Client sends:  InitRequest → TextChunk* → TextComplete
  Server sends:  AudioChunk* → StatusUpdate(done)

Uses grpcio.aio for async compatibility with the engine's asyncio event loop.

To generate proto stubs:
    python -m grpc_tools.protoc -I engine/gateway \
        --python_out=engine/gateway \
        --grpc_python_out=engine/gateway \
        engine/gateway/tts.proto
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..server import TTSEngine

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24000


class TTSServicer:
    """gRPC servicer that bridges streaming requests to TTSEngine."""

    def __init__(self, engine: TTSEngine):
        self._engine = engine

    async def SynthesizeStream(self, request_iterator, context):
        """Handle one bidirectional stream.

        Spawns two concurrent tasks:
          1. Reader:  consumes client messages (init → text → done)
          2. Writer:  yields audio chunks from engine to client
        """
        session_id = None
        audio_queue: asyncio.Queue = asyncio.Queue(maxsize=256)

        try:
            async for request in request_iterator:
                msg_type = request.WhichOneof("request")

                if msg_type == "init":
                    init = request.init
                    session_id = init.session_id or str(uuid.uuid4())

                    async def on_audio(sid, data):
                        await audio_queue.put(("audio", data))

                    async def on_done(sid, metrics):
                        await audio_queue.put(("done", metrics))

                    await self._engine._dispatcher.create_session(
                        session_id,
                        speaker_key=init.speaker or None,
                        task_type=init.task_type or "custom_voice",
                        ref_audio=init.ref_audio or None,
                        on_audio=on_audio,
                        on_done=on_done,
                    )
                    logger.info("gRPC session started: %s", session_id)

                elif msg_type == "text":
                    if session_id:
                        await self._engine.feed_text(session_id, request.text.text)

                elif msg_type == "done":
                    if session_id:
                        await self._engine.text_complete(session_id)

                elif msg_type == "cancel":
                    if session_id:
                        await self._engine.cancel(session_id)
                        break

                while not audio_queue.empty():
                    msg_type_q, payload = audio_queue.get_nowait()
                    if msg_type_q == "audio":
                        yield self._make_audio_response(payload)
                    elif msg_type_q == "done":
                        yield self._make_status_response("done", "Synthesis complete")
                        return

        except asyncio.CancelledError:
            logger.info("gRPC stream cancelled: %s", session_id)
        except Exception as e:
            logger.error("gRPC stream error: %s: %s", session_id, e)
            yield self._make_status_response("error", str(e))
        finally:
            if session_id:
                await self._engine.cancel(session_id)

        while not audio_queue.empty():
            msg_type_q, payload = audio_queue.get_nowait()
            if msg_type_q == "audio":
                yield self._make_audio_response(payload)

        yield self._make_status_response("done", "Stream ended")

    def _make_audio_response(self, pcm_bytes: bytes):
        """Build a SynthesizeResponse with audio data.

        Returns a dict-like object; actual proto construction depends on
        generated stubs.
        """
        return {
            "audio": {
                "pcm_data": pcm_bytes,
                "sample_rate": SAMPLE_RATE,
            }
        }

    def _make_status_response(self, event: str, message: str = ""):
        return {
            "status": {
                "event": event,
                "message": message,
            }
        }


async def serve(engine: TTSEngine, port: int = 50051) -> None:
    """Start gRPC aio server. Call from within an asyncio event loop."""
    try:
        import grpc
        from grpc import aio as grpc_aio
    except ImportError:
        logger.error("grpcio not installed. Run: pip install grpcio grpcio-tools")
        return

    server = grpc_aio.server()

    servicer = TTSServicer(engine)

    # When proto stubs are generated, register like this:
    # from . import tts_pb2_grpc
    # tts_pb2_grpc.add_TTSServiceServicer_to_server(servicer, server)

    server.add_insecure_port(f"[::]:{port}")
    await server.start()
    logger.info("gRPC server listening on port %d", port)
    await server.wait_for_termination()

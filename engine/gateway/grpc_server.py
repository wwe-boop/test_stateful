"""gRPC streaming gateway for the TTS engine.

Handles bidirectional streaming: client sends text chunks, server sends audio.

Protocol:
  Client sends:  InitRequest → TextChunk* → TextComplete
  Server sends:  AudioChunk* → StatusUpdate(done)

Uses grpcio.aio for async compatibility with the engine's asyncio event loop.

To regenerate proto stubs:
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

from . import tts_pb2, tts_pb2_grpc

if TYPE_CHECKING:
    from ..server import TTSEngine

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24000


class TTSServicer(tts_pb2_grpc.TTSServiceServicer):
    """gRPC servicer that bridges streaming requests to TTSEngine."""

    def __init__(self, engine: TTSEngine):
        self._engine = engine

    async def SynthesizeStream(self, request_iterator, context):
        """Handle one bidirectional stream.

        Text buffering strategy:
        - Text chunks are accumulated in a buffer
        - When ``done`` arrives, if no text has been streamed yet (all text
          arrived before ``done``), we use ``feed_full_text`` for optimal
          offline pre-splitting.  Otherwise we finalize streaming with
          ``text_complete``.
        - When audio is produced before ``done``, we flush buffered text
          via ``feed_text`` and switch to streaming mode.
        """
        session_id = None
        audio_queue: asyncio.Queue = asyncio.Queue(maxsize=256)
        got_done = False
        got_cancel = False
        text_buffer: list[str] = []
        text_flushed = False

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
                        if text_flushed:
                            await self._engine.feed_text(session_id, request.text.text)
                        else:
                            text_buffer.append(request.text.text)

                elif msg_type == "done":
                    if session_id:
                        if not text_flushed and text_buffer:
                            full_text = "".join(text_buffer)
                            text_buffer.clear()
                            text_flushed = True
                            await self._engine.feed_full_text(session_id, full_text)
                            logger.info("gRPC oneshot: %s (%d chars via feed_full_text)",
                                        session_id, len(full_text))
                        else:
                            await self._engine.text_complete(session_id)
                    got_done = True

                elif msg_type == "cancel":
                    if session_id:
                        await self._engine.cancel(session_id)
                    got_cancel = True
                    break

                # Drain available audio; if audio arrives while text is
                # still buffered, flush the buffer into streaming mode.
                while not audio_queue.empty():
                    msg_type_q, payload = audio_queue.get_nowait()
                    if msg_type_q == "audio":
                        if not text_flushed and text_buffer and session_id:
                            for chunk in text_buffer:
                                await self._engine.feed_text(session_id, chunk)
                            text_buffer.clear()
                            text_flushed = True
                        yield _make_audio_response(payload)
                    elif msg_type_q == "done":
                        err = payload.get("error") if isinstance(payload, dict) else None
                        if err:
                            yield _make_status_response("error", str(err))
                        else:
                            yield _make_status_response("done", "Synthesis complete")
                        return

            if got_done and not got_cancel and session_id:
                while True:
                    try:
                        msg_type_q, payload = await asyncio.wait_for(
                            audio_queue.get(), timeout=300.0,
                        )
                    except asyncio.TimeoutError:
                        logger.warning("gRPC session %s: audio wait timeout", session_id)
                        break
                    if msg_type_q == "audio":
                        yield _make_audio_response(payload)
                    elif msg_type_q == "done":
                        err = payload.get("error") if isinstance(payload, dict) else None
                        if err:
                            yield _make_status_response("error", str(err))
                        else:
                            yield _make_status_response("done", "Synthesis complete")
                        return

        except asyncio.CancelledError:
            logger.info("gRPC stream cancelled: %s", session_id)
        except Exception as e:
            logger.error("gRPC stream error: %s: %s", session_id, e)
            yield _make_status_response("error", str(e))
        finally:
            if session_id:
                await self._engine.cancel(session_id)

        yield _make_status_response("done", "Stream ended")


def _make_audio_response(pcm_bytes: bytes) -> tts_pb2.SynthesizeResponse:
    return tts_pb2.SynthesizeResponse(
        audio=tts_pb2.AudioChunk(
            pcm_data=pcm_bytes,
            sample_rate=SAMPLE_RATE,
        )
    )


def _make_status_response(event: str, message: str = "") -> tts_pb2.SynthesizeResponse:
    return tts_pb2.SynthesizeResponse(
        status=tts_pb2.StatusUpdate(
            event=event,
            message=message,
        )
    )


async def serve(engine: TTSEngine, port: int = 50051, *, stop_event: asyncio.Event) -> None:
    """Start gRPC aio server. Call from within an asyncio event loop.

    Waits on ``stop_event`` then calls ``server.stop()`` so SIGINT/SIGTERM can shut
    down cleanly. ``wait_for_termination()`` alone does not reliably react to
    asyncio task cancellation.
    """
    try:
        import grpc
        from grpc import aio as grpc_aio
    except ImportError:
        logger.error("grpcio not installed. Run: pip install grpcio grpcio-tools")
        return

    server = grpc_aio.server()

    servicer = TTSServicer(engine)
    tts_pb2_grpc.add_TTSServiceServicer_to_server(servicer, server)

    server.add_insecure_port(f"[::]:{port}")
    await server.start()
    logger.info("gRPC server listening on port %d", port)
    await stop_event.wait()
    await server.stop(5.0)

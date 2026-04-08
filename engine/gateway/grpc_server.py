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

    async def _create_session(
        self,
        session_id: str | None,
        *,
        speaker: str,
        task_type: str,
        ref_audio: bytes,
        audio_queue: asyncio.Queue,
    ) -> str:
        session_id = session_id or str(uuid.uuid4())

        async def on_audio(sid, data):
            await audio_queue.put(("audio", data))

        async def on_done(sid, metrics):
            await audio_queue.put(("done", metrics))

        await self._engine._dispatcher.create_session(
            session_id,
            speaker_key=speaker or None,
            task_type=task_type or "custom_voice",
            ref_audio=ref_audio or None,
            on_audio=on_audio,
            on_done=on_done,
        )
        logger.info("gRPC session started: %s", session_id)
        return session_id

    async def _drain_available_audio(self, audio_queue: asyncio.Queue):
        while not audio_queue.empty():
            msg_type_q, payload = audio_queue.get_nowait()
            response = _queue_message_to_response(msg_type_q, payload)
            if response is not None:
                yield response
            if msg_type_q == "done":
                return

    async def _drain_until_done(
        self,
        session_id: str,
        audio_queue: asyncio.Queue,
        *,
        timeout: float = 300.0,
    ):
        while True:
            try:
                msg_type_q, payload = await asyncio.wait_for(
                    audio_queue.get(), timeout=timeout,
                )
            except asyncio.TimeoutError:
                logger.warning("gRPC session %s: audio wait timeout", session_id)
                break
            response = _queue_message_to_response(msg_type_q, payload)
            if response is not None:
                yield response
            if msg_type_q == "done":
                return

    async def SynthesizeOnce(self, request, context):
        """Handle unary full-text requests with streamed audio output."""
        session_id = None
        audio_queue: asyncio.Queue = asyncio.Queue(maxsize=256)

        try:
            session_id = await self._create_session(
                request.session_id,
                speaker=request.speaker,
                task_type=request.task_type,
                ref_audio=request.ref_audio,
                audio_queue=audio_queue,
            )
            await self._engine.feed_full_text(session_id, request.text)
            async for response in self._drain_until_done(session_id, audio_queue):
                yield response
            return
        except asyncio.CancelledError:
            logger.info("gRPC oneshot cancelled: %s", session_id)
        except Exception as e:
            logger.error("gRPC oneshot error: %s: %s", session_id, e)
            yield _make_status_response("error", str(e))
        finally:
            if session_id:
                await self._engine.cancel(session_id)

        yield _make_status_response("done", "Stream ended")

    async def SynthesizeStream(self, request_iterator, context):
        """Handle one bidirectional stream.

        Stream semantics:
        - Each ``TextChunk`` is forwarded to the engine immediately.
        - ``TextComplete`` only signals that no more text will arrive.

        This keeps bidirectional streaming truly incremental. Offline
        pre-splitting should use a dedicated oneshot path instead of being
        inferred from delayed chunks in this RPC.
        """
        session_id = None
        audio_queue: asyncio.Queue = asyncio.Queue(maxsize=256)
        got_done = False
        got_cancel = False

        try:
            async for request in request_iterator:
                msg_type = request.WhichOneof("request")

                if msg_type == "init":
                    init = request.init
                    session_id = await self._create_session(
                        init.session_id,
                        speaker=init.speaker,
                        task_type=init.task_type,
                        ref_audio=init.ref_audio,
                        audio_queue=audio_queue,
                    )

                elif msg_type == "text":
                    if session_id:
                        await self._engine.feed_text(session_id, request.text.text)

                elif msg_type == "done":
                    if session_id:
                        await self._engine.text_complete(session_id)
                    got_done = True

                elif msg_type == "cancel":
                    if session_id:
                        await self._engine.cancel(session_id)
                    got_cancel = True
                    break

                # Drain available audio promptly to reduce streaming latency.
                async for response in self._drain_available_audio(audio_queue):
                    yield response
                    if _is_done_response(response):
                        return

            if got_done and not got_cancel and session_id:
                async for response in self._drain_until_done(session_id, audio_queue):
                    yield response
                    if _is_done_response(response):
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


def _queue_message_to_response(
    msg_type_q: str, payload,
) -> tts_pb2.SynthesizeResponse | None:
    if msg_type_q == "audio":
        return _make_audio_response(payload)
    if msg_type_q == "done":
        err = payload.get("error") if isinstance(payload, dict) else None
        if err:
            return _make_status_response("error", str(err))
        return _make_status_response("done", "Synthesis complete")
    return None


def _is_done_response(response: tts_pb2.SynthesizeResponse) -> bool:
    return (
        response.WhichOneof("response") == "status"
        and response.status.event in {"done", "error"}
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

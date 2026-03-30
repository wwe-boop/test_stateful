#!/usr/bin/env python3
"""
TTS Gateway — gRPC server that bridges TTSService to Triton tts_orchestrator.

Streaming text input: TextChunk messages are forwarded to the orchestrator
immediately via append_text actions, enabling token-level streaming from
upstream LLMs. The gateway no longer waits for TextComplete to collect all text.

Run from gateway/: python server.py [--triton host:port] [--port 50051]
Requires: Triton server with tts_orchestrator loaded; generate proto first (see README).
"""

import argparse
import base64
import json
import logging
import threading
import time
import uuid
from typing import Iterator

import grpc
import numpy as np

from proto import tts_service_pb2 as pb
from proto import tts_service_pb2_grpc as pb_grpc

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("gateway")

SAMPLE_RATE = 24000
TRITON_MODEL = "tts_orchestrator"


def _float32_to_pcm16(audio: np.ndarray) -> bytes:
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767).astype(np.int16)
    return pcm.tobytes()


class _TritonStreamClient:
    """Manages a Triton gRPC streaming connection for a single TTS session.

    Sends init/append_text/text_complete actions and yields audio responses.
    """

    def __init__(self, triton_url: str):
        try:
            import tritonclient.grpc as grpcclient
        except ImportError:
            raise RuntimeError("tritonclient[grpc] not installed")
        self._grpcclient = grpcclient
        self._client = grpcclient.InferenceServerClient(url=triton_url)
        if not self._client.is_server_ready():
            raise RuntimeError(f"Triton not ready at {triton_url}")
        self._chunks: list[tuple[np.ndarray, bool]] = []
        self._errors: list[str] = []
        self._lock = threading.Lock()
        self._client.start_stream(callback=self._callback)

    def _callback(self, result, error):
        with self._lock:
            if error:
                err_str = str(error)
                if "CAPABILITIES:" not in err_str:
                    self._errors.append(err_str)
                return
            audio = result.as_numpy("audio_chunk")
            is_final = result.as_numpy("is_final")
            final = bool(is_final.flatten()[0]) if is_final is not None and is_final.size else False
            self._chunks.append((audio.flatten(), final))

    def send_request(self, req_dict: dict):
        gc = self._grpcclient
        req_json = json.dumps(req_dict)
        req_input = gc.InferInput("request", [1], "BYTES")
        req_input.set_data_from_numpy(np.array([req_json], dtype=object))
        audio_out = gc.InferRequestedOutput("audio_chunk")
        final_out = gc.InferRequestedOutput("is_final")
        self._client.async_stream_infer(
            model_name=TRITON_MODEL,
            inputs=[req_input],
            outputs=[audio_out, final_out],
        )

    def poll_chunks(self) -> list[tuple[np.ndarray, bool]]:
        with self._lock:
            if self._errors:
                err = self._errors.pop(0)
                raise RuntimeError(err)
            result = list(self._chunks)
            self._chunks.clear()
            return result

    def stop(self):
        try:
            self._client.stop_stream()
        except Exception:
            pass


def _build_init_request(init: pb.InitRequest, session_id: str, first_text: str = "") -> dict:
    req = {
        "action": "init",
        "session_id": session_id,
        "task_type": init.task_type or "voice_design",
        "language": init.language or "auto",
        "text": first_text,
    }
    if init.ref_audio:
        req["ref_audio"] = base64.b64encode(init.ref_audio).decode("ascii")
    if init.ref_text:
        req["ref_text"] = init.ref_text
    if init.x_vector_only:
        req["x_vector_only"] = True
    if init.speaker:
        req["speaker"] = init.speaker
    if init.instruct:
        req["instruct"] = init.instruct
    return req


class TTSGatewayServicer(pb_grpc.TTSServiceServicer):
    def __init__(self, triton_url: str):
        self.triton_url = triton_url

    def StreamingSynthesize(
        self,
        request_iterator: Iterator[pb.TTSRequest],
        context: grpc.ServicerContext,
    ) -> Iterator[pb.TTSResponse]:
        session_id = uuid.uuid4().hex[:12]
        stream_client = None
        init_sent = False

        try:
            stream_client = _TritonStreamClient(self.triton_url)

            for req in request_iterator:
                if context.is_active() is False:
                    break

                if req.HasField("init"):
                    init_req = _build_init_request(req.init, session_id)
                    stream_client.send_request(init_req)
                    init_sent = True

                elif req.HasField("text"):
                    text = req.text.text or ""
                    if not init_sent:
                        yield pb.TTSResponse(
                            error=pb.TTSError(code=400, message="TextChunk before InitRequest"))
                        return
                    if text.strip():
                        stream_client.send_request({
                            "action": "append_text",
                            "session_id": session_id,
                            "text": text,
                        })

                elif req.HasField("complete"):
                    if init_sent:
                        stream_client.send_request({
                            "action": "text_complete",
                            "session_id": session_id,
                        })
                    break

                # Yield any audio chunks that have arrived so far
                for audio_f32, is_final in stream_client.poll_chunks():
                    pcm_bytes = _float32_to_pcm16(audio_f32)
                    yield pb.TTSResponse(
                        audio=pb.AudioChunk(
                            pcm_data=pcm_bytes,
                            sample_rate=SAMPLE_RATE,
                            is_final=is_final,
                        )
                    )
                    if is_final:
                        return

            if not init_sent:
                yield pb.TTSResponse(
                    error=pb.TTSError(code=400, message="missing InitRequest"))
                return

            # Drain remaining audio after input stream ends
            timeout = 120
            t0 = time.perf_counter()
            while (time.perf_counter() - t0) < timeout:
                try:
                    for audio_f32, is_final in stream_client.poll_chunks():
                        pcm_bytes = _float32_to_pcm16(audio_f32)
                        yield pb.TTSResponse(
                            audio=pb.AudioChunk(
                                pcm_data=pcm_bytes,
                                sample_rate=SAMPLE_RATE,
                                is_final=is_final,
                            )
                        )
                        if is_final:
                            return
                except RuntimeError as e:
                    yield pb.TTSResponse(
                        error=pb.TTSError(code=500, message=str(e)))
                    return
                if not context.is_active():
                    break
                time.sleep(0.02)

            yield pb.TTSResponse(
                error=pb.TTSError(code=504, message="Triton stream timeout"))

        except Exception as e:
            logger.exception("Streaming synthesis failed")
            yield pb.TTSResponse(
                error=pb.TTSError(code=500, message=str(e)))
        finally:
            if stream_client is not None:
                stream_client.stop()


def main():
    p = argparse.ArgumentParser(description="TTS Gateway (gRPC -> Triton)")
    p.add_argument("--triton", default="localhost:8001", help="Triton gRPC address")
    p.add_argument("--port", type=int, default=50051, help="Gateway listen port")
    args = p.parse_args()
    server = grpc.server()
    pb_grpc.add_TTSServiceServicer_to_server(TTSGatewayServicer(args.triton), server)
    server.add_insecure_port(f"[::]:{args.port}")
    server.start()
    logger.info("TTS Gateway listening on 0.0.0.0:%s (Triton %s)", args.port, args.triton)
    server.wait_for_termination()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Unified full E2E tool for standalone engine and Triton serving interfaces.

Default matrix:
  - standalone engine gRPC: full suite close to tests/e2e/test_engine_standalone.py
  - standalone engine WebSocket: same engine suite over websocket transport
  - Triton gRPC: health + real synthesis request
  - Triton HTTP: health + model/config introspection + infer behavior check

Typical usage:
    python scripts/python/test_serving_endpoints.py
    python scripts/python/test_serving_endpoints.py --targets engine-grpc,engine-websocket
    python scripts/python/test_serving_endpoints.py --targets triton-grpc,triton-http
"""

from __future__ import annotations

import argparse
import base64
import json
import socket
import statistics
import sys
import time
import uuid
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import requests
from raw_websocket import (
    RawWebSocketError,
    ws_close,
    ws_connect,
    ws_recv_frame,
    ws_send_json,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine.gateway import tts_pb2, tts_pb2_grpc

DEFAULT_ENGINE_GRPC = "localhost:50051"
DEFAULT_ENGINE_WS = "ws://localhost:50052/v1/ws"
DEFAULT_TRITON_HTTP = "http://localhost:8000"
DEFAULT_TRITON_GRPC = "localhost:8001"
DEFAULT_TRITON_MODEL = "tts_orchestrator"
DEFAULT_TRITON_HTTP_MODEL = "tts_orchestrator_http"
DEFAULT_SAMPLE_RATE = 24000
TRITON_EXPECTED_INPUTS = {"request"}
TRITON_EXPECTED_OUTPUTS = {"audio_chunk", "event_type", "event_json", "is_final"}

TEST_TEXTS = [
    "你好，今天天气真好。",
    "欢迎来到人工智能语音合成的世界。",
    "技术创新推动着社会不断前进。",
    "我们正在测试多路并发的语音合成能力。",
]
LONG_TEXT = (
    "人工智能正在深刻改变我们的世界。"
    "从语音识别到自然语言处理，从计算机视觉到机器人技术，"
    "AI的应用已经渗透到生活的方方面面。"
    "在医疗领域，AI可以辅助诊断疾病、发现新药物。"
    "在教育领域，AI可以提供个性化的学习方案。"
    "在交通领域，自动驾驶技术正在逐步成熟。"
    "未来，人工智能将继续推动社会进步，"
    "为人类创造更多的可能性。"
)
VERY_LONG_TEXT = (
    "在遥远的古代，人类就开始仰望星空，思考宇宙的奥秘。"
    "从古希腊的哲学家到中国的天文学家，人们不断探索着这个世界的本质。"
    "随着科学技术的发展，我们对宇宙的认知越来越深入。"
    "从伽利略的望远镜到哈勃太空望远镜，每一次技术的突破都让我们看到了更广阔的世界。"
    "如今，人工智能技术的发展为科学研究带来了前所未有的机遇。"
    "机器学习算法可以处理海量的天文数据，发现人眼无法察觉的规律。"
    "深度学习网络能够分析复杂的光谱数据，帮助我们理解恒星的演化过程。"
    "在医学领域，AI辅助诊断系统已经在某些疾病的识别上达到了专家水平。"
    "在交通出行方面，自动驾驶技术正在逐步走向成熟，有望彻底改变我们的出行方式。"
    "在教育领域，个性化学习系统能够根据每个学生的特点量身定制学习方案。"
    "在金融领域，智能风控系统可以实时监测交易异常，保护用户的资金安全。"
    "然而，技术的发展也带来了新的挑战和思考。"
    "如何确保人工智能的安全性和可控性？"
    "如何在自动化浪潮中保障就业和社会公平？"
    "这些问题需要我们全社会共同面对和解决。"
    "只有在技术进步与人文关怀之间找到平衡，"
    "才能让人工智能真正造福全人类，"
    "创造一个更加美好的未来。"
)
STREAMING_LONG_TEXT_CHUNKS = [
    "人工智能正在深刻改变我们的世界。",
    "从语音识别到自然语言处理，从计算机视觉到机器人技术，AI的应用已经渗透到生活的方方面面。",
    "在医疗领域，AI可以辅助诊断疾病、发现新药物。在教育领域，AI可以提供个性化的学习方案。",
    "在交通领域，自动驾驶技术正在逐步成熟。未来，人工智能将继续推动社会进步，为人类创造更多的可能性。",
]
CUSTOM_VOICE_INSTRUCT_ZH = "用温柔、舒缓的语气朗读。"
CUSTOM_VOICE_INSTRUCT_EN = "Speak in a calm and friendly tone."


@dataclass
class RequestSpec:
    task_type: str = "custom_voice"
    language: str = "auto"
    speaker: str | None = "Serena"
    instruct: str | None = None
    ref_audio_bytes: bytes | None = None
    ref_text: str | None = None
    x_vector_only: bool = False
    sample_rate: int = DEFAULT_SAMPLE_RATE
    channels: int = 1
    encoding: str = "pcm_f32"


@dataclass
class SynthesisResult:
    transport: str
    session_id: str
    text: str
    sample_rate: int = DEFAULT_SAMPLE_RATE
    encoding: str = "pcm_f32"
    first_chunk_ms: float | None = None
    ttft_ms: float | None = None
    start_to_first_audio_ms: float | None = None
    total_ms: float = 0.0
    num_chunks: int = 0
    total_samples: int = 0
    error: str | None = None
    warnings: list[str] = field(default_factory=list)
    events: list[str] = field(default_factory=list)
    audio_chunk_intervals_ms: list[float] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)
    audio: np.ndarray | None = None

    @property
    def duration_sec(self) -> float:
        return self.total_samples / self.sample_rate if self.total_samples > 0 else 0.0

    @property
    def rtf(self) -> float:
        if self.duration_sec <= 0 or self.total_ms <= 0:
            return 0.0
        return (self.total_ms / 1000.0) / self.duration_sec

    @property
    def decode_step_mean_ms(self) -> float | None:
        if not self.audio_chunk_intervals_ms:
            return None
        return statistics.mean(self.audio_chunk_intervals_ms)

    @property
    def decode_step_p50_ms(self) -> float | None:
        if not self.audio_chunk_intervals_ms:
            return None
        return statistics.median(self.audio_chunk_intervals_ms)

    @property
    def decode_step_p95_ms(self) -> float | None:
        if not self.audio_chunk_intervals_ms:
            return None
        vals = sorted(self.audio_chunk_intervals_ms)
        idx = min(len(vals) - 1, max(0, int(round(0.95 * (len(vals) - 1)))))
        return vals[idx]

    def to_summary(self) -> str:
        if self.error:
            return f"ERROR: {self.error}"
        fc = f"{self.first_chunk_ms:.0f}ms" if self.first_chunk_ms is not None else "N/A"
        ttft = f"{self.ttft_ms:.0f}ms" if self.ttft_ms is not None else "N/A"
        start_to_first = (
            f"{self.start_to_first_audio_ms:.0f}ms"
            if self.start_to_first_audio_ms is not None else "N/A"
        )
        step_mean = (
            f"{self.decode_step_mean_ms:.1f}ms"
            if self.decode_step_mean_ms is not None else "N/A"
        )
        step_p50 = (
            f"{self.decode_step_p50_ms:.1f}ms"
            if self.decode_step_p50_ms is not None else "N/A"
        )
        step_p95 = (
            f"{self.decode_step_p95_ms:.1f}ms"
            if self.decode_step_p95_ms is not None else "N/A"
        )
        return (
            f"first_chunk={fc} ttft={ttft} start_to_first_audio={start_to_first} "
            f"decode_step_mean={step_mean} p50={step_p50} p95={step_p95} "
            f"total={self.total_ms:.0f}ms chunks={self.num_chunks} "
            f"audio={self.duration_sec:.2f}s rtf={self.rtf:.2f}"
        )


@dataclass
class CaseResult:
    target: str
    case: str
    ok: bool
    skipped: bool = False
    summary: str = ""
    error: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    synthesis: SynthesisResult | None = None

    def to_json(self) -> dict[str, Any]:
        synth = None
        if self.synthesis is not None:
            synth = {
                "transport": self.synthesis.transport,
                "session_id": self.synthesis.session_id,
                "sample_rate": self.synthesis.sample_rate,
                "encoding": self.synthesis.encoding,
                "first_chunk_ms": self.synthesis.first_chunk_ms,
                "ttft_ms": self.synthesis.ttft_ms,
                "start_to_first_audio_ms": self.synthesis.start_to_first_audio_ms,
                "decode_step_mean_ms": self.synthesis.decode_step_mean_ms,
                "decode_step_p50_ms": self.synthesis.decode_step_p50_ms,
                "decode_step_p95_ms": self.synthesis.decode_step_p95_ms,
                "total_ms": self.synthesis.total_ms,
                "num_chunks": self.synthesis.num_chunks,
                "total_samples": self.synthesis.total_samples,
                "duration_sec": self.synthesis.duration_sec,
                "rtf": self.synthesis.rtf,
                "error": self.synthesis.error,
                "warnings": list(self.synthesis.warnings),
                "events": list(self.synthesis.events),
                "details": dict(self.synthesis.details),
            }
        return {
            "target": self.target,
            "case": self.case,
            "ok": self.ok,
            "skipped": self.skipped,
            "summary": self.summary,
            "error": self.error,
            "details": self.details,
            "synthesis": synth,
        }


def _compute_intervals_ms(timestamps: list[float]) -> list[float]:
    if len(timestamps) < 2:
        return []
    return [(timestamps[i] - timestamps[i - 1]) * 1000.0 for i in range(1, len(timestamps))]


def _error_text(exc: BaseException) -> str:
    return str(exc).strip() or exc.__class__.__name__


def _save_wav(audio: np.ndarray, path: Path, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.clip(audio, -1.0, 1.0)
    pcm16 = (audio * 32767.0).astype(np.int16)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())


def _decode_audio_bytes(raw: bytes, encoding: str) -> np.ndarray:
    if encoding == "pcm_s16le":
        return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
    return np.frombuffer(raw, dtype=np.float32)


def _load_ref_audio(path: str) -> bytes:
    return Path(path).read_bytes()


def _encode_ref_audio(raw: bytes | None) -> str | None:
    if raw is None:
        return None
    return base64.b64encode(raw).decode("ascii")


def _parse_host_port(endpoint: str, default_port: int) -> tuple[str, int]:
    host, sep, port_text = endpoint.strip().rpartition(":")
    if not sep:
        return endpoint.strip(), default_port
    if not host:
        raise ValueError(f"invalid endpoint: {endpoint!r}")
    return host, int(port_text)


def _make_engine_request_spec(args: argparse.Namespace, loaded_model_type: str = "") -> RequestSpec:
    task_type = (args.task_type or loaded_model_type or "custom_voice").strip()
    speaker = args.speaker.strip() if args.speaker else ""
    instruct = args.instruct.strip() if args.instruct else ""
    ref_audio_bytes = _load_ref_audio(args.ref_audio_path) if args.ref_audio_path else None
    ref_text = args.ref_text.strip() if args.ref_text else None

    if task_type == "custom_voice" and not speaker:
        speaker = "Serena"
    if task_type in {"voice_design", "instruct"} and not instruct:
        instruct = CUSTOM_VOICE_INSTRUCT_ZH
    if task_type in {"base", "icl", "voice_clone"} and ref_audio_bytes is None:
        raise ValueError(
            f"task_type '{task_type}' requires --ref-audio-path for a full synthesis test"
        )
    if task_type == "icl" and not ref_text:
        raise ValueError("task_type 'icl' requires --ref-text for a full synthesis test")

    return RequestSpec(
        task_type=task_type,
        language=args.language,
        speaker=speaker or None,
        instruct=instruct or None,
        ref_audio_bytes=ref_audio_bytes,
        ref_text=ref_text,
        x_vector_only=args.x_vector_only,
        sample_rate=args.sample_rate,
        channels=1,
        encoding=args.audio_encoding,
    )


def _custom_voice_instruct_supported(capabilities: dict[str, Any]) -> bool:
    model_type = str(capabilities.get("loaded_model_type", "") or "").strip()
    variant = str(capabilities.get("variant", "") or "").lower()
    if model_type != "custom_voice":
        return False
    if "0.6" in variant or "0b6" in variant:
        return False
    return True


def _engine_audio_encoding_to_proto(name: str) -> int:
    return (
        tts_pb2.AUDIO_ENCODING_PCM_S16LE
        if name == "pcm_s16le"
        else tts_pb2.AUDIO_ENCODING_PCM_F32
    )


def _case_from_synth(
    target: str,
    case: str,
    synthesis: SynthesisResult,
    *,
    save_path: Path | None = None,
    min_audio_sec: float = 0.1,
) -> CaseResult:
    ok = synthesis.error is None and synthesis.total_samples >= int(synthesis.sample_rate * min_audio_sec)
    if ok and save_path is not None and synthesis.audio is not None and synthesis.audio.size > 0:
        _save_wav(synthesis.audio, save_path, synthesis.sample_rate)
    summary = synthesis.to_summary()
    return CaseResult(
        target=target,
        case=case,
        ok=ok,
        summary=summary,
        error=synthesis.error or "",
        synthesis=synthesis,
    )


def _story_path(args: argparse.Namespace) -> Path:
    if args.story_path:
        return Path(args.story_path)
    return REPO_ROOT / "tests" / "data" / "story.txt"


def _read_story_text(args: argparse.Namespace) -> str:
    story_path = _story_path(args)
    if not story_path.is_file():
        return ""
    return story_path.read_text(encoding="utf-8").strip()


class EngineGrpcTransport:
    name = "engine-grpc"

    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self.host, self.port = _parse_host_port(endpoint, 50051)

    def get_capabilities(self, timeout: float) -> dict[str, Any]:
        import grpc

        channel = grpc.insecure_channel(f"{self.host}:{self.port}")
        stub = tts_pb2_grpc.TTSServiceStub(channel)
        try:
            grpc.channel_ready_future(channel).result(timeout=timeout)
            resp = stub.GetCapabilities(tts_pb2.GetCapabilitiesRequest(), timeout=timeout)
            return {
                "variant": resp.variant,
                "loaded_model_type": resp.loaded_model_type,
                "declared_supported_task_types": list(resp.declared_supported_task_types),
                "supported_audio_formats": [
                    {"encoding": fmt.encoding, "sample_rate": fmt.sample_rate, "channels": fmt.channels}
                    for fmt in resp.supported_audio_formats
                ],
            }
        finally:
            channel.close()

    def _make_session_config(self, spec: RequestSpec, *, input_mode: int) -> tts_pb2.SessionConfig:
        return tts_pb2.SessionConfig(
            task_type=spec.task_type,
            language=spec.language,
            speaker=spec.speaker or "",
            instruct=spec.instruct or "",
            ref_audio=spec.ref_audio_bytes or b"",
            ref_text=spec.ref_text or "",
            x_vector_only=bool(spec.x_vector_only),
            input_mode=input_mode,
            group_policy=tts_pb2.GROUP_POLICY_AUTO,
            audio=tts_pb2.AudioFormat(
                encoding=_engine_audio_encoding_to_proto(spec.encoding),
                sample_rate=spec.sample_rate,
                channels=spec.channels,
            ),
        )

    def synthesize_oneshot(
        self,
        spec: RequestSpec,
        text: str,
        *,
        timeout: float,
        session_id: str | None = None,
    ) -> SynthesisResult:
        import grpc

        sid = session_id or uuid.uuid4().hex[:12]
        result = SynthesisResult(self.name, sid, text, sample_rate=spec.sample_rate, encoding=spec.encoding)
        channel = grpc.insecure_channel(f"{self.host}:{self.port}")
        stub = tts_pb2_grpc.TTSServiceStub(channel)
        chunks: list[np.ndarray] = []
        timestamps: list[float] = []
        first_ts: float | None = None
        started = time.perf_counter()
        try:
            request = tts_pb2.SynthesizeOnceRequest(
                session_id=sid,
                text=text,
                config=self._make_session_config(spec, input_mode=tts_pb2.INPUT_MODE_FULL_TEXT),
            )
            for resp in stub.SynthesizeOnce(request, timeout=timeout):
                which = resp.WhichOneof("response")
                if which == "audio":
                    now = time.perf_counter()
                    if first_ts is None:
                        first_ts = now
                    timestamps.append(now)
                    chunks.append(_decode_audio_bytes(resp.audio.pcm_data, spec.encoding))
                    result.sample_rate = int(resp.audio.sample_rate or spec.sample_rate)
                elif which == "event":
                    result.events.append(resp.event.type)
                    if resp.event.type == "warning" and resp.event.message:
                        result.warnings.append(resp.event.message)
                    if resp.event.type == "error":
                        result.error = resp.event.message or "gRPC event error"
                        break
                    if resp.event.type in {"done", "end"}:
                        break
                elif which == "status":
                    if resp.status.event == "error":
                        result.error = resp.status.message or "gRPC status error"
                        break
                    if resp.status.event == "done":
                        break
        except grpc.RpcError as exc:
            result.error = f"gRPC {exc.code().name}: {exc.details()}"
        finally:
            channel.close()

        result.total_ms = (time.perf_counter() - started) * 1000.0
        result.first_chunk_ms = (first_ts - started) * 1000.0 if first_ts is not None else None
        result.start_to_first_audio_ms = result.first_chunk_ms
        result.ttft_ms = result.first_chunk_ms
        result.audio_chunk_intervals_ms = _compute_intervals_ms(timestamps)
        result.num_chunks = len(chunks)
        if chunks:
            result.audio = np.concatenate(chunks)
            result.total_samples = int(result.audio.size)
        return result

    def synthesize_streaming(
        self,
        spec: RequestSpec,
        text_chunks: list[str],
        *,
        timeout: float,
        session_id: str | None = None,
        chunk_delay_ms: float = 200.0,
        input_mode: int = tts_pb2.INPUT_MODE_CLAUSE,
    ) -> SynthesisResult:
        import grpc

        sid = session_id or uuid.uuid4().hex[:12]
        result = SynthesisResult(
            self.name,
            sid,
            " ".join(text_chunks),
            sample_rate=spec.sample_rate,
            encoding=spec.encoding,
        )
        channel = grpc.insecure_channel(f"{self.host}:{self.port}")
        stub = tts_pb2_grpc.TTSServiceStub(channel)
        chunks: list[np.ndarray] = []
        timestamps: list[float] = []
        first_ts: float | None = None
        marks = {"start_sent": None, "first_text_sent": None}

        def request_gen():
            marks["start_sent"] = time.perf_counter()
            yield tts_pb2.SynthesizeRequest(
                start=tts_pb2.StartRequest(
                    session_id=sid,
                    config=self._make_session_config(spec, input_mode=input_mode),
                )
            )
            for chunk in text_chunks:
                if chunk_delay_ms > 0:
                    time.sleep(chunk_delay_ms / 1000.0)
                now = time.perf_counter()
                if marks["first_text_sent"] is None:
                    marks["first_text_sent"] = now
                yield tts_pb2.SynthesizeRequest(text=tts_pb2.TextChunk(text=chunk))
            yield tts_pb2.SynthesizeRequest(end=tts_pb2.EndRequest())

        started = time.perf_counter()
        try:
            for resp in stub.SynthesizeStream(request_gen(), timeout=timeout):
                which = resp.WhichOneof("response")
                if which == "audio":
                    now = time.perf_counter()
                    if first_ts is None:
                        first_ts = now
                    timestamps.append(now)
                    chunks.append(_decode_audio_bytes(resp.audio.pcm_data, spec.encoding))
                    result.sample_rate = int(resp.audio.sample_rate or spec.sample_rate)
                elif which == "event":
                    result.events.append(resp.event.type)
                    if resp.event.type == "warning" and resp.event.message:
                        result.warnings.append(resp.event.message)
                    if resp.event.type == "error":
                        result.error = resp.event.message or "gRPC stream error"
                        break
                    if resp.event.type in {"done", "end"}:
                        break
        except grpc.RpcError as exc:
            result.error = f"gRPC {exc.code().name}: {exc.details()}"
        finally:
            channel.close()

        result.total_ms = (time.perf_counter() - started) * 1000.0
        result.first_chunk_ms = (first_ts - started) * 1000.0 if first_ts is not None else None
        result.start_to_first_audio_ms = (
            (first_ts - marks["start_sent"]) * 1000.0
            if first_ts is not None and marks["start_sent"] is not None else None
        )
        result.ttft_ms = (
            (first_ts - marks["first_text_sent"]) * 1000.0
            if first_ts is not None and marks["first_text_sent"] is not None else None
        )
        result.audio_chunk_intervals_ms = _compute_intervals_ms(timestamps)
        result.num_chunks = len(chunks)
        if chunks:
            result.audio = np.concatenate(chunks)
            result.total_samples = int(result.audio.size)
        return result

    def cancel_midstream(self, spec: RequestSpec, *, timeout: float) -> SynthesisResult:
        import grpc

        sid = uuid.uuid4().hex[:12]
        result = SynthesisResult(self.name, sid, "(cancel)")
        channel = grpc.insecure_channel(f"{self.host}:{self.port}")
        stub = tts_pb2_grpc.TTSServiceStub(channel)
        chunks: list[np.ndarray] = []
        started = time.perf_counter()

        def request_gen():
            yield tts_pb2.SynthesizeRequest(
                start=tts_pb2.StartRequest(
                    session_id=sid,
                    config=self._make_session_config(spec, input_mode=tts_pb2.INPUT_MODE_LONG_SEGMENT),
                )
            )
            yield tts_pb2.SynthesizeRequest(text=tts_pb2.TextChunk(text="这段文字将被取消。"))
            time.sleep(0.1)
            yield tts_pb2.SynthesizeRequest(cancel=tts_pb2.CancelRequest())

        try:
            for resp in stub.SynthesizeStream(request_gen(), timeout=timeout):
                which = resp.WhichOneof("response")
                if which == "audio":
                    chunks.append(_decode_audio_bytes(resp.audio.pcm_data, spec.encoding))
                elif which == "event":
                    result.events.append(resp.event.type)
                    if resp.event.type == "error":
                        result.error = resp.event.message or "cancel stream error"
                        break
                    if resp.event.type in {"done", "end"}:
                        break
        except grpc.RpcError:
            pass
        finally:
            channel.close()

        result.total_ms = (time.perf_counter() - started) * 1000.0
        result.num_chunks = len(chunks)
        if chunks:
            result.audio = np.concatenate(chunks)
            result.total_samples = int(result.audio.size)
        return result


class EngineWebSocketTransport:
    name = "engine-websocket"

    def __init__(self, url: str):
        self.url = url

    def get_capabilities(self, timeout: float) -> dict[str, Any]:
        conn = ws_connect(self.url, timeout=timeout)
        try:
            ws_send_json(conn, {"type": "get_capabilities"})
            deadline = time.perf_counter() + timeout
            while time.perf_counter() < deadline:
                conn.sock.settimeout(max(0.05, min(0.2, deadline - time.perf_counter())))
                opcode, payload = ws_recv_frame(conn)
                if opcode == 0x1:
                    message = json.loads(payload.decode("utf-8"))
                    if message.get("type") != "capabilities":
                        raise RawWebSocketError(
                            f"unexpected websocket response type: {message.get('type')!r}"
                        )
                    return message.get("capabilities", {})
            raise TimeoutError("websocket capabilities timed out")
        finally:
            ws_close(conn)

    def _config_payload(self, spec: RequestSpec, *, input_mode: str) -> dict[str, Any]:
        payload = {
            "task_type": spec.task_type,
            "language": spec.language,
            "speaker": spec.speaker or "",
            "instruct": spec.instruct or "",
            "ref_audio": _encode_ref_audio(spec.ref_audio_bytes) or "",
            "ref_text": spec.ref_text or "",
            "x_vector_only": bool(spec.x_vector_only),
            "input_mode": input_mode,
            "group_policy": "auto",
            "audio": {
                "encoding": spec.encoding,
                "sample_rate": spec.sample_rate,
                "channels": spec.channels,
            },
        }
        return payload

    def _consume_frame(
        self,
        result: SynthesisResult,
        chunks: list[np.ndarray],
        timestamps: list[float],
        state: dict[str, Any],
        opcode: int,
        payload: bytes,
        *,
        treat_close_as_ok: bool = False,
    ) -> bool:
        if opcode == 0x2:
            now = time.perf_counter()
            if state["first_ts"] is None:
                state["first_ts"] = now
            timestamps.append(now)
            chunks.append(_decode_audio_bytes(payload, state["encoding"]))
            return False
        if opcode == 0x1:
            message = json.loads(payload.decode("utf-8"))
            if message.get("type") != "event":
                return False
            event = message.get("event", {})
            event_type = str(event.get("type", "") or "")
            result.events.append(event_type)
            if event_type == "start" and isinstance(event.get("audio"), dict):
                audio = event["audio"]
                state["encoding"] = str(audio.get("encoding") or state["encoding"])
                result.encoding = state["encoding"]
                result.sample_rate = int(audio.get("sample_rate") or result.sample_rate)
            if event_type == "warning" and event.get("message"):
                result.warnings.append(str(event.get("message")))
            if event_type == "error":
                result.error = str(event.get("message") or "websocket error")
                return True
            if event_type in {"done", "end"}:
                return True
            return False
        if opcode == 0x8:
            return True if treat_close_as_ok else result.error is not None
        if opcode == 0x9:
            return False
        return False

    def _read_until_timeout(
        self,
        conn,
        result: SynthesisResult,
        chunks: list[np.ndarray],
        timestamps: list[float],
        state: dict[str, Any],
        *,
        deadline: float,
        stop_on_idle: bool,
        treat_close_as_ok: bool = False,
    ) -> bool:
        while time.perf_counter() < deadline:
            remaining = deadline - time.perf_counter()
            conn.sock.settimeout(max(0.02, min(0.1 if stop_on_idle else 0.5, remaining)))
            try:
                opcode, payload = ws_recv_frame(conn)
            except socket.timeout:
                return False if stop_on_idle else False
            terminal = self._consume_frame(
                result,
                chunks,
                timestamps,
                state,
                opcode,
                payload,
                treat_close_as_ok=treat_close_as_ok,
            )
            if terminal:
                return True
        if not stop_on_idle:
            result.error = result.error or "websocket timed out waiting for terminal event"
        return bool(result.error)

    def synthesize_oneshot(
        self,
        spec: RequestSpec,
        text: str,
        *,
        timeout: float,
        session_id: str | None = None,
    ) -> SynthesisResult:
        sid = session_id or uuid.uuid4().hex[:12]
        result = SynthesisResult(self.name, sid, text, sample_rate=spec.sample_rate, encoding=spec.encoding)
        conn = ws_connect(self.url, timeout=timeout)
        chunks: list[np.ndarray] = []
        timestamps: list[float] = []
        state = {"first_ts": None, "encoding": spec.encoding}
        started = time.perf_counter()
        try:
            ws_send_json(
                conn,
                {
                    "type": "oneshot",
                    "session_id": sid,
                    "text": text,
                    "config": self._config_payload(spec, input_mode="full_text"),
                },
            )
            deadline = started + timeout
            self._read_until_timeout(
                conn,
                result,
                chunks,
                timestamps,
                state,
                deadline=deadline,
                stop_on_idle=False,
            )
        except Exception as exc:
            result.error = _error_text(exc)
        finally:
            ws_close(conn)

        result.total_ms = (time.perf_counter() - started) * 1000.0
        first_ts = state["first_ts"]
        result.first_chunk_ms = (first_ts - started) * 1000.0 if first_ts is not None else None
        result.start_to_first_audio_ms = result.first_chunk_ms
        result.ttft_ms = result.first_chunk_ms
        result.audio_chunk_intervals_ms = _compute_intervals_ms(timestamps)
        result.num_chunks = len(chunks)
        if chunks:
            result.audio = np.concatenate(chunks)
            result.total_samples = int(result.audio.size)
        return result

    def synthesize_streaming(
        self,
        spec: RequestSpec,
        text_chunks: list[str],
        *,
        timeout: float,
        session_id: str | None = None,
        chunk_delay_ms: float = 200.0,
        input_mode: str = "clause",
    ) -> SynthesisResult:
        sid = session_id or uuid.uuid4().hex[:12]
        result = SynthesisResult(
            self.name,
            sid,
            " ".join(text_chunks),
            sample_rate=spec.sample_rate,
            encoding=spec.encoding,
        )
        conn = ws_connect(self.url, timeout=timeout)
        chunks: list[np.ndarray] = []
        timestamps: list[float] = []
        state = {"first_ts": None, "encoding": spec.encoding}
        marks = {"start_sent": None, "first_text_sent": None}
        started = time.perf_counter()
        try:
            marks["start_sent"] = time.perf_counter()
            ws_send_json(
                conn,
                {
                    "type": "start",
                    "session_id": sid,
                    "config": self._config_payload(spec, input_mode=input_mode),
                },
            )
            deadline = started + timeout
            self._read_until_timeout(
                conn, result, chunks, timestamps, state,
                deadline=min(deadline, time.perf_counter() + 0.1),
                stop_on_idle=True,
            )

            for chunk in text_chunks:
                if chunk_delay_ms > 0:
                    time.sleep(chunk_delay_ms / 1000.0)
                now = time.perf_counter()
                if marks["first_text_sent"] is None:
                    marks["first_text_sent"] = now
                ws_send_json(conn, {"type": "text", "text": chunk})
                terminal = self._read_until_timeout(
                    conn,
                    result,
                    chunks,
                    timestamps,
                    state,
                    deadline=min(deadline, time.perf_counter() + 0.1),
                    stop_on_idle=True,
                )
                if terminal:
                    break

            if result.error is None and "done" not in result.events:
                ws_send_json(conn, {"type": "end"})
                self._read_until_timeout(
                    conn,
                    result,
                    chunks,
                    timestamps,
                    state,
                    deadline=deadline,
                    stop_on_idle=False,
                )
        except Exception as exc:
            result.error = _error_text(exc)
        finally:
            ws_close(conn)

        result.total_ms = (time.perf_counter() - started) * 1000.0
        first_ts = state["first_ts"]
        result.first_chunk_ms = (first_ts - started) * 1000.0 if first_ts is not None else None
        result.start_to_first_audio_ms = (
            (first_ts - marks["start_sent"]) * 1000.0
            if first_ts is not None and marks["start_sent"] is not None else None
        )
        result.ttft_ms = (
            (first_ts - marks["first_text_sent"]) * 1000.0
            if first_ts is not None and marks["first_text_sent"] is not None else None
        )
        result.audio_chunk_intervals_ms = _compute_intervals_ms(timestamps)
        result.num_chunks = len(chunks)
        if chunks:
            result.audio = np.concatenate(chunks)
            result.total_samples = int(result.audio.size)
        return result

    def cancel_midstream(self, spec: RequestSpec, *, timeout: float) -> SynthesisResult:
        sid = uuid.uuid4().hex[:12]
        result = SynthesisResult(self.name, sid, "(cancel)")
        conn = ws_connect(self.url, timeout=timeout)
        chunks: list[np.ndarray] = []
        timestamps: list[float] = []
        state = {"first_ts": None, "encoding": spec.encoding}
        started = time.perf_counter()
        try:
            ws_send_json(
                conn,
                {
                    "type": "start",
                    "session_id": sid,
                    "config": self._config_payload(spec, input_mode="long_segment"),
                },
            )
            ws_send_json(conn, {"type": "text", "text": "这段文字将被取消。"})
            time.sleep(0.1)
            ws_send_json(conn, {"type": "cancel"})
            self._read_until_timeout(
                conn,
                result,
                chunks,
                timestamps,
                state,
                deadline=started + timeout,
                stop_on_idle=False,
                treat_close_as_ok=True,
            )
            if result.error == "websocket timed out waiting for terminal event":
                result.error = None
        except Exception as exc:
            result.error = _error_text(exc)
        finally:
            ws_close(conn)

        result.total_ms = (time.perf_counter() - started) * 1000.0
        result.num_chunks = len(chunks)
        if chunks:
            result.audio = np.concatenate(chunks)
            result.total_samples = int(result.audio.size)
        return result


class TritonGrpcTransport:
    name = "triton-grpc"

    def __init__(self, endpoint: str, model_name: str):
        self.endpoint = endpoint
        self.model_name = model_name

    def health(self) -> dict[str, Any]:
        import tritonclient.grpc as grpcclient

        client = grpcclient.InferenceServerClient(url=self.endpoint)
        return {
            "server_live": bool(client.is_server_live()),
            "server_ready": bool(client.is_server_ready()),
            "model_ready": bool(client.is_model_ready(self.model_name)),
        }

    def synthesize(self, spec: RequestSpec, text: str, timeout: float) -> SynthesisResult:
        import threading
        import tritonclient.grpc as grpcclient

        synth = SynthesisResult(self.name, uuid.uuid4().hex[:12], text)
        request = _build_triton_request(spec, text=text)
        req_json = json.dumps(request, ensure_ascii=False)

        client = grpcclient.InferenceServerClient(url=self.endpoint)
        req_input = grpcclient.InferInput("request", [1], "BYTES")
        req_input.set_data_from_numpy(np.array([req_json], dtype=object))
        audio_out = grpcclient.InferRequestedOutput("audio_chunk")
        event_type_out = grpcclient.InferRequestedOutput("event_type")
        event_json_out = grpcclient.InferRequestedOutput("event_json")
        final_out = grpcclient.InferRequestedOutput("is_final")

        done = threading.Event()
        chunks: list[np.ndarray] = []
        timestamps: list[float] = []
        first_ts: float | None = None
        started = time.perf_counter()

        def _decode_obj(value):
            if isinstance(value, bytes):
                return value.decode("utf-8")
            return str(value)

        def callback(result=None, error=None, **kwargs):
            nonlocal first_ts
            infer_result = kwargs.get("result", result)
            error = kwargs.get("error", error)
            if error is not None:
                synth.error = str(error)
                done.set()
                return
            if infer_result is None:
                return
            event_type = infer_result.as_numpy("event_type")
            event_json = infer_result.as_numpy("event_json")
            audio = infer_result.as_numpy("audio_chunk")
            is_final = infer_result.as_numpy("is_final")
            et = _decode_obj(event_type.flatten()[0]) if event_type is not None and event_type.size else ""
            payload = {}
            if event_json is not None and event_json.size:
                raw_json = _decode_obj(event_json.flatten()[0])
                if raw_json:
                    payload = json.loads(raw_json)
            if et:
                synth.events.append(et)
            if et == "start":
                audio_fmt = payload.get("audio_format", {}) or {}
                synth.encoding = str(audio_fmt.get("encoding") or synth.encoding)
                synth.sample_rate = int(audio_fmt.get("sample_rate") or synth.sample_rate)
            elif et == "audio" and audio is not None and audio.size:
                raw = audio.flatten()[0]
                now = time.perf_counter()
                if first_ts is None:
                    first_ts = now
                timestamps.append(now)
                chunks.append(_decode_audio_bytes(raw, synth.encoding))
            elif et == "warning" and payload.get("message"):
                synth.warnings.append(str(payload.get("message")))
            elif et == "error":
                synth.error = str(payload.get("message") or "triton grpc error")
                done.set()
                return
            if is_final is not None and is_final.size and bool(is_final.flatten()[0]):
                done.set()

        client.start_stream(callback=callback)
        try:
            client.async_stream_infer(
                model_name=self.model_name,
                inputs=[req_input],
                outputs=[audio_out, event_type_out, event_json_out, final_out],
            )
            if not done.wait(timeout):
                synth.error = f"triton grpc timed out after {timeout:.1f}s"
        finally:
            client.stop_stream()

        synth.total_ms = (time.perf_counter() - started) * 1000.0
        synth.first_chunk_ms = (first_ts - started) * 1000.0 if first_ts is not None else None
        synth.ttft_ms = synth.first_chunk_ms
        synth.start_to_first_audio_ms = synth.first_chunk_ms
        synth.audio_chunk_intervals_ms = _compute_intervals_ms(timestamps)
        synth.num_chunks = len(chunks)
        if chunks:
            synth.audio = np.concatenate(chunks)
            synth.total_samples = int(synth.audio.size)
        return synth


class TritonHttpTransport:
    name = "triton-http"

    def __init__(self, base_url: str, model_name: str):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name

    def health(self, timeout: float) -> dict[str, Any]:
        live = requests.get(f"{self.base_url}/v2/health/live", timeout=timeout)
        ready = requests.get(f"{self.base_url}/v2/health/ready", timeout=timeout)
        model = requests.get(f"{self.base_url}/v2/models/{self.model_name}/ready", timeout=timeout)
        return {
            "server_live": live.status_code == 200,
            "server_ready": ready.status_code == 200,
            "model_ready": model.status_code == 200,
        }

    def describe_model(self, timeout: float) -> dict[str, Any]:
        metadata_resp = requests.get(
            f"{self.base_url}/v2/models/{self.model_name}",
            timeout=timeout,
        )
        config_resp = requests.get(
            f"{self.base_url}/v2/models/{self.model_name}/config",
            timeout=timeout,
        )
        metadata_resp.raise_for_status()
        config_resp.raise_for_status()
        return {
            "metadata": metadata_resp.json(),
            "config": config_resp.json(),
        }

    def synthesize(
        self,
        spec: RequestSpec,
        text: str,
        timeout: float,
    ) -> SynthesisResult:
        req_json = json.dumps(_build_triton_request(spec, text=text), ensure_ascii=False)
        payload = {
            "inputs": [
                {
                    "name": "request",
                    "shape": [1],
                    "datatype": "BYTES",
                    "data": [req_json],
                }
            ],
            "outputs": [
                {"name": "audio_chunk"},
                {"name": "event_type"},
                {"name": "event_json"},
                {"name": "is_final"},
            ],
        }
        started = time.perf_counter()
        synth = SynthesisResult(self.name, uuid.uuid4().hex[:12], text)
        try:
            resp = requests.post(
                f"{self.base_url}/v2/models/{self.model_name}/versions/1/infer",
                json=payload,
                timeout=timeout,
            )
            if resp.status_code != 200:
                synth.error = f"HTTP {resp.status_code}: {resp.text[:500]}"
                synth.total_ms = (time.perf_counter() - started) * 1000.0
                return synth

            body = resp.json()
            outputs = body.get("outputs", [])
            by_name = {item.get("name"): item for item in outputs}
            missing = TRITON_EXPECTED_OUTPUTS - set(by_name)
            if missing:
                synth.error = f"missing outputs: {sorted(missing)}"
                synth.details["response_keys"] = list(body.keys())
                synth.details["outputs"] = outputs
                synth.total_ms = (time.perf_counter() - started) * 1000.0
                return synth

            event_type = self._first_output_scalar(by_name["event_type"])
            event_json = self._first_output_scalar(by_name["event_json"])
            audio_chunk = self._first_output_scalar(by_name["audio_chunk"])
            is_final = self._first_output_scalar(by_name["is_final"])

            meta = {}
            if event_json:
                meta = json.loads(event_json)
                if not isinstance(meta, dict):
                    meta = {}
            synth.details["meta"] = meta
            if event_type:
                synth.events.append(str(event_type))
            if meta.get("warnings"):
                synth.warnings.extend(str(x) for x in meta.get("warnings") or [])

            if str(event_type or "") == "error":
                synth.error = str(meta.get("message") or "triton http error")
            elif not bool(is_final):
                synth.error = "triton http response was not final"
            else:
                encoding = str(meta.get("audio_chunk_encoding") or "").strip().lower()
                audio_fmt = meta.get("audio_format", {}) or {}
                synth.encoding = str(audio_fmt.get("encoding") or synth.encoding)
                synth.sample_rate = int(audio_fmt.get("sample_rate") or synth.sample_rate)
                raw_bytes = b""
                if audio_chunk:
                    if encoding == "base64":
                        raw_bytes = base64.b64decode(str(audio_chunk))
                    else:
                        raw_bytes = str(audio_chunk).encode("utf-8")
                if raw_bytes:
                    synth.audio = _decode_audio_bytes(raw_bytes, synth.encoding)
                    synth.total_samples = int(synth.audio.size)
                    synth.num_chunks = 1
        except Exception as exc:
            synth.error = _error_text(exc)
        synth.total_ms = (time.perf_counter() - started) * 1000.0
        if synth.total_samples > 0:
            synth.first_chunk_ms = synth.total_ms
            synth.ttft_ms = synth.total_ms
            synth.start_to_first_audio_ms = synth.total_ms
        return synth

    @staticmethod
    def _first_output_scalar(output: dict[str, Any]) -> Any:
        data = output.get("data")
        if isinstance(data, list) and data:
            return data[0]
        contents = output.get("contents")
        if isinstance(contents, dict):
            for key in (
                "bytes_contents",
                "string_contents",
                "bool_contents",
                "int_contents",
                "int64_contents",
                "uint_contents",
                "uint64_contents",
                "fp32_contents",
                "fp64_contents",
            ):
                values = contents.get(key)
                if isinstance(values, list) and values:
                    return values[0]
        return None


def _build_triton_request(spec: RequestSpec, *, text: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "text": text,
        "language": spec.language,
    }
    if spec.task_type:
        payload["task_type"] = spec.task_type
    if spec.speaker:
        payload["speaker"] = spec.speaker
    if spec.instruct:
        payload["instruct"] = spec.instruct
    if spec.ref_audio_bytes:
        payload["ref_audio"] = _encode_ref_audio(spec.ref_audio_bytes)
    if spec.ref_text:
        payload["ref_text"] = spec.ref_text
    if spec.x_vector_only:
        payload["x_vector_only"] = True
    return payload


def _make_case(
    target: str,
    case: str,
    ok: bool,
    *,
    summary: str = "",
    error: str = "",
    details: dict[str, Any] | None = None,
    skipped: bool = False,
) -> CaseResult:
    return CaseResult(
        target=target,
        case=case,
        ok=ok,
        skipped=skipped,
        summary=summary,
        error=error,
        details=details or {},
    )


def _run_engine_suite(
    transport,
    args: argparse.Namespace,
    *,
    output_dir: Path,
) -> list[CaseResult]:
    results: list[CaseResult] = []
    target = transport.name
    try:
        caps = transport.get_capabilities(args.timeout)
        results.append(
            _make_case(
                target,
                "capabilities",
                ok=bool(caps.get("loaded_model_type") or caps.get("variant")),
                summary=f"variant={caps.get('variant') or 'unknown'}, model_type={caps.get('loaded_model_type') or 'unknown'}",
                details=caps,
            )
        )
    except Exception as exc:
        results.append(_make_case(target, "capabilities", ok=False, error=_error_text(exc)))
        return results

    try:
        spec = _make_engine_request_spec(args, caps.get("loaded_model_type", ""))
    except Exception as exc:
        results.append(_make_case(target, "request-spec", ok=False, error=_error_text(exc)))
        return results

    if not args.skip_single:
        synth = transport.synthesize_oneshot(spec, "你好，这是单路测试。", timeout=args.timeout, session_id=f"{target}-single")
        results.append(
            _case_from_synth(
                target,
                "single-smoke",
                synth,
                save_path=output_dir / f"{target}_single_smoke.wav",
            )
        )

    if not args.skip_streaming:
        synth = transport.synthesize_streaming(
            spec,
            ["你好，这是流式文本输入测试。", "我们正在验证", "文本追加功能", "是否工作正常。"],
            timeout=args.timeout,
            session_id=f"{target}-stream",
            chunk_delay_ms=args.chunk_delay_ms,
        )
        results.append(
            _case_from_synth(
                target,
                "streaming-text",
                synth,
                save_path=output_dir / f"{target}_streaming_text.wav",
            )
        )

    if not args.skip_custom_instruct:
        if _custom_voice_instruct_supported(caps):
            instruct_spec = RequestSpec(**{**spec.__dict__, "instruct": CUSTOM_VOICE_INSTRUCT_ZH})
            synth = transport.synthesize_oneshot(
                instruct_spec,
                "你好，这是带指令的预置音色测试。",
                timeout=args.timeout,
                session_id=f"{target}-instruct-one",
            )
            results.append(
                _case_from_synth(
                    target,
                    "custom-instruct-oneshot",
                    synth,
                    save_path=output_dir / f"{target}_custom_instruct_oneshot.wav",
                )
            )
        else:
            results.append(
                _make_case(
                    target,
                    "custom-instruct-oneshot",
                    ok=True,
                    skipped=True,
                    summary="skipped: current loaded model does not support custom_voice instruct",
                )
            )

    if not args.skip_concurrent:
        levels = [int(x.strip()) for x in args.concurrency.split(",") if x.strip()]
        for level in levels:
            concurrent_results: list[SynthesisResult] = []
            with ThreadPoolExecutor(max_workers=level) as executor:
                futures = {
                    executor.submit(
                        transport.synthesize_oneshot,
                        spec,
                        TEST_TEXTS[idx % len(TEST_TEXTS)],
                        timeout=args.timeout,
                        session_id=f"{target}-c{level}-{idx}",
                    ): idx
                    for idx in range(level)
                }
                for future in as_completed(futures):
                    concurrent_results.append(future.result())
            ok = all(r.error is None and r.total_samples > 0 for r in concurrent_results)
            summary = (
                f"ok={sum(1 for r in concurrent_results if r.error is None)}/{len(concurrent_results)} "
                f"median_first_chunk={statistics.median([r.first_chunk_ms for r in concurrent_results if r.first_chunk_ms is not None]):.0f}ms"
                if concurrent_results and any(r.first_chunk_ms is not None for r in concurrent_results)
                else f"ok={sum(1 for r in concurrent_results if r.error is None)}/{len(concurrent_results)}"
            )
            results.append(
                _make_case(
                    target,
                    f"concurrent-x{level}",
                    ok=ok,
                    summary=summary,
                    details={"results": [r.to_summary() for r in concurrent_results]},
                )
            )

    if not args.skip_long:
        for case_name, text, min_audio_sec in [
            ("medium-long", LONG_TEXT, 1.0),
            ("very-long", VERY_LONG_TEXT, 3.0),
        ]:
            synth = transport.synthesize_oneshot(
                spec,
                text,
                timeout=args.long_timeout,
                session_id=f"{target}-{case_name}",
            )
            case = _case_from_synth(
                target,
                case_name,
                synth,
                save_path=output_dir / f"{target}_{case_name}.wav",
                min_audio_sec=min_audio_sec,
            )
            results.append(case)

        stream_long = transport.synthesize_streaming(
            spec,
            STREAMING_LONG_TEXT_CHUNKS,
            timeout=args.long_timeout,
            session_id=f"{target}-stream-long",
            chunk_delay_ms=max(args.chunk_delay_ms, 300.0),
            input_mode=tts_pb2.INPUT_MODE_LONG_SEGMENT
            if target == "engine-grpc"
            else "long_segment",
        )
        results.append(
            _case_from_synth(
                target,
                "stream-long",
                stream_long,
                save_path=output_dir / f"{target}_stream_long.wav",
                min_audio_sec=1.0,
            )
        )

        story_text = _read_story_text(args)
        if story_text:
            story_spec = spec
            if _custom_voice_instruct_supported(caps):
                story_spec = RequestSpec(**{**spec.__dict__, "instruct": CUSTOM_VOICE_INSTRUCT_ZH})
            story = transport.synthesize_oneshot(
                story_spec,
                story_text,
                timeout=args.story_timeout,
                session_id=f"{target}-story",
            )
            results.append(
                _case_from_synth(
                    target,
                    "story",
                    story,
                    save_path=output_dir / f"{target}_story.wav",
                    min_audio_sec=3.0,
                )
            )
        else:
            results.append(
                _make_case(
                    target,
                    "story",
                    ok=True,
                    skipped=True,
                    summary=f"skipped: story file not found at {_story_path(args)}",
                )
            )

    if not args.skip_badcase:
        empty = transport.synthesize_oneshot(
            spec, "", timeout=min(args.timeout, 15.0), session_id=f"{target}-bad-empty"
        )
        results.append(
            _make_case(
                target,
                "badcase-empty-text",
                ok=bool(empty.error) or empty.total_samples == 0,
                summary=empty.to_summary(),
                error=empty.error or "",
                details={"samples": empty.total_samples},
                )
        )
        whitespace = transport.synthesize_oneshot(
            spec, "   \n\t  ", timeout=min(args.timeout, 15.0), session_id=f"{target}-bad-space"
        )
        results.append(
            _make_case(
                target,
                "badcase-whitespace",
                ok=bool(whitespace.error) or whitespace.total_samples == 0,
                summary=whitespace.to_summary(),
                error=whitespace.error or "",
                details={"samples": whitespace.total_samples},
            )
        )
        invalid_spec = RequestSpec(**{**spec.__dict__, "task_type": "nonexistent_task"})
        invalid = transport.synthesize_oneshot(
            invalid_spec, "测试", timeout=min(args.timeout, 15.0), session_id=f"{target}-bad-task"
        )
        results.append(
            _make_case(
                target,
                "badcase-invalid-task",
                ok=bool(invalid.error),
                summary=invalid.to_summary(),
                error=invalid.error or "",
            )
        )
        single_char = transport.synthesize_oneshot(
            spec, "好", timeout=min(args.timeout, 30.0), session_id=f"{target}-single-char"
        )
        results.append(
            _case_from_synth(
                target,
                "badcase-single-char",
                single_char,
                save_path=output_dir / f"{target}_single_char.wav",
            )
        )
        cancel = transport.cancel_midstream(spec, timeout=min(args.timeout, 15.0))
        results.append(
            _make_case(
                target,
                "badcase-cancel",
                ok=cancel.error is None,
                summary=cancel.to_summary(),
                error=cancel.error or "",
            )
        )
        slow_stream = transport.synthesize_streaming(
            spec,
            ["你好，", "这是一个", "慢速输入的测试。"],
            timeout=max(args.timeout, 60.0),
            session_id=f"{target}-slow-stream",
            chunk_delay_ms=1000.0,
        )
        results.append(
            _case_from_synth(
                target,
                "badcase-slow-stream",
                slow_stream,
                save_path=output_dir / f"{target}_slow_stream.wav",
            )
        )

    return results


def _run_triton_grpc_suite(transport: TritonGrpcTransport, args: argparse.Namespace, output_dir: Path) -> list[CaseResult]:
    results: list[CaseResult] = []
    try:
        health = transport.health()
        results.append(
            _make_case(
                transport.name,
                "health",
                ok=all(health.values()),
                summary=", ".join(f"{k}={v}" for k, v in health.items()),
                details=health,
            )
        )
    except Exception as exc:
        results.append(_make_case(transport.name, "health", ok=False, error=_error_text(exc)))
        return results

    spec = _make_engine_request_spec(args)
    synth = transport.synthesize(spec, "今天天气真好，我们一起出去玩吧。", timeout=args.timeout)
    case = _case_from_synth(
        transport.name,
        "grpc-synthesize",
        synth,
        save_path=output_dir / "triton_grpc_synthesize.wav",
    )
    results.append(case)

    if not args.skip_long:
        for case_name, text, min_audio_sec in [
            ("medium-long", LONG_TEXT, 1.0),
            ("very-long", VERY_LONG_TEXT, 3.0),
        ]:
            synth = transport.synthesize(spec, text, timeout=args.long_timeout)
            results.append(
                _case_from_synth(
                    transport.name,
                    case_name,
                    synth,
                    save_path=output_dir / f"triton_grpc_{case_name}.wav",
                    min_audio_sec=min_audio_sec,
                )
            )
        story_text = _read_story_text(args)
        if story_text:
            story = transport.synthesize(spec, story_text, timeout=args.story_timeout)
            results.append(
                _case_from_synth(
                    transport.name,
                    "story",
                    story,
                    save_path=output_dir / "triton_grpc_story.wav",
                    min_audio_sec=3.0,
                )
            )
        else:
            results.append(
                _make_case(
                    transport.name,
                    "story",
                    ok=True,
                    skipped=True,
                    summary=f"skipped: story file not found at {_story_path(args)}",
                )
            )
    return results


def _run_triton_http_suite(
    transport: TritonHttpTransport,
    args: argparse.Namespace,
    output_dir: Path,
) -> list[CaseResult]:
    results: list[CaseResult] = []
    try:
        health = transport.health(args.timeout)
        results.append(
            _make_case(
                transport.name,
                "health",
                ok=all(health.values()),
                summary=", ".join(f"{k}={v}" for k, v in health.items()),
                details=health,
            )
        )
    except Exception as exc:
        results.append(_make_case(transport.name, "health", ok=False, error=_error_text(exc)))
        return results

    describe: dict[str, Any] | None = None
    try:
        describe = transport.describe_model(args.timeout)
        metadata = describe.get("metadata", {}) or {}
        config = describe.get("config", {}) or {}
        meta_inputs = {item.get("name") for item in metadata.get("inputs", [])}
        meta_outputs = {item.get("name") for item in metadata.get("outputs", [])}
        backend = str(config.get("backend") or metadata.get("backend") or "")
        decoupled = bool((config.get("model_transaction_policy", {}) or {}).get("decoupled"))
        ok = (
            str(metadata.get("name") or "") == transport.model_name
            and TRITON_EXPECTED_INPUTS.issubset(meta_inputs)
            and TRITON_EXPECTED_OUTPUTS.issubset(meta_outputs)
            and bool(backend)
        )
        results.append(
            _make_case(
                transport.name,
                "model-config",
                ok=ok,
                summary=(
                    f"backend={backend or 'unknown'}, decoupled={decoupled}, "
                    f"inputs={sorted(meta_inputs)}, outputs={sorted(meta_outputs)}"
                ),
                details=describe,
                error="" if ok else "metadata/config validation failed",
            )
        )
    except Exception as exc:
        results.append(_make_case(transport.name, "model-config", ok=False, error=_error_text(exc)))
        return results

    spec = _make_engine_request_spec(args)
    synth = transport.synthesize(
        spec,
        "今天天气真好，我们一起出去玩吧。",
        timeout=args.timeout,
    )
    results.append(
        _case_from_synth(
            transport.name,
            "http-synthesize",
            synth,
            save_path=output_dir / "triton_http_synthesize.wav",
        )
    )

    if not args.skip_long:
        for case_name, text, min_audio_sec in [
            ("medium-long", LONG_TEXT, 1.0),
            ("very-long", VERY_LONG_TEXT, 3.0),
        ]:
            synth = transport.synthesize(spec, text, timeout=args.long_timeout)
            results.append(
                _case_from_synth(
                    transport.name,
                    case_name,
                    synth,
                    save_path=output_dir / f"triton_http_{case_name}.wav",
                    min_audio_sec=min_audio_sec,
                )
            )
        story_text = _read_story_text(args)
        if story_text:
            synth = transport.synthesize(spec, story_text, timeout=args.story_timeout)
            results.append(
                _case_from_synth(
                    transport.name,
                    "story",
                    synth,
                    save_path=output_dir / "triton_http_story.wav",
                    min_audio_sec=3.0,
                )
            )
        else:
            results.append(
                _make_case(
                    transport.name,
                    "story",
                    ok=True,
                    skipped=True,
                    summary=f"skipped: story file not found at {_story_path(args)}",
                )
            )
    return results


def _print_cases(cases: list[CaseResult]) -> None:
    for case in cases:
        status = "SKIP" if case.skipped else ("PASS" if case.ok else "FAIL")
        detail = case.summary or case.error or ""
        print(f"[{status}] {case.target:16} {case.case:28} {detail}")


def _print_final_summary(all_cases: list[CaseResult]) -> None:
    print("\n" + "=" * 72)
    print("Final Summary")
    print("=" * 72)
    grouped: dict[str, list[CaseResult]] = {}
    for case in all_cases:
        grouped.setdefault(case.target, []).append(case)
    for target, cases in grouped.items():
        ok = sum(1 for case in cases if case.ok and not case.skipped)
        fail = sum(1 for case in cases if not case.ok and not case.skipped)
        skip = sum(1 for case in cases if case.skipped)
        print(f"{target:16} ok={ok} fail={fail} skip={skip}")
        synths = [
            case.synthesis
            for case in cases
            if case.synthesis is not None and case.ok and not case.skipped and case.synthesis.error is None
        ]
        if not synths:
            continue
        first_chunks = [s.first_chunk_ms for s in synths if s.first_chunk_ms is not None]
        ttfts = [s.ttft_ms for s in synths if s.ttft_ms is not None]
        totals = [s.total_ms for s in synths if s.total_ms > 0]
        rtfs = [s.rtf for s in synths if s.rtf > 0]
        decode_steps = [step for s in synths for step in s.audio_chunk_intervals_ms]
        total_audio = sum(s.duration_sec for s in synths)
        wall_time = max((s.total_ms for s in synths), default=0.0) / 1000.0
        throughput = total_audio / wall_time if wall_time > 0 else 0.0
        if first_chunks:
            print(
                f"  first_chunk: min={min(first_chunks):.0f}ms "
                f"median={statistics.median(first_chunks):.0f}ms "
                f"max={max(first_chunks):.0f}ms mean={statistics.mean(first_chunks):.0f}ms"
            )
        if ttfts:
            print(
                f"  ttft:        min={min(ttfts):.0f}ms "
                f"median={statistics.median(ttfts):.0f}ms "
                f"max={max(ttfts):.0f}ms mean={statistics.mean(ttfts):.0f}ms"
            )
        if totals:
            print(
                f"  total:       min={min(totals):.0f}ms "
                f"median={statistics.median(totals):.0f}ms "
                f"max={max(totals):.0f}ms mean={statistics.mean(totals):.0f}ms"
            )
        if decode_steps:
            vals = sorted(decode_steps)
            idx = min(len(vals) - 1, max(0, int(round(0.95 * (len(vals) - 1)))))
            print(
                f"  decode_step: min={min(vals):.1f}ms "
                f"p50={statistics.median(vals):.1f}ms "
                f"p95={vals[idx]:.1f}ms mean={statistics.mean(vals):.1f}ms"
            )
        if rtfs:
            print(
                f"  rtf:         min={min(rtfs):.2f} "
                f"median={statistics.median(rtfs):.2f} max={max(rtfs):.2f}"
            )
        if total_audio > 0 and wall_time > 0:
            print(
                f"  throughput:  {total_audio:.1f}s audio / {wall_time:.1f}s wall "
                f"= {throughput:.2f}x realtime"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified full E2E test tool for engine and Triton endpoints.")
    parser.add_argument(
        "--targets",
        default="engine-grpc,engine-websocket,triton-grpc,triton-http",
        help="Comma-separated targets: engine-grpc,engine-websocket,triton-grpc,triton-http",
    )
    parser.add_argument("--engine-grpc", default=DEFAULT_ENGINE_GRPC)
    parser.add_argument("--engine-websocket", default=DEFAULT_ENGINE_WS)
    parser.add_argument("--triton-http", default=DEFAULT_TRITON_HTTP)
    parser.add_argument("--triton-grpc", default=DEFAULT_TRITON_GRPC)
    parser.add_argument("--triton-model", default=DEFAULT_TRITON_MODEL)
    parser.add_argument("--triton-http-model", default=DEFAULT_TRITON_HTTP_MODEL)
    parser.add_argument("--task-type", default="custom_voice")
    parser.add_argument("--speaker", default="Serena")
    parser.add_argument("--instruct", default="")
    parser.add_argument("--language", default="auto")
    parser.add_argument("--ref-audio-path", default="")
    parser.add_argument("--ref-text", default="")
    parser.add_argument("--x-vector-only", action="store_true")
    parser.add_argument("--audio-encoding", choices=["pcm_f32", "pcm_s16le"], default="pcm_f32")
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--long-timeout", type=float, default=300.0)
    parser.add_argument("--story-timeout", type=float, default=600.0)
    parser.add_argument("--story-path", default="")
    parser.add_argument("--chunk-delay-ms", type=float, default=200.0)
    parser.add_argument("--concurrency", default="1,2,4")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "workspace" / "audio_samples" / "serving_e2e"))
    parser.add_argument("--skip-single", action="store_true")
    parser.add_argument("--skip-streaming", action="store_true")
    parser.add_argument("--skip-custom-instruct", action="store_true")
    parser.add_argument("--skip-concurrent", action="store_true")
    parser.add_argument("--skip-long", action="store_true")
    parser.add_argument("--skip-badcase", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    targets = [item.strip() for item in args.targets.split(",") if item.strip()]
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    all_cases: list[CaseResult] = []

    for target in targets:
        if target == "engine-grpc":
            all_cases.extend(
                _run_engine_suite(
                    EngineGrpcTransport(args.engine_grpc),
                    args,
                    output_dir=output_root / target,
                )
            )
        elif target == "engine-websocket":
            all_cases.extend(
                _run_engine_suite(
                    EngineWebSocketTransport(args.engine_websocket),
                    args,
                    output_dir=output_root / target,
                )
            )
        elif target == "triton-grpc":
            all_cases.extend(
                _run_triton_grpc_suite(
                    TritonGrpcTransport(args.triton_grpc, args.triton_model),
                    args,
                    output_root / target,
                )
            )
        elif target == "triton-http":
            all_cases.extend(
                _run_triton_http_suite(
                    TritonHttpTransport(args.triton_http, args.triton_http_model),
                    args,
                    output_root / target,
                )
            )
        else:
            all_cases.append(_make_case(target, "target-parse", ok=False, error=f"unknown target: {target}"))

    if args.json:
        print(json.dumps([case.to_json() for case in all_cases], ensure_ascii=False, indent=2))
    else:
        _print_cases(all_cases)
        _print_final_summary(all_cases)

    return 0 if all(case.ok or case.skipped for case in all_cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())

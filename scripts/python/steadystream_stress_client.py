#!/usr/bin/env python3
"""Poisson-arrival concurrent stress client for SteadyStream Table 4.

Replays a fixed upstream trace at concurrency levels 1/8/16/32/64/128 with
async gRPC streaming sessions. Records per-packet timestamps for FASL/Jitter/Stutter.

Usage:
    python scripts/python/steadystream_stress_client.py \\
        --triton localhost:8001 \\
        --trace eval/fixtures/steadystream_stress_v1.json \\
        --concurrency 8 \\
        --seed 42 \\
        --out-dir workspace/table4_runs/c8_seed42
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
import uuid
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.stress_metrics import aggregate_stress_runs, compute_session_metrics

SAMPLE_RATE = 24000
DEFAULT_CONCURRENCY = [1, 8, 16, 32, 64, 128]


@dataclass
class SessionRecord:
    session_id: str
    variant: str
    concurrency: int
    seed: int
    trace_id: str
    session_start_ts: float = 0.0
    first_text_ts: float | None = None
    first_pcm_ts: float | None = None
    ttfb_ms: float | None = None
    server_ttft_ms: float | None = None
    trace_pause_after_total_ms: float = 0.0
    audio_packets: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    total_ms: float = 0.0
    total_samples: int = 0

    def to_dict(self) -> dict[str, Any]:
        packets = []
        for p in self.audio_packets:
            packets.append(
                {
                    "client_ts": p["client_ts"],
                    "n_samples": len(p["samples"]),
                    # Omit raw samples in JSON; kept in companion npz if requested
                }
            )
        d = {
            "session_id": self.session_id,
            "variant": self.variant,
            "concurrency": self.concurrency,
            "seed": self.seed,
            "trace_id": self.trace_id,
            "session_start_ts": self.session_start_ts,
            "first_text_ts": self.first_text_ts,
            "first_token_ts": self.first_text_ts,
            "first_pcm_ts": self.first_pcm_ts,
            "ttfb_ms": self.ttfb_ms,
            "ttft_ms": self.ttfb_ms,
            "server_ttft_ms": self.server_ttft_ms,
            "trace_pause_after_total_ms": self.trace_pause_after_total_ms,
            "audio_packets": packets,
            "events": self.events,
            "error": self.error,
            "total_ms": round(self.total_ms, 2),
            "total_samples": self.total_samples,
        }
        if hasattr(self, "_metrics") and self._metrics:
            d["metrics"] = self._metrics
        return d


def _decode_obj(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _decode_audio_bytes(raw: bytes, audio_format: dict) -> np.ndarray:
    if (audio_format.get("encoding") or "pcm_f32") == "pcm_s16le":
        return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
    return np.frombuffer(raw, dtype=np.float32)


def load_trace(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        trace = json.load(f)
    if "chunks" not in trace or not trace["chunks"]:
        raise ValueError(f"Trace {path} must contain non-empty 'chunks'")
    return trace


def poisson_session_starts(
    n_sessions: int,
    *,
    duration_sec: float,
    seed: int,
) -> list[float]:
    """Return relative start offsets (seconds) for n_sessions over duration_sec."""
    rng = random.Random(seed)
    if n_sessions <= 1:
        return [0.0]
    lam = n_sessions / max(duration_sec, 1.0)
    offsets = [0.0]
    t = 0.0
    while len(offsets) < n_sessions and t < duration_sec * 1.5:
        t += rng.expovariate(lam)
        offsets.append(t)
    while len(offsets) < n_sessions:
        offsets.append(offsets[-1] + 1.0 / lam)
    return sorted(offsets[:n_sessions])


def _run_engine_streaming_session(
    endpoint: str,
    trace: dict[str, Any],
    *,
    session_id: str,
    variant: str,
    concurrency: int,
    seed: int,
    start_delay_sec: float,
    timeout: float,
) -> SessionRecord:
    import grpc
    from engine.gateway import tts_pb2, tts_pb2_grpc

    rec = SessionRecord(
        session_id=session_id,
        variant=variant,
        concurrency=concurrency,
        seed=seed,
        trace_id=trace.get("trace_id", "unknown"),
        trace_pause_after_total_ms=sum(c.get("pause_after_ms", 0) for c in trace["chunks"]),
    )
    if start_delay_sec > 0:
        time.sleep(start_delay_sec)

    mode_map = {
        "token": tts_pb2.INPUT_MODE_TOKEN,
        "clause": tts_pb2.INPUT_MODE_CLAUSE,
        "long_segment": tts_pb2.INPUT_MODE_LONG_SEGMENT,
        "full_text": tts_pb2.INPUT_MODE_FULL_TEXT,
    }
    input_mode = mode_map.get(trace.get("stream_input_mode", "clause"), tts_pb2.INPUT_MODE_CLAUSE)
    chunks = trace["chunks"]

    def request_gen():
        yield tts_pb2.SynthesizeRequest(
            start=tts_pb2.StartRequest(
                session_id=session_id,
                config=tts_pb2.SessionConfig(
                    task_type=trace.get("task_type", "custom_voice"),
                    speaker=str(trace.get("speaker", "serena")).lower(),
                    language=trace.get("language", "Chinese"),
                    instruct=trace.get("instruct", ""),
                    input_mode=input_mode,
                    group_policy=tts_pb2.GROUP_POLICY_NONE,
                    audio=tts_pb2.AudioFormat(
                        encoding=tts_pb2.AUDIO_ENCODING_PCM_F32,
                        sample_rate=SAMPLE_RATE,
                        channels=1,
                    ),
                ),
            )
        )
        for chunk in chunks:
            pause_ms = float(chunk.get("pause_after_ms", 0))
            if pause_ms > 0:
                time.sleep(pause_ms / 1000.0)
            if rec.first_text_ts is None:
                rec.first_text_ts = time.perf_counter()
            yield tts_pb2.SynthesizeRequest(text=tts_pb2.TextChunk(text=chunk["text"]))
        yield tts_pb2.SynthesizeRequest(end=tts_pb2.EndRequest())

    t0 = time.perf_counter()
    rec.session_start_ts = t0
    channel = grpc.insecure_channel(endpoint)
    stub = tts_pb2_grpc.TTSServiceStub(channel)
    try:
        grpc.channel_ready_future(channel).result(timeout=10.0)
        for response in stub.SynthesizeStream(request_gen(), timeout=timeout):
            now = time.perf_counter()
            which = response.WhichOneof("response")
            if which == "audio":
                samples = np.frombuffer(response.audio.pcm_data, dtype=np.float32)
                if rec.first_pcm_ts is None:
                    rec.first_pcm_ts = now
                    rec.ttfb_ms = (now - t0) * 1000.0
                rec.audio_packets.append({"client_ts": now, "samples": samples})
                rec.events.append({"ts": now, "type": "audio", "n_samples": len(samples)})
            elif which == "event":
                payload = {"type": response.event.type, "message": response.event.message}
                rec.events.append({"ts": now, "type": "event", "payload": payload})
                if response.event.type == "error":
                    rec.error = response.event.message or "engine stream error"
                    break
    except grpc.RpcError as exc:
        rec.error = str(exc)
    finally:
        channel.close()

    rec.total_ms = (time.perf_counter() - t0) * 1000.0
    rec.total_samples = int(sum(len(p["samples"]) for p in rec.audio_packets))
    return rec


def run_streaming_session(
    backend: str,
    endpoint: str,
    trace: dict[str, Any],
    *,
    session_id: str,
    variant: str,
    concurrency: int,
    seed: int,
    start_delay_sec: float,
    timeout: float,
) -> SessionRecord:
    if backend == "engine":
        return _run_engine_streaming_session(
            endpoint, trace,
            session_id=session_id, variant=variant, concurrency=concurrency,
            seed=seed, start_delay_sec=start_delay_sec, timeout=timeout,
        )
    return _run_streaming_session(
        endpoint, trace,
        session_id=session_id, variant=variant, concurrency=concurrency,
        seed=seed, start_delay_sec=start_delay_sec, timeout=timeout,
    )


def _run_streaming_session(
    triton_url: str,
    trace: dict[str, Any],
    *,
    session_id: str,
    variant: str,
    concurrency: int,
    seed: int,
    start_delay_sec: float,
    timeout: float,
) -> SessionRecord:
    try:
        import tritonclient.grpc as grpcclient
    except ImportError as exc:
        raise RuntimeError("tritonclient[grpc] required") from exc

    rec = SessionRecord(
        session_id=session_id,
        variant=variant,
        concurrency=concurrency,
        seed=seed,
        trace_id=trace.get("trace_id", "unknown"),
        trace_pause_after_total_ms=sum(c.get("pause_after_ms", 0) for c in trace["chunks"]),
    )

    if start_delay_sec > 0:
        time.sleep(start_delay_sec)

    client = grpcclient.InferenceServerClient(url=triton_url)
    chunks = trace["chunks"]
    audio_format = {"encoding": "pcm_f32", "sample_rate": SAMPLE_RATE}
    done = threading.Event()
    errors: list[str] = []
    all_samples: list[np.ndarray] = []

    def callback(result=None, error=None):
        if error:
            err_str = str(error)
            if "CAPABILITIES:" not in err_str:
                errors.append(err_str)
                done.set()
            return
        if result is None:
            return
        event_type = result.as_numpy("event_type")
        event_json = result.as_numpy("event_json")
        audio = result.as_numpy("audio_chunk")
        is_final = result.as_numpy("is_final")
        et = _decode_obj(event_type.flatten()[0]) if event_type is not None and event_type.size else ""
        payload: dict[str, Any] = {}
        if event_json is not None and event_json.size:
            raw_json = _decode_obj(event_json.flatten()[0])
            if raw_json:
                payload = json.loads(raw_json)
        now = time.perf_counter()
        if et == "start":
            audio_format.update(payload.get("audio_format", {}) or {})
            rec.events.append({"ts": now, "type": "start", "payload": payload})
        elif et == "warning":
            rec.events.append({"ts": now, "type": "warning", "payload": payload})
        elif et == "text_token":
            rec.events.append({"ts": now, "type": "text_token", "payload": payload})
        elif et == "audio" and audio is not None and audio.size:
            samples = _decode_audio_bytes(audio.flatten()[0], audio_format)
            if rec.first_pcm_ts is None:
                rec.first_pcm_ts = now
                if rec.session_start_ts:
                    rec.ttfb_ms = (now - rec.session_start_ts) * 1000.0
                if rec.server_ttft_ms is None:
                    raw = payload.get("triton_adapter_ttft_ms") or payload.get("server_ttft_ms")
                    if raw is not None:
                        try:
                            rec.server_ttft_ms = float(raw)
                        except (TypeError, ValueError):
                            pass
            rec.audio_packets.append({"client_ts": now, "samples": samples})
            all_samples.append(samples)
            rec.events.append({"ts": now, "type": "audio", "n_samples": len(samples), "payload": payload})
        elif et == "error":
            errors.append(payload.get("message", "unknown error"))
            done.set()
            return
        final = bool(is_final.flatten()[0]) if is_final is not None and is_final.size else False
        if final:
            done.set()

    def _send_one(req_dict: dict[str, Any]) -> None:
        req_json = json.dumps(req_dict)
        req_input = grpcclient.InferInput("request", [1], "BYTES")
        req_input.set_data_from_numpy(np.array([req_json], dtype=object))
        client.async_stream_infer(
            model_name="tts_orchestrator",
            inputs=[req_input],
            outputs=[
                grpcclient.InferRequestedOutput("audio_chunk"),
                grpcclient.InferRequestedOutput("event_type"),
                grpcclient.InferRequestedOutput("event_json"),
                grpcclient.InferRequestedOutput("is_final"),
            ],
        )

    t0 = time.perf_counter()
    rec.session_start_ts = t0
    client.start_stream(callback=callback)

    init_req = {
        "action": "init",
        "session_id": session_id,
        "task_type": trace.get("task_type", "custom_voice"),
        "speaker": trace.get("speaker", "Serena"),
        "language": trace.get("language", "Chinese"),
        "instruct": trace.get("instruct", ""),
        "text": "",
    }
    if trace.get("stream_input_mode"):
        init_req["stream_input_mode"] = trace["stream_input_mode"]
    _send_one(init_req)

    for chunk in chunks:
        pause_ms = float(chunk.get("pause_after_ms", 0))
        if pause_ms > 0:
            time.sleep(pause_ms / 1000.0)
        if rec.first_text_ts is None:
            rec.first_text_ts = time.perf_counter()
        _send_one({"action": "append_text", "session_id": session_id, "text": chunk["text"]})

    _send_one({"action": "text_complete", "session_id": session_id})
    done.wait(timeout=timeout)
    client.stop_stream()

    rec.total_ms = (time.perf_counter() - t0) * 1000.0
    if errors:
        rec.error = errors[0]
    if all_samples:
        rec.total_samples = int(sum(s.size for s in all_samples))
    return rec


def run_stress(
    endpoint: str,
    trace: dict[str, Any],
    *,
    backend: str = "engine",
    concurrency: int,
    seed: int,
    variant: str,
    out_dir: Path,
    warmup: int,
    timeout: float,
    save_wav: bool,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    duration_sec = float(trace.get("expected_session_duration_sec") or 45.0)
    starts = poisson_session_starts(concurrency, duration_sec=duration_sec, seed=seed)

    # Warmup (single-session, not included in metrics)
    for i in range(warmup):
        sid = f"warmup-{i}-{uuid.uuid4().hex[:8]}"
        try:
            run_streaming_session(
                backend, endpoint,
                trace,
                session_id=sid,
                variant=variant,
                concurrency=1,
                seed=seed,
                start_delay_sec=0.0,
                timeout=timeout,
            )
        except Exception:
            pass

    wall_t0 = time.perf_counter()
    by_idx: dict[int, SessionRecord] = {}

    def _worker(idx: int) -> SessionRecord:
        sid = f"stress-c{concurrency}-s{seed}-{idx}-{uuid.uuid4().hex[:8]}"
        return run_streaming_session(
            backend, endpoint,
            trace,
            session_id=sid,
            variant=variant,
            concurrency=concurrency,
            seed=seed,
            start_delay_sec=starts[idx],
            timeout=timeout,
        )

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(_worker, i): i for i in range(concurrency)}
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                rec = fut.result()
            except Exception as exc:
                rec = SessionRecord(
                    session_id=f"failed-{idx}",
                    variant=variant,
                    concurrency=concurrency,
                    seed=seed,
                    trace_id=trace.get("trace_id", "unknown"),
                    error=str(exc),
                )
            by_idx[idx] = rec

    wall_sec = time.perf_counter() - wall_t0
    summaries: list[dict[str, Any]] = []
    for idx in sorted(by_idx):
        rec = by_idx[idx]
        metrics = compute_session_metrics(
            {
                **rec.to_dict(),
                "audio_packets": rec.audio_packets,
                "events": rec.events,
                "total_ms": rec.total_ms,
                "total_samples": rec.total_samples,
            }
        )
        rec._metrics = metrics
        summaries.append(metrics)
        session_path = out_dir / f"session_{idx:03d}.json"
        with open(session_path, "w", encoding="utf-8") as f:
            json.dump(rec.to_dict(), f, ensure_ascii=False, indent=2)
        if save_wav and rec.audio_packets:
            wav_path = out_dir / f"session_{idx:03d}.wav"
            _save_wav(np.concatenate([p["samples"] for p in rec.audio_packets]), wav_path)

    aggregate = aggregate_stress_runs(summaries, wall_sec=wall_sec)
    aggregate["concurrency"] = concurrency
    aggregate["seed"] = seed
    aggregate["variant"] = variant
    aggregate["wall_sec"] = round(wall_sec, 2)
    aggregate["trace_id"] = trace.get("trace_id")

    run_summary = {
        "aggregate": aggregate,
        "sessions": summaries,
    }
    with open(out_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(run_summary, f, ensure_ascii=False, indent=2)
    return run_summary


def _save_wav(audio: np.ndarray, path: Path) -> None:
    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * 32767).astype(np.int16)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_int16.tobytes())


def postprocess_run_dir(run_dir: Path) -> dict[str, Any]:
    """Recompute metrics from session JSON + optional sidecar npz (future)."""
    sessions = []
    for path in sorted(run_dir.glob("session_*.json")):
        with open(path, encoding="utf-8") as f:
            sessions.append(json.load(f))
    summaries = [compute_session_metrics(s) for s in sessions]
    wall_sec = None
    summary_path = run_dir / "run_summary.json"
    if summary_path.exists():
        with open(summary_path, encoding="utf-8") as f:
            wall_sec = (json.load(f).get("aggregate") or {}).get("wall_sec")
    return {"aggregate": aggregate_stress_runs(summaries, wall_sec=wall_sec), "sessions": summaries}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", default="engine", choices=["engine", "triton"])
    p.add_argument("--endpoint", default="", help="engine gRPC or triton gRPC host:port")
    p.add_argument("--triton", default="localhost:8001", help="Alias when --backend triton")
    p.add_argument("--trace", default=str(REPO_ROOT / "eval/fixtures/steadystream_stress_v1.json"))
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--variant", default="stateful_triton", choices=["pad_baseline", "stateful_triton", "full_steadystream"])
    p.add_argument("--out-dir", default="")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--save-wav", action="store_true")
    p.add_argument("--postprocess-only", action="store_true", help="Recompute metrics from existing run dir")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else (
        REPO_ROOT / "workspace" / "table4_runs" / f"{args.variant}_c{args.concurrency}_seed{args.seed}"
    )

    if args.postprocess_only:
        summary = postprocess_run_dir(out_dir)
        print(json.dumps(summary["aggregate"], ensure_ascii=False, indent=2))
        return 0

    trace = load_trace(Path(args.trace))
    endpoint = args.endpoint or (args.triton if args.backend == "triton" else "127.0.0.1:50051")
    print(
        f"Trace: {trace.get('trace_id')}  backend={args.backend}  endpoint={endpoint}  "
        f"concurrency={args.concurrency}  seed={args.seed}"
    )
    summary = run_stress(
        endpoint,
        trace,
        backend=args.backend,
        concurrency=args.concurrency,
        seed=args.seed,
        variant=args.variant,
        out_dir=out_dir,
        warmup=args.warmup,
        timeout=args.timeout,
        save_wav=args.save_wav,
    )
    agg = summary["aggregate"]
    print(json.dumps(agg, ensure_ascii=False, indent=2))
    print(f"Results written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

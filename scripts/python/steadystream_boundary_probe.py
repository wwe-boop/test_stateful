#!/usr/bin/env python3
"""Boundary continuity probe for SteadyStream-style experiments.

This script compares three synthesis modes against the standalone engine gRPC:

1. `stateless_once`: synthesize each segment independently via `SynthesizeOnce`
2. `stateful_stream`: synthesize all segments inside one `SynthesizeStream` session
3. `offline_full`: synthesize the concatenated full text with `SynthesizeOnce`

It saves WAVs plus a compact `results.json` with boundary F0 / energy probes.
The probe prefers exact engine `text_boundary_commit` events when available and
falls back to proportional boundary placement only when the response path does
not expose boundary events.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Any

import grpc
import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine.gateway import tts_pb2, tts_pb2_grpc
from eval.boundary_metrics import (
    boundary_positions_from_concat,
    boundary_positions_proportional,
    measure_boundary_metrics,
    summarize_boundary_metrics,
)


DEFAULT_SEGMENTS = [
    "其实我真的有发现，",
    "你是不是也有过这种感觉？",
    "明明刚才还很平静，",
    "下一秒却突然紧张起来！",
    "比如看到一串数字和英文缩写时，",
    "大脑会先停顿半秒，然后才继续往下读。",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="127.0.0.1:50051", help="Engine gRPC endpoint")
    parser.add_argument("--speaker", default="001")
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--instruct", default="")
    parser.add_argument("--task-type", default="custom_voice")
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--stream-input-mode", choices=["token", "clause", "long_segment", "full_text"], default="clause")
    parser.add_argument("--stream-group-policy", choices=["none", "auto"], default="none")
    parser.add_argument("--stream-timeout-sec", type=float, default=600.0)
    parser.add_argument("--segments-json", default="", help="JSON list of text segments")
    parser.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "workspace" / "steadystream_boundary_probe"),
        help="Directory for WAVs and results.json",
    )
    return parser.parse_args()


def load_segments(args: argparse.Namespace) -> list[str]:
    if args.segments_json:
        raw = json.loads(args.segments_json)
        if not isinstance(raw, list) or not all(isinstance(x, str) for x in raw):
            raise ValueError("--segments-json must be a JSON string list")
        parts = [x for x in raw if x]
        if not parts:
            raise ValueError("--segments-json yielded no non-empty segments")
        return parts
    return list(DEFAULT_SEGMENTS)


def input_mode_to_proto(name: str) -> int:
    mapping = {
        "token": tts_pb2.INPUT_MODE_TOKEN,
        "clause": tts_pb2.INPUT_MODE_CLAUSE,
        "long_segment": tts_pb2.INPUT_MODE_LONG_SEGMENT,
        "full_text": tts_pb2.INPUT_MODE_FULL_TEXT,
    }
    return mapping[name]


def group_policy_to_proto(name: str) -> int:
    mapping = {
        "none": tts_pb2.GROUP_POLICY_NONE,
        "auto": tts_pb2.GROUP_POLICY_AUTO,
    }
    return mapping[name]


def make_session_config(args: argparse.Namespace, *, input_mode: str, group_policy: str) -> Any:
    return tts_pb2.SessionConfig(
        task_type=args.task_type,
        speaker=args.speaker,
        language=args.language,
        instruct=args.instruct,
        input_mode=input_mode_to_proto(input_mode),
        group_policy=group_policy_to_proto(group_policy),
        audio=tts_pb2.AudioFormat(
            encoding=tts_pb2.AUDIO_ENCODING_PCM_F32,
            sample_rate=args.sample_rate,
            channels=1,
        ),
    )


def synthesize_once(
    stub: Any,
    args: argparse.Namespace,
    text: str,
    *,
    force_input_mode: str = "full_text",
) -> tuple[np.ndarray, int]:
    request = tts_pb2.SynthesizeOnceRequest(
        session_id=uuid.uuid4().hex[:12],
        text=text,
        config=make_session_config(args, input_mode=force_input_mode, group_policy="none"),
    )
    sample_rate = args.sample_rate
    chunks: list[np.ndarray] = []
    for response in stub.SynthesizeOnce(request, timeout=args.stream_timeout_sec):
        which = response.WhichOneof("response")
        if which == "audio":
            sample_rate = int(response.audio.sample_rate or sample_rate)
            chunks.append(np.frombuffer(response.audio.pcm_data, dtype=np.float32))
    wav = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    return wav, sample_rate


def synthesize_stream(
    stub: Any,
    args: argparse.Namespace,
    segments: list[str],
) -> tuple[np.ndarray, int, list[int]]:
    session_id = uuid.uuid4().hex[:12]

    def request_gen():
        yield tts_pb2.SynthesizeRequest(
            start=tts_pb2.StartRequest(
                session_id=session_id,
                config=make_session_config(
                    args,
                    input_mode=args.stream_input_mode,
                    group_policy=args.stream_group_policy,
                ),
            )
        )
        for text in segments:
            yield tts_pb2.SynthesizeRequest(text=tts_pb2.TextChunk(text=text))
        yield tts_pb2.SynthesizeRequest(end=tts_pb2.EndRequest())

    sample_rate = args.sample_rate
    chunks: list[np.ndarray] = []
    exact_boundaries: list[int] = []
    audio_samples_seen = 0
    for response in stub.SynthesizeStream(request_gen(), timeout=args.stream_timeout_sec):
        which = response.WhichOneof("response")
        if which == "audio":
            sample_rate = int(response.audio.sample_rate or sample_rate)
            chunk = np.frombuffer(response.audio.pcm_data, dtype=np.float32)
            chunks.append(chunk)
            audio_samples_seen += len(chunk)
        elif which == "event":
            if response.event.type == "error":
                raise RuntimeError(response.event.message or "engine stream error")
            if response.event.type == "text_boundary_commit":
                exact_boundaries.append(audio_samples_seen)
    wav = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    if len(exact_boundaries) > max(0, len(segments) - 1):
        exact_boundaries = exact_boundaries[: len(segments) - 1]
    return wav, sample_rate, exact_boundaries


def main() -> None:
    args = parse_args()
    segments = load_segments(args)
    full_text = "".join(segments)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    channel = grpc.insecure_channel(args.endpoint)
    grpc.channel_ready_future(channel).result(timeout=10.0)
    stub = tts_pb2_grpc.TTSServiceStub(channel)

    stateless_parts: list[np.ndarray] = []
    sample_rate = args.sample_rate
    for seg in segments:
        wav, sample_rate = synthesize_once(stub, args, seg)
        stateless_parts.append(wav)
    stateless_audio = np.concatenate(stateless_parts) if stateless_parts else np.zeros(0, dtype=np.float32)
    stateless_boundaries = boundary_positions_from_concat(stateless_parts)
    stateless_metrics = measure_boundary_metrics(
        stateless_audio,
        sample_rate,
        stateless_boundaries,
        boundary_source="exact_concat",
    )
    sf.write(out_dir / "stateless_once.wav", stateless_audio, sample_rate)

    stream_audio, stream_sr, stream_exact_boundaries = synthesize_stream(stub, args, segments)
    stream_boundaries = (
        stream_exact_boundaries
        if len(stream_exact_boundaries) == max(0, len(segments) - 1)
        else boundary_positions_proportional(stream_audio, segments)
    )
    stream_metrics = measure_boundary_metrics(
        stream_audio,
        stream_sr,
        stream_boundaries,
        boundary_source="exact_event" if len(stream_exact_boundaries) == max(0, len(segments) - 1) else "proxy_proportional",
    )
    sf.write(out_dir / "stateful_stream.wav", stream_audio, stream_sr)

    full_audio, full_sr = synthesize_once(stub, args, full_text)
    full_boundaries = boundary_positions_proportional(full_audio, segments)
    full_metrics = measure_boundary_metrics(
        full_audio,
        full_sr,
        full_boundaries,
        boundary_source="proxy_proportional",
    )
    sf.write(out_dir / "offline_full.wav", full_audio, full_sr)

    result = {
        "endpoint": args.endpoint,
        "speaker": args.speaker,
        "language": args.language,
        "segments": segments,
        "stream_input_mode": args.stream_input_mode,
        "stream_group_policy": args.stream_group_policy,
        "stateless_once": {
            "audio_sec": round(len(stateless_audio) / sample_rate, 3),
            "boundary_metrics": stateless_metrics,
            "boundary_summary": summarize_boundary_metrics(stateless_metrics),
        },
        "stateful_stream": {
            "audio_sec": round(len(stream_audio) / stream_sr, 3),
            "boundary_metrics": stream_metrics,
            "boundary_summary": summarize_boundary_metrics(stream_metrics),
            "exact_boundary_count": len(stream_exact_boundaries),
        },
        "offline_full": {
            "audio_sec": round(len(full_audio) / full_sr, 3),
            "boundary_metrics_proxy": full_metrics,
            "boundary_summary_proxy": summarize_boundary_metrics(full_metrics),
        },
    }
    with open(out_dir / "results.json", "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

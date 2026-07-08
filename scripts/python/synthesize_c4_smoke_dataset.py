#!/usr/bin/env python3
"""Synthesize a small API-backed SteadyStream C4 smoke dataset.

This script intentionally creates a *synthetic smoke* dataset, not final C4
training evidence. It is useful for validating the continuation manifest,
prepare_data input, collate layout, and short LoRA plumbing before real
long-recording data is available.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SAMPLE_RATE = 24000
PAUSE_MS_BY_PUNCT = {
    "comma": 220,
    "semicolon": 300,
    "colon": 300,
    "period": 460,
    "question": 380,
    "exclamation": 380,
    "dash": 260,
    "switch": 480,
    "none": 0,
}


@dataclass
class SegmentAudio:
    audio: np.ndarray
    sample_rate: int
    events: list[dict[str, Any]]


def pause_ms_for_punct(punct_class: str, *, is_last: bool = False) -> int:
    if is_last:
        return 0
    return int(PAUSE_MS_BY_PUNCT.get(str(punct_class or "period"), 460))


def append_silence(audio: np.ndarray, sample_rate: int, pause_ms: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if pause_ms <= 0:
        return audio
    silence = np.zeros(int(round(sample_rate * pause_ms / 1000.0)), dtype=np.float32)
    return np.concatenate([audio, silence])


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    audio = np.asarray(audio, dtype=np.float32)
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767.0).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _decode_audio(raw: bytes, encoding: int) -> np.ndarray:
    # tts_pb2.AUDIO_ENCODING_PCM_S16LE has value 2 in the generated proto.
    if encoding == 2:
        return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
    return np.frombuffer(raw, dtype=np.float32)


def synthesize_stream(
    *,
    endpoint: str,
    text: str,
    speaker: str,
    language: str,
    instruct: str,
    timeout: float,
) -> SegmentAudio:
    import grpc
    from engine.gateway import tts_pb2, tts_pb2_grpc

    session_id = f"c4-smoke-{uuid.uuid4().hex[:12]}"
    channel = grpc.insecure_channel(endpoint)
    stub = tts_pb2_grpc.TTSServiceStub(channel)
    grpc.channel_ready_future(channel).result(timeout=min(timeout, 10.0))

    def request_gen():
        yield tts_pb2.SynthesizeRequest(
            start=tts_pb2.StartRequest(
                session_id=session_id,
                config=tts_pb2.SessionConfig(
                    task_type="custom_voice",
                    language=language,
                    speaker=speaker,
                    instruct=instruct,
                    input_mode=tts_pb2.INPUT_MODE_FULL_TEXT,
                    group_policy=tts_pb2.GROUP_POLICY_NONE,
                    audio=tts_pb2.AudioFormat(
                        encoding=tts_pb2.AUDIO_ENCODING_PCM_F32,
                        sample_rate=SAMPLE_RATE,
                        channels=1,
                    ),
                ),
            )
        )
        yield tts_pb2.SynthesizeRequest(text=tts_pb2.TextChunk(text=text, seq_no=0))
        yield tts_pb2.SynthesizeRequest(end=tts_pb2.EndRequest())

    audio_chunks: list[np.ndarray] = []
    events: list[dict[str, Any]] = []
    sample_rate = SAMPLE_RATE
    for response in stub.SynthesizeStream(request_gen(), timeout=timeout):
        which = response.WhichOneof("response")
        if which == "audio":
            sample_rate = int(response.audio.sample_rate or sample_rate)
            audio_chunks.append(_decode_audio(response.audio.pcm_data, response.audio.encoding))
        elif which == "event":
            event = {
                "type": response.event.type,
                "message": response.event.message,
                "segment_id": int(response.event.segment_id),
                "meta": dict(response.event.meta),
            }
            events.append(event)
            if response.event.type == "error":
                raise RuntimeError(response.event.message or "TTS stream error")

    audio = np.concatenate(audio_chunks) if audio_chunks else np.zeros((0,), dtype=np.float32)
    if audio.size == 0:
        raise RuntimeError("TTS stream returned no audio")
    return SegmentAudio(audio=audio, sample_rate=sample_rate, events=events)


def build_manifest_row(
    source_row: dict[str, Any],
    *,
    sample_dir: Path,
    segment_records: list[dict[str, Any]],
    endpoint: str,
) -> dict[str, Any]:
    speaker = str(source_row.get("speaker") or source_row.get("speaker_name") or "unknown")
    return {
        "sample_id": str(source_row["sample_id"]),
        "speaker_name": speaker,
        "language": str(source_row.get("language") or "Chinese"),
        "instruct": str(source_row.get("instruct") or ""),
        "segments": segment_records,
        "instruct_switch": None,
        "meta": {
            "source": "api_synthetic_c4_smoke",
            "source_sample_id": str(source_row["sample_id"]),
            "source_scenario": str(source_row.get("scenario") or ""),
            "synth_endpoint": endpoint,
            "full_audio": str(sample_dir / "full.wav"),
            "synthetic": True,
            "not_for_final_table2_c4": True,
            "created_at_unix": int(time.time()),
        },
    }


def synthesize_dataset(args: argparse.Namespace) -> dict[str, Any]:
    input_rows = read_jsonl(args.input_jsonl)
    selected_rows = input_rows[args.start_index : args.start_index + args.limit if args.limit else None]
    manifest_rows: list[dict[str, Any]] = []
    prepare_rows: list[dict[str, Any]] = []
    run_items: list[dict[str, Any]] = []

    for row_index, row in enumerate(selected_rows, start=args.start_index):
        sample_id = str(row.get("sample_id") or f"sample_{row_index:05d}")
        sample_dir = args.out_dir / "wav" / sample_id
        segments = row.get("segments") or []
        if not isinstance(segments, list) or not segments:
            raise ValueError(f"{sample_id}: missing non-empty segments")

        speaker = str(row.get("speaker") or row.get("speaker_name") or args.default_speaker)
        if not args.keep_speaker_case:
            speaker = speaker.lower()
        language = str(row.get("language") or "Chinese")
        instruct = str(row.get("instruct") or "")

        print(f"[sample] {sample_id} speaker={speaker} segments={len(segments)}", flush=True)
        full_audio_parts: list[np.ndarray] = []
        segment_records: list[dict[str, Any]] = []
        ref_audio = ""

        for segment_index, segment in enumerate(segments):
            text = str(segment.get("text") or "").strip()
            if not text:
                raise ValueError(f"{sample_id} segment {segment_index}: empty text")
            punct_class = str(segment.get("punct_class") or "period")
            pause_ms = pause_ms_for_punct(punct_class, is_last=segment_index == len(segments) - 1)

            seg_audio = synthesize_stream(
                endpoint=args.endpoint,
                text=text,
                speaker=speaker,
                language=language,
                instruct=instruct,
                timeout=args.timeout,
            )
            with_pause = append_silence(seg_audio.audio, seg_audio.sample_rate, pause_ms)
            rel_wav = Path("wav") / sample_id / f"segment_{segment_index:03d}.wav"
            wav_path = args.out_dir / rel_wav
            write_wav(wav_path, with_pause, seg_audio.sample_rate)
            if not ref_audio:
                ref_audio = str(wav_path)

            full_audio_parts.append(with_pause)
            segment_records.append(
                {
                    "text": text,
                    "audio": str(wav_path),
                    "pause_ms": pause_ms,
                    "punct_class": punct_class,
                    "synthetic": True,
                }
            )
            prepare_rows.append(
                {
                    "sample_id": sample_id,
                    "segment_index": segment_index,
                    "audio": str(wav_path),
                    "text": text,
                    "ref_audio": ref_audio,
                    "speaker_name": speaker,
                    "language": language,
                    "pause_ms": pause_ms,
                    "punct_class": punct_class,
                    "synthetic": True,
                }
            )
            run_items.append(
                {
                    "sample_id": sample_id,
                    "segment_index": segment_index,
                    "text": text,
                    "audio": str(wav_path),
                    "audio_sec": round(float(with_pause.size) / seg_audio.sample_rate, 3),
                    "pause_ms": pause_ms,
                    "event_types": [event["type"] for event in seg_audio.events],
                }
            )

        full_audio = np.concatenate(full_audio_parts)
        write_wav(sample_dir / "full.wav", full_audio, SAMPLE_RATE)
        manifest_rows.append(
            build_manifest_row(
                row,
                sample_dir=sample_dir,
                segment_records=segment_records,
                endpoint=args.endpoint,
            )
        )

    manifest_path = args.out_dir / "c4_synthetic_smoke_manifest.jsonl"
    prepare_path = args.out_dir / "prepare_data_input.jsonl"
    summary_path = args.out_dir / "summary.json"
    write_jsonl(manifest_path, manifest_rows)
    write_jsonl(prepare_path, prepare_rows)

    summary = {
        "input_jsonl": str(args.input_jsonl),
        "out_dir": str(args.out_dir),
        "endpoint": args.endpoint,
        "samples": len(manifest_rows),
        "segments": len(prepare_rows),
        "manifest_jsonl": str(manifest_path),
        "prepare_data_input_jsonl": str(prepare_path),
        "synthetic": True,
        "not_for_final_table2_c4": True,
        "items": run_items,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=Path("workspace/datasets/test-prosody-mini-smoke.jsonl"),
        help="Segmented text JSONL generated by eval/data_synth.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("workspace/c4_synthetic_smoke"),
    )
    parser.add_argument("--endpoint", default="127.0.0.1:50051")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--default-speaker", default="serena")
    parser.add_argument("--keep-speaker-case", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = synthesize_dataset(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("[warn] synthetic smoke data is not valid final C4 evidence", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

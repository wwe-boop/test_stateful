#!/usr/bin/env python3
"""Synthesize C4 full-passage wavs via engine gRPC SynthesizeOnce.

This script is for Phase 1.3 of the SteadyStream C4 playbook: synthesize the
whole paragraph once. It intentionally does not synthesize per segment.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
import wave
from pathlib import Path
from typing import Any, Iterable

import numpy as np

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2] if len(SCRIPT_PATH.parents) > 2 else Path.cwd()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SAMPLE_RATE = 24000


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    audio = np.asarray(audio, dtype=np.float32)
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767.0).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def decode_audio(raw: bytes, encoding: int) -> np.ndarray:
    # tts_pb2.AUDIO_ENCODING_PCM_S16LE == 2; PCM_F32 == 1.
    if int(encoding) == 2:
        return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
    return np.frombuffer(raw, dtype=np.float32).astype(np.float32, copy=False)


def chinese_char_count(text: str) -> int:
    return sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")


def synthesize_once(stub: Any, tts_pb2: Any, row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    sample_id = str(row["sample_id"])
    text = str(row.get("text") or "")
    if not text:
        segments = row.get("segments") or []
        text = "".join(str(seg.get("text") or "") for seg in segments if isinstance(seg, dict))
    if not text.strip():
        raise ValueError(f"{sample_id}: empty text")

    wav_path = args.out_dir / "wav" / f"{sample_id}.wav"
    events_path = args.out_dir / "events" / f"{sample_id}.json"
    if args.resume and wav_path.exists() and events_path.exists():
        event_payload = json.loads(events_path.read_text(encoding="utf-8"))
        return {**event_payload["manifest_row"], "resume_hit": True}

    sampling = tts_pb2.SamplingParams(
        do_sample=bool(args.do_sample),
        temperature=float(args.temperature),
        top_k=int(args.top_k),
        top_p=float(args.top_p),
        repetition_penalty=float(args.repetition_penalty),
        max_new_tokens=int(args.max_new_tokens),
    )
    request = tts_pb2.SynthesizeOnceRequest(
        session_id=f"c4-full-{sample_id}-{uuid.uuid4().hex[:8]}",
        text=text,
        config=tts_pb2.SessionConfig(
            task_type=args.task_type,
            speaker=args.speaker,
            language=str(row.get("language") or args.language),
            instruct=str(row.get("instruct") if row.get("instruct") is not None else args.instruct),
            input_mode=tts_pb2.INPUT_MODE_FULL_TEXT,
            group_policy=tts_pb2.GROUP_POLICY_NONE,
            audio=tts_pb2.AudioFormat(
                encoding=tts_pb2.AUDIO_ENCODING_PCM_F32,
                sample_rate=int(args.sample_rate),
                channels=1,
            ),
            sampling=sampling,
        ),
    )

    t0 = time.perf_counter()
    audio_chunks: list[np.ndarray] = []
    events: list[dict[str, Any]] = []
    sample_rate = int(args.sample_rate)
    first_audio_ms: float | None = None
    for response in stub.SynthesizeOnce(request, timeout=float(args.timeout)):
        now = time.perf_counter()
        which = response.WhichOneof("response")
        if which == "audio":
            if first_audio_ms is None:
                first_audio_ms = (now - t0) * 1000.0
            sample_rate = int(response.audio.sample_rate or sample_rate)
            audio_chunks.append(decode_audio(response.audio.pcm_data, response.audio.encoding))
        elif which == "event":
            event = {
                "type": response.event.type,
                "session_id": response.event.session_id,
                "segment_id": int(response.event.segment_id),
                "text": response.event.text,
                "message": response.event.message,
                "meta": dict(response.event.meta),
                "dt_ms": round((now - t0) * 1000.0, 2),
            }
            events.append(event)
            if response.event.type == "error":
                raise RuntimeError(response.event.message or f"{sample_id}: synth error")

    audio = np.concatenate(audio_chunks) if audio_chunks else np.zeros((0,), dtype=np.float32)
    if audio.size == 0:
        raise RuntimeError(f"{sample_id}: no audio returned")
    write_wav(wav_path, audio, sample_rate)

    audio_sec = float(audio.size) / float(sample_rate)
    chars = chinese_char_count(text)
    sec_per_char = audio_sec / max(chars, 1)
    manifest_row = {
        "sample_id": sample_id,
        "speaker": args.speaker,
        "language": str(row.get("language") or args.language),
        "instruct": str(row.get("instruct") if row.get("instruct") is not None else args.instruct),
        "text": text,
        "segments": row.get("segments") or [],
        "boundary_punct_classes": row.get("boundary_punct_classes") or [],
        "full_audio": str(wav_path),
        "sample_rate": sample_rate,
        "audio_sec": round(audio_sec, 3),
        "chinese_chars": chars,
        "sec_per_char": round(sec_per_char, 4),
        "scenario": row.get("scenario"),
        "source_text_jsonl": str(args.input_jsonl),
        "synthetic_full_passage": True,
        "synth_endpoint": args.endpoint,
        "sampling": {
            "do_sample": bool(args.do_sample),
            "temperature": float(args.temperature),
            "top_k": int(args.top_k),
            "top_p": float(args.top_p),
            "repetition_penalty": float(args.repetition_penalty),
            "max_new_tokens": int(args.max_new_tokens),
        },
        "timing": {
            "first_audio_ms": round(first_audio_ms, 2) if first_audio_ms is not None else None,
            "total_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            "audio_chunks": len(audio_chunks),
        },
    }
    events_path.parent.mkdir(parents=True, exist_ok=True)
    events_path.write_text(
        json.dumps({"events": events, "manifest_row": manifest_row}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest_row


def summarize(rows: list[dict[str, Any]], failures: list[dict[str, str]], args: argparse.Namespace) -> dict[str, Any]:
    sec_per_char = [float(row["sec_per_char"]) for row in rows]
    audio_sec = [float(row["audio_sec"]) for row in rows]
    qc_ok = [row for row in rows if 0.15 <= float(row["sec_per_char"]) <= 0.35]
    return {
        "input_jsonl": str(args.input_jsonl),
        "out_dir": str(args.out_dir),
        "manifest_jsonl": str(args.out_dir / "c4_synth_full_manifest.jsonl"),
        "endpoint": args.endpoint,
        "speaker": args.speaker,
        "task_type": args.task_type,
        "sample_rate": args.sample_rate,
        "sampling": {
            "do_sample": bool(args.do_sample),
            "temperature": float(args.temperature),
            "top_k": int(args.top_k),
            "top_p": float(args.top_p),
            "repetition_penalty": float(args.repetition_penalty),
            "max_new_tokens": int(args.max_new_tokens),
        },
        "requested": args.limit,
        "succeeded": len(rows),
        "failed": len(failures),
        "failures": failures[:20],
        "total_audio_sec": round(sum(audio_sec), 3),
        "total_audio_hours": round(sum(audio_sec) / 3600.0, 4),
        "sec_per_char_min": round(min(sec_per_char), 4) if sec_per_char else None,
        "sec_per_char_max": round(max(sec_per_char), 4) if sec_per_char else None,
        "sec_per_char_qc_ok": len(qc_ok),
        "sec_per_char_qc_bad": len(rows) - len(qc_ok),
        "qc_rule": "0.15 <= sec_per_char <= 0.35; ASR CER QC is a later Phase 1.3 gate",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("workspace/c4_synth_v1"))
    parser.add_argument("--endpoint", default="127.0.0.1:50071")
    parser.add_argument("--speaker", default="001")
    parser.add_argument("--task-type", default="custom_voice")
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--instruct", default="")
    parser.add_argument("--sample-rate", type=int, default=SAMPLE_RATE)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--limit", type=int, default=0, help="0 means all rows")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--do-sample", action="store_true", help="Explicitly set sampling.do_sample=true; default false matches engine config.")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import grpc
    from engine.gateway import tts_pb2, tts_pb2_grpc

    all_rows = read_jsonl(args.input_jsonl)
    end = None if args.limit <= 0 else args.start_index + args.limit
    selected = all_rows[args.start_index:end]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    channel = grpc.insecure_channel(args.endpoint)
    grpc.channel_ready_future(channel).result(timeout=10.0)
    stub = tts_pb2_grpc.TTSServiceStub(channel)

    manifest_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    manifest_path = args.out_dir / "c4_synth_full_manifest.jsonl"
    for idx, row in enumerate(selected, start=args.start_index):
        sample_id = str(row.get("sample_id") or f"row_{idx:05d}")
        print(f"[synth] {idx} {sample_id}", flush=True)
        try:
            manifest_row = synthesize_once(stub, tts_pb2, row, args)
            manifest_rows.append(manifest_row)
            write_jsonl(manifest_path, manifest_rows)
        except Exception as exc:  # noqa: BLE001
            failure = {"sample_id": sample_id, "error": str(exc)}
            failures.append(failure)
            print(f"[error] {sample_id}: {exc}", file=sys.stderr, flush=True)
            if args.stop_on_error:
                raise

    channel.close()
    summary = summarize(manifest_rows, failures, args)
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())

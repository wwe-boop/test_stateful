#!/usr/bin/env python3
"""Build a C4 continuation manifest from full-passage synth event logs.

This is the pragmatic Phase 1.4 bridge for synthetic full-passage data: use the
engine's own full_text chunk boundaries (`text_boundary_commit` + `segment_end`)
instead of fake character-proportional cuts. Audio slicing uses only the Python
standard library because the 5090 host does not have numpy/soundfile.
"""

from __future__ import annotations

import argparse
import audioop
import json
import wave
from pathlib import Path
from typing import Any


PUNCT_TO_CLASS = {
    "，": "comma",
    ",": "comma",
    "、": "comma",
    "；": "semicolon",
    ";": "semicolon",
    "。": "period",
    ".": "period",
    "？": "question",
    "?": "question",
    "！": "exclamation",
    "!": "exclamation",
    "：": "colon",
    ":": "colon",
    "—": "dash",
    "-": "dash",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def wav_info(path: Path) -> tuple[int, int, int, bytes]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sampwidth = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frames = handle.getnframes()
        raw = handle.readframes(frames)
    if channels != 1 or sampwidth != 2:
        raise ValueError(f"{path}: expected mono s16 wav, got channels={channels} sampwidth={sampwidth}")
    return sample_rate, sampwidth, frames, raw


def write_wav_slice(path: Path, *, sample_rate: int, sampwidth: int, raw: bytes, start_frame: int, end_frame: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_bytes = sampwidth
    payload = raw[start_frame * frame_bytes : end_frame * frame_bytes]
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(sampwidth)
        handle.setframerate(sample_rate)
        handle.writeframes(payload)


def infer_punct_class(text: str) -> str:
    stripped = str(text or "").strip()
    for ch in reversed(stripped):
        if ch in PUNCT_TO_CLASS:
            return PUNCT_TO_CLASS[ch]
    return "none"


def estimate_trailing_silence_ms(raw: bytes, *, sample_rate: int, sampwidth: int, max_ms: int, frame_ms: int, rms_threshold: int) -> int:
    frame_bytes = int(sample_rate * frame_ms / 1000.0) * sampwidth
    if frame_bytes <= 0:
        return 0
    max_bytes = int(sample_rate * max_ms / 1000.0) * sampwidth
    tail = raw[-max_bytes:] if len(raw) > max_bytes else raw
    silent = 0
    for end in range(len(tail), 0, -frame_bytes):
        start = max(0, end - frame_bytes)
        chunk = tail[start:end]
        if len(chunk) < sampwidth:
            break
        if audioop.rms(chunk, sampwidth) <= rms_threshold:
            silent += frame_ms
        else:
            break
    return min(max_ms, silent)


def load_event_segments(events_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(events_path.read_text(encoding="utf-8"))
    events = payload.get("events") or []
    commits = [e for e in events if e.get("type") == "text_boundary_commit"]
    ends = [e for e in events if e.get("type") == "segment_end"]
    by_id: dict[int, dict[str, Any]] = {}
    for e in commits:
        by_id.setdefault(int(e.get("segment_id", -1)), {})["text"] = str(e.get("text") or "")
    for e in ends:
        seg_id = int(e.get("segment_id", -1))
        meta = e.get("meta") or {}
        by_id.setdefault(seg_id, {})["audio_steps"] = int(meta.get("audio_steps") or 0)
        by_id[seg_id]["event_meta"] = meta
    out = []
    for seg_id in sorted(k for k in by_id if k >= 0):
        item = by_id[seg_id]
        if item.get("text") and item.get("audio_steps"):
            out.append({"segment_id": seg_id, **item})
    return out


def build_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_rows = read_jsonl(args.full_manifest_jsonl)
    rows: list[dict[str, Any]] = []
    stats = {
        "source_rows": len(source_rows),
        "usable_rows": 0,
        "skipped_too_few_chunks": 0,
        "segments": 0,
        "audio_seconds": 0.0,
    }
    for src in source_rows:
        sample_id = str(src["sample_id"])
        events_path = args.events_dir / f"{sample_id}.json"
        wav_path = Path(src["full_audio"])
        if not wav_path.exists() and args.wav_dir:
            wav_path = args.wav_dir / f"{sample_id}.wav"
        if not events_path.exists() or not wav_path.exists():
            stats["skipped_too_few_chunks"] += 1
            continue
        event_segments = load_event_segments(events_path)
        if len(event_segments) < args.min_chunks:
            stats["skipped_too_few_chunks"] += 1
            continue
        sample_rate, sampwidth, total_frames, raw = wav_info(wav_path)
        total_steps = sum(int(seg["audio_steps"]) for seg in event_segments)
        if total_steps <= 0:
            stats["skipped_too_few_chunks"] += 1
            continue
        start_frame = 0
        out_segments: list[dict[str, Any]] = []
        for idx, seg in enumerate(event_segments):
            if idx == len(event_segments) - 1:
                end_frame = total_frames
            else:
                end_frame = round((sum(int(s["audio_steps"]) for s in event_segments[: idx + 1]) / total_steps) * total_frames)
            end_frame = max(start_frame + 1, min(total_frames, int(end_frame)))
            rel_audio = Path(sample_id) / f"segment_{idx:03d}.wav"
            out_audio = args.audio_out / rel_audio
            write_wav_slice(out_audio, sample_rate=sample_rate, sampwidth=sampwidth, raw=raw, start_frame=start_frame, end_frame=end_frame)
            seg_raw = raw[start_frame * sampwidth : end_frame * sampwidth]
            pause_ms = 0 if idx == len(event_segments) - 1 else estimate_trailing_silence_ms(
                seg_raw,
                sample_rate=sample_rate,
                sampwidth=sampwidth,
                max_ms=args.max_pause_ms,
                frame_ms=args.silence_frame_ms,
                rms_threshold=args.silence_rms_threshold,
            )
            text = str(seg["text"])
            out_segments.append(
                {
                    "text": text,
                    "audio": str(out_audio),
                    "pause_ms": pause_ms,
                    "punct_class": infer_punct_class(text),
                    "source_utterance_id": sample_id,
                    "engine_segment_id": int(seg["segment_id"]),
                    "audio_steps": int(seg["audio_steps"]),
                    "start_ms": round(start_frame * 1000.0 / sample_rate),
                    "audio_end_ms": round(end_frame * 1000.0 / sample_rate),
                    "split_source": "engine_full_text_events",
                }
            )
            start_frame = end_frame
        rows.append(
            {
                "sample_id": sample_id,
                "speaker_name": str(src.get("speaker") or "001"),
                "language": str(src.get("language") or "Chinese"),
                "segments": out_segments,
                "meta": {
                    "source": "c4_synth_full_passage",
                    "source_manifest": str(args.full_manifest_jsonl),
                    "original_full_audio": str(wav_path),
                    "split_mode": "engine_full_text_events",
                    "original_segments": src.get("segments") or [],
                },
            }
        )
        stats["usable_rows"] += 1
        stats["segments"] += len(out_segments)
        stats["audio_seconds"] += total_frames / sample_rate
    return rows, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-manifest-jsonl", type=Path, required=True)
    parser.add_argument("--events-dir", type=Path, required=True)
    parser.add_argument("--wav-dir", type=Path)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--audio-out", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--min-chunks", type=int, default=2)
    parser.add_argument("--max-pause-ms", type=int, default=600)
    parser.add_argument("--silence-frame-ms", type=int, default=20)
    parser.add_argument("--silence-rms-threshold", type=int, default=96)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows, stats = build_rows(args)
    write_jsonl(args.output_jsonl, rows)
    summary = {**stats, "output_jsonl": str(args.output_jsonl), "audio_out": str(args.audio_out), "samples": len(rows)}
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

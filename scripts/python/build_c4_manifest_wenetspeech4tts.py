#!/usr/bin/env python3
"""Build C4 continuation manifests from WenetSpeech4TTS Premium shards.

The preferred C4 pilot shape is intra-record continuation: each Premium wav is
already a same-speaker, quality-filtered merged segment with text timestamps. We
split that real continuous utterance at punctuation, keep the boundary silence
at the end of the previous segment, and emit a SteadyStream continuation JSONL.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

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
    "-": "dash",
    "—": "dash",
}
PUNCT_CHARS = set(PUNCT_TO_CLASS)
SOURCE_RE = re.compile(r"^(?P<source>[XY]\d+_[^_]+)_S(?P<start>\d+)(?:-S(?P<end>\d+))?$")


@dataclass(frozen=True)
class TextSegment:
    text: str
    punct_class: str
    start_ms: int
    text_end_ms: int
    audio_end_ms: int
    pause_ms: int


def read_dnsmos(path: Path | None) -> dict[str, float]:
    if path is None or not path.exists():
        return {}
    scores: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) >= 2:
            try:
                scores[parts[0]] = float(parts[1])
            except ValueError:
                continue
    return scores


def parse_wenet_txt(path: Path) -> tuple[str, str, list[tuple[int, int]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2 or "\t" not in lines[0]:
        raise ValueError(f"{path}: expected id/text line plus timestamp line")
    utt_id, text = lines[0].split("\t", 1)
    raw_times = ast.literal_eval(lines[1].strip())
    times = [(int(start), int(end)) for start, end in raw_times]
    return utt_id, text.strip(), times


def split_text_by_punctuation(
    text: str,
    times: list[tuple[int, int]],
    *,
    max_pause_ms: int,
    min_segment_ms: int,
    max_segment_ms: int,
    max_timestamp_mismatch: int,
) -> list[TextSegment]:
    non_punct_chars = [ch for ch in text if ch.strip() and ch not in PUNCT_CHARS]
    if abs(len(non_punct_chars) - len(times)) > max_timestamp_mismatch:
        return []

    raw_segments: list[dict[str, Any]] = []
    chars: list[str] = []
    first_time_idx: int | None = None
    last_time_idx: int | None = None
    time_idx = 0
    last_punct = ""

    def flush() -> None:
        nonlocal chars, first_time_idx, last_time_idx, last_punct
        seg_text = "".join(chars).strip()
        if seg_text and first_time_idx is not None and last_time_idx is not None:
            start_ms = int(times[first_time_idx][0])
            end_ms = int(times[last_time_idx][1])
            duration = end_ms - start_ms
            if min_segment_ms <= duration <= max_segment_ms:
                raw_segments.append(
                    {
                        "text": seg_text,
                        "punct_class": PUNCT_TO_CLASS.get(last_punct, "none"),
                        "start_ms": start_ms,
                        "text_end_ms": end_ms,
                    }
                )
        chars = []
        first_time_idx = None
        last_time_idx = None
        last_punct = ""

    for ch in text:
        if not ch.strip():
            continue
        chars.append(ch)
        if ch in PUNCT_CHARS:
            last_punct = ch
            flush()
            continue
        if time_idx >= len(times):
            return []
        if first_time_idx is None:
            first_time_idx = time_idx
        last_time_idx = time_idx
        time_idx += 1
    flush()

    if len(raw_segments) < 2:
        return []

    segments: list[TextSegment] = []
    for idx, seg in enumerate(raw_segments):
        next_start = raw_segments[idx + 1]["start_ms"] if idx + 1 < len(raw_segments) else None
        if next_start is None:
            pause_ms = 0
            audio_end_ms = int(seg["text_end_ms"])
        else:
            natural_pause = max(0, int(next_start) - int(seg["text_end_ms"]))
            pause_ms = min(int(max_pause_ms), natural_pause)
            audio_end_ms = int(seg["text_end_ms"]) + pause_ms
        segments.append(
            TextSegment(
                text=str(seg["text"]),
                punct_class=str(seg["punct_class"]),
                start_ms=int(seg["start_ms"]),
                text_end_ms=int(seg["text_end_ms"]),
                audio_end_ms=audio_end_ms,
                pause_ms=pause_ms,
            )
        )
    return segments


def source_id_from_utt_id(utt_id: str) -> str:
    match = SOURCE_RE.match(utt_id)
    if match:
        return match.group("source")
    return utt_id.rsplit("_S", 1)[0]


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def write_audio_slice(
    *,
    audio: np.ndarray,
    sample_rate: int,
    start_ms: int,
    end_ms: int,
    out_path: Path,
) -> None:
    start = max(0, int(round(start_ms * sample_rate / 1000.0)))
    end = min(len(audio), int(round(end_ms * sample_rate / 1000.0)))
    if end <= start:
        raise ValueError(f"empty audio slice for {out_path}: {start_ms}-{end_ms} ms")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, audio[start:end], sample_rate)


def iter_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    shard_dir = args.raw_root / "Premium" / args.shard
    txt_dir = shard_dir / "txts"
    wav_dir = shard_dir / "wavs"
    if not txt_dir.exists() or not wav_dir.exists():
        raise FileNotFoundError(f"missing extracted shard dirs under {shard_dir}")

    dnsmos = read_dnsmos(args.dnsmos)
    rows: list[dict[str, Any]] = []
    stats = {
        "txt_files": 0,
        "candidate_utterances": 0,
        "skipped_low_dnsmos": 0,
        "skipped_parse_or_alignment": 0,
        "windows": 0,
        "segments": 0,
    }

    for txt_path in sorted(txt_dir.glob("*.txt")):
        if args.limit and len(rows) >= args.limit:
            break
        stats["txt_files"] += 1
        try:
            utt_id, text, times = parse_wenet_txt(txt_path)
        except Exception:
            stats["skipped_parse_or_alignment"] += 1
            continue
        score = dnsmos.get(utt_id)
        if score is not None and score < args.min_dnsmos:
            stats["skipped_low_dnsmos"] += 1
            continue
        segments = split_text_by_punctuation(
            text,
            times,
            max_pause_ms=args.max_pause_ms,
            min_segment_ms=args.min_segment_ms,
            max_segment_ms=args.max_segment_ms,
            max_timestamp_mismatch=args.max_timestamp_mismatch,
        )
        if len(segments) < args.min_segments:
            continue
        stats["candidate_utterances"] += 1

        wav_path = wav_dir / f"{utt_id}.wav"
        if not wav_path.exists():
            stats["skipped_parse_or_alignment"] += 1
            continue
        audio = None
        sample_rate = None
        if args.audio_out:
            audio, sample_rate = sf.read(wav_path, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)

        source_id = source_id_from_utt_id(utt_id)
        stride = args.window_stride or args.max_segments
        for window_index, start in enumerate(range(0, len(segments) - args.min_segments + 1, stride)):
            if args.limit and len(rows) >= args.limit:
                break
            window = segments[start : start + args.max_segments]
            if len(window) < args.min_segments:
                continue
            sample_id = f"{args.shard}_{safe_name(utt_id)}_w{window_index:03d}"
            out_segments = []
            for seg_index, segment in enumerate(window):
                item = {
                    "text": segment.text,
                    "pause_ms": segment.pause_ms,
                    "punct_class": segment.punct_class,
                    "source_utterance_id": utt_id,
                    "start_ms": segment.start_ms,
                    "text_end_ms": segment.text_end_ms,
                    "audio_end_ms": segment.audio_end_ms,
                }
                if args.audio_out:
                    assert audio is not None and sample_rate is not None
                    rel_audio = Path(sample_id) / f"segment_{seg_index:03d}.wav"
                    out_audio = args.audio_out / rel_audio
                    slice_start = 0 if start == 0 and seg_index == 0 else segment.start_ms
                    write_audio_slice(
                        audio=audio,
                        sample_rate=sample_rate,
                        start_ms=slice_start,
                        end_ms=segment.audio_end_ms,
                        out_path=out_audio,
                    )
                    item["audio"] = str(out_audio)
                else:
                    item["audio"] = str(wav_path)
                out_segments.append(item)
            rows.append(
                {
                    "sample_id": sample_id,
                    "speaker_name": f"{args.speaker_prefix}_{safe_name(source_id)}",
                    "language": "Chinese",
                    "segments": out_segments,
                    "meta": {
                        "source": "WenetSpeech4TTS",
                        "subset": "Premium",
                        "shard": args.shard,
                        "source_id": source_id,
                        "source_utterance_id": utt_id,
                        "original_wav": str(wav_path),
                        "dnsmos": score,
                        "split_mode": "intra_utterance_punctuation",
                    },
                }
            )
            stats["windows"] += 1
            stats["segments"] += len(out_segments)
    return rows, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("workspace/datasets/raw/WenetSpeech4TTS"))
    parser.add_argument("--shard", default="WenetSpeech4TTS_Premium_0")
    parser.add_argument("--dnsmos", type=Path, default=Path("workspace/datasets/raw/WenetSpeech4TTS/DNSMOS_P808Scores/Premium_DNSMOS.lst"))
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--audio-out", type=Path, help="Optional directory for clipped segment wavs.")
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--min-segments", type=int, default=4)
    parser.add_argument("--max-segments", type=int, default=8)
    parser.add_argument("--window-stride", type=int, default=0)
    parser.add_argument("--max-pause-ms", type=int, default=600)
    parser.add_argument("--min-segment-ms", type=int, default=300)
    parser.add_argument("--max-segment-ms", type=int, default=15000)
    parser.add_argument("--max-timestamp-mismatch", type=int, default=2)
    parser.add_argument("--min-dnsmos", type=float, default=3.5)
    parser.add_argument("--speaker-prefix", default="wenet")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows, stats = iter_rows(args)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        **stats,
        "output_jsonl": str(args.output_jsonl),
        "audio_out": str(args.audio_out) if args.audio_out else None,
        "samples": len(rows),
    }
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

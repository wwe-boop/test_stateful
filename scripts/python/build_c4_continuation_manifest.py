#!/usr/bin/env python3
"""Validate SteadyStream C4 continuation JSONL manifests."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_PUNCT_CLASSES = {
    "comma",
    "semicolon",
    "period",
    "question",
    "exclamation",
    "colon",
    "dash",
    "switch",
    "none",
}


@dataclass
class ManifestIssue:
    line_no: int
    field: str
    message: str


@dataclass
class ManifestStats:
    samples: int = 0
    valid_samples: int = 0
    segments: int = 0
    coded_segments: int = 0
    code_frames: int = 0
    speaker_counts: Counter[str] = field(default_factory=Counter)
    punct_counts: Counter[str] = field(default_factory=Counter)
    source_counts: Counter[str] = field(default_factory=Counter)

    def to_dict(self, *, frame_hz: float) -> dict[str, Any]:
        speech_seconds = self.code_frames / frame_hz if frame_hz > 0 else 0.0
        return {
            "samples": self.samples,
            "valid_samples": self.valid_samples,
            "segments": self.segments,
            "coded_segments": self.coded_segments,
            "code_frames": self.code_frames,
            "speech_seconds_from_codes": round(speech_seconds, 3),
            "speech_hours_from_codes": round(speech_seconds / 3600.0, 6),
            "speaker_counts": dict(sorted(self.speaker_counts.items())),
            "punct_counts": dict(sorted(self.punct_counts.items())),
            "source_counts": dict(sorted(self.source_counts.items())),
        }


def _is_non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_code_frame(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, int) and item >= 0 for item in value)
    )


def _count_valid_code_frames(codes: Any) -> int:
    if not isinstance(codes, list):
        return 0
    count = 0
    for frame in codes:
        if not _is_code_frame(frame):
            return 0
        count += 1
    return count


def _source_keys(record: dict[str, Any]) -> set[str]:
    meta = record.get("meta")
    if not isinstance(meta, dict):
        return set()
    keys = set()
    for field_name in ("source", "source_id", "book_id", "program", "dataset"):
        value = meta.get(field_name)
        if _is_non_empty_string(value):
            keys.add(value.strip())
    return keys


def load_eval_sources(path: Path | None) -> set[str]:
    if path is None:
        return set()
    sources = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            sources.add(line)
    return sources


def validate_record(
    record: Any,
    *,
    line_no: int,
    require_codes: bool,
    max_pause_ms: float,
    min_segments: int,
    punct_classes: set[str],
    eval_sources: set[str],
    stats: ManifestStats,
) -> list[ManifestIssue]:
    issues: list[ManifestIssue] = []
    stats.samples += 1

    if not isinstance(record, dict):
        return [ManifestIssue(line_no, "$", "record must be a JSON object")]

    for field_name in ("sample_id", "speaker_name", "language"):
        if not _is_non_empty_string(record.get(field_name)):
            issues.append(
                ManifestIssue(line_no, field_name, "must be a non-empty string")
            )

    speaker_name = record.get("speaker_name")
    if _is_non_empty_string(speaker_name):
        stats.speaker_counts[speaker_name.strip()] += 1

    source_overlap = _source_keys(record) & eval_sources
    if source_overlap:
        issues.append(
            ManifestIssue(
                line_no,
                "meta",
                "source overlaps eval set: " + ", ".join(sorted(source_overlap)),
            )
        )
    for source in _source_keys(record):
        stats.source_counts[source] += 1

    segments = record.get("segments")
    if not isinstance(segments, list):
        issues.append(ManifestIssue(line_no, "segments", "must be a list"))
        return issues
    if len(segments) < min_segments:
        issues.append(
            ManifestIssue(
                line_no,
                "segments",
                f"must contain at least {min_segments} segments",
            )
        )

    stats.segments += len(segments)
    for idx, segment in enumerate(segments):
        prefix = f"segments[{idx}]"
        if not isinstance(segment, dict):
            issues.append(ManifestIssue(line_no, prefix, "must be an object"))
            continue

        if not _is_non_empty_string(segment.get("text")):
            issues.append(
                ManifestIssue(line_no, f"{prefix}.text", "must be non-empty")
            )

        pause_ms = segment.get("pause_ms")
        if not isinstance(pause_ms, (int, float)) or pause_ms < 0:
            issues.append(
                ManifestIssue(
                    line_no,
                    f"{prefix}.pause_ms",
                    "must be a non-negative number",
                )
            )
        elif pause_ms > max_pause_ms:
            issues.append(
                ManifestIssue(
                    line_no,
                    f"{prefix}.pause_ms",
                    f"must be <= {max_pause_ms:g} ms",
                )
            )

        punct_class = segment.get("punct_class")
        if not _is_non_empty_string(punct_class):
            issues.append(
                ManifestIssue(
                    line_no,
                    f"{prefix}.punct_class",
                    "must be a non-empty string",
                )
            )
        elif punct_class not in punct_classes:
            issues.append(
                ManifestIssue(
                    line_no,
                    f"{prefix}.punct_class",
                    "unknown punct class",
                )
            )
        else:
            stats.punct_counts[punct_class] += 1

        code_frame_count = _count_valid_code_frames(segment.get("codes"))
        if code_frame_count:
            stats.coded_segments += 1
            stats.code_frames += code_frame_count
        elif require_codes:
            issues.append(
                ManifestIssue(
                    line_no,
                    f"{prefix}.codes",
                    "must be a non-empty list of codec frames",
                )
            )

    if not issues:
        stats.valid_samples += 1
    return issues


def validate_manifest(
    input_jsonl: Path,
    *,
    require_codes: bool = False,
    max_pause_ms: float = 600.0,
    min_segments: int = 2,
    punct_classes: set[str] | None = None,
    eval_sources: set[str] | None = None,
) -> tuple[ManifestStats, list[ManifestIssue]]:
    stats = ManifestStats()
    issues: list[ManifestIssue] = []
    punct_classes = punct_classes or DEFAULT_PUNCT_CLASSES
    eval_sources = eval_sources or set()

    with input_jsonl.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                stats.samples += 1
                issues.append(ManifestIssue(line_no, "$", f"invalid JSON: {exc}"))
                continue
            issues.extend(
                validate_record(
                    record,
                    line_no=line_no,
                    require_codes=require_codes,
                    max_pause_ms=max_pause_ms,
                    min_segments=min_segments,
                    punct_classes=punct_classes,
                    eval_sources=eval_sources,
                    stats=stats,
                )
            )
    return stats, issues


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a SteadyStream C4 continuation JSONL manifest."
    )
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path)
    parser.add_argument("--require-codes", action="store_true")
    parser.add_argument("--max-pause-ms", type=float, default=600.0)
    parser.add_argument("--min-segments", type=int, default=2)
    parser.add_argument("--frame-hz", type=float, default=12.5)
    parser.add_argument(
        "--punct-class",
        action="append",
        dest="punct_classes",
        help="Allowed punctuation class. Can be passed multiple times.",
    )
    parser.add_argument(
        "--eval-source-list",
        type=Path,
        help="Optional newline-separated source keys reserved for evaluation.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    punct_classes = set(args.punct_classes or DEFAULT_PUNCT_CLASSES)
    eval_sources = load_eval_sources(args.eval_source_list)

    stats, issues = validate_manifest(
        args.input_jsonl,
        require_codes=args.require_codes,
        max_pause_ms=args.max_pause_ms,
        min_segments=args.min_segments,
        punct_classes=punct_classes,
        eval_sources=eval_sources,
    )
    summary = stats.to_dict(frame_hz=args.frame_hz)
    summary["issue_count"] = len(issues)
    summary["issues"] = [
        {"line_no": issue.line_no, "field": issue.field, "message": issue.message}
        for issue in issues
    ]

    text = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.output_summary:
        args.output_summary.parent.mkdir(parents=True, exist_ok=True)
        args.output_summary.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)

    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())

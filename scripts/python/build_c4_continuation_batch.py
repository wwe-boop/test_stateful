#!/usr/bin/env python3
"""Build a SteadyStream C4 continuation batch for layout dry-runs.

This is the bridge between the validated continuation JSONL and the future C4
training loop. It builds the same tensor families as the official single-
sentence collate path, but lays multiple text/code segments into one sequence
and masks loss to speech codec tokens plus segment boundary/EOS tokens.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch


ASSISTANT_TEMPLATE = "<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"


class TextTokenizer(Protocol):
    def __call__(self, text: str, return_tensors: str = "pt", padding: bool = True) -> Any:
        ...


@dataclass(frozen=True)
class C4SpecialIds:
    tts_pad_token_id: int
    tts_bos_token_id: int
    tts_eos_token_id: int
    codec_pad_id: int
    codec_bos_id: int
    codec_eos_token_id: int
    codec_nothink_id: int
    codec_think_bos_id: int
    codec_think_eos_id: int


def load_special_ids(config_path: Path) -> C4SpecialIds:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    talker = cfg.get("talker_config") or cfg
    return C4SpecialIds(
        tts_pad_token_id=int(cfg["tts_pad_token_id"]),
        tts_bos_token_id=int(cfg["tts_bos_token_id"]),
        tts_eos_token_id=int(cfg["tts_eos_token_id"]),
        codec_pad_id=int(talker["codec_pad_id"]),
        codec_bos_id=int(talker["codec_bos_id"]),
        codec_eos_token_id=int(talker["codec_eos_token_id"]),
        codec_nothink_id=int(talker["codec_nothink_id"]),
        codec_think_bos_id=int(talker["codec_think_bos_id"]),
        codec_think_eos_id=int(talker["codec_think_eos_id"]),
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def assistant_text_ids(tokenizer: TextTokenizer, text: str) -> list[int]:
    encoded = tokenizer(
        ASSISTANT_TEMPLATE.format(text=text),
        return_tensors="pt",
        padding=True,
    )["input_ids"]
    if encoded.dim() == 1:
        ids = encoded.tolist()
    else:
        ids = encoded[0].tolist()
    if len(ids) <= 5:
        raise ValueError(f"tokenized text too short for assistant template: {text!r}")
    return ids[:-5]


def validate_codes(codes: Any, *, sample_id: str, segment_index: int) -> torch.Tensor:
    if not isinstance(codes, list) or not codes:
        raise ValueError(f"{sample_id}:{segment_index}: missing non-empty codes")
    tensor = torch.tensor(codes, dtype=torch.long)
    if tensor.ndim != 2 or tensor.shape[1] != 16:
        raise ValueError(
            f"{sample_id}:{segment_index}: codes must have shape [frames, 16], got {tuple(tensor.shape)}"
        )
    return tensor


def _make_empty_batch(batch_size: int, max_length: int) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.zeros((batch_size, max_length, 2), dtype=torch.long),
        "codec_ids": torch.zeros((batch_size, max_length, 16), dtype=torch.long),
        "text_embedding_mask": torch.zeros((batch_size, max_length), dtype=torch.bool),
        "codec_embedding_mask": torch.zeros((batch_size, max_length), dtype=torch.bool),
        "codec_mask": torch.zeros((batch_size, max_length), dtype=torch.bool),
        "attention_mask": torch.zeros((batch_size, max_length), dtype=torch.long),
        "codec_0_labels": torch.full((batch_size, max_length), -100, dtype=torch.long),
    }


def estimate_row_length(row: dict[str, Any], tokenizer: TextTokenizer) -> int:
    length = 8
    for segment in row.get("segments") or []:
        text_ids = assistant_text_ids(tokenizer, str(segment.get("text") or ""))
        codes = validate_codes(
            segment.get("codes"),
            sample_id=str(row.get("sample_id") or ""),
            segment_index=0,
        )
        segment_text_len = max(0, len(text_ids) - 3)
        # text ids + text EOS + codec BOS + speech codec frames + boundary EOS
        length += segment_text_len + 1 + 1 + int(codes.shape[0]) + 1
    return length


def build_continuation_batch(
    rows: list[dict[str, Any]],
    *,
    tokenizer: TextTokenizer,
    special_ids: C4SpecialIds,
    max_segments: int | None = None,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    if not rows:
        raise ValueError("rows must not be empty")

    max_length = max(estimate_row_length(row, tokenizer) for row in rows)
    batch = _make_empty_batch(len(rows), max_length)
    layouts: list[dict[str, Any]] = []

    for batch_index, row in enumerate(rows):
        sample_id = str(row.get("sample_id") or f"sample_{batch_index}")
        segments = row.get("segments")
        if not isinstance(segments, list) or not segments:
            raise ValueError(f"{sample_id}: missing non-empty segments")
        if max_segments is not None:
            segments = segments[:max_segments]

        first_text_ids = assistant_text_ids(tokenizer, str(segments[0].get("text") or ""))
        if len(first_text_ids) < 3:
            raise ValueError(f"{sample_id}: first text ids shorter than prefix")

        # Shared CustomVoice/chat prefix, matching the official finetuning collate slots.
        batch["input_ids"][batch_index, :3, 0] = torch.tensor(first_text_ids[:3])
        batch["input_ids"][batch_index, 3:7, 0] = special_ids.tts_pad_token_id
        batch["input_ids"][batch_index, 7, 0] = special_ids.tts_bos_token_id
        batch["input_ids"][batch_index, 3:8, 1] = torch.tensor(
            [
                special_ids.codec_nothink_id,
                special_ids.codec_think_bos_id,
                special_ids.codec_think_eos_id,
                0,
                special_ids.codec_pad_id,
            ],
            dtype=torch.long,
        )

        spans: list[dict[str, Any]] = []
        pos = 8
        for segment_index, segment in enumerate(segments):
            text = str(segment.get("text") or "")
            text_ids = assistant_text_ids(tokenizer, text)
            text_body = torch.tensor(text_ids[3:], dtype=torch.long)
            codes = validate_codes(
                segment.get("codes"),
                sample_id=sample_id,
                segment_index=segment_index,
            )
            codec0 = codes[:, 0]

            text_start = pos
            text_end = text_start + int(text_body.numel())
            text_eos = text_end
            codec_bos = text_eos + 1
            codec_start = codec_bos + 1
            codec_end = codec_start + int(codec0.numel())
            boundary = codec_end
            pos = boundary + 1

            batch["input_ids"][batch_index, text_start:text_end, 0] = text_body
            batch["input_ids"][batch_index, text_eos, 0] = special_ids.tts_eos_token_id
            batch["input_ids"][batch_index, codec_bos:boundary + 1, 0] = special_ids.tts_pad_token_id

            batch["input_ids"][batch_index, text_start:codec_bos, 1] = special_ids.codec_pad_id
            batch["input_ids"][batch_index, codec_bos, 1] = special_ids.codec_bos_id
            batch["input_ids"][batch_index, codec_start:codec_end, 1] = codec0
            batch["input_ids"][batch_index, boundary, 1] = special_ids.codec_eos_token_id

            batch["codec_ids"][batch_index, codec_start:codec_end, :] = codes
            batch["codec_mask"][batch_index, codec_start:codec_end] = True
            # Match the official collate: train codec_0 speech frames and the
            # codec EOS/boundary position. codec_mask stays speech-only because
            # sub-talker residual-code loss has no target at EOS.
            batch["codec_0_labels"][batch_index, codec_start:codec_end] = codec0
            batch["codec_0_labels"][batch_index, boundary] = special_ids.codec_eos_token_id

            spans.append(
                {
                    "segment_index": segment_index,
                    "text": text,
                    "punct_class": segment.get("punct_class"),
                    "pause_ms": segment.get("pause_ms"),
                    "text_span": [text_start, text_end],
                    "text_eos": text_eos,
                    "codec_bos": codec_bos,
                    "codec_span": [codec_start, codec_end],
                    "boundary_codec_eos": boundary,
                    "codec_frames": int(codec0.numel()),
                    "loss_positions": int(codec0.numel()) + 1,
                }
            )

        batch["text_embedding_mask"][batch_index, :pos] = True
        batch["codec_embedding_mask"][batch_index, 3:pos] = True
        batch["codec_embedding_mask"][batch_index, 6] = False
        batch["attention_mask"][batch_index, :pos] = True

        layouts.append(
            {
                "sample_id": sample_id,
                "sequence_length": pos,
                "segments": spans,
                "total_codec_frames": sum(span["codec_frames"] for span in spans),
                "total_loss_positions": int((batch["codec_0_labels"][batch_index] != -100).sum().item()),
                "boundary_positions": [span["boundary_codec_eos"] for span in spans],
            }
        )

    return batch, layouts


def summarize_batch(batch: dict[str, torch.Tensor], layouts: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "batch_size": int(batch["input_ids"].shape[0]),
        "max_length": int(batch["input_ids"].shape[1]),
        "input_ids_shape": list(batch["input_ids"].shape),
        "codec_ids_shape": list(batch["codec_ids"].shape),
        "codec_mask_true": int(batch["codec_mask"].sum().item()),
        "loss_positions": int((batch["codec_0_labels"] != -100).sum().item()),
        "attention_tokens": int(batch["attention_mask"].sum().item()),
        "layouts": layouts,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-jsonl", type=Path, required=True)
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=Path("workspace/model_repository/tts_orchestrator/1/tokenizer"),
    )
    parser.add_argument(
        "--config-json",
        type=Path,
        default=Path("workspace/model_repository/tts_orchestrator/1/tokenizer/config.json"),
    )
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--max-segments", type=int)
    parser.add_argument("--output-summary", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from transformers import AutoTokenizer

    rows = read_jsonl(args.manifest_jsonl)
    if args.limit:
        rows = rows[: args.limit]
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_dir,
        trust_remote_code=True,
        fix_mistral_regex=True,
    )
    special_ids = load_special_ids(args.config_json)
    batch, layouts = build_continuation_batch(
        rows,
        tokenizer=tokenizer,
        special_ids=special_ids,
        max_segments=args.max_segments,
    )
    summary = summarize_batch(batch, layouts)
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.output_summary:
        args.output_summary.parent.mkdir(parents=True, exist_ok=True)
        args.output_summary.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

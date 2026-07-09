#!/usr/bin/env python3
"""Compare C4 continuation vs single-segment teacher-forced codec_0 NLL."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qwen_tts import Qwen3TTSModel  # noqa: E402
from scripts.python.build_c4_continuation_batch import build_continuation_batch, read_jsonl  # noqa: E402
from scripts.python.run_c4_forward_smoke import (  # noqa: E402
    first_ref_audio,
    load_ref_mels,
    special_ids_from_model_config,
)


def normalized_chars(text: str) -> int:
    return sum(1 for ch in str(text or "") if not ch.isspace())


def row_for_single_segment(row: dict[str, Any], segment_index: int) -> dict[str, Any]:
    out = copy.deepcopy(row)
    segments = out.get("segments") or []
    out["segments"] = [copy.deepcopy(segments[segment_index])]
    out["sample_id"] = f"{row.get('sample_id')}__single_seg{segment_index}"
    return out


def build_embeddings(
    *,
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    ref_mels: torch.Tensor,
) -> torch.Tensor:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    with torch.no_grad():
        speaker_embedding = model.speaker_encoder(ref_mels.to(device=device, dtype=dtype)).detach()

    input_ids = batch["input_ids"].to(device)
    codec_ids = batch["codec_ids"].to(device)
    text_embedding_mask = batch["text_embedding_mask"].to(device).unsqueeze(-1)
    codec_embedding_mask = batch["codec_embedding_mask"].to(device).unsqueeze(-1)
    codec_mask = batch["codec_mask"].to(device)

    input_text_ids = input_ids[:, :, 0]
    input_codec_ids = input_ids[:, :, 1]

    input_codec_embedding = model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
    input_text_embedding = model.talker.model.text_embedding(input_text_ids)
    if input_text_embedding.shape[-1] != input_codec_embedding.shape[-1]:
        input_text_embedding = model.talker.text_projection(input_text_embedding)
    input_text_embedding = input_text_embedding * text_embedding_mask
    if input_codec_embedding.shape[1] > 6:
        input_codec_embedding[:, 6, :] = speaker_embedding
    input_embeddings = input_text_embedding + input_codec_embedding

    for codebook_index in range(1, 16):
        codec_i_embedding = model.talker.code_predictor.get_input_embeddings()[codebook_index - 1](
            codec_ids[:, :, codebook_index]
        )
        codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
        input_embeddings = input_embeddings + codec_i_embedding

    return input_embeddings


@torch.inference_mode()
def codec0_nll_for_segment(
    *,
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    layout: dict[str, Any],
    segment_index: int,
    ref_mels: torch.Tensor,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    input_embeddings = build_embeddings(model=model, batch=batch, ref_mels=ref_mels)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["codec_0_labels"].to(device)

    outputs = model.talker(
        inputs_embeds=input_embeddings[:, :-1, :],
        attention_mask=attention_mask[:, :-1],
        labels=labels[:, 1:],
        output_hidden_states=False,
    )
    logits = outputs.logits
    shifted_labels = labels[:, 1:]
    positions = torch.arange(1, labels.shape[1], device=device).unsqueeze(0)
    seg = layout["segments"][segment_index]
    codec_start, codec_end = [int(x) for x in seg["codec_span"]]
    boundary = int(seg["boundary_codec_eos"])
    target_mask = (
        (positions >= codec_start)
        & (positions <= boundary)
        & (shifted_labels != -100)
    )
    if int(target_mask.sum().item()) <= 0:
        raise RuntimeError(f"no target NLL positions for {layout.get('sample_id')} segment {segment_index}")

    flat_loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        shifted_labels.reshape(-1),
        reduction="none",
        ignore_index=-100,
    ).reshape_as(shifted_labels)
    target_loss = flat_loss[target_mask]
    return {
        "sample_id": layout["sample_id"],
        "segment_index": segment_index,
        "target_text": seg["text"],
        "target_chars": normalized_chars(seg["text"]),
        "target_positions": int(target_mask.sum().item()),
        "codec_frames": int(seg["codec_frames"]),
        "nll_mean": round(float(target_loss.mean().detach().cpu()), 6),
        "nll_sum": round(float(target_loss.sum().detach().cpu()), 6),
        "ppl": round(float(torch.exp(target_loss.mean()).detach().cpu()), 6)
        if float(target_loss.mean().detach().cpu()) < 20
        else math.inf,
        "layout": seg,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    deltas = [float(r["delta_nll_cont_minus_single"]) for r in rows]
    ratios = [float(r["ratio_cont_over_single"]) for r in rows]
    cont = [float(r["continuation"]["nll_mean"]) for r in rows]
    single = [float(r["single_segment"]["nll_mean"]) for r in rows]
    return {
        "n": len(rows),
        "single_nll_mean": round(sum(single) / len(single), 6),
        "continuation_nll_mean": round(sum(cont) / len(cont), 6),
        "delta_nll_mean": round(sum(deltas) / len(deltas), 6),
        "delta_nll_min": round(min(deltas), 6),
        "delta_nll_max": round(max(deltas), 6),
        "ratio_mean": round(sum(ratios) / len(ratios), 6),
        "continuation_worse_count": sum(1 for d in deltas if d > 0),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    all_rows = read_jsonl(args.manifest_jsonl)
    selected = []
    for row in all_rows:
        segments = row.get("segments") or []
        if len(segments) <= args.target_segment_index:
            continue
        text = str(segments[args.target_segment_index].get("text") or "")
        if normalized_chars(text) < args.min_target_chars:
            continue
        selected.append(row)
        if args.limit and len(selected) >= args.limit:
            break
    if not selected:
        raise ValueError("no rows matched target segment/min chars filters")

    qwen3tts = Qwen3TTSModel.from_pretrained(
        str(args.model_dir),
        device_map=args.device_map,
        dtype=torch.bfloat16,
    )
    model = qwen3tts.model.eval()
    device = next(model.parameters()).device
    special_ids = special_ids_from_model_config(model.config)

    comparisons = []
    for idx, row in enumerate(selected, start=1):
        ref_mels = load_ref_mels(first_ref_audio(row, repo_root=args.repo_root))
        continuation_batch, continuation_layouts = build_continuation_batch(
            [row],
            tokenizer=qwen3tts.processor,
            special_ids=special_ids,
            max_segments=args.target_segment_index + 1,
        )
        continuation = codec0_nll_for_segment(
            model=model,
            batch=continuation_batch,
            layout=continuation_layouts[0],
            segment_index=args.target_segment_index,
            ref_mels=ref_mels,
        )

        single_row = row_for_single_segment(row, args.target_segment_index)
        single_batch, single_layouts = build_continuation_batch(
            [single_row],
            tokenizer=qwen3tts.processor,
            special_ids=special_ids,
            max_segments=1,
        )
        single = codec0_nll_for_segment(
            model=model,
            batch=single_batch,
            layout=single_layouts[0],
            segment_index=0,
            ref_mels=ref_mels,
        )
        delta = float(continuation["nll_mean"]) - float(single["nll_mean"])
        ratio = float(continuation["nll_mean"]) / max(float(single["nll_mean"]), 1e-8)
        item = {
            "index": idx,
            "sample_id": str(row.get("sample_id")),
            "target_segment_index": args.target_segment_index,
            "target_text": continuation["target_text"],
            "single_segment": single,
            "continuation": continuation,
            "delta_nll_cont_minus_single": round(delta, 6),
            "ratio_cont_over_single": round(ratio, 6),
        }
        comparisons.append(item)
        print(
            f"[NLL] {idx}/{len(selected)} {item['sample_id']} "
            f"single={single['nll_mean']:.4f} cont={continuation['nll_mean']:.4f} "
            f"delta={delta:.4f}",
            flush=True,
        )

    return {
        "model_dir": str(args.model_dir),
        "manifest_jsonl": str(args.manifest_jsonl),
        "device": str(device),
        "target_segment_index": args.target_segment_index,
        "min_target_chars": args.min_target_chars,
        "limit": args.limit,
        "summary": summarize(comparisons),
        "comparisons": comparisons,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-jsonl", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--device-map", default="cuda:1")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--target-segment-index", type=int, default=1)
    parser.add_argument("--min-target-chars", type=int, default=8)
    parser.add_argument("--output-summary", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run(args)
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["summary"], ensure_ascii=False, indent=2))
    print(f"Saved: {args.output_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

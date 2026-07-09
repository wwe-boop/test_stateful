#!/usr/bin/env python3
"""Small C4 continuation teachability diagnostic.

This is intentionally a smoke trainer, not a production fine-tuning recipe. It
uses the validated C4 continuation batch layout and records target-segment NLL
before/after a few full-parameter optimization steps on a tiny manifest.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW

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
from scripts.python.run_c4_teacher_forcing_nll import codec0_nll_for_segment  # noqa: E402


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def build_training_embeddings(
    *,
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    ref_mels: torch.Tensor,
) -> torch.Tensor:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

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


def restrict_loss_to_segment(
    batch: dict[str, torch.Tensor],
    layout: dict[str, Any],
    segment_index: int,
) -> dict[str, torch.Tensor]:
    out = {key: value.clone() for key, value in batch.items()}
    segment = layout["segments"][segment_index]
    codec_start, codec_end = [int(x) for x in segment["codec_span"]]
    boundary = int(segment["boundary_codec_eos"])

    labels = torch.full_like(out["codec_0_labels"], -100)
    labels[:, codec_start:codec_end] = out["codec_0_labels"][:, codec_start:codec_end]
    labels[:, boundary] = out["codec_0_labels"][:, boundary]
    out["codec_0_labels"] = labels

    codec_mask = torch.zeros_like(out["codec_mask"], dtype=torch.bool)
    codec_mask[:, codec_start:codec_end] = out["codec_mask"][:, codec_start:codec_end]
    out["codec_mask"] = codec_mask
    return out


def train_step(
    *,
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    ref_mels: torch.Tensor,
) -> torch.Tensor:
    device = next(model.parameters()).device
    input_embeddings = build_training_embeddings(model=model, batch=batch, ref_mels=ref_mels)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["codec_0_labels"].to(device)
    codec_ids = batch["codec_ids"].to(device)
    codec_mask = batch["codec_mask"].to(device)

    outputs = model.talker(
        inputs_embeds=input_embeddings[:, :-1, :],
        attention_mask=attention_mask[:, :-1],
        labels=labels[:, 1:],
        output_hidden_states=True,
    )
    hidden_states = outputs.hidden_states[0][-1]
    talker_hidden_states = hidden_states[codec_mask[:, :-1]]
    talker_codec_ids = codec_ids[codec_mask]
    _, sub_talker_loss = model.talker.forward_sub_talker_finetune(talker_codec_ids, talker_hidden_states)
    return outputs.loss + 0.3 * sub_talker_loss


@torch.inference_mode()
def evaluate_rows(
    *,
    model: torch.nn.Module,
    rows: list[dict[str, Any]],
    processor: Any,
    special_ids: Any,
    target_segment_index: int,
    repo_root: Path,
) -> dict[str, Any]:
    model.eval()
    items = []
    for row in rows:
        ref_mels = load_ref_mels(first_ref_audio(row, repo_root=repo_root))
        batch, layouts = build_continuation_batch(
            [row],
            tokenizer=processor,
            special_ids=special_ids,
            max_segments=target_segment_index + 1,
        )
        item = codec0_nll_for_segment(
            model=model,
            batch=batch,
            layout=layouts[0],
            segment_index=target_segment_index,
            ref_mels=ref_mels,
        )
        items.append(item)
    nlls = [float(item["nll_mean"]) for item in items]
    return {
        "n": len(items),
        "target_nll_mean": round(sum(nlls) / len(nlls), 6) if nlls else None,
        "target_nll_min": round(min(nlls), 6) if nlls else None,
        "target_nll_max": round(max(nlls), 6) if nlls else None,
        "items": items,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-jsonl", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--device-map", default="cuda:1")
    parser.add_argument("--target-segment-index", type=int, default=2)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--loss-scope", choices=["all", "target"], default="target")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--eval-limit", type=int, default=5)
    parser.add_argument("--output-summary", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows = read_jsonl(args.manifest_jsonl)[: args.limit]
    if not rows:
        raise ValueError("empty training manifest")
    eval_rows_subset = rows[: min(args.eval_limit, len(rows))]

    qwen3tts = Qwen3TTSModel.from_pretrained(
        str(args.model_dir),
        device_map=args.device_map,
        dtype=torch.bfloat16,
    )
    model = qwen3tts.model
    special_ids = special_ids_from_model_config(model.config)
    device = next(model.parameters()).device
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    before = evaluate_rows(
        model=model,
        rows=eval_rows_subset,
        processor=qwen3tts.processor,
        special_ids=special_ids,
        target_segment_index=args.target_segment_index,
        repo_root=args.repo_root,
    )

    losses = []
    model.train()
    for epoch in range(args.epochs):
        order = list(rows)
        random.shuffle(order)
        for step, row in enumerate(order, start=1):
            ref_mels = load_ref_mels(first_ref_audio(row, repo_root=args.repo_root))
            batch, layouts = build_continuation_batch(
                [row],
                tokenizer=qwen3tts.processor,
                special_ids=special_ids,
                max_segments=args.target_segment_index + 1,
            )
            if args.loss_scope == "target":
                batch = restrict_loss_to_segment(batch, layouts[0], args.target_segment_index)
            optimizer.zero_grad(set_to_none=True)
            loss = train_step(model=model, batch=batch, ref_mels=ref_mels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_value = float(loss.detach().cpu())
            losses.append(loss_value)
            print(f"[train] epoch={epoch} step={step}/{len(order)} loss={loss_value:.6f}", flush=True)

    after = evaluate_rows(
        model=model,
        rows=eval_rows_subset,
        processor=qwen3tts.processor,
        special_ids=special_ids,
        target_segment_index=args.target_segment_index,
        repo_root=args.repo_root,
    )

    summary = {
        "model_dir": str(args.model_dir),
        "manifest_jsonl": str(args.manifest_jsonl),
        "device": str(device),
        "target_segment_index": args.target_segment_index,
        "train_samples": len(rows),
        "eval_samples": len(eval_rows_subset),
        "epochs": args.epochs,
        "lr": args.lr,
        "loss_scope": args.loss_scope,
        "loss_first": round(losses[0], 6) if losses else None,
        "loss_last": round(losses[-1], 6) if losses else None,
        "loss_mean": round(sum(losses) / len(losses), 6) if losses else None,
        "before": before,
        "after": after,
        "delta_after_minus_before": round(
            float(after["target_nll_mean"]) - float(before["target_nll_mean"]), 6
        ) if before["target_nll_mean"] is not None and after["target_nll_mean"] is not None else None,
    }
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in {"before", "after"}}, ensure_ascii=False, indent=2))
    print(f"Saved: {args.output_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

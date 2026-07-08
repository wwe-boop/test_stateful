#!/usr/bin/env python3
"""Run one frozen-backbone optimizer step on a C4 continuation smoke batch."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qwen_tts import Qwen3TTSModel
from scripts.python.build_c4_continuation_batch import build_continuation_batch, read_jsonl
from scripts.python.run_c4_forward_smoke import (
    first_ref_audio,
    load_ref_mels,
    special_ids_from_model_config,
)


DEFAULT_TRAINABLE_PREFIXES = (
    "talker.codec_head",
    "talker.code_predictor.lm_head",
)


def parse_prefixes(value: str) -> tuple[str, ...]:
    return tuple(prefix.strip() for prefix in value.split(",") if prefix.strip())


def configure_trainable(model: torch.nn.Module, prefixes: tuple[str, ...]) -> dict[str, Any]:
    trainable_names: list[str] = []
    trainable_params = 0
    total_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        enabled = any(name.startswith(prefix) for prefix in prefixes)
        param.requires_grad_(enabled)
        if enabled:
            trainable_names.append(name)
            trainable_params += param.numel()
    if trainable_params == 0:
        raise ValueError(f"no trainable parameters matched prefixes: {prefixes}")
    return {
        "trainable_prefixes": list(prefixes),
        "trainable_names": trainable_names,
        "trainable_params": int(trainable_params),
        "total_params": int(total_params),
        "trainable_ratio": round(float(trainable_params) / float(total_params), 6),
    }


def build_embeddings_and_losses(
    *,
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    ref_mels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    with torch.no_grad():
        speaker_embedding = model.speaker_encoder(ref_mels.to(device=device, dtype=dtype)).detach()

    input_ids = batch["input_ids"].to(device)
    codec_ids = batch["codec_ids"].to(device)
    text_embedding_mask = batch["text_embedding_mask"].to(device).unsqueeze(-1)
    codec_embedding_mask = batch["codec_embedding_mask"].to(device).unsqueeze(-1)
    attention_mask = batch["attention_mask"].to(device)
    codec_0_labels = batch["codec_0_labels"].to(device)
    codec_mask = batch["codec_mask"].to(device)

    input_text_ids = input_ids[:, :, 0]
    input_codec_ids = input_ids[:, :, 1]

    input_codec_embedding = model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
    input_text_embedding = model.talker.model.text_embedding(input_text_ids)
    if input_text_embedding.shape[-1] != input_codec_embedding.shape[-1]:
        input_text_embedding = model.talker.text_projection(input_text_embedding)
    input_text_embedding = input_text_embedding * text_embedding_mask
    input_codec_embedding[:, 6, :] = speaker_embedding
    input_embeddings = input_text_embedding + input_codec_embedding

    for codebook_index in range(1, 16):
        codec_i_embedding = model.talker.code_predictor.get_input_embeddings()[codebook_index - 1](
            codec_ids[:, :, codebook_index]
        )
        codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
        input_embeddings = input_embeddings + codec_i_embedding

    outputs = model.talker(
        inputs_embeds=input_embeddings[:, :-1, :],
        attention_mask=attention_mask[:, :-1],
        labels=codec_0_labels[:, 1:],
        output_hidden_states=True,
    )
    hidden_states = outputs.hidden_states[0][-1]
    talker_hidden_states = hidden_states[codec_mask[:, :-1]]
    talker_codec_ids = codec_ids[codec_mask]
    _, sub_talker_loss = model.talker.forward_sub_talker_finetune(
        talker_codec_ids,
        talker_hidden_states,
    )
    combined_loss = outputs.loss + 0.3 * sub_talker_loss
    return outputs.loss, sub_talker_loss, combined_loss


def run_train_step(args: argparse.Namespace) -> dict[str, Any]:
    rows = read_jsonl(args.manifest_jsonl)
    if args.limit:
        rows = rows[: args.limit]
    if len(rows) != 1:
        raise ValueError("train-step smoke currently expects exactly one manifest row")

    qwen3tts = Qwen3TTSModel.from_pretrained(
        str(args.model_dir),
        device_map=args.device_map,
        dtype=torch.bfloat16,
    )
    model = qwen3tts.model
    trainable = configure_trainable(model, parse_prefixes(args.trainable_prefixes))
    model.train()

    batch, layouts = build_continuation_batch(
        rows,
        tokenizer=qwen3tts.processor,
        special_ids=special_ids_from_model_config(model.config),
        max_segments=args.max_segments,
    )
    ref_mels = load_ref_mels(first_ref_audio(rows[0], repo_root=args.repo_root))

    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    optimizer.zero_grad(set_to_none=True)
    talker_loss, sub_talker_loss, combined_loss = build_embeddings_and_losses(
        model=model,
        batch=batch,
        ref_mels=ref_mels,
    )
    combined_loss.backward()

    grad_norm = torch.nn.utils.clip_grad_norm_(
        [param for param in model.parameters() if param.requires_grad],
        args.max_grad_norm,
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    if args.save_trainable_state:
        args.save_trainable_state.parent.mkdir(parents=True, exist_ok=True)
        state = {
            name: param.detach().cpu()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        torch.save(state, args.save_trainable_state)

    device = next(model.parameters()).device
    return {
        "model_dir": str(args.model_dir),
        "manifest_jsonl": str(args.manifest_jsonl),
        "device": str(device),
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        **trainable,
        "batch_shape": list(batch["input_ids"].shape),
        "codec_frames": int(batch["codec_mask"].sum().item()),
        "loss_positions": int((batch["codec_0_labels"] != -100).sum().item()),
        "talker_loss": round(float(talker_loss.detach().cpu()), 6),
        "sub_talker_loss": round(float(sub_talker_loss.detach().cpu()), 6),
        "combined_loss": round(float(combined_loss.detach().cpu()), 6),
        "grad_norm": round(float(grad_norm.detach().cpu()), 6),
        "saved_trainable_state": str(args.save_trainable_state) if args.save_trainable_state else None,
        "layouts": layouts,
        "cuda_mem_allocated_bytes": int(torch.cuda.memory_allocated(device)) if device.type == "cuda" else None,
        "cuda_mem_reserved_bytes": int(torch.cuda.memory_reserved(device)) if device.type == "cuda" else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-jsonl", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--max-segments", type=int)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--trainable-prefixes",
        default=",".join(DEFAULT_TRAINABLE_PREFIXES),
    )
    parser.add_argument("--save-trainable-state", type=Path)
    parser.add_argument("--output-summary", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_train_step(args)
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.output_summary:
        args.output_summary.parent.mkdir(parents=True, exist_ok=True)
        args.output_summary.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

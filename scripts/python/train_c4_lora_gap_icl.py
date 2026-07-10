#!/usr/bin/env python3
"""Train/evaluate an ICL-style C4 LoRA layout for target codec_0 NLL."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from qwen_tts import Qwen3TTSModel
except ImportError:
    qwen_root = REPO_ROOT.parent / "Qwen3-TTS"
    if qwen_root.exists():
        sys.path.insert(0, str(qwen_root))
    from qwen_tts import Qwen3TTSModel

from scripts.python.build_c4_continuation_batch import read_jsonl  # noqa: E402
from scripts.python.run_c4_forward_smoke import first_ref_audio, load_ref_mels, special_ids_from_model_config  # noqa: E402
from scripts.python.run_c4_teacher_forcing_nll import (  # noqa: E402
    codec0_nll_for_segment,
    row_for_single_segment,
)


ASSISTANT_TEMPLATE = "<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"


def normalized_chars(text: str) -> int:
    return sum(1 for ch in str(text or "") if not ch.isspace())


def assistant_ids(processor: Any, text: str, device: torch.device) -> torch.Tensor:
    return processor(
        text=[ASSISTANT_TEMPLATE.format(text=text)],
        return_tensors="pt",
        padding=True,
    )["input_ids"].to(device)


def codec_frame_embed(model: torch.nn.Module, codes: torch.Tensor, tts_pad_embed: torch.Tensor) -> torch.Tensor:
    pieces = [model.talker.get_input_embeddings()(codes[:, 0])]
    for codebook_index in range(1, 16):
        pieces.append(model.talker.code_predictor.get_input_embeddings()[codebook_index - 1](codes[:, codebook_index]))
    return tts_pad_embed.expand(1, codes.shape[0], -1) + torch.stack(pieces, dim=0).sum(dim=0).unsqueeze(0)


def custom_prefix(model: torch.nn.Module, input_id: torch.Tensor, speaker: str, language: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cfg = model.config
    talk = cfg.talker_config
    device = input_id.device
    speaker_key = str(speaker).lower()
    speaker_embed = model.talker.get_input_embeddings()(
        torch.tensor(talk.spk_id[speaker_key], device=device, dtype=input_id.dtype)
    )
    language_key = str(language or "Auto").lower()
    if language_key == "auto":
        codec_prefill = [[talk.codec_nothink_id, talk.codec_think_bos_id, talk.codec_think_eos_id]]
    else:
        codec_prefill = [[talk.codec_think_id, talk.codec_think_bos_id, talk.codec_language_id[language_key], talk.codec_think_eos_id]]
    tts_bos_embed, tts_eos_embed, tts_pad_embed = model.talker.text_projection(
        model.talker.get_text_embeddings()(
            torch.tensor(
                [[cfg.tts_bos_token_id, cfg.tts_eos_token_id, cfg.tts_pad_token_id]],
                device=device,
                dtype=input_id.dtype,
            )
        )
    ).chunk(3, dim=1)
    codec_input_0 = model.talker.get_input_embeddings()(torch.tensor(codec_prefill, device=device, dtype=input_id.dtype))
    codec_input_1 = model.talker.get_input_embeddings()(
        torch.tensor([[talk.codec_pad_id, talk.codec_bos_id]], device=device, dtype=input_id.dtype)
    )
    codec_input = torch.cat([codec_input_0, speaker_embed.view(1, 1, -1), codec_input_1], dim=1)
    role = model.talker.text_projection(model.talker.get_text_embeddings()(input_id[:, :3]))
    prefix = torch.cat((tts_pad_embed.expand(-1, codec_input.shape[1] - 2, -1), tts_bos_embed), dim=1) + codec_input[:, :-1]
    return torch.cat([role, prefix], dim=1), tts_eos_embed, tts_pad_embed


def build_icl_batch(
    *,
    model: torch.nn.Module,
    processor: Any,
    row: dict[str, Any],
    target_segment_index: int,
    speaker: str,
    language: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    device = next(model.parameters()).device
    talk = model.config.talker_config
    segments = row.get("segments") or []
    if target_segment_index <= 0 or target_segment_index >= len(segments):
        raise ValueError(f"{row.get('sample_id')}: ICL layout needs history and target")
    history = segments[target_segment_index - 1]
    target = segments[target_segment_index]
    combined_text = str(history["text"]) + str(target["text"])
    ids = assistant_ids(processor, combined_text, device)
    prefix, tts_eos_embed, tts_pad_embed = custom_prefix(model, ids, speaker, language)

    text_embed = torch.cat(
        (model.talker.text_projection(model.talker.get_text_embeddings()(ids[:, 3:-5])), tts_eos_embed),
        dim=1,
    )
    codec_pad = model.talker.get_input_embeddings()(
        torch.tensor([[talk.codec_pad_id] * text_embed.shape[1]], device=device, dtype=ids.dtype)
    )
    codec_bos = tts_pad_embed + model.talker.get_input_embeddings()(
        torch.tensor([[talk.codec_bos_id]], device=device, dtype=ids.dtype)
    )

    history_codes = torch.tensor(history["codes"], dtype=torch.long, device=device)
    target_codes = torch.tensor(target["codes"], dtype=torch.long, device=device)
    target_codec_start = prefix.shape[1] + text_embed.shape[1] + 1 + history_codes.shape[0]
    pieces = [
        prefix,
        text_embed + codec_pad,
        codec_bos,
        codec_frame_embed(model, history_codes, tts_pad_embed),
        codec_frame_embed(model, target_codes, tts_pad_embed),
        tts_pad_embed + model.talker.get_input_embeddings()(
            torch.tensor([[talk.codec_eos_token_id]], device=device, dtype=ids.dtype)
        ),
    ]
    inputs_embeds = torch.cat(pieces, dim=1)
    labels = torch.full((1, inputs_embeds.shape[1]), -100, dtype=torch.long, device=device)
    target_codec_end = target_codec_start + target_codes.shape[0]
    boundary = target_codec_end
    labels[:, target_codec_start:target_codec_end] = target_codes[:, 0]
    labels[:, boundary] = talk.codec_eos_token_id
    codec_ids = torch.zeros((1, inputs_embeds.shape[1], 16), dtype=torch.long, device=device)
    codec_ids[:, target_codec_start:target_codec_end, :] = target_codes
    layout = {
        "sample_id": row.get("sample_id"),
        "history_text": history["text"],
        "target_text": target["text"],
        "target_chars": normalized_chars(target["text"]),
        "target_codec_start": int(target_codec_start),
        "target_codec_end": int(target_codec_end),
        "boundary": int(boundary),
        "target_frames": int(target_codes.shape[0]),
        "sequence_length": int(inputs_embeds.shape[1]),
    }
    return inputs_embeds, labels, codec_ids, layout


def icl_nll(
    *,
    model: torch.nn.Module,
    processor: Any,
    row: dict[str, Any],
    target_segment_index: int,
    speaker: str,
    language: str,
) -> dict[str, Any]:
    model.eval()
    inputs_embeds, labels, _, layout = build_icl_batch(
        model=model,
        processor=processor,
        row=row,
        target_segment_index=target_segment_index,
        speaker=speaker,
        language=language,
    )
    attention_mask = torch.ones(inputs_embeds.shape[:2], device=inputs_embeds.device, dtype=torch.long)
    outputs = model.talker(
        inputs_embeds=inputs_embeds[:, :-1, :],
        attention_mask=attention_mask[:, :-1],
        labels=labels[:, 1:],
        output_hidden_states=False,
    )
    shifted = labels[:, 1:]
    logits = outputs.logits
    positions = torch.arange(1, labels.shape[1], device=labels.device).unsqueeze(0)
    mask = (
        (positions >= layout["target_codec_start"])
        & (positions <= layout["boundary"])
        & (shifted != -100)
    )
    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        shifted.reshape(-1),
        reduction="none",
        ignore_index=-100,
    ).reshape_as(shifted)[mask]
    return {
        "sample_id": str(row.get("sample_id")),
        "nll_mean": round(float(loss.mean().detach().cpu()), 6),
        "target_text": layout["target_text"],
        "target_chars": layout["target_chars"],
        "target_frames": layout["target_frames"],
        "layout": layout,
    }


@torch.inference_mode()
def evaluate_gap(
    *,
    model: torch.nn.Module,
    rows: list[dict[str, Any]],
    processor: Any,
    special_ids: Any,
    target_segment_index: int,
    repo_root: Path,
    speaker: str,
    language: str,
) -> dict[str, Any]:
    items = []
    for row in rows:
        ref_mels = load_ref_mels(first_ref_audio(row, repo_root=repo_root))
        single_row = row_for_single_segment(row, target_segment_index)
        from scripts.python.build_c4_continuation_batch import build_continuation_batch

        single_batch, single_layouts = build_continuation_batch([single_row], tokenizer=processor, special_ids=special_ids, max_segments=1)
        single = codec0_nll_for_segment(
            model=model,
            batch=single_batch,
            layout=single_layouts[0],
            segment_index=0,
            ref_mels=ref_mels,
            speaker=speaker,
        )
        cont = icl_nll(
            model=model,
            processor=processor,
            row=row,
            target_segment_index=target_segment_index,
            speaker=speaker,
            language=language,
        )
        gap = float(cont["nll_mean"]) - float(single["nll_mean"])
        items.append(
            {
                "sample_id": str(row.get("sample_id")),
                "single_nll": single["nll_mean"],
                "icl_nll": cont["nll_mean"],
                "gap": round(gap, 6),
                "target_text": cont["target_text"],
            }
        )
    singles = [float(x["single_nll"]) for x in items]
    conts = [float(x["icl_nll"]) for x in items]
    gaps = [float(x["gap"]) for x in items]
    return {
        "n": len(items),
        "single_nll_mean": round(sum(singles) / len(singles), 6),
        "continuation_nll_mean": round(sum(conts) / len(conts), 6),
        "gap_mean": round(sum(gaps) / len(gaps), 6),
        "gap_negative_count": sum(1 for x in gaps if x <= 0),
        "items": items,
    }


def train_loss(
    *,
    model: torch.nn.Module,
    processor: Any,
    row: dict[str, Any],
    target_segment_index: int,
    speaker: str,
    language: str,
) -> torch.Tensor:
    model.train()
    inputs_embeds, labels, _, _ = build_icl_batch(
        model=model,
        processor=processor,
        row=row,
        target_segment_index=target_segment_index,
        speaker=speaker,
        language=language,
    )
    attention_mask = torch.ones(inputs_embeds.shape[:2], device=inputs_embeds.device, dtype=torch.long)
    outputs = model.talker(
        inputs_embeds=inputs_embeds[:, :-1, :],
        attention_mask=attention_mask[:, :-1],
        labels=labels[:, 1:],
        output_hidden_states=False,
    )
    return outputs.loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest-jsonl", type=Path, required=True)
    parser.add_argument("--eval-manifest-jsonl", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--speaker", default="001")
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--target-segment-index", type=int, default=1)
    parser.add_argument("--train-limit", type=int, default=200)
    parser.add_argument("--eval-limit", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--adapter-out", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_rows = read_jsonl(args.train_manifest_jsonl)[: args.train_limit]
    eval_rows = read_jsonl(args.eval_manifest_jsonl)[: args.eval_limit]
    train_ids = {str(row.get("sample_id")) for row in train_rows}
    eval_ids = {str(row.get("sample_id")) for row in eval_rows}
    overlap = sorted(train_ids & eval_ids)
    if overlap:
        raise ValueError(f"train/eval overlap: {overlap[:5]}")

    qwen3tts = Qwen3TTSModel.from_pretrained(str(args.model_dir), device_map="cuda:0", dtype=torch.bfloat16)
    model = get_peft_model(
        qwen3tts.model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=args.lora_dropout,
            bias="none",
        ),
    )
    special_ids = special_ids_from_model_config(model.config)
    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=0.01)

    before = evaluate_gap(
        model=model,
        rows=eval_rows,
        processor=qwen3tts.processor,
        special_ids=special_ids,
        target_segment_index=args.target_segment_index,
        repo_root=args.repo_root,
        speaker=args.speaker,
        language=args.language,
    )
    losses = []
    order = list(train_rows)
    random.shuffle(order)
    for step, row in enumerate(order[: args.max_steps], start=1):
        optimizer.zero_grad(set_to_none=True)
        loss = train_loss(
            model=model,
            processor=qwen3tts.processor,
            row=row,
            target_segment_index=args.target_segment_index,
            speaker=args.speaker,
            language=args.language,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if step == 1 or step % 25 == 0 or step == args.max_steps:
            print(f"[icl-train] step={step}/{args.max_steps} loss={losses[-1]:.6f}", flush=True)

    after = evaluate_gap(
        model=model,
        rows=eval_rows,
        processor=qwen3tts.processor,
        special_ids=special_ids,
        target_segment_index=args.target_segment_index,
        repo_root=args.repo_root,
        speaker=args.speaker,
        language=args.language,
    )
    before_gap = float(before["gap_mean"])
    after_gap = float(after["gap_mean"])
    closure = (before_gap - after_gap) / before_gap if before_gap else 0.0
    args.adapter_out.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.adapter_out))
    summary = {
        "layout": "icl_text12_codes1_target_codes2",
        "model_dir": str(args.model_dir),
        "train_manifest_jsonl": str(args.train_manifest_jsonl),
        "eval_manifest_jsonl": str(args.eval_manifest_jsonl),
        "adapter_out": str(args.adapter_out),
        "speaker": args.speaker,
        "language": args.language,
        "target_segment_index": args.target_segment_index,
        "train_samples": len(train_rows),
        "eval_samples": len(eval_rows),
        "actual_steps": len(losses),
        "lr": args.lr,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "loss_first": round(losses[0], 6) if losses else None,
        "loss_last": round(losses[-1], 6) if losses else None,
        "loss_mean": round(sum(losses) / len(losses), 6) if losses else None,
        "before": before,
        "after": after,
        "gap_closure_rate": round(closure, 6),
        "single_degradation": round(float(after["single_nll_mean"]) - float(before["single_nll_mean"]), 6),
    }
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in {"before", "after"}}, ensure_ascii=False, indent=2))
    print(f"Saved: {args.output_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

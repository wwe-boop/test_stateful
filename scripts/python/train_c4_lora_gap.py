#!/usr/bin/env python3
"""Train a small C4 LoRA adapter and report frozen-eval NLL gap closure."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qwen_tts import Qwen3TTSModel  # noqa: E402
from scripts.python.build_c4_continuation_batch import build_continuation_batch, read_jsonl  # noqa: E402
from scripts.python.run_c4_forward_smoke import first_ref_audio, load_ref_mels, special_ids_from_model_config  # noqa: E402
from scripts.python.run_c4_teacher_forcing_nll import codec0_nll_for_segment, row_for_single_segment  # noqa: E402
from scripts.python.train_c4_continuation_smoke import restrict_loss_to_segment, train_step  # noqa: E402


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


@torch.inference_mode()
def evaluate_gap(
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
        cont_batch, cont_layouts = build_continuation_batch(
            [row], tokenizer=processor, special_ids=special_ids, max_segments=target_segment_index + 1
        )
        continuation = codec0_nll_for_segment(
            model=model,
            batch=cont_batch,
            layout=cont_layouts[0],
            segment_index=target_segment_index,
            ref_mels=ref_mels,
        )
        single_row = row_for_single_segment(row, target_segment_index)
        single_batch, single_layouts = build_continuation_batch(
            [single_row], tokenizer=processor, special_ids=special_ids, max_segments=1
        )
        single = codec0_nll_for_segment(
            model=model,
            batch=single_batch,
            layout=single_layouts[0],
            segment_index=0,
            ref_mels=ref_mels,
        )
        items.append(
            {
                "sample_id": str(row.get("sample_id")),
                "single_nll": single["nll_mean"],
                "continuation_nll": continuation["nll_mean"],
                "gap": round(float(continuation["nll_mean"]) - float(single["nll_mean"]), 6),
                "target_text": continuation["target_text"],
                "target_chars": continuation["target_chars"],
            }
        )
    singles = [float(item["single_nll"]) for item in items]
    conts = [float(item["continuation_nll"]) for item in items]
    gaps = [float(item["gap"]) for item in items]
    return {
        "n": len(items),
        "single_nll_mean": round(mean(singles), 6) if singles else None,
        "continuation_nll_mean": round(mean(conts), 6) if conts else None,
        "gap_mean": round(mean(gaps), 6) if gaps else None,
        "gap_min": round(min(gaps), 6) if gaps else None,
        "gap_max": round(max(gaps), 6) if gaps else None,
        "gap_negative_count": sum(1 for gap in gaps if gap <= 0),
        "items": items,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest-jsonl", type=Path, required=True)
    parser.add_argument("--eval-manifest-jsonl", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--device-map", default="cuda:1")
    parser.add_argument("--target-segment-index", type=int, default=2)
    parser.add_argument("--train-limit", type=int, default=200)
    parser.add_argument("--eval-limit", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--loss-scope", choices=["all", "target"], default="target")
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", default="q_proj,v_proj")
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
    if not train_rows or not eval_rows:
        raise ValueError("empty train/eval rows")
    train_ids = {str(row.get("sample_id")) for row in train_rows}
    eval_ids = {str(row.get("sample_id")) for row in eval_rows}
    overlap = sorted(train_ids & eval_ids)
    if overlap:
        raise ValueError(f"train/eval overlap: {overlap[:5]}")

    qwen3tts = Qwen3TTSModel.from_pretrained(str(args.model_dir), device_map=args.device_map, dtype=torch.bfloat16)
    base_model = qwen3tts.model
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[item.strip() for item in args.target_modules.split(",") if item.strip()],
        lora_dropout=args.lora_dropout,
        bias="none",
    )
    model = get_peft_model(base_model, lora_config)
    special_ids = special_ids_from_model_config(model.config)
    device = next(model.parameters()).device
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=0.01)

    before = evaluate_gap(
        model=model,
        rows=eval_rows,
        processor=qwen3tts.processor,
        special_ids=special_ids,
        target_segment_index=args.target_segment_index,
        repo_root=args.repo_root,
    )

    losses = []
    model.train()
    for epoch in range(args.epochs):
        order = list(train_rows)
        random.shuffle(order)
        for step, row in enumerate(order, start=1):
            ref_mels = load_ref_mels(first_ref_audio(row, repo_root=args.repo_root))
            batch, layouts = build_continuation_batch(
                [row], tokenizer=qwen3tts.processor, special_ids=special_ids, max_segments=args.target_segment_index + 1
            )
            if args.loss_scope == "target":
                batch = restrict_loss_to_segment(batch, layouts[0], args.target_segment_index)
            optimizer.zero_grad(set_to_none=True)
            loss = train_step(model=model, batch=batch, ref_mels=ref_mels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
            optimizer.step()
            loss_value = float(loss.detach().cpu())
            losses.append(loss_value)
            if step == 1 or step % 25 == 0 or step == len(order):
                print(f"[lora-train] epoch={epoch} step={step}/{len(order)} loss={loss_value:.6f}", flush=True)

    after = evaluate_gap(
        model=model,
        rows=eval_rows,
        processor=qwen3tts.processor,
        special_ids=special_ids,
        target_segment_index=args.target_segment_index,
        repo_root=args.repo_root,
    )
    before_gap = float(before["gap_mean"])
    after_gap = float(after["gap_mean"])
    closure = (before_gap - after_gap) / before_gap if before_gap else 0.0

    args.adapter_out.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.adapter_out))
    summary = {
        "model_dir": str(args.model_dir),
        "train_manifest_jsonl": str(args.train_manifest_jsonl),
        "eval_manifest_jsonl": str(args.eval_manifest_jsonl),
        "adapter_out": str(args.adapter_out),
        "device": str(device),
        "target_segment_index": args.target_segment_index,
        "train_samples": len(train_rows),
        "eval_samples": len(eval_rows),
        "epochs": args.epochs,
        "lr": args.lr,
        "loss_scope": args.loss_scope,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "target_modules": [item.strip() for item in args.target_modules.split(",") if item.strip()],
        "trainable_params": int(trainable_params),
        "total_params": int(total_params),
        "trainable_percent": round(100.0 * trainable_params / max(total_params, 1), 6),
        "loss_first": round(losses[0], 6) if losses else None,
        "loss_last": round(losses[-1], 6) if losses else None,
        "loss_mean": round(sum(losses) / len(losses), 6) if losses else None,
        "before": before,
        "after": after,
        "gap_closure_rate": round(closure, 6),
    }
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in {"before", "after"}}, ensure_ascii=False, indent=2))
    print(f"Saved: {args.output_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

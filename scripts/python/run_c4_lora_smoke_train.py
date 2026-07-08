#!/usr/bin/env python3
"""Run a tiny package-free LoRA smoke train on a C4 continuation batch."""

from __future__ import annotations

import argparse
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

from qwen_tts import Qwen3TTSModel
from scripts.python.build_c4_continuation_batch import build_continuation_batch, read_jsonl
from scripts.python.run_c4_forward_smoke import (
    first_ref_audio,
    load_ref_mels,
    special_ids_from_model_config,
)
from scripts.python.run_c4_train_step_smoke import build_embeddings_and_losses


DEFAULT_TARGET_SUFFIXES = (
    "self_attn.q_proj",
    "self_attn.v_proj",
)


class LoRALinear(torch.nn.Module):
    def __init__(self, base: torch.nn.Linear, *, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be > 0")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = float(alpha) / float(rank)
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity()
        for param in self.base.parameters():
            param.requires_grad_(False)

        self.lora_A = torch.nn.Parameter(
            torch.empty(
                self.rank,
                base.in_features,
                device=base.weight.device,
                dtype=base.weight.dtype,
            )
        )
        self.lora_B = torch.nn.Parameter(
            torch.zeros(
                base.out_features,
                self.rank,
                device=base.weight.device,
                dtype=base.weight.dtype,
            )
        )
        torch.nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_hidden = F.linear(self.dropout(x), self.lora_A)
        lora_out = F.linear(lora_hidden, self.lora_B)
        return base_out + lora_out * self.scaling


def parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _get_parent_module(root: torch.nn.Module, module_name: str) -> tuple[torch.nn.Module, str]:
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def inject_lora(
    model: torch.nn.Module,
    *,
    target_suffixes: tuple[str, ...],
    rank: int,
    alpha: float,
    dropout: float,
) -> dict[str, Any]:
    for param in model.parameters():
        param.requires_grad_(False)

    matched: list[str] = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, torch.nn.Linear):
            continue
        if not any(name.endswith(suffix) for suffix in target_suffixes):
            continue
        parent, child_name = _get_parent_module(model, name)
        setattr(parent, child_name, LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout))
        matched.append(name)

    if not matched:
        raise ValueError(f"no linear modules matched suffixes: {target_suffixes}")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    return {
        "target_suffixes": list(target_suffixes),
        "lora_rank": int(rank),
        "lora_alpha": float(alpha),
        "lora_dropout": float(dropout),
        "matched_modules": matched,
        "matched_module_count": len(matched),
        "trainable_params": int(trainable_params),
        "total_params": int(total_params),
        "trainable_ratio": round(float(trainable_params) / float(total_params), 8),
    }


def lora_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: param.detach().cpu()
        for name, param in model.named_parameters()
        if ".lora_A" in name or ".lora_B" in name
    }


def prepare_cycle_items(
    *,
    rows: list[dict[str, Any]],
    qwen3tts: Qwen3TTSModel,
    model: torch.nn.Module,
    max_segments: int | None,
    repo_root: Path,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        batch, layouts = build_continuation_batch(
            [row],
            tokenizer=qwen3tts.processor,
            special_ids=special_ids_from_model_config(model.config),
            max_segments=max_segments,
        )
        items.append(
            {
                "row_index": row_index,
                "sample_id": str(row.get("sample_id") or f"sample_{row_index}"),
                "batch": batch,
                "layouts": layouts,
                "ref_mels": load_ref_mels(first_ref_audio(row, repo_root=repo_root)),
            }
        )
    return items


def run_lora_smoke(args: argparse.Namespace) -> dict[str, Any]:
    rows = read_jsonl(args.manifest_jsonl)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("manifest contains no rows after applying --limit")

    qwen3tts = Qwen3TTSModel.from_pretrained(
        str(args.model_dir),
        device_map=args.device_map,
        dtype=torch.bfloat16,
    )
    model = qwen3tts.model
    lora_info = inject_lora(
        model,
        target_suffixes=parse_csv(args.target_suffixes),
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
    )
    model.train()

    cycle_items = prepare_cycle_items(
        rows=rows,
        qwen3tts=qwen3tts,
        model=model,
        max_segments=args.max_segments,
        repo_root=args.repo_root,
    )

    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    loss_history: list[dict[str, Any]] = []
    for step in range(args.steps):
        item = cycle_items[step % len(cycle_items)]
        optimizer.zero_grad(set_to_none=True)
        talker_loss, sub_talker_loss, combined_loss = build_embeddings_and_losses(
            model=model,
            batch=item["batch"],
            ref_mels=item["ref_mels"],
        )
        combined_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [param for param in model.parameters() if param.requires_grad],
            args.max_grad_norm,
        )
        optimizer.step()
        loss_history.append(
            {
                "step": float(step),
                "row_index": float(item["row_index"]),
                "sample_id": item["sample_id"],
                "talker_loss": round(float(talker_loss.detach().cpu()), 6),
                "sub_talker_loss": round(float(sub_talker_loss.detach().cpu()), 6),
                "combined_loss": round(float(combined_loss.detach().cpu()), 6),
                "grad_norm": round(float(grad_norm.detach().cpu()), 6),
            }
        )

    if args.save_adapter:
        args.save_adapter.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "metadata": {
                    "model_dir": str(args.model_dir),
                    "target_suffixes": list(parse_csv(args.target_suffixes)),
                    "rank": args.rank,
                    "alpha": args.alpha,
                    "dropout": args.dropout,
                    "steps": args.steps,
                    "lr": args.lr,
                    "sample_strategy": "cycle_rows_batch_size_1",
                    "manifest_rows": len(rows),
                },
                "state_dict": lora_state_dict(model),
            },
            args.save_adapter,
        )

    device = next(model.parameters()).device
    all_layouts = [
        layout
        for item in cycle_items
        for layout in item["layouts"]
    ]
    sample_batches = [
        {
            "sample_id": item["sample_id"],
            "batch_shape": list(item["batch"]["input_ids"].shape),
            "codec_frames": int(item["batch"]["codec_mask"].sum().item()),
            "loss_positions": int((item["batch"]["codec_0_labels"] != -100).sum().item()),
        }
        for item in cycle_items
    ]
    return {
        "model_dir": str(args.model_dir),
        "manifest_jsonl": str(args.manifest_jsonl),
        "device": str(device),
        "steps": args.steps,
        "lr": args.lr,
        "sample_strategy": "cycle_rows_batch_size_1",
        "manifest_rows": len(rows),
        **lora_info,
        "batch_shape": sample_batches[0]["batch_shape"] if len(sample_batches) == 1 else None,
        "sample_batches": sample_batches,
        "codec_frames": sum(item["codec_frames"] for item in sample_batches),
        "loss_positions": sum(item["loss_positions"] for item in sample_batches),
        "loss_history": loss_history,
        "saved_adapter": str(args.save_adapter) if args.save_adapter else None,
        "layouts": all_layouts,
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
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--target-suffixes", default=",".join(DEFAULT_TARGET_SUFFIXES))
    parser.add_argument("--save-adapter", type=Path)
    parser.add_argument("--output-summary", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_lora_smoke(args)
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.output_summary:
        args.output_summary.parent.mkdir(parents=True, exist_ok=True)
        args.output_summary.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

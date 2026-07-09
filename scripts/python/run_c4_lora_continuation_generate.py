#!/usr/bin/env python3
"""Generate C4 single/continuation wavs with an optional PEFT LoRA adapter."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import soundfile as sf
import torch
from peft import PeftModel

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qwen_tts import Qwen3TTSModel  # noqa: E402
from scripts.python.build_c4_continuation_batch import build_continuation_batch, read_jsonl  # noqa: E402
from scripts.python.run_c4_forward_smoke import first_ref_audio, load_ref_mels, special_ids_from_model_config  # noqa: E402
from scripts.python.run_c4_teacher_forcing_nll import row_for_single_segment  # noqa: E402


def build_embeddings(
    *,
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    ref_mels: torch.Tensor,
    prefill_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    with torch.no_grad():
        speaker_embedding = model.speaker_encoder(ref_mels.to(device=device, dtype=dtype)).detach()

    input_ids = batch["input_ids"][:, :prefill_len, :].to(device)
    codec_ids = batch["codec_ids"][:, :prefill_len, :].to(device)
    text_embedding_mask = batch["text_embedding_mask"][:, :prefill_len].to(device).unsqueeze(-1)
    codec_embedding_mask = batch["codec_embedding_mask"][:, :prefill_len].to(device).unsqueeze(-1)
    codec_mask = batch["codec_mask"][:, :prefill_len].to(device)

    input_text_ids = input_ids[:, :, 0]
    input_codec_ids = input_ids[:, :, 1]
    input_codec_embedding = model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
    input_text_embedding = model.talker.model.text_embedding(input_text_ids)
    if input_text_embedding.shape[-1] != input_codec_embedding.shape[-1]:
        input_text_embedding = model.talker.text_projection(input_text_embedding)
    input_text_embedding = input_text_embedding * text_embedding_mask
    if prefill_len > 6:
        input_codec_embedding[:, 6, :] = speaker_embedding
    input_embeddings = input_text_embedding + input_codec_embedding

    for codebook_index in range(1, 16):
        codec_i_embedding = model.talker.code_predictor.get_input_embeddings()[codebook_index - 1](
            codec_ids[:, :, codebook_index]
        )
        codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
        input_embeddings = input_embeddings + codec_i_embedding

    tts_pad_id = int(model.config.tts_pad_token_id)
    tts_pad_embed = model.talker.model.text_embedding(torch.tensor([[tts_pad_id]], device=device, dtype=torch.long))
    if tts_pad_embed.shape[-1] != input_embeddings.shape[-1]:
        tts_pad_embed = model.talker.text_projection(tts_pad_embed)
    return input_embeddings, tts_pad_embed.to(device=device, dtype=dtype)


def build_generation_batch(
    *,
    row: dict[str, Any],
    processor: Any,
    special_ids: Any,
    args: argparse.Namespace,
) -> tuple[dict[str, torch.Tensor], dict[str, Any], int, str]:
    if args.generation_mode == "single":
        build_row = row_for_single_segment(row, args.target_segment_index)
        target_index = 0
        max_segments = 1
    else:
        build_row = row
        target_index = args.target_segment_index
        max_segments = args.target_segment_index + 1

    batch, layouts = build_continuation_batch(
        [build_row],
        tokenizer=processor,
        special_ids=special_ids,
        max_segments=max_segments,
    )
    return batch, layouts[0], target_index, str(row.get("sample_id") or "sample")


def summarize_generation_stop(result: Any, codes: torch.Tensor, eos_id: int, max_new_tokens: int) -> tuple[torch.Tensor, dict[str, Any]]:
    sequences = getattr(result, "sequences", None)
    sequence_tokens: list[int] = []
    if sequences is not None:
        sequence_tokens = [int(x) for x in sequences[0].detach().cpu().tolist()]

    sequence_eos_positions = [i for i, token in enumerate(sequence_tokens) if token == eos_id]
    sequence_eos_step = int(sequence_eos_positions[0]) if sequence_eos_positions else -1
    hidden_eos_positions = (codes[:, 0] == eos_id).nonzero(as_tuple=False)
    hidden_eos_step = int(hidden_eos_positions[0].item()) if hidden_eos_positions.numel() else -1

    if 0 <= sequence_eos_step <= int(codes.shape[0]):
        effective = codes[:sequence_eos_step]
        stop_reason = "sequence_eos"
    elif 0 <= hidden_eos_step < int(codes.shape[0]):
        effective = codes[:hidden_eos_step]
        stop_reason = "hidden_state_eos"
    else:
        effective = codes
        stop_reason = "max_new_tokens" if int(codes.shape[0]) >= max_new_tokens - 1 else "unknown_short"

    return effective, {
        "raw_generated_steps": int(codes.shape[0]),
        "sequence_len": len(sequence_tokens),
        "sequence_last_token": int(sequence_tokens[-1]) if sequence_tokens else None,
        "sequence_eos_step": sequence_eos_step,
        "legacy_hidden_state_eos_step": hidden_eos_step,
        "eos_step": sequence_eos_step,
        "stop_reason": stop_reason,
    }


@torch.inference_mode()
def generate_one(
    *,
    row: dict[str, Any],
    qwen3tts: Any,
    model: torch.nn.Module,
    special_ids: Any,
    args: argparse.Namespace,
    index: int,
) -> dict[str, Any]:
    batch, layout, build_target_index, source_sample_id = build_generation_batch(
        row=row,
        processor=qwen3tts.processor,
        special_ids=special_ids,
        args=args,
    )
    target = layout["segments"][build_target_index]
    prefill_len = int(target["codec_bos"]) + 1
    target_frames = int(target["codec_frames"])
    max_new_tokens = args.max_new_tokens or args.official_max_new_tokens

    ref_mels = load_ref_mels(first_ref_audio(row, repo_root=args.repo_root))
    input_embeddings, tts_pad_embed = build_embeddings(
        model=model,
        batch=batch,
        ref_mels=ref_mels,
        prefill_len=prefill_len,
    )
    device = next(model.parameters()).device
    attention_mask = torch.ones(input_embeddings.shape[:2], device=device, dtype=torch.long)
    trailing_text_hidden = tts_pad_embed.expand(1, max_new_tokens + 4, -1).contiguous()

    result = model.talker.generate(
        inputs_embeds=input_embeddings,
        attention_mask=attention_mask,
        trailing_text_hidden=trailing_text_hidden,
        tts_pad_embed=tts_pad_embed,
        max_new_tokens=max_new_tokens,
        min_new_tokens=2,
        do_sample=args.do_sample,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        subtalker_dosample=args.subtalker_dosample,
        subtalker_top_k=args.subtalker_top_k,
        subtalker_top_p=args.subtalker_top_p,
        subtalker_temperature=args.subtalker_temperature,
        eos_token_id=int(model.config.talker_config.codec_eos_token_id),
        repetition_penalty=args.repetition_penalty,
        suppress_tokens=[
            i
            for i in range(int(model.config.talker_config.vocab_size) - 1024, int(model.config.talker_config.vocab_size))
            if i != int(model.config.talker_config.codec_eos_token_id)
        ],
        output_hidden_states=True,
        return_dict_in_generate=True,
    )
    code_steps = [hid[-1] for hid in result.hidden_states if hid[-1] is not None]
    if not code_steps:
        raise RuntimeError(f"{layout['sample_id']}: no codec steps")
    codes = torch.stack(code_steps, dim=1)[0]
    eos_id = int(model.config.talker_config.codec_eos_token_id)
    effective, stop_info = summarize_generation_stop(result, codes, eos_id, max_new_tokens)

    sample_dir = args.output_dir / f"{index:03d}_{args.generation_mode}_{source_sample_id}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    codes_path = sample_dir / "generated_codes.pt"
    wav_path = sample_dir / "hf_continuation.wav"
    torch.save(effective.detach().cpu(), codes_path)
    wavs, sample_rate = model.speech_tokenizer.decode([{"audio_codes": effective}])
    sf.write(str(wav_path), wavs[0], sample_rate)

    reference = str((row.get("segments") or [])[args.target_segment_index].get("text") or "")
    item = {
        "key": f"{args.variant}:{args.generation_mode}:{source_sample_id}",
        "sample_id": source_sample_id,
        "variant": args.variant,
        "generation_mode": args.generation_mode,
        "seed": int(args.seed),
        "wav_path": str(wav_path),
        "reference": reference,
        "target_segment_index": args.target_segment_index,
        "build_target_index": build_target_index,
        "prefill_len": prefill_len,
        "target_frames": target_frames,
        "max_new_tokens": max_new_tokens,
        "generated_frames": int(effective.shape[0]),
        **stop_info,
        "layout": layout,
    }
    (sample_dir / "summary.json").write_text(json.dumps(item, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-jsonl", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--device-map", default="cuda:1")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--target-segment-index", type=int, default=2)
    parser.add_argument("--generation-mode", choices=("continuation", "single"), default="continuation")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--official-max-new-tokens", type=int, default=4096)
    parser.add_argument("--extra-tokens", type=int, default=32, help="Deprecated; kept for old command compatibility.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--subtalker-dosample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--subtalker-top-k", type=int, default=50)
    parser.add_argument("--subtalker-top-p", type=float, default=1.0)
    parser.add_argument("--subtalker-temperature", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--variant", default="hf_c4_lora_continuation")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rows = read_jsonl(args.manifest_jsonl)[: args.limit]
    qwen3tts = Qwen3TTSModel.from_pretrained(str(args.model_dir), device_map=args.device_map, dtype=torch.bfloat16)
    model = qwen3tts.model
    if args.adapter_dir:
        model = PeftModel.from_pretrained(model, str(args.adapter_dir)).eval()
        qwen3tts.model = model
    model.eval()
    special_ids = special_ids_from_model_config(model.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    items = []
    for index, row in enumerate(rows):
        item = generate_one(row=row, qwen3tts=qwen3tts, model=model, special_ids=special_ids, args=args, index=index)
        items.append(item)
        print(
            f"[generated] {index + 1}/{len(rows)} {item['generation_mode']} {item['sample_id']} "
            f"frames={item['generated_frames']} raw={item['raw_generated_steps']} "
            f"stop={item['stop_reason']} seq_eos={item['sequence_eos_step']}",
            flush=True,
        )
    manifest = {"rows": [{k: v for k, v in item.items() if k != "layout"} for item in items]}
    (args.output_dir / "asr_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "model_dir": str(args.model_dir),
        "adapter_dir": str(args.adapter_dir) if args.adapter_dir else None,
        "manifest_jsonl": str(args.manifest_jsonl),
        "variant": args.variant,
        "limit": args.limit,
        "target_segment_index": args.target_segment_index,
        "generation_mode": args.generation_mode,
        "max_new_tokens": args.max_new_tokens,
        "official_max_new_tokens": args.official_max_new_tokens,
        "sampling": {
            "do_sample": args.do_sample,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "temperature": args.temperature,
            "subtalker_dosample": args.subtalker_dosample,
            "subtalker_top_k": args.subtalker_top_k,
            "subtalker_top_p": args.subtalker_top_p,
            "subtalker_temperature": args.subtalker_temperature,
            "repetition_penalty": args.repetition_penalty,
        },
        "items": items,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "items"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

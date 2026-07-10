#!/usr/bin/env python3
"""Generate C4 single/continuation audio with official CustomVoice prefill style."""

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

try:
    from qwen_tts import Qwen3TTSModel
except ImportError:
    qwen_root = REPO_ROOT.parent / "Qwen3-TTS"
    if qwen_root.exists():
        sys.path.insert(0, str(qwen_root))
    from qwen_tts import Qwen3TTSModel


ASSISTANT_TEMPLATE = "<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


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


def build_custom_prefix(
    *,
    model: torch.nn.Module,
    input_id: torch.Tensor,
    speaker: str,
    language: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cfg = model.config
    talk = cfg.talker_config
    device = input_id.device

    speaker_key = str(speaker).lower()
    if speaker_key not in talk.spk_id:
        raise ValueError(f"speaker {speaker!r} not in spk_id={talk.spk_id}")
    speaker_embed = model.talker.get_input_embeddings()(
        torch.tensor(talk.spk_id[speaker_key], device=device, dtype=input_id.dtype)
    )

    language_key = str(language or "Auto").lower()
    if language_key == "auto":
        language_id = None
    else:
        if language_key not in talk.codec_language_id:
            raise ValueError(f"language {language!r} not in codec_language_id")
        language_id = talk.codec_language_id[language_key]

    tts_bos_embed, tts_eos_embed, tts_pad_embed = model.talker.text_projection(
        model.talker.get_text_embeddings()(
            torch.tensor(
                [[cfg.tts_bos_token_id, cfg.tts_eos_token_id, cfg.tts_pad_token_id]],
                device=device,
                dtype=input_id.dtype,
            )
        )
    ).chunk(3, dim=1)

    if language_id is None:
        codec_prefill = [[talk.codec_nothink_id, talk.codec_think_bos_id, talk.codec_think_eos_id]]
    else:
        codec_prefill = [[talk.codec_think_id, talk.codec_think_bos_id, language_id, talk.codec_think_eos_id]]
    codec_input_0 = model.talker.get_input_embeddings()(
        torch.tensor(codec_prefill, device=device, dtype=input_id.dtype)
    )
    codec_input_1 = model.talker.get_input_embeddings()(
        torch.tensor([[talk.codec_pad_id, talk.codec_bos_id]], device=device, dtype=input_id.dtype)
    )
    codec_input = torch.cat([codec_input_0, speaker_embed.view(1, 1, -1), codec_input_1], dim=1)
    role_embed = model.talker.text_projection(model.talker.get_text_embeddings()(input_id[:, :3]))
    prefix_embed = torch.cat(
        (tts_pad_embed.expand(-1, codec_input.shape[1] - 2, -1), tts_bos_embed),
        dim=1,
    ) + codec_input[:, :-1]
    return torch.cat([role_embed, prefix_embed], dim=1), tts_eos_embed, tts_pad_embed


def build_prefill_embeddings(
    *,
    model: torch.nn.Module,
    processor: Any,
    row: dict[str, Any],
    target_segment_index: int,
    generation_mode: str,
    speaker: str,
    language: str,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    device = next(model.parameters()).device
    talk = model.config.talker_config
    all_segments = row.get("segments") or []
    if target_segment_index >= len(all_segments):
        raise ValueError(f"{row.get('sample_id')}: missing target segment {target_segment_index}")

    if generation_mode == "single":
        history_segments: list[dict[str, Any]] = []
        target_segment = all_segments[target_segment_index]
    else:
        history_segments = all_segments[:target_segment_index]
        target_segment = all_segments[target_segment_index]

    first_ids = assistant_ids(processor, str((history_segments or [target_segment])[0]["text"]), device)
    prefix, tts_eos_embed, tts_pad_embed = build_custom_prefix(
        model=model,
        input_id=first_ids,
        speaker=speaker,
        language=language,
    )
    pieces = [prefix]
    layout: dict[str, Any] = {
        "sample_id": row.get("sample_id"),
        "generation_mode": generation_mode,
        "history_segments": [],
        "target_text": target_segment["text"],
    }

    def append_text(text: str) -> None:
        ids = assistant_ids(processor, text, device)
        text_embed = torch.cat(
            (model.talker.text_projection(model.talker.get_text_embeddings()(ids[:, 3:-5])), tts_eos_embed),
            dim=1,
        )
        codec_pad = model.talker.get_input_embeddings()(
            torch.tensor([[talk.codec_pad_id] * text_embed.shape[1]], device=device, dtype=ids.dtype)
        )
        pieces.append(text_embed + codec_pad)

    def append_codec_bos(dtype: torch.dtype) -> None:
        pieces.append(
            tts_pad_embed
            + model.talker.get_input_embeddings()(
                torch.tensor([[talk.codec_bos_id]], device=device, dtype=dtype)
            )
        )

    for history_index, segment in enumerate(history_segments):
        append_text(str(segment["text"]))
        append_codec_bos(first_ids.dtype)
        codes = torch.tensor(segment["codes"], dtype=torch.long, device=device)
        pieces.append(codec_frame_embed(model, codes, tts_pad_embed))
        pieces.append(
            tts_pad_embed
            + model.talker.get_input_embeddings()(
                torch.tensor([[talk.codec_eos_token_id]], device=device, dtype=first_ids.dtype)
            )
        )
        layout["history_segments"].append(
            {
                "segment_index": history_index,
                "text": segment["text"],
                "frames": int(codes.shape[0]),
            }
        )

    append_text(str(target_segment["text"]))
    append_codec_bos(first_ids.dtype)
    inputs_embeds = torch.cat(pieces, dim=1)
    layout.update(
        {
            "prefill_len": int(inputs_embeds.shape[1]),
            "target_segment_index": target_segment_index,
            "target_frames": len(target_segment.get("codes") or []),
        }
    )
    return inputs_embeds, tts_pad_embed, layout


def generate_one(
    *,
    model: torch.nn.Module,
    processor: Any,
    row: dict[str, Any],
    args: argparse.Namespace,
    index: int,
) -> dict[str, Any]:
    inputs_embeds, tts_pad_embed, layout = build_prefill_embeddings(
        model=model,
        processor=processor,
        row=row,
        target_segment_index=args.target_segment_index,
        generation_mode=args.generation_mode,
        speaker=args.speaker,
        language=args.language,
    )
    device = inputs_embeds.device
    talk = model.config.talker_config
    attention_mask = torch.ones(inputs_embeds.shape[:2], device=device, dtype=torch.long)
    result = model.talker.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        trailing_text_hidden=tts_pad_embed,
        tts_pad_embed=tts_pad_embed,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=2,
        do_sample=args.do_sample,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        subtalker_dosample=args.subtalker_dosample,
        subtalker_top_k=args.subtalker_top_k,
        subtalker_top_p=args.subtalker_top_p,
        subtalker_temperature=args.subtalker_temperature,
        eos_token_id=talk.codec_eos_token_id,
        repetition_penalty=args.repetition_penalty,
        suppress_tokens=[i for i in range(talk.vocab_size - 1024, talk.vocab_size) if i != talk.codec_eos_token_id],
        output_hidden_states=True,
        return_dict_in_generate=True,
    )
    codes = torch.stack([hid[-1] for hid in result.hidden_states if hid[-1] is not None], dim=1)[0]
    first_codebook = codes[:, 0]
    stop_positions = (first_codebook == talk.codec_eos_token_id).nonzero(as_tuple=False)
    eos_step = int(stop_positions[0].item()) if stop_positions.numel() else -1
    effective = codes[:eos_step] if eos_step >= 0 else codes
    stop_reason = "sequence_eos" if eos_step >= 0 else "max_new_tokens"

    sample_id = str(row["sample_id"])
    sample_dir = args.out_dir / f"{index:03d}_{args.generation_mode}_{sample_id}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    wav_path = sample_dir / "official_prefill.wav"
    wavs, sample_rate = model.speech_tokenizer.decode([{"audio_codes": effective}])
    sf.write(str(wav_path), wavs[0], sample_rate)
    reference = str(row["segments"][args.target_segment_index]["text"])
    item = {
        "key": f"{args.variant}:{args.generation_mode}:{sample_id}",
        "variant": args.variant,
        "generation_mode": args.generation_mode,
        "sample_id": sample_id,
        "speaker": args.speaker,
        "language": args.language,
        "seed": args.seed,
        "wav_path": str(wav_path),
        "reference": reference,
        "seconds": len(wavs[0]) / sample_rate,
        "sample_rate": sample_rate,
        "generated_frames": int(effective.shape[0]),
        "raw_generated_steps": int(codes.shape[0]),
        "eos_step": eos_step,
        "stop_reason": stop_reason,
        **layout,
    }
    (sample_dir / "summary.json").write_text(json.dumps(item, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-jsonl", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--generation-mode", choices=["single", "continuation"], default="continuation")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--target-segment-index", type=int, default=1)
    parser.add_argument("--speaker", default="001")
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--subtalker-dosample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--subtalker-top-k", type=int, default=50)
    parser.add_argument("--subtalker-top-p", type=float, default=1.0)
    parser.add_argument("--subtalker-temperature", type=float, default=0.9)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows = read_jsonl(args.manifest_jsonl)[: args.limit]
    qwen3tts = Qwen3TTSModel.from_pretrained(str(args.model_dir), device_map="cuda:0", dtype=torch.bfloat16)
    model = qwen3tts.model
    if args.adapter_dir:
        model = PeftModel.from_pretrained(model, str(args.adapter_dir)).eval()
        qwen3tts.model = model
    model.eval()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    items = []
    for index, row in enumerate(rows):
        item = generate_one(model=model, processor=qwen3tts.processor, row=row, args=args, index=index)
        items.append(item)
        print(json.dumps({k: v for k, v in item.items() if k != "history_segments"}, ensure_ascii=False), flush=True)

    payload = {
        "variant": args.variant,
        "model_dir": str(args.model_dir),
        "adapter_dir": str(args.adapter_dir) if args.adapter_dir else None,
        "manifest_jsonl": str(args.manifest_jsonl),
        "generation_mode": args.generation_mode,
        "items": items,
    }
    (args.out_dir / "asr_manifest.json").write_text(
        json.dumps({"rows": [{k: v for k, v in item.items() if k != "history_segments"} for item in items]}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

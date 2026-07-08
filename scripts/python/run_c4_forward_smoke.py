#!/usr/bin/env python3
"""Run a no-backward C4 continuation forward smoke test on a local Qwen3-TTS model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qwen_tts import Qwen3TTSModel
from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram
from scripts.python.build_c4_continuation_batch import (
    C4SpecialIds,
    build_continuation_batch,
    read_jsonl,
)


def special_ids_from_model_config(config: Any) -> C4SpecialIds:
    talker = config.talker_config
    return C4SpecialIds(
        tts_pad_token_id=int(config.tts_pad_token_id),
        tts_bos_token_id=int(config.tts_bos_token_id),
        tts_eos_token_id=int(config.tts_eos_token_id),
        codec_pad_id=int(talker.codec_pad_id),
        codec_bos_id=int(talker.codec_bos_id),
        codec_eos_token_id=int(talker.codec_eos_token_id),
        codec_nothink_id=int(talker.codec_nothink_id),
        codec_think_bos_id=int(talker.codec_think_bos_id),
        codec_think_eos_id=int(talker.codec_think_eos_id),
    )


def resolve_audio_path(path_value: str, *, repo_root: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return repo_root / path


def load_ref_mels(path: Path, *, sample_rate: int = 24000) -> torch.Tensor:
    audio, sr = librosa.load(path, sr=None, mono=True)
    if sr != sample_rate:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
    audio = np.asarray(audio, dtype=np.float32)
    return mel_spectrogram(
        torch.from_numpy(audio).unsqueeze(0),
        n_fft=1024,
        num_mels=128,
        sampling_rate=sample_rate,
        hop_size=256,
        win_size=1024,
        fmin=0,
        fmax=12000,
    ).transpose(1, 2)


def first_ref_audio(row: dict[str, Any], *, repo_root: Path) -> Path:
    segments = row.get("segments") or []
    if not segments:
        raise ValueError(f"{row.get('sample_id')}: missing segments")
    audio = segments[0].get("audio")
    if not isinstance(audio, str) or not audio:
        raise ValueError(f"{row.get('sample_id')}: first segment missing audio")
    return resolve_audio_path(audio, repo_root=repo_root)


@torch.inference_mode()
def run_forward_smoke(args: argparse.Namespace) -> dict[str, Any]:
    rows = read_jsonl(args.manifest_jsonl)
    if args.limit:
        rows = rows[: args.limit]
    if len(rows) != 1:
        raise ValueError("forward smoke currently expects exactly one manifest row")

    qwen3tts = Qwen3TTSModel.from_pretrained(
        str(args.model_dir),
        device_map=args.device_map,
        dtype=torch.bfloat16,
    )
    model = qwen3tts.model.eval()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    batch, layouts = build_continuation_batch(
        rows,
        tokenizer=qwen3tts.processor,
        special_ids=special_ids_from_model_config(model.config),
        max_segments=args.max_segments,
    )

    ref_mels = load_ref_mels(first_ref_audio(rows[0], repo_root=args.repo_root))
    ref_mels = ref_mels.to(device=device, dtype=dtype)
    speaker_embedding = model.speaker_encoder(ref_mels).detach()

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
    talker_hidden_states = hidden_states[codec_mask[:, 1:]]
    talker_codec_ids = codec_ids[codec_mask]
    _, sub_talker_loss = model.talker.forward_sub_talker_finetune(
        talker_codec_ids,
        talker_hidden_states,
    )
    loss = outputs.loss + 0.3 * sub_talker_loss

    return {
        "model_dir": str(args.model_dir),
        "manifest_jsonl": str(args.manifest_jsonl),
        "device": str(device),
        "dtype": str(dtype),
        "batch_shape": list(batch["input_ids"].shape),
        "codec_frames": int(batch["codec_mask"].sum().item()),
        "loss_positions": int((batch["codec_0_labels"] != -100).sum().item()),
        "talker_loss": round(float(outputs.loss.detach().cpu()), 6),
        "sub_talker_loss": round(float(sub_talker_loss.detach().cpu()), 6),
        "combined_loss": round(float(loss.detach().cpu()), 6),
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
    parser.add_argument("--output-summary", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_forward_smoke(args)
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.output_summary:
        args.output_summary.parent.mkdir(parents=True, exist_ok=True)
        args.output_summary.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

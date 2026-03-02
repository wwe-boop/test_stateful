#!/usr/bin/env python3
"""
[Phase 2, Item 12 — Part 1] Generate FP32 PyTorch reference for TRT E2E verification.

Runs full decode loop (Talker prefill + N steps of Talker→CP→codec_sum→Talker) in PyTorch
and saves reference data for comparison with container-side TRT pipeline (verify_e2e_trt.py).

Usage (host, conda activate qwen3-tts):
  python scripts/python/verify_e2e_trt_ref.py --variant design-1.7b --steps 50
  python scripts/python/verify_e2e_trt_ref.py --variant design-1.7b --steps 1000  # long sequence
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "export"))
sys.path.insert(0, str(REPO_ROOT / "third_party" / "Qwen3-TTS"))

from utils import (
    setup_logging,
    resolve_model_path,
    load_tts_model,
    resolve_device,
    DEFAULT_OUTPUT_DIR,
    has_model_weights,
)

logger = logging.getLogger("e2e_trt_ref")


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="E2E TRT reference generator")
    parser.add_argument("--variant", default="design-1.7b", help="Model variant")
    parser.add_argument("--steps", type=int, default=50, help="Decode steps (N)")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = resolve_device(args.device)
    variant = args.variant
    n_steps = args.steps

    path = resolve_model_path(variant)
    if not has_model_weights(path):
        logger.error(f"No weights for {variant}")
        sys.exit(1)

    weights_dir = DEFAULT_OUTPUT_DIR / variant / "weights"
    if not weights_dir.exists():
        logger.error(f"Exported weights not found: {weights_dir}. Run export_models.sh first.")
        sys.exit(1)
    cfg = json.load(open(weights_dir / "config.json"))
    hidden_size = int(cfg["talker_hidden_size"])
    vocab_size = int(cfg["talker_vocab_size"])
    codec_bos_id = int(cfg.get("codec_bos_id", 2149))

    logger.info(f"Loading {variant} on {device} ...")
    model = load_tts_model(path, device=device, dtype=torch.float32)
    tts_pad_token_id = getattr(model.config, "tts_pad_token_id", 0)
    talker = model.talker
    cp = talker.code_predictor

    # Prefill: same as verify_talker_trtllm_ref (fixed seed)
    n_text = 20
    torch.manual_seed(42)
    text_ids = torch.randint(100, 5000, (1, n_text), device=device)
    with torch.no_grad():
        text_embed = talker.text_projection(talker.model.text_embedding(text_ids))
        codec_bos_embed = talker.model.codec_embedding(
            torch.tensor([[codec_bos_id]], device=device))
        inputs_embeds = torch.cat([text_embed, codec_bos_embed], dim=1)
    seq_len = inputs_embeds.shape[1]
    logger.info(f"Prefill: seq_len={seq_len}, hidden={hidden_size}, vocab={vocab_size}")

    with torch.no_grad():
        out = talker.model(
            inputs_embeds=inputs_embeds,
            use_cache=True,
            return_dict=True,
        )
        prefill_hidden = out.last_hidden_state
        prefill_logits = talker.codec_head(prefill_hidden)
        past_kv = out.past_key_values
        past_hidden = prefill_hidden[:, -1:, :]
        codec_token_0 = prefill_logits[:, -1, :].argmax(dim=-1)
    logger.info(f"Initial codec_token_0: {codec_token_0.item()}")

    # Pad embed for next-input (same as verify_e2e)
    with torch.no_grad():
        pad_id = torch.tensor([[tts_pad_token_id]], device=device)
        pad_embed = talker.text_projection(talker.model.text_embedding(pad_id))

    step_past_hiddens = []
    step_talker_logits = []
    all_codec_tokens = []
    step_cp_tokens = []

    kv = past_kv
    current_past_hidden = past_hidden
    current_codec_token_0 = codec_token_0

    for step in range(n_steps):
        with torch.no_grad():
            # Code Predictor
            embed_0 = talker.model.codec_embedding(current_codec_token_0).unsqueeze(1)
            seq = cp.small_to_mtp_projection(
                torch.cat([current_past_hidden, embed_0], dim=1))
            pt_tokens = []
            for stage in range(len(cp.lm_head)):
                B, S, D = seq.shape
                pos_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
                pos_embs = cp.model.rotary_emb(seq, pos_ids)
                causal = torch.triu(
                    torch.full((S, S), float("-inf"), device=device, dtype=seq.dtype),
                    diagonal=1,
                ).unsqueeze(0).unsqueeze(0)
                h = seq
                for layer in cp.model.layers:
                    h = layer(
                        h,
                        attention_mask=causal,
                        position_ids=pos_ids,
                        past_key_values=None,
                        output_attentions=False,
                        use_cache=False,
                        cache_position=torch.arange(S, device=device),
                        position_embeddings=pos_embs,
                    )[0]
                h = cp.model.norm(h)
                logits = cp.lm_head[stage](h[:, -1:, :])
                token = logits.argmax(dim=-1).squeeze(-1)
                pt_tokens.append(token.item())
                if stage < len(cp.lm_head) - 1:
                    next_emb = cp.small_to_mtp_projection(
                        cp.model.codec_embedding[stage](token).unsqueeze(1)
                    )
                    seq = torch.cat([seq, next_emb], dim=1)

            full_codec = [current_codec_token_0.item()] + pt_tokens
            all_codec_tokens.append(full_codec)
            step_cp_tokens.append(pt_tokens)
            step_past_hiddens.append(current_past_hidden.cpu().float().numpy())

            # Next Talker input: codec_sum + pad_embed
            all_tokens = torch.tensor([full_codec], device=device, dtype=torch.long)
            codec_sum = torch.zeros(1, 1, hidden_size, device=device, dtype=current_past_hidden.dtype)
            for i in range(min(16, all_tokens.shape[1])):
                if i == 0:
                    codec_sum += talker.model.codec_embedding(all_tokens[:, i]).unsqueeze(1)
                elif i - 1 < len(cp.model.codec_embedding):
                    codec_sum += cp.model.codec_embedding[i - 1](
                        all_tokens[:, i]
                    ).unsqueeze(1)
            next_input = codec_sum + pad_embed

            step_out = talker.model(
                inputs_embeds=next_input,
                past_key_values=kv,
                use_cache=True,
                return_dict=True,
            )
            current_past_hidden = step_out.last_hidden_state
            kv = step_out.past_key_values
            step_logits = talker.codec_head(current_past_hidden)
            step_talker_logits.append(step_logits[:, -1, :].cpu().float().numpy())
            current_codec_token_0 = step_logits[:, -1, :].argmax(dim=-1)

        if (step + 1) % 10 == 0 or step == 0:
            logger.info(f"  step {step}: codec_tokens[0:3]={full_codec[:3]}")

    step_past_hiddens = np.stack([x[0] for x in step_past_hiddens], axis=0)
    step_talker_logits = np.stack(step_talker_logits, axis=0)
    all_codec_tokens = np.array(all_codec_tokens, dtype=np.int64)
    step_cp_tokens = np.array(step_cp_tokens, dtype=np.int64)

    out_dir = DEFAULT_OUTPUT_DIR / variant
    ref_path = out_dir / "e2e_trt_ref.npz"
    np.savez(
        ref_path,
        inputs_embeds=inputs_embeds.cpu().float().numpy(),
        prefill_logits=prefill_logits.cpu().float().numpy(),
        prefill_hidden=prefill_hidden.cpu().float().numpy(),
        hidden_size=np.array(hidden_size),
        vocab_size=np.array(vocab_size),
        seq_len=np.array(seq_len),
        n_text=np.array(n_text),
        codec_bos_id=np.array(codec_bos_id),
        text_ids=text_ids.cpu().numpy(),
        n_steps=np.array(n_steps),
        step_past_hidden=step_past_hiddens,
        step_talker_logits=step_talker_logits,
        all_codec_tokens=all_codec_tokens,
        step_cp_tokens=step_cp_tokens,
    )
    logger.info(f"Reference saved: {ref_path}")
    logger.info(f"  inputs_embeds: {inputs_embeds.shape}")
    logger.info(f"  all_codec_tokens: {all_codec_tokens.shape}")
    logger.info(f"  step_talker_logits: {step_talker_logits.shape}")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

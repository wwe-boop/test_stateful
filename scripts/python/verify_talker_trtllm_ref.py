#!/usr/bin/env python3
"""
[Phase 2, Item 9 — Part 1] Generate PyTorch reference for Talker TRT-LLM verification.

Runs the Talker Backbone in PyTorch and saves reference inputs/outputs as .npz
for comparison with TRT-LLM engine inference (run inside container).

Usage (host, with conda activate qwen3-tts):
  python scripts/python/verify_talker_trtllm_ref.py --variant design-1.7b
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
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

logger = logging.getLogger("talker_ref")


def main():
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="design-1.7b")
    parser.add_argument("--device", default=None)
    parser.add_argument("--n-decode-steps", type=int, default=5)
    args = parser.parse_args()

    device = resolve_device(args.device)
    variant = args.variant

    path = resolve_model_path(variant)
    if not has_model_weights(path):
        logger.error(f"No weights for {variant}")
        return

    logger.info(f"Loading {variant} on {device} ...")
    model = load_tts_model(path, device=device, dtype=torch.float32)
    talker = model.talker

    weights_dir = DEFAULT_OUTPUT_DIR / variant / "weights"
    cfg = json.load(open(weights_dir / "config.json"))
    hidden_size = cfg["talker_hidden_size"]
    codec_bos_id = cfg.get("codec_bos_id", 2149)

    # Build prefill: random text tokens → text_projection → concat with codec_bos embed
    n_text = 20
    torch.manual_seed(42)
    text_ids = torch.randint(100, 5000, (1, n_text), device=device)

    with torch.no_grad():
        text_embed = talker.text_projection(talker.model.text_embedding(text_ids))
        codec_bos_embed = talker.model.codec_embedding(
            torch.tensor([[codec_bos_id]], device=device))
        inputs_embeds = torch.cat([text_embed, codec_bos_embed], dim=1)
        seq_len = inputs_embeds.shape[1]

        logger.info(f"Prefill: {seq_len} tokens (text={n_text}, codec_bos=1)")
        logger.info(f"inputs_embeds: {inputs_embeds.shape}, range=[{inputs_embeds.min():.3f}, {inputs_embeds.max():.3f}]")

        # Prefill forward
        out = talker.model(
            inputs_embeds=inputs_embeds,
            use_cache=True,
            return_dict=True,
        )
        prefill_hidden = out.last_hidden_state
        prefill_logits = talker.codec_head(prefill_hidden)

        logger.info(f"Prefill hidden: {prefill_hidden.shape}")
        logger.info(f"Prefill logits: {prefill_logits.shape}")

        # First decode token
        codec_token_0 = prefill_logits[:, -1, :].argmax(dim=-1)
        logger.info(f"codec_token_0 = {codec_token_0.item()}")

        # Decode steps
        past_kv = out.past_key_values
        decode_logits_list = []
        decode_tokens = [codec_token_0.item()]

        for step in range(args.n_decode_steps):
            cur_token = torch.tensor([[decode_tokens[-1]]], device=device)
            cur_embed = talker.model.codec_embedding(cur_token)
            pos = seq_len + step

            step_out = talker.model(
                inputs_embeds=cur_embed,
                past_key_values=past_kv,
                use_cache=True,
                return_dict=True,
            )
            step_hidden = step_out.last_hidden_state
            step_logits = talker.codec_head(step_hidden)
            past_kv = step_out.past_key_values

            next_token = step_logits[:, -1, :].argmax(dim=-1).item()
            decode_tokens.append(next_token)
            decode_logits_list.append(step_logits[:, -1, :].cpu().float().numpy())

            logger.info(f"  decode step {step}: token={next_token}, "
                        f"logits_range=[{step_logits.min():.2f}, {step_logits.max():.2f}]")

    # Save reference
    out_dir = DEFAULT_OUTPUT_DIR / variant
    ref_path = out_dir / "talker_trtllm_ref.npz"

    np.savez(
        ref_path,
        inputs_embeds=inputs_embeds.cpu().float().numpy(),
        prefill_logits=prefill_logits.cpu().float().numpy(),
        prefill_hidden=prefill_hidden.cpu().float().numpy(),
        codec_token_0=np.array(codec_token_0.item()),
        decode_tokens=np.array(decode_tokens),
        decode_logits=np.stack(decode_logits_list),
        hidden_size=np.array(hidden_size),
        n_text=np.array(n_text),
        codec_bos_id=np.array(codec_bos_id),
        text_ids=text_ids.cpu().numpy(),
    )
    logger.info(f"Reference saved: {ref_path}")
    logger.info(f"Decode tokens: {decode_tokens}")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

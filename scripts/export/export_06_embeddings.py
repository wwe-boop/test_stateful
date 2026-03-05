#!/usr/bin/env python3
"""
[Step 06] Export Embedding weights for the TTS Orchestrator.

Exports both .pt (legacy) and ONNX/npz for lightweight Triton deploy:
  - text_embedder.onnx, codec_embedder.onnx (variant dir) — BLS sub-models, bf16-safe
  - special_embeddings.npz, codec_embeddings_3d.npz (weights dir) — numpy for BLS

Legacy .pt components (unchanged):
  1. text_embedding.pt, text_projection.pt, codec_embeddings.pt, codec_embeddings_3d.pt
  2. special_embeddings.pt, codec_head.pt, code_predictor_lm_heads.pt, config.json

Applies to: ALL variants
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Allow importing from scripts/python (for CodecEmbeddingSum)
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from python.codec_embedding_sum import CodecEmbeddingSum

from utils import (
    setup_logging,
    resolve_model_path,
    ensure_output_dir,
    load_tts_model,
    export_onnx,
    to_numpy,
    verify_onnx,
    resolve_device,
    resolve_dtype,
    add_common_args,
    MODEL_VARIANTS,
    DEFAULT_DTYPE,
    DTYPE_NAMES,
    ONNX_EXPORT_DTYPE,
)

logger = logging.getLogger("onnx_export")


def _cast_state_dict(state_dict: dict, dtype: torch.dtype) -> dict:
    """Cast all float tensors in a state_dict to the target dtype."""
    return {k: v.to(dtype) if v.is_floating_point() else v for k, v in state_dict.items()}


class _TextEmbedderONNX(nn.Module):
    """Embedding + ResizeMLP for ONNX export: token_ids [B,S] -> [B,S,H]."""

    def __init__(self, text_embedding: nn.Module, text_projection: nn.Module):
        super().__init__()
        self.text_embedding = text_embedding
        self.text_projection = text_projection

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.text_projection(self.text_embedding(token_ids))


class _CodecEmbedderONNX(nn.Module):
    """Codec embedding lookup for ONNX export: codec_ids [B,S] -> [B,S,H]."""

    def __init__(self, codec_embedding: nn.Module):
        super().__init__()
        self.codec_embedding = codec_embedding

    def forward(self, codec_ids: torch.Tensor) -> torch.Tensor:
        return self.codec_embedding(codec_ids)


def export_embeddings(
    variant: str,
    models_dir: str = None,
    output_dir: str = None,
    device: str = "cpu",
    dtype: torch.dtype = DEFAULT_DTYPE,
) -> Path:
    model_path = resolve_model_path(variant, models_dir)
    out_dir = ensure_output_dir(output_dir, variant) / "weights"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading model: {variant} from {model_path} (fp32 for precision)")
    logger.info(f"  Weights will be saved as {DTYPE_NAMES[dtype]} for inference")
    model = load_tts_model(model_path, device=device, dtype=torch.float32)

    talker = model.talker
    config = model.config

    # 1. Text Embedding
    text_emb = _cast_state_dict(talker.model.text_embedding.state_dict(), dtype)
    path = out_dir / "text_embedding.pt"
    torch.save(text_emb, path)
    n_params = sum(p.numel() for p in talker.model.text_embedding.parameters())
    logger.info(f"  text_embedding.pt: {n_params/1e6:.1f}M params, saved to {path}")

    # 2. Text Projection (ResizeMLP)
    text_proj = _cast_state_dict(talker.text_projection.state_dict(), dtype)
    path = out_dir / "text_projection.pt"
    torch.save(text_proj, path)
    n_params = sum(p.numel() for p in talker.text_projection.parameters())
    logger.info(f"  text_projection.pt: {n_params/1e6:.1f}M params, saved to {path}")

    # 3. Codec Embeddings
    codec_embs = {
        "talker_codec_embedding": _cast_state_dict(
            talker.model.codec_embedding.state_dict(), dtype),
        "code_predictor_codec_embeddings": [
            _cast_state_dict(emb.state_dict(), dtype)
            for emb in talker.code_predictor.model.codec_embedding
        ],
    }
    path = out_dir / "codec_embeddings.pt"
    torch.save(codec_embs, path)
    n_talker = sum(p.numel() for p in talker.model.codec_embedding.parameters())
    n_cp = sum(
        sum(p.numel() for p in emb.parameters())
        for emb in talker.code_predictor.model.codec_embedding
    )
    logger.info(
        f"  codec_embeddings.pt: talker={n_talker/1e6:.1f}M + "
        f"code_predictor={n_cp/1e6:.1f}M params, saved to {path}"
    )

    # 3b. Pre-stacked 3D codec embedding (for optimized gather+sum, §2.1)
    stacked = CodecEmbeddingSum.stack_weights(
        talker.model.codec_embedding.weight,
        [emb.weight for emb in talker.code_predictor.model.codec_embedding],
        dtype=dtype,
    )
    path_3d = out_dir / "codec_embeddings_3d.pt"
    torch.save(stacked.cpu(), path_3d)
    logger.info(f"  codec_embeddings_3d.pt: shape={tuple(stacked.shape)}, saved to {path_3d}")
    # Lightweight deploy: numpy format for BLS (cupy load)
    path_3d_npz = out_dir / "codec_embeddings_3d.npz"
    np.savez_compressed(
        path_3d_npz,
        data=stacked.float().cpu().numpy(),
    )
    logger.info(f"  codec_embeddings_3d.npz: saved to {path_3d_npz}")

    # 4. Special Embeddings (tts_pad, tts_bos, tts_eos) — computed in FP32, saved in target dtype
    with torch.no_grad():
        special_ids = torch.tensor(
            [[config.tts_pad_token_id, config.tts_bos_token_id, config.tts_eos_token_id]],
            device=device,
        )
        special_text_embed = talker.model.text_embedding(special_ids)
        special_projected = talker.text_projection(special_text_embed)
        tts_pad_embed = special_projected[:, 0:1, :].to(dtype)
        tts_bos_embed = special_projected[:, 1:2, :].to(dtype)
        tts_eos_embed = special_projected[:, 2:3, :].to(dtype)

    special = {
        "tts_pad_embed": tts_pad_embed.cpu(),
        "tts_bos_embed": tts_bos_embed.cpu(),
        "tts_eos_embed": tts_eos_embed.cpu(),
        "tts_pad_token_id": config.tts_pad_token_id,
        "tts_bos_token_id": config.tts_bos_token_id,
        "tts_eos_token_id": config.tts_eos_token_id,
    }
    path = out_dir / "special_embeddings.pt"
    torch.save(special, path)
    logger.info(f"  special_embeddings.pt: shape={tts_pad_embed.shape}, saved to {path}")
    # Lightweight deploy: numpy format for BLS (no torch)
    path_special_npz = out_dir / "special_embeddings.npz"
    np.savez(
        path_special_npz,
        tts_pad_embed=tts_pad_embed.float().cpu().numpy(),
        tts_bos_embed=tts_bos_embed.float().cpu().numpy(),
        tts_eos_embed=tts_eos_embed.float().cpu().numpy(),
    )
    logger.info(f"  special_embeddings.npz: saved to {path_special_npz}")

    # 5. Codec Head
    codec_head = _cast_state_dict(talker.codec_head.state_dict(), dtype)
    path = out_dir / "codec_head.pt"
    torch.save(codec_head, path)
    n_params = sum(p.numel() for p in talker.codec_head.parameters())
    logger.info(f"  codec_head.pt: {n_params/1e6:.1f}M params, saved to {path}")

    # 6. Code Predictor lm_heads (num_code_groups-1 heads, needed for fallback mode)
    lm_heads = [_cast_state_dict(head.state_dict(), dtype)
                for head in talker.code_predictor.lm_head]
    path = out_dir / "code_predictor_lm_heads.pt"
    torch.save(lm_heads, path)
    n_params = sum(
        sum(p.numel() for p in head.parameters())
        for head in talker.code_predictor.lm_head
    )
    logger.info(f"  code_predictor_lm_heads.pt: {n_params/1e6:.1f}M params ({len(lm_heads)} heads)")

    # 7. Model config metadata
    metadata = {
        "variant": variant,
        "target_dtype": DTYPE_NAMES[dtype],
        "tts_model_type": config.tts_model_type,
        "tts_model_size": config.tts_model_size,
        "tokenizer_type": config.tokenizer_type,
        "talker_hidden_size": talker.config.hidden_size,
        "talker_text_hidden_size": talker.config.text_hidden_size,
        "talker_vocab_size": talker.config.vocab_size,
        "talker_num_layers": talker.config.num_hidden_layers,
        "talker_num_heads": talker.config.num_attention_heads,
        "talker_num_kv_heads": talker.config.num_key_value_heads,
        "num_code_groups": talker.config.num_code_groups,
        "code_predictor_hidden_size": talker.config.code_predictor_config.hidden_size,
        "code_predictor_num_layers": talker.config.code_predictor_config.num_hidden_layers,
        "code_predictor_vocab_size": talker.config.code_predictor_config.vocab_size,
        "codec_eos_token_id": talker.config.codec_eos_token_id,
        "codec_bos_id": talker.config.codec_bos_id,
        "codec_pad_id": talker.config.codec_pad_id,
        "codec_nothink_id": getattr(talker.config, "codec_nothink_id", 2155),
        "codec_think_bos_id": getattr(talker.config, "codec_think_bos_id", 2156),
        "codec_think_eos_id": getattr(talker.config, "codec_think_eos_id", 2157),
        "codec_think_id": getattr(talker.config, "codec_think_id", 2154),
        "hidden_act": getattr(talker.config, "hidden_act", "silu"),
        "im_start_token_id": config.im_start_token_id,
        "im_end_token_id": config.im_end_token_id,
    }
    if hasattr(talker.config, "spk_id") and talker.config.spk_id:
        metadata["spk_id"] = talker.config.spk_id
    if hasattr(talker.config, "codec_language_id") and talker.config.codec_language_id:
        metadata["codec_language_id"] = talker.config.codec_language_id

    path = out_dir / "config.json"
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info(f"  config.json: saved to {path}")

    # --- Optional: ONNX sub-models for BLS fallback (orchestrator uses in-process torch by default) ---
    variant_dir = out_dir.parent
    _device = next(talker.parameters()).device

    # 1. text_embedder.onnx
    text_embedder_wrapper = _TextEmbedderONNX(
        talker.model.text_embedding.to(_device),
        talker.text_projection.to(_device),
    ).eval()
    dummy_token_ids = torch.randint(0, 1000, (1, 10), device=_device, dtype=torch.int64)
    with torch.no_grad():
        ref_text_emb = text_embedder_wrapper(dummy_token_ids)
    text_embedder_onnx = str(variant_dir / "text_embedder.onnx")
    export_onnx(
        model=text_embedder_wrapper,
        dummy_inputs=(dummy_token_ids,),
        input_names=["token_ids"],
        output_names=["embeddings"],
        dynamic_axes={
            "token_ids": {0: "batch", 1: "seq_len"},
            "embeddings": {0: "batch", 1: "seq_len"},
        },
        onnx_path=text_embedder_onnx,
        simplify=True,
    )
    test_inputs = {"token_ids": dummy_token_ids.cpu().numpy()}
    torch_outputs = {"embeddings": to_numpy(ref_text_emb)}
    try:
        ok = verify_onnx(text_embedder_onnx, test_inputs, torch_outputs, atol=1e-4)
        if ok:
            logger.info("  text_embedder.onnx verification PASSED")
        else:
            logger.warning("  text_embedder.onnx verification FAILED")
    except Exception as e:
        logger.warning("  text_embedder.onnx verification skip: %s", e)
    del text_embedder_wrapper, ref_text_emb, dummy_token_ids

    # 2. codec_embedder.onnx
    codec_embedder_wrapper = _CodecEmbedderONNX(
        talker.model.codec_embedding.to(_device),
    ).eval()
    dummy_codec_ids = torch.randint(0, 100, (1, 5), device=_device, dtype=torch.int64)
    with torch.no_grad():
        ref_codec_emb = codec_embedder_wrapper(dummy_codec_ids)
    codec_embedder_onnx = str(variant_dir / "codec_embedder.onnx")
    export_onnx(
        model=codec_embedder_wrapper,
        dummy_inputs=(dummy_codec_ids,),
        input_names=["codec_ids"],
        output_names=["embeddings"],
        dynamic_axes={
            "codec_ids": {0: "batch", 1: "seq_len"},
            "embeddings": {0: "batch", 1: "seq_len"},
        },
        onnx_path=codec_embedder_onnx,
        simplify=True,
    )
    test_inputs_ce = {"codec_ids": dummy_codec_ids.cpu().numpy()}
    torch_outputs_ce = {"embeddings": to_numpy(ref_codec_emb)}
    try:
        ok = verify_onnx(codec_embedder_onnx, test_inputs_ce, torch_outputs_ce, atol=1e-4)
        if ok:
            logger.info("  codec_embedder.onnx verification PASSED")
        else:
            logger.warning("  codec_embedder.onnx verification FAILED")
    except Exception as e:
        logger.warning("  codec_embedder.onnx verification skip: %s", e)
    del codec_embedder_wrapper, ref_codec_emb, dummy_codec_ids

    del model
    if device != "cpu":
        torch.cuda.empty_cache()
    return out_dir


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Export Embedding weights")
    parser.add_argument("--variant", type=str, default=None,
                        help="Model variant. Default: export all variants")
    add_common_args(parser)
    args = parser.parse_args()

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)

    variants = [args.variant] if args.variant else list(MODEL_VARIANTS.keys())

    for variant in variants:
        if variant not in MODEL_VARIANTS:
            logger.error(f"Unknown variant: {variant}")
            continue
        try:
            path = export_embeddings(variant, args.models_dir, args.output_dir, device, dtype)
            logger.info(f"[{variant}] Embeddings exported to: {path}")
        except FileNotFoundError as e:
            logger.warning(f"[{variant}] Skipped: {e}")
        except Exception as e:
            logger.error(f"[{variant}] Failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()

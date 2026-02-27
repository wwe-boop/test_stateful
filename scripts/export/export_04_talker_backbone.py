#!/usr/bin/env python3
"""
[Step 04] Convert Talker Backbone weights to TRT-LLM checkpoint.

Component: Talker Backbone
Architecture: Qwen3-style Transformer, 20L, h=1024, GQA(16h/2kv), head_dim=64
              QK-Norm (RMSNorm on head_dim), SwiGLU MLP, RoPE
Input:  inputs_embeds [B, S, 1024]
Output: hidden_states [B, S, 1024], logits [B, S, 3072]
Engine: TRT-LLM (KV Cache managed by TRT-LLM runtime)
Usage: Called every decode step (hottest path)

This script produces a TRT-LLM checkpoint (config.json + rank0.safetensors).
Engine compilation (trtllm-build) is a separate step performed inside a
TRT-LLM container via build_engines.sh — see docs/architecture.md.

Workflow:
  1. Extract sub-module weights from Qwen3TTSForConditionalGeneration
  2. Map weight names to TRT-LLM Qwen checkpoint format
  3. Write TRT-LLM checkpoint (config.json + rank0.safetensors)
  4. (Optional) Export ONNX correctness baseline for verification

The Talker Backbone maps to Qwen3ForCausalLM in TRT-LLM:
  - 20 decoder layers with GQA (16 Q heads, 2 KV heads)
  - head_dim=64, hidden_size=1024, intermediate_size=2048
  - QK-Norm (q_norm, k_norm per layer) — Qwen3 feature
  - codec_head → lm_head in TRT-LLM (Linear 1024→3072)

Applies to: ALL variants (each has a talker backbone)
"""

import argparse
import json
import logging
from pathlib import Path
from collections import OrderedDict

import torch
import torch.nn as nn

from utils import (
    setup_logging,
    resolve_model_path,
    ensure_output_dir,
    load_tts_model,
    export_onnx,
    verify_onnx,
    to_numpy,
    resolve_device,
    resolve_dtype,
    add_common_args,
    MODEL_VARIANTS,
    ONNX_EXPORT_DTYPE,
    DTYPE_NAMES,
)

logger = logging.getLogger("onnx_export")


def _extract_talker_weights(model) -> tuple[dict, dict]:
    """Extract Talker Backbone weights and config from the full TTS model.

    Returns:
        (state_dict, config_dict): Talker weights in HF naming + Talker config
    """
    talker = model.talker
    config = talker.config

    config_dict = {
        "vocab_size": config.vocab_size,
        "hidden_size": config.hidden_size,
        "intermediate_size": config.intermediate_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "head_dim": getattr(config, "head_dim", config.hidden_size // config.num_attention_heads),
        "hidden_act": config.hidden_act,
        "max_position_embeddings": config.max_position_embeddings,
        "rms_norm_eps": config.rms_norm_eps,
        "rope_theta": config.rope_theta,
        "rope_scaling": config.rope_scaling,
        "attention_bias": config.attention_bias,
        "use_sliding_window": config.use_sliding_window,
        "sliding_window": config.sliding_window,
    }

    state_dict = OrderedDict()

    talker_model = talker.model
    for name, param in talker_model.named_parameters():
        state_dict[f"model.{name}"] = param.detach().cpu()

    codec_head = talker.codec_head
    for name, param in codec_head.named_parameters():
        state_dict[f"lm_head.{name}"] = param.detach().cpu()

    logger.info(f"  Extracted {len(state_dict)} Talker weight tensors")
    total_params = sum(p.numel() for p in state_dict.values())
    logger.info(f"  Total parameters: {total_params / 1e6:.1f}M")

    return state_dict, config_dict


# Weight name mapping: HF Talker → TRT-LLM Qwen checkpoint format
#
# HF (Talker sub-module):
#   model.layers.{i}.self_attn.{q,k,v,o}_proj.weight
#   model.layers.{i}.self_attn.{q,k}_norm.weight       (Qwen3 QK-Norm)
#   model.layers.{i}.mlp.{gate,up,down}_proj.weight
#   model.layers.{i}.input_layernorm.weight
#   model.layers.{i}.post_attention_layernorm.weight
#   model.norm.weight
#   lm_head.weight
#
# TRT-LLM Qwen:
#   transformer.layers.{i}.attention.qkv.weight         (fused Q+K+V)
#   transformer.layers.{i}.attention.dense.weight        (o_proj)
#   transformer.layers.{i}.attention.q_norm.weight       (Qwen3 QK-Norm)
#   transformer.layers.{i}.attention.k_norm.weight       (Qwen3 QK-Norm)
#   transformer.layers.{i}.mlp.gate.weight               (gate_up fused)
#   transformer.layers.{i}.mlp.proj.weight               (down_proj)
#   transformer.layers.{i}.input_layernorm.weight
#   transformer.layers.{i}.post_layernorm.weight
#   transformer.vocab_embedding.weight                   (unused, we receive inputs_embeds)
#   transformer.ln_f.weight
#   lm_head.weight

def _map_weights_to_trtllm(
    hf_state: dict,
    config: dict,
    dtype: torch.dtype = torch.bfloat16,
) -> dict:
    """Map Talker HF weights to TRT-LLM checkpoint naming convention.

    Fuses separate Q, K, V projections into a single QKV weight tensor
    as required by TRT-LLM's optimized attention kernels.
    """
    trtllm_state = OrderedDict()
    n_layers = config["num_hidden_layers"]
    num_heads = config["num_attention_heads"]
    num_kv_heads = config["num_key_value_heads"]
    head_dim = config["head_dim"]

    for i in range(n_layers):
        prefix_hf = f"model.layers.{i}"
        prefix_trt = f"transformer.layers.{i}"

        # Fuse Q + K + V into single QKV tensor
        q_w = hf_state[f"{prefix_hf}.self_attn.q_proj.weight"].to(dtype)
        k_w = hf_state[f"{prefix_hf}.self_attn.k_proj.weight"].to(dtype)
        v_w = hf_state[f"{prefix_hf}.self_attn.v_proj.weight"].to(dtype)
        qkv_w = torch.cat([q_w, k_w, v_w], dim=0)
        trtllm_state[f"{prefix_trt}.attention.qkv.weight"] = qkv_w

        # O projection
        trtllm_state[f"{prefix_trt}.attention.dense.weight"] = (
            hf_state[f"{prefix_hf}.self_attn.o_proj.weight"].to(dtype)
        )

        # QK-Norm (Qwen3 feature)
        trtllm_state[f"{prefix_trt}.attention.q_norm.weight"] = (
            hf_state[f"{prefix_hf}.self_attn.q_norm.weight"].to(dtype)
        )
        trtllm_state[f"{prefix_trt}.attention.k_norm.weight"] = (
            hf_state[f"{prefix_hf}.self_attn.k_norm.weight"].to(dtype)
        )

        # Fuse gate + up into single tensor for TRT-LLM SwiGLU
        gate_w = hf_state[f"{prefix_hf}.mlp.gate_proj.weight"].to(dtype)
        up_w = hf_state[f"{prefix_hf}.mlp.up_proj.weight"].to(dtype)
        gate_up_w = torch.cat([gate_w, up_w], dim=0)
        trtllm_state[f"{prefix_trt}.mlp.gate.weight"] = gate_up_w

        # Down projection
        trtllm_state[f"{prefix_trt}.mlp.proj.weight"] = (
            hf_state[f"{prefix_hf}.mlp.down_proj.weight"].to(dtype)
        )

        # LayerNorms
        trtllm_state[f"{prefix_trt}.input_layernorm.weight"] = (
            hf_state[f"{prefix_hf}.input_layernorm.weight"].to(dtype)
        )
        trtllm_state[f"{prefix_trt}.post_layernorm.weight"] = (
            hf_state[f"{prefix_hf}.post_attention_layernorm.weight"].to(dtype)
        )

    # Final LayerNorm
    trtllm_state["transformer.ln_f.weight"] = (
        hf_state["model.norm.weight"].to(dtype)
    )

    # LM head (codec_head in TTS, maps to lm_head in TRT-LLM)
    trtllm_state["lm_head.weight"] = (
        hf_state["lm_head.weight"].to(dtype)
    )

    return trtllm_state


def _write_trtllm_checkpoint(
    trtllm_state: dict,
    config: dict,
    output_dir: Path,
    dtype: torch.dtype = torch.bfloat16,
) -> Path:
    """Write TRT-LLM checkpoint directory (config.json + rank0.safetensors)."""
    ckpt_dir = output_dir / "trtllm_checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    dtype_str = {
        torch.bfloat16: "bfloat16",
        torch.float16: "float16",
        torch.float32: "float32",
    }[dtype]

    trtllm_config = {
        "architecture": "QWenForCausalLM",
        "dtype": dtype_str,
        "logits_dtype": "float32",
        "vocab_size": config["vocab_size"],
        "hidden_size": config["hidden_size"],
        "intermediate_size": config["intermediate_size"],
        "num_hidden_layers": config["num_hidden_layers"],
        "num_attention_heads": config["num_attention_heads"],
        "num_key_value_heads": config["num_key_value_heads"],
        "head_size": config["head_dim"],
        "hidden_act": config["hidden_act"],
        "max_position_embeddings": config["max_position_embeddings"],
        "norm_epsilon": config["rms_norm_eps"],
        "position_embedding_type": "rope_gpt_neox",
        "rotary_base": config["rope_theta"],
        "rotary_scaling": config.get("rope_scaling"),
        "qwen_type": "qwen3",
        "qk_layernorm": True,
        "mapping": {
            "world_size": 1,
            "tp_size": 1,
            "pp_size": 1,
        },
        "quantization": {
            "quant_algo": None,
            "kv_cache_quant_algo": None,
        },
    }

    config_path = ckpt_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(trtllm_config, f, indent=2)
    logger.info(f"  Wrote TRT-LLM config: {config_path}")

    weights_path = ckpt_dir / "rank0.safetensors"
    try:
        from safetensors.torch import save_file
        save_file(trtllm_state, str(weights_path))
    except ImportError:
        weights_path = ckpt_dir / "rank0.bin"
        torch.save(trtllm_state, str(weights_path))
        logger.warning("  safetensors not available, saved as .bin (trtllm-build prefers .safetensors)")

    weight_mb = sum(v.numel() * v.element_size() for v in trtllm_state.values()) / (1024 * 1024)
    logger.info(f"  Wrote TRT-LLM weights: {weights_path} ({weight_mb:.1f} MB)")

    return ckpt_dir


# ── ONNX correctness baseline ──

class TalkerBackbonePrefillWrapper(nn.Module):
    """Wraps the Talker backbone for prefill-mode ONNX export (no KV cache).

    Exports the decoder layers + norm + codec_head as a single graph.
    No KV cache I/O — TRT-LLM manages KV cache natively at deployment time.
    This wrapper is only used to produce a correctness reference.
    """

    def __init__(self, talker_model, codec_head):
        super().__init__()
        self.layers = talker_model.layers
        self.norm = talker_model.norm
        self.rotary_emb = talker_model.rotary_emb
        self.codec_head = codec_head

    def forward(self, inputs_embeds: torch.Tensor) -> tuple:
        """
        Args:
            inputs_embeds: [B, S, hidden_size] - combined text+codec embeddings
        Returns:
            hidden_states: [B, S, hidden_size]
            logits: [B, S, vocab_size]
        """
        B, S, D = inputs_embeds.shape
        device = inputs_embeds.device

        cache_position = torch.arange(S, device=device)
        position_ids = cache_position.view(1, 1, -1).expand(3, B, -1)

        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        from transformers.masking_utils import create_causal_mask

        causal_mask = create_causal_mask(
            config=self.layers[0].self_attn.config if hasattr(self.layers[0].self_attn, 'config') else None,
            input_embeds=inputs_embeds,
            attention_mask=None,
            cache_position=cache_position,
            past_key_values=None,
        )

        hidden = inputs_embeds
        for layer in self.layers:
            layer_out = layer(
                hidden,
                attention_mask=causal_mask,
                position_ids=position_ids[0],
                past_key_values=None,
                output_attentions=False,
                use_cache=False,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            hidden = layer_out[0]

        hidden = self.norm(hidden)
        logits = self.codec_head(hidden)
        return hidden, logits


def _export_onnx_baseline(
    model,
    variant: str,
    output_dir: Path,
    device: str = "cpu",
) -> str:
    """Export ONNX correctness baseline (fp32, no KV cache)."""
    talker = model.talker
    talker_model = talker.model.to(device).eval()
    codec_head = talker.codec_head.to(device).eval()

    wrapper = TalkerBackbonePrefillWrapper(talker_model, codec_head).to(device).eval()

    hidden_size = talker.config.hidden_size
    B, S = 1, 16
    dummy_embeds = torch.randn(B, S, hidden_size, device=device, dtype=torch.float32)

    with torch.no_grad():
        ref_hidden, ref_logits = wrapper(dummy_embeds)

    logger.info(f"  ONNX baseline shapes: hidden={ref_hidden.shape}, logits={ref_logits.shape}")

    onnx_path = str(output_dir / "talker_backbone_prefill.onnx")
    export_onnx(
        model=wrapper,
        dummy_inputs=(dummy_embeds,),
        input_names=["inputs_embeds"],
        output_names=["hidden_states", "logits"],
        dynamic_axes={
            "inputs_embeds": {0: "batch", 1: "seq_len"},
            "hidden_states": {0: "batch", 1: "seq_len"},
            "logits": {0: "batch", 1: "seq_len"},
        },
        onnx_path=onnx_path,
    )

    test_inputs = {"inputs_embeds": to_numpy(dummy_embeds)}
    torch_outputs = {
        "hidden_states": to_numpy(ref_hidden),
        "logits": to_numpy(ref_logits),
    }
    ok = verify_onnx(onnx_path, test_inputs, torch_outputs, atol=1e-4)
    if ok:
        logger.info("  ONNX baseline verification PASSED")
    else:
        logger.warning("  ONNX baseline verification FAILED (non-blocking for TRT-LLM path)")

    return onnx_path


def export_talker_backbone(
    variant: str,
    models_dir: str = None,
    output_dir: str = None,
    device: str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    export_onnx_baseline: bool = False,
) -> dict:
    """Export Talker Backbone weights as TRT-LLM checkpoint.

    Produces a TRT-LLM checkpoint directory (config.json + rank0.safetensors)
    ready for engine compilation via trtllm-build (run inside a TRT-LLM
    container — see build_engines.sh).

    Steps:
      1. Load full TTS model, extract Talker sub-module weights
      2. Map weights to TRT-LLM Qwen checkpoint format
      3. Write TRT-LLM checkpoint (config.json + rank0.safetensors)
      4. (Optional) Export ONNX correctness baseline

    Returns:
        dict with paths: {"checkpoint": ..., "onnx_baseline": ...}
    """
    model_path = resolve_model_path(variant, models_dir)
    out_dir = ensure_output_dir(output_dir, variant)
    results = {}

    logger.info(f"Loading model: {variant} from {model_path}")
    model = load_tts_model(model_path, device="cpu", dtype=torch.float32)

    # Step 1-2: Extract and map weights
    logger.info("Extracting Talker Backbone weights ...")
    hf_state, talker_config = _extract_talker_weights(model)

    logger.info("Mapping weights to TRT-LLM checkpoint format ...")
    trtllm_state = _map_weights_to_trtllm(hf_state, talker_config, dtype=dtype)
    del hf_state

    logger.info(f"  TRT-LLM checkpoint tensors: {len(trtllm_state)}")
    for name, tensor in list(trtllm_state.items())[:5]:
        logger.info(f"    {name}: {tensor.shape} {tensor.dtype}")
    if len(trtllm_state) > 5:
        logger.info(f"    ... ({len(trtllm_state) - 5} more)")

    # Step 3: Write checkpoint
    ckpt_dir = _write_trtllm_checkpoint(trtllm_state, talker_config, out_dir, dtype)
    results["checkpoint"] = str(ckpt_dir)
    del trtllm_state

    logger.info(
        "TRT-LLM checkpoint ready. To build engine, run:\n"
        "  bash scripts/bash/build_engines.sh"
    )

    # Step 4: ONNX baseline (optional)
    if export_onnx_baseline:
        logger.info("Exporting ONNX correctness baseline ...")
        model_fp32 = model.to("cpu").float()
        onnx_path = _export_onnx_baseline(model_fp32, variant, out_dir, device="cpu")
        results["onnx_baseline"] = onnx_path

    del model
    if device != "cpu":
        torch.cuda.empty_cache()

    return results


def main():
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Export Talker Backbone to TRT-LLM checkpoint"
    )
    parser.add_argument("--variant", type=str, default=None,
                        help="Model variant. Default: export all variants")
    add_common_args(parser)
    parser.add_argument("--onnx-baseline", action="store_true",
                        help="Also export ONNX correctness baseline (fp32)")

    args = parser.parse_args()

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)

    variants = [args.variant] if args.variant else list(MODEL_VARIANTS.keys())

    for variant in variants:
        if variant not in MODEL_VARIANTS:
            logger.error(f"Unknown variant: {variant}")
            continue
        try:
            results = export_talker_backbone(
                variant, args.models_dir, args.output_dir, device, dtype,
                export_onnx_baseline=args.onnx_baseline,
            )
            for key, path in results.items():
                logger.info(f"[{variant}] Talker Backbone {key}: {path}")
        except FileNotFoundError as e:
            logger.warning(f"[{variant}] Skipped: {e}")
        except Exception as e:
            logger.error(f"[{variant}] Failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()

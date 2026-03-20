#!/usr/bin/env python3
"""
[Step 04] Export Talker Context Fused (prefill + CP + codec_sum) to ONNX.

Deprecated: Prefer export_08_talker_unified.py / export_09_talker_code2wav_fused.py for deployment.

Component: Talker prefill + argmax + Code Predictor (15-step) + Codec Embedding Sum.
Input:  input_embeds [B, S, H], position_ids [3, B, S] (3D RoPE)
Output: codec_sum [B, 1, H], full_codec [B, 16], last_hidden [B, 1, H], last_logits [B, 1, V],
        present_kv_{i}_k, present_kv_{i}_v for i=0..num_layers-1

Output format matches talker_decode_fused so BLS can use the same flow for context and decode.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

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
    CodePredictorUnrolled,
)

logger = logging.getLogger("onnx_export")


class PrefillKVCache:
    """Cache that stores present key/value per layer for prefill (no past concat).

    Used so we can retrieve key_states, value_states after each layer forward
    for ONNX export. Implements the interface expected by HuggingFace attention
    (get_seq_length, update).
    """

    def __init__(self, num_layers: int):
        self._cache: List[Tuple[torch.Tensor, torch.Tensor]] = [None] * num_layers
        self._num_layers = num_layers

    def get_seq_length(self) -> int:
        """Return current sequence length (0 for prefill, no past)."""
        return 0

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Store key/value for this layer and return as-is (no concat with past)."""
        self._cache[layer_idx] = (key_states, value_states)
        return key_states, value_states

    def get_layer(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._cache[layer_idx]


def _build_codec_embedding_sum_from_model(model, device, dtype=torch.float32):
    """Build CodecEmbeddingSum from live model (stacked [16, 3072, H])."""
    try:
        from codec_embedding_sum import CodecEmbeddingSum
    except ImportError:
        repo_root = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(repo_root / "scripts" / "python"))
        from codec_embedding_sum import CodecEmbeddingSum
    return CodecEmbeddingSum.from_model(model, dtype=dtype).to(device).eval()


class TalkerContextONNX(nn.Module):
    """Talker prefill: process S tokens, output last hidden + logits + KV cache.

    Uses explicit causal mask (triu) for ONNX compatibility; avoids
    create_causal_mask (vmap). Captures present_key_values via PrefillKVCache.
    """

    def __init__(self, talker_model, codec_head):
        super().__init__()
        self.layers = talker_model.layers
        self.norm = talker_model.norm
        self.rotary_emb = talker_model.rotary_emb
        self.codec_head = codec_head
        self.num_layers = len(self.layers)

    def forward(
        self,
        input_embeds: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, ...]:
        """
        Args:
            input_embeds: [B, S, H]
            position_ids: [3, B, S] for 3D multimodal RoPE (TTS: same value per dim)
        Returns:
            last_hidden: [B, 1, H]
            last_logits: [B, 1, V]
            present_kv_0_k, present_kv_0_v, ..., present_kv_{N-1}_k, present_kv_{N-1}_v
        """
        B, S, H = input_embeds.shape
        device = input_embeds.device
        dtype = input_embeds.dtype

        # position_embeddings: cos, sin from 3D RoPE
        position_embeddings = self.rotary_emb(input_embeds, position_ids)

        # Causal mask: [B, 1, S, S], lower triangle valid
        causal_mask = torch.triu(
            torch.full((S, S), float("-inf"), device=device, dtype=dtype),
            diagonal=1,
        )
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(B, 1, S, S)

        cache = PrefillKVCache(self.num_layers)
        hidden = input_embeds

        # text_position_ids for layer: [B, S] (first of the 3 RoPE dims)
        text_position_ids = position_ids[0] if position_ids.dim() == 3 else position_ids

        for layer in self.layers:
            layer_out = layer(
                hidden,
                attention_mask=causal_mask,
                position_ids=text_position_ids,
                past_key_values=cache,
                output_attentions=False,
                use_cache=True,
                cache_position=torch.arange(S, device=device),
                position_embeddings=position_embeddings,
            )
            hidden = layer_out[0]

        hidden = self.norm(hidden)
        logits = self.codec_head(hidden)

        last_hidden = hidden[:, -1:, :]
        last_logits = logits[:, -1:, :]

        outputs: list = [last_hidden, last_logits]
        for i in range(self.num_layers):
            k, v = cache.get_layer(i)
            outputs.append(k)
            outputs.append(v)

        return tuple(outputs)


class TalkerContextFusedONNX(nn.Module):
    """Context prefill + argmax + CP 15-step + Codec Embedding Sum. Output format matches decode_fused."""

    def __init__(
        self,
        talker_context: TalkerContextONNX,
        code_predictor_unrolled: CodePredictorUnrolled,
        codec_embedding_sum: nn.Module,
    ):
        super().__init__()
        self.talker_context = talker_context
        self.cp = code_predictor_unrolled
        self.codec_sum = codec_embedding_sum
        self.num_layers = talker_context.num_layers

    def forward(
        self,
        input_embeds: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, ...]:
        # ① Talker context (prefill)
        out = self.talker_context(input_embeds, position_ids)
        last_hidden = out[0]   # [B, 1, H]
        last_logits = out[1]   # [B, 1, V]
        present_kv = list(out[2:])

        # ② argmax → codec_token_0
        codec_token_0 = last_logits[:, -1, :].argmax(dim=-1)

        # ③ Code Predictor 15-step unrolled
        cp_tokens = self.cp(last_hidden, codec_token_0)

        # ④ full_codec [B, 16]
        full_codec = torch.cat(
            [codec_token_0.unsqueeze(1), cp_tokens.long()], dim=1
        )

        # ⑤ Codec Embedding Sum → codec_sum [B, 1, H]
        codec_sum = self.codec_sum(full_codec).unsqueeze(1)

        return (codec_sum, full_codec, last_hidden, last_logits, *present_kv)


def _export_talker_context_onnx(
    model,
    variant: str,
    output_dir: Path,
    device: str = "cpu",
    opset_version: int = 18,
) -> str:
    """Export Talker context fused (prefill + CP + codec_sum) ONNX. Output format matches decode_fused."""
    talker = model.talker
    talker_config = talker.model.config
    setattr(talker_config, "_attn_implementation", "eager")
    if hasattr(talker.code_predictor, "model") and hasattr(talker.code_predictor.model, "config"):
        setattr(talker.code_predictor.model.config, "_attn_implementation", "eager")

    talker_model = talker.model.to(device).eval()
    codec_head = talker.codec_head.to(device).eval()
    context_step = TalkerContextONNX(talker_model, codec_head).to(device).eval()

    code_predictor = talker.code_predictor.to(device).eval()
    talker_codec_emb = talker.model.codec_embedding.to(device).eval()
    cp_unrolled = CodePredictorUnrolled(code_predictor, talker_codec_emb).to(device).eval()

    codec_sum_module = _build_codec_embedding_sum_from_model(model, device)

    fused = TalkerContextFusedONNX(context_step, cp_unrolled, codec_sum_module).to(device).eval()

    hidden_size = talker.config.hidden_size
    num_layers = talker.config.num_hidden_layers
    B, S = 1, 32

    dummy_embeds = torch.randn(B, S, hidden_size, device=device, dtype=ONNX_EXPORT_DTYPE)
    position_ids = torch.arange(S, device=device, dtype=torch.long)
    position_ids = position_ids.unsqueeze(0).unsqueeze(0).expand(3, B, S)

    with torch.no_grad():
        out = fused(dummy_embeds, position_ids)

    codec_sum, full_codec, last_hidden, last_logits = out[0], out[1], out[2], out[3]
    logger.info(
        f"  Context fused shapes: codec_sum={codec_sum.shape}, full_codec={full_codec.shape}, "
        f"last_hidden={last_hidden.shape}, last_logits={last_logits.shape}, present_kv layers={num_layers}"
    )

    output_names = ["codec_sum", "full_codec", "last_hidden", "last_logits"]
    for i in range(num_layers):
        output_names.append(f"present_kv_{i}_k")
        output_names.append(f"present_kv_{i}_v")

    dynamic_axes = {
        "input_embeds": {0: "batch", 1: "seq_len"},
        "position_ids": {0: "three", 1: "batch", 2: "seq_len"},
        "codec_sum": {0: "batch"},
        "full_codec": {0: "batch"},
        "last_hidden": {0: "batch"},
        "last_logits": {0: "batch"},
    }
    for i in range(num_layers):
        dynamic_axes[f"present_kv_{i}_k"] = {0: "batch", 2: "seq_len"}
        dynamic_axes[f"present_kv_{i}_v"] = {0: "batch", 2: "seq_len"}

    onnx_path = str(output_dir / "talker_context.onnx")
    export_onnx(
        model=fused,
        dummy_inputs=(dummy_embeds, position_ids),
        input_names=["input_embeds", "position_ids"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        onnx_path=onnx_path,
        opset_version=opset_version,
        simplify=True,
    )

    test_inputs = {
        "input_embeds": to_numpy(dummy_embeds),
        "position_ids": position_ids.cpu().numpy(),
    }
    torch_outputs = {output_names[i]: to_numpy(out[i]) for i in range(len(output_names))}
    atol = 2e-3 if ONNX_EXPORT_DTYPE == torch.float32 else 1e-1
    ok = verify_onnx(onnx_path, test_inputs, torch_outputs, atol=atol, rtol=1e-2)
    if ok:
        logger.info("  Context fused ONNX verification PASSED")
    else:
        logger.warning("  Context fused ONNX verification had differences (argmax path)")

    return onnx_path


def export_talker_context(
    variant: str,
    models_dir: str = None,
    output_dir: str = None,
    device: str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> dict:
    """Export Talker context (prefill) to ONNX with KV cache outputs."""
    model_path = resolve_model_path(variant, models_dir)
    out_dir = ensure_output_dir(output_dir, variant)

    logger.info(f"Loading model: {variant} from {model_path}")
    model = load_tts_model(model_path, device="cpu", dtype=torch.float32)

    # Ensure eager attention for ONNX-friendly ops (avoids SDPA/vmap in mask)
    talker_config = model.talker.model.config
    setattr(talker_config, "_attn_implementation", "eager")

    onnx_path = _export_talker_context_onnx(model, variant, out_dir, device=device)

    del model
    if device != "cpu":
        torch.cuda.empty_cache()

    return {"onnx": onnx_path}


def main():
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Export Talker Backbone context (prefill) to ONNX with KV cache"
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Model variant. Default: export all variants",
    )
    parser.add_argument("--verify", action="store_true", help="Run verification only (no export)")
    add_common_args(parser)
    args = parser.parse_args()

    device = resolve_device(args.device)
    variants = [args.variant] if args.variant else list(MODEL_VARIANTS.keys())

    for variant in variants:
        if variant not in MODEL_VARIANTS:
            logger.error(f"Unknown variant: {variant}")
            continue
        try:
            results = export_talker_context(
                variant,
                args.models_dir,
                args.output_dir,
                device,
            )
            for key, path in results.items():
                logger.info(f"[{variant}] Talker context {key}: {path}")
        except FileNotFoundError as e:
            logger.warning(f"[{variant}] Skipped: {e}")
        except Exception as e:
            logger.error(f"[{variant}] Failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()

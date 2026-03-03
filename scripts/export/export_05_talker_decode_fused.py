#!/usr/bin/env python3
"""
[Step 05] Export Fused Talker Decode (1 step) + Code Predictor + Codec Sum to ONNX.

Deprecated: Prefer export_04_talker_unified.py for single-engine deployment (no weight duplication).

Single decode step: Talker 1-token forward (KV cache reuse) → argmax → CP 15-step
→ codec_sum. All in one ONNX for minimal GPU round-trips.

Inputs:  input_embeds [B, 1, H], position_id [3, B, 1],
         past_kv_{i}_k, past_kv_{i}_v (i=0..num_layers-1)
Outputs: codec_sum [B, 1, H], full_codec [B, 16], hidden [B, 1, H], logits [B, 1, V],
         present_kv_{i}_k, present_kv_{i}_v

See plan: Fused TRT Decode Engine with Self-Managed KV Cache, Phase 2.
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
    add_common_args,
    MODEL_VARIANTS,
    ONNX_EXPORT_DTYPE,
    CodePredictorUnrolled,
)

logger = logging.getLogger("onnx_export")


class DecodeKVCache:
    """Cache that concatenates past KV with new KV per layer (for decode step)."""

    def __init__(self, past_key_values: List[Tuple[torch.Tensor, torch.Tensor]]):
        # past_key_values: list of (past_k [B,kv_heads,S,head_dim], past_v)
        self._past = list(past_key_values)

    def get_seq_length(self) -> int:
        if self._past[0] is None or self._past[0][0] is None:
            return 0
        return self._past[0][0].shape[2]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        past_k, past_v = self._past[layer_idx]
        full_k = torch.cat([past_k, key_states], dim=2)
        full_v = torch.cat([past_v, value_states], dim=2)
        self._past[layer_idx] = (full_k, full_v)
        return full_k, full_v

    def get_present(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._past[layer_idx]


class TalkerDecodeStepONNX(nn.Module):
    """Single Talker decode step with KV cache: 1 token in, hidden + logits + present_kv out."""

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
        *past_key_values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, ...]:
        # input_embeds: [B, 1, H], position_ids: [3, B, 1]
        # past_key_values: 2*num_layers tensors (k0, v0, k1, v1, ...)
        B, one, H = input_embeds.shape
        device = input_embeds.device
        dtype = input_embeds.dtype
        n = self.num_layers
        past_list = [
            (past_key_values[2 * i], past_key_values[2 * i + 1])
            for i in range(n)
        ]
        cache = DecodeKVCache(past_list)
        S_total = cache.get_seq_length() + 1
        position_embeddings = self.rotary_emb(input_embeds, position_ids)
        causal_mask = torch.zeros(
            B, 1, 1, S_total, device=device, dtype=dtype
        )
        text_position_ids = position_ids[0] if position_ids.dim() == 3 else position_ids
        cache_position = torch.arange(
            S_total - 1, S_total, device=device, dtype=torch.long
        )

        hidden = input_embeds
        for layer in self.layers:
            layer_out = layer(
                hidden,
                attention_mask=causal_mask,
                position_ids=text_position_ids,
                past_key_values=cache,
                output_attentions=False,
                use_cache=True,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            hidden = layer_out[0]

        hidden = self.norm(hidden)
        logits = self.codec_head(hidden)
        outputs = [hidden, logits]
        for i in range(n):
            k, v = cache.get_present(i)
            outputs.append(k)
            outputs.append(v)
        return tuple(outputs)


class TalkerDecodeFusedONNX(nn.Module):
    """Fused: Talker decode step + argmax + Code Predictor (15-step) + Codec Embedding Sum."""

    def __init__(
        self,
        talker_decode_step: TalkerDecodeStepONNX,
        code_predictor_unrolled: CodePredictorUnrolled,
        codec_embedding_sum: nn.Module,
    ):
        super().__init__()
        self.talker_decode = talker_decode_step
        self.cp = code_predictor_unrolled
        self.codec_sum = codec_embedding_sum
        self.num_layers = talker_decode_step.num_layers

    def forward(
        self,
        input_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        *past_key_values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, ...]:
        # ① Talker decode step
        talker_out = self.talker_decode(input_embeds, position_ids, *past_key_values)
        hidden = talker_out[0]
        logits = talker_out[1]
        present_kv = list(talker_out[2:])

        # ② argmax → codec_token_0
        codec_token_0 = logits[:, -1, :].argmax(dim=-1)

        # ③ Code Predictor 15-step unrolled
        cp_tokens = self.cp(hidden, codec_token_0)

        # ④ full_codec [B, 16] (int64 for embedding lookup)
        full_codec = torch.cat(
            [codec_token_0.unsqueeze(1), cp_tokens.long()], dim=1
        )

        # ⑤ Codec Embedding Sum → codec_sum [B, H] → [B, 1, H]
        codec_sum = self.codec_sum(full_codec).unsqueeze(1)

        return (codec_sum, full_codec, hidden, logits, *present_kv)


def _build_codec_embedding_sum_from_model(model, device, dtype=torch.float32):
    """Build CodecEmbeddingSum from live model (stacked [16, 3072, H])."""
    try:
        from codec_embedding_sum import CodecEmbeddingSum
    except ImportError:
        repo_root = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(repo_root / "scripts" / "python"))
        from codec_embedding_sum import CodecEmbeddingSum
    return CodecEmbeddingSum.from_model(model, dtype=dtype).to(device).eval()


def _export_fused_decode_onnx(
    model,
    variant: str,
    output_dir: Path,
    device: str = "cpu",
    opset_version: int = 18,
) -> str:
    """Export fused decode ONNX."""
    talker = model.talker
    talker_config = talker.model.config
    setattr(talker_config, "_attn_implementation", "eager")
    if hasattr(talker.code_predictor, "model") and hasattr(talker.code_predictor.model, "config"):
        setattr(talker.code_predictor.model.config, "_attn_implementation", "eager")

    talker_model = talker.model.to(device).eval()
    codec_head = talker.codec_head.to(device).eval()
    talker_decode = TalkerDecodeStepONNX(talker_model, codec_head).to(device).eval()

    code_predictor = talker.code_predictor.to(device).eval()
    talker_codec_emb = talker.model.codec_embedding.to(device).eval()
    cp_unrolled = CodePredictorUnrolled(code_predictor, talker_codec_emb).to(
        device
    ).eval()

    codec_sum_module = _build_codec_embedding_sum_from_model(model, device)

    fused = TalkerDecodeFusedONNX(talker_decode, cp_unrolled, codec_sum_module).to(
        device
    ).eval()

    num_layers = talker_config.num_hidden_layers
    hidden_size = talker_config.hidden_size
    num_kv_heads = talker_config.num_key_value_heads
    head_dim = getattr(
        talker_config, "head_dim", talker_config.hidden_size // talker_config.num_attention_heads
    )
    B = 1
    S_past = 4

    dummy_embeds = torch.randn(B, 1, hidden_size, device=device, dtype=ONNX_EXPORT_DTYPE)
    position_ids = torch.zeros(3, B, 1, device=device, dtype=torch.long)
    position_ids[0, 0, 0] = S_past
    position_ids[1, 0, 0] = S_past
    position_ids[2, 0, 0] = S_past

    past_list = []
    for _ in range(num_layers):
        past_list.append(
            torch.randn(B, num_kv_heads, S_past, head_dim, device=device, dtype=ONNX_EXPORT_DTYPE)
        )
        past_list.append(
            torch.randn(B, num_kv_heads, S_past, head_dim, device=device, dtype=ONNX_EXPORT_DTYPE)
        )

    with torch.no_grad():
        out = fused(dummy_embeds, position_ids, *past_list)

    codec_sum, full_codec, hidden, logits = out[0], out[1], out[2], out[3]
    logger.info(
        f"  Fused decode shapes: codec_sum={codec_sum.shape}, full_codec={full_codec.shape}, "
        f"hidden={hidden.shape}, logits={logits.shape}"
    )

    output_names = ["codec_sum", "full_codec", "hidden", "logits"]
    for i in range(num_layers):
        output_names.append(f"present_kv_{i}_k")
        output_names.append(f"present_kv_{i}_v")

    input_names = ["input_embeds", "position_ids"]
    for i in range(num_layers):
        input_names.append(f"past_kv_{i}_k")
        input_names.append(f"past_kv_{i}_v")

    dynamic_axes = {
        "input_embeds": {0: "batch"},
        "position_ids": {0: "three", 1: "batch"},
        "codec_sum": {0: "batch"},
        "full_codec": {0: "batch"},
        "hidden": {0: "batch"},
        "logits": {0: "batch"},
    }
    for i in range(num_layers):
        dynamic_axes[f"past_kv_{i}_k"] = {0: "batch", 2: "S_past"}
        dynamic_axes[f"past_kv_{i}_v"] = {0: "batch", 2: "S_past"}
        dynamic_axes[f"present_kv_{i}_k"] = {0: "batch", 2: "S_past_plus_1"}
        dynamic_axes[f"present_kv_{i}_v"] = {0: "batch", 2: "S_past_plus_1"}

    onnx_path = str(output_dir / "talker_decode_fused.onnx")
    dummy_inputs = (dummy_embeds, position_ids, *past_list)
    export_onnx(
        model=fused,
        dummy_inputs=dummy_inputs,
        input_names=input_names,
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
    for i, t in enumerate(past_list):
        test_inputs[input_names[2 + i]] = to_numpy(t)
    torch_outputs = {output_names[i]: to_numpy(out[i]) for i in range(len(output_names))}
    atol = 2e-3 if ONNX_EXPORT_DTYPE == torch.float32 else 1e-1
    ok = verify_onnx(onnx_path, test_inputs, torch_outputs, atol=atol, rtol=1e-2)
    if ok:
        logger.info("  Fused decode ONNX verification PASSED")
    else:
        logger.warning("  Fused decode ONNX verification had differences (argmax path)")

    return onnx_path


def export_talker_decode_fused(
    variant: str,
    models_dir: str = None,
    output_dir: str = None,
    device: str = "cpu",
) -> dict:
    """Export fused Talker decode + CP + codec_sum to ONNX."""
    model_path = resolve_model_path(variant, models_dir)
    out_dir = ensure_output_dir(output_dir, variant)

    logger.info(f"Loading model: {variant} from {model_path}")
    model = load_tts_model(model_path, device="cpu", dtype=torch.float32)

    onnx_path = _export_fused_decode_onnx(model, variant, out_dir, device=device)

    del model
    if device != "cpu":
        torch.cuda.empty_cache()

    return {"onnx": onnx_path}


def main():
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Export Fused Talker Decode + CP + Codec Sum to ONNX"
    )
    parser.add_argument("--variant", type=str, default=None)
    add_common_args(parser)
    args = parser.parse_args()

    device = resolve_device(args.device)
    variants = [args.variant] if args.variant else list(MODEL_VARIANTS.keys())

    for variant in variants:
        if variant not in MODEL_VARIANTS:
            logger.error(f"Unknown variant: {variant}")
            continue
        try:
            results = export_talker_decode_fused(
                variant, args.models_dir, args.output_dir, device
            )
            for key, path in results.items():
                logger.info(f"[{variant}] Fused decode {key}: {path}")
        except FileNotFoundError as e:
            logger.warning(f"[{variant}] Skipped: {e}")
        except Exception as e:
            logger.error(f"[{variant}] Failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()

"""
Shared Talker ONNX building blocks: backbone (layers + codec_head) and fused (+ CP + codec sum).

Used by export_07_talker_backbone, export_08_talker_unified, export_09_talker_code2wav_fused.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple, TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    pass

from utils import CodePredictorUnrolled


class UnifiedKVCache:
    """Single cache that concats past + new KV per layer. S_past=0: cat([], new) -> new."""

    def __init__(self, past_key_values: List[Tuple[torch.Tensor, torch.Tensor]]):
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


def build_codec_embedding_sum_from_model(model, device, dtype=torch.float32):
    """Build CodecEmbeddingSum from live model (stacked [16, 3072, H])."""
    try:
        from codec_embedding_sum import CodecEmbeddingSum
    except ImportError:
        repo_root = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(repo_root / "scripts" / "python"))
        from codec_embedding_sum import CodecEmbeddingSum
    return CodecEmbeddingSum.from_model(model, dtype=dtype).to(device).eval()


class TalkerUnifiedONNX(nn.Module):
    """Talker backbone for both prefill and decode: single path with dynamic S and S_past."""

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
        *inputs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, ...]:
        B, S, H = input_embeds.shape
        device = input_embeds.device
        dtype = input_embeds.dtype
        n = self.num_layers
        if position_ids.dim() == 3 and position_ids.size(0) != 3:
            position_ids = position_ids.permute(1, 0, 2)
        attention_bias = None
        past_seq_lens = None
        if len(inputs) == 2 * n:
            past_key_values = inputs
        elif len(inputs) == 2 + 2 * n:
            attention_bias = inputs[0]
            past_seq_lens = inputs[1]
            past_key_values = inputs[2:]
        else:
            raise ValueError(
                f"Expected {2 * n} or {2 + 2 * n} extra tensors, got {len(inputs)}"
            )
        past_list = [
            (past_key_values[2 * i], past_key_values[2 * i + 1])
            for i in range(n)
        ]
        cache = UnifiedKVCache(past_list)
        S_past = cache.get_seq_length()
        S_total = S_past + S

        position_embeddings = self.rotary_emb(input_embeds, position_ids)

        row_idx = torch.arange(S, device=device, dtype=torch.long).unsqueeze(1) + S_past
        col_idx = torch.arange(S_total, device=device, dtype=torch.long).unsqueeze(0)
        neg_val = -1.0e4
        causal_mask = (col_idx > row_idx).to(dtype=dtype) * neg_val
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(B, 1, S, S_total)
        if attention_bias is not None:
            causal_mask = causal_mask + attention_bias.to(device=device, dtype=dtype)
        if past_seq_lens is not None:
            # Keep past_seq_lens as a live ONNX input for BLS bookkeeping while
            # letting attention_bias carry the actual padded-key masking semantics.
            causal_mask = causal_mask + (
                past_seq_lens.to(device=device, dtype=dtype).sum() * 0.0
            )

        text_position_ids = position_ids[0] if position_ids.dim() == 3 else position_ids
        cache_position = torch.arange(S_past, S_past + S, device=device, dtype=torch.long)

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


class TalkerUnifiedFusedONNX(nn.Module):
    """Talker (prefill or decode) + argmax + CP 15-step + Codec Embedding Sum."""

    def __init__(
        self,
        talker_unified: TalkerUnifiedONNX,
        code_predictor_unrolled: CodePredictorUnrolled,
        codec_embedding_sum: nn.Module,
    ):
        super().__init__()
        self.talker_unified = talker_unified
        self.cp = code_predictor_unrolled
        self.codec_sum = codec_embedding_sum
        self.num_layers = talker_unified.num_layers

    def forward(
        self,
        input_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        *inputs: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        talker_out = self.talker_unified(input_embeds, position_ids, *inputs)
        hidden = talker_out[0]
        logits = talker_out[1]
        present_kv = list(talker_out[2:])

        codec_token_0 = logits[:, -1, :].argmax(dim=-1)
        cp_tokens = self.cp(hidden[:, -1:, :], codec_token_0)
        full_codec = torch.cat(
            [codec_token_0.unsqueeze(1), cp_tokens.long()], dim=1
        )
        codec_sum = self.codec_sum(full_codec).unsqueeze(1)

        return (codec_sum, full_codec, hidden, logits, *present_kv)


def build_talker_backbone_module(model, device: str = "cpu") -> Tuple[TalkerUnifiedONNX, int, int, int, int]:
    """Build TalkerUnifiedONNX only (no CP / codec sum)."""
    talker = model.talker
    talker_config = talker.model.config
    setattr(talker_config, "_attn_implementation", "eager")

    talker_model = talker.model.to(device).eval()
    codec_head = talker.codec_head.to(device).eval()
    backbone = TalkerUnifiedONNX(talker_model, codec_head).to(device).eval()

    num_layers = talker_config.num_hidden_layers
    hidden_size = talker_config.hidden_size
    num_kv_heads = talker_config.num_key_value_heads
    head_dim = getattr(
        talker_config, "head_dim",
        talker_config.hidden_size // talker_config.num_attention_heads,
    )
    return backbone, num_layers, hidden_size, num_kv_heads, head_dim


def build_talker_unified_fused_module(model, device: str = "cpu") -> Tuple[TalkerUnifiedFusedONNX, int, int, int, int]:
    """Build TalkerUnifiedFusedONNX (prefill+decode+CP+codec_sum)."""
    talker = model.talker
    talker_config = talker.model.config
    setattr(talker_config, "_attn_implementation", "eager")
    if hasattr(talker.code_predictor, "model") and hasattr(talker.code_predictor.model, "config"):
        setattr(talker.code_predictor.model.config, "_attn_implementation", "eager")

    talker_model = talker.model.to(device).eval()
    codec_head = talker.codec_head.to(device).eval()
    backbone = TalkerUnifiedONNX(talker_model, codec_head).to(device).eval()

    code_predictor = talker.code_predictor.to(device).eval()
    talker_codec_emb = talker.model.codec_embedding.to(device).eval()
    cp_unrolled = CodePredictorUnrolled(code_predictor, talker_codec_emb).to(device).eval()

    codec_sum_module = build_codec_embedding_sum_from_model(model, device)

    fused = TalkerUnifiedFusedONNX(backbone, cp_unrolled, codec_sum_module).to(device).eval()

    num_layers = talker_config.num_hidden_layers
    hidden_size = talker_config.hidden_size
    num_kv_heads = talker_config.num_key_value_heads
    head_dim = getattr(
        talker_config, "head_dim",
        talker_config.hidden_size // talker_config.num_attention_heads,
    )
    return fused, num_layers, hidden_size, num_kv_heads, head_dim

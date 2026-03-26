"""
Stateful code2wav streaming wrapper for ONNX/TRT export.

Unified chunk_T=4: input codes [B, 16, 4], output wav [B, 7680] + 37 updated state tensors.
Uses SlidingWindowKVCache (window=72), streaming CausalConvNet (left-context buffer),
and streaming CausalTransConvNet (overlap-add) for incremental decode.
"""

from __future__ import annotations

import types
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F


# -----------------------------------------------------------------------------
# Sliding-window KV cache (concat past + new, then truncate to last 72)
# -----------------------------------------------------------------------------


class SlidingWindowKVCache:
    """KV cache that truncates to last window_size positions after each update."""

    def __init__(
        self,
        past_key_values: List[Tuple[torch.Tensor, torch.Tensor]],
        window_size: int = 72,
    ):
        self._past = list(past_key_values)
        self.window_size = window_size

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
        if self.window_size is not None and self.window_size > 0:
            full_k = full_k[:, :, -self.window_size :, :]
            full_v = full_v[:, :, -self.window_size :, :]
        self._past[layer_idx] = (full_k, full_v)
        return full_k, full_v

    def get_present(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._past[layer_idx]


# -----------------------------------------------------------------------------
# Streaming causal conv: left-context buffer instead of zero-pad
# -----------------------------------------------------------------------------


def streaming_causal_conv(
    conv_module: nn.Module,
    x: torch.Tensor,
    state: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run causal conv with left context from state. state shape [B, C_in, padding].
    Returns (out [B, C_out, T], new_state [B, C_in, padding]).
    """
    padding = conv_module.padding  # left padding length
    full = torch.cat([state, x], dim=-1)
    out = conv_module.conv(full)
    new_state = full[..., -padding:]
    return out, new_state


# -----------------------------------------------------------------------------
# Streaming causal transconv: overlap-add at chunk boundary
# -----------------------------------------------------------------------------


def streaming_causal_transconv(
    transconv_module: nn.Module,
    x: torch.Tensor,
    overlap_state: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run causal transconv and overlap-add with previous chunk tail.
    overlap_state shape [B, C_out, right_pad]. right_pad = kernel_size - stride.
    overlap_state is bias-free to avoid double-counting bias in the overlap region.
    Returns (out [B, C_out, T*stride], new_overlap [B, C_out, right_pad]).
    Note: this implementation avoids https://github.com/NVIDIA/TensorRT-Incubator/issues/565 @ 2026.03.26
    """
    raw = transconv_module.conv(x)  # length (M-1)*S + K, includes bias
    rp = transconv_module.right_pad
    if rp > 0:
        # Functional form (no in-place slice write) is more stable for ONNX/TRT.
        head = raw[..., :rp] + overlap_state
        body = raw[..., rp:-rp]
        output = torch.cat([head, body], dim=-1)
    else:
        output = raw
    if rp > 0:
        new_overlap = raw[..., -rp:].clone()
        bias = getattr(transconv_module.conv, "bias", None)
        if bias is not None:
            new_overlap = new_overlap - bias.view(1, -1, 1)
    else:
        new_overlap = overlap_state
    return output, new_overlap


# -----------------------------------------------------------------------------
# Code2Wav streaming wrapper (chunk_T=4 fixed)
# -----------------------------------------------------------------------------

# State layout: 2 * num_hidden_layers KV + NUM_CONV + NUM_TRANSCONV (e.g. 8 layers → 37 tensors).
NUM_CONV = 17
NUM_TRANSCONV = 4
CHUNK_T = 4
SAMPLES_PER_CHUNK = CHUNK_T * 1920  # 7680
COLD_START_DUMMY_PAST_LEN = 1

def num_code2wav_hidden_layers(decoder: nn.Module) -> int:
    """Match the actual pre_transformer depth (checkpoint may disagree with config)."""
    pt = getattr(decoder, "pre_transformer", None)
    layers = getattr(pt, "layers", None) if pt is not None else None
    if layers is not None and len(layers) > 0:
        return int(len(layers))
    return int(getattr(decoder.config, "num_hidden_layers", 8))


def count_code2wav_state_tensors(decoder: nn.Module) -> int:
    """Total streaming state tensors (KV + conv + transconv)."""
    return 2 * num_code2wav_hidden_layers(decoder) + NUM_CONV + NUM_TRANSCONV


def get_initial_state_shapes(
    decoder: nn.Module,
    batch_size: int = 1,
    past_kv_len: int = COLD_START_DUMMY_PAST_LEN,
):
    """Return list of (name, shape) for code2wav streaming state tensors. Used for ONNX export."""
    cfg = decoder.config
    codebook_dim = getattr(cfg, "codebook_dim", 512)
    latent_dim = cfg.latent_dim
    decoder_dim = cfg.decoder_dim
    num_kv_heads = cfg.num_key_value_heads
    head_dim = getattr(cfg, "head_dim", getattr(cfg, "hidden_size", 512) // cfg.num_attention_heads)
    B = batch_size
    n_layers = num_code2wav_hidden_layers(decoder)
    shapes = []
    for i in range(n_layers):
        shapes.append((f"past_kv_{i}_k", (B, num_kv_heads, past_kv_len, head_dim)))
        shapes.append((f"past_kv_{i}_v", (B, num_kv_heads, past_kv_len, head_dim)))
    shapes.append(("conv_state_0", (B, codebook_dim, 2)))
    shapes.append(("conv_state_1", (B, latent_dim, 6)))
    shapes.append(("conv_state_2", (B, latent_dim, 6)))
    shapes.append(("conv_state_3", (B, latent_dim, 6)))
    for block_idx in range(4):
        out_dim = decoder_dim // (2 ** (block_idx + 1))
        shapes.append((f"conv_state_{4+block_idx*3+0}", (B, out_dim, 6)))
        shapes.append((f"conv_state_{4+block_idx*3+1}", (B, out_dim, 18)))
        shapes.append((f"conv_state_{4+block_idx*3+2}", (B, out_dim, 54)))
    shapes.append(("conv_state_16", (B, decoder_dim // (2 ** 4), 6)))
    for block_idx in range(4):
        out_dim = decoder_dim // (2 ** (block_idx + 1))
        rp = [8, 5, 4, 3][block_idx]
        shapes.append((f"transconv_overlap_{block_idx}", (B, out_dim, rp)))
    return shapes


def create_initial_states(decoder: nn.Module, device: torch.device, dtype: torch.dtype, batch_size: int = 1):
    """Create zero-initialized state tensors for the streaming wrapper."""
    shapes = get_initial_state_shapes(
        decoder,
        batch_size=batch_size,
        past_kv_len=COLD_START_DUMMY_PAST_LEN,
    )
    return [torch.zeros(s, device=device, dtype=dtype) for _, s in shapes]


def freeze_quantizer_codebooks_for_export(decoder: nn.Module) -> None:
    """
    Materialize codebook embedding tables for export so ONNX graph does not carry
    per-call clamp/divide subgraphs from cluster_usage + embedding_sum.
    """
    quantizer = getattr(decoder, "quantizer", None)
    if quantizer is None:
        return

    rvq_modules = []
    for name in ("rvq_first", "rvq_rest"):
        mod = getattr(quantizer, name, None)
        if mod is not None:
            rvq_modules.append(mod)

    for rvq in rvq_modules:
        vq = getattr(rvq, "vq", None)
        layers = getattr(vq, "layers", None)
        if layers is None:
            continue
        for layer in layers:
            codebook = getattr(layer, "_codebook", None)
            if codebook is None:
                continue
            with torch.no_grad():
                embedding = codebook.embedding_sum / codebook.cluster_usage.clamp(min=codebook.epsilon).unsqueeze(1)
            if hasattr(codebook, "inference_embedding"):
                codebook.inference_embedding = embedding
            else:
                codebook.register_buffer("inference_embedding", embedding)
            if not hasattr(codebook, "_orig_decode_for_export"):
                codebook._orig_decode_for_export = codebook.decode

            def _decode_with_frozen_embedding(self, codes: torch.Tensor) -> torch.Tensor:
                return F.embedding(codes, self.inference_embedding)

            codebook.decode = types.MethodType(_decode_with_frozen_embedding, codebook)


class Code2WavStreamingWrapper(nn.Module):
    """
    Stateful code2wav for ONNX/TRT: codes [B,16,T] + cache_position + c2w_attention_bias + state tensors,
    output wav [B, 7680] + updated states (layout matches get_initial_state_shapes).
    """

    def __init__(self, decoder: nn.Module, window_size: int = 72):
        super().__init__()
        self.decoder = decoder
        freeze_quantizer_codebooks_for_export(self.decoder)
        self.window_size = window_size
        cfg = decoder.config
        self.num_layers = num_code2wav_hidden_layers(decoder)
        # KV states are 2 * num_layers; conv/transconv follow immediately (not a fixed 16-slot pad).
        self._num_kv_state_tensors = 2 * self.num_layers
        self.hidden_size = getattr(cfg, "hidden_size", 512)
        self.latent_dim = cfg.latent_dim
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = getattr(cfg, "head_dim", self.hidden_size // cfg.num_attention_heads)
        # Force eager attention for ONNX
        if hasattr(cfg, "_attn_implementation"):
            self._saved_attn_impl = getattr(cfg, "_attn_implementation", None)
        setattr(cfg, "_attn_implementation", "eager")

    def forward(
        self,
        codes: torch.Tensor,
        cache_position: torch.Tensor,
        c2w_attention_bias: torch.Tensor,
        *states: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        """
        codes: [B, 16, T], cache_position: [T] or [B, T] (absolute positions for this chunk).
        c2w_attention_bias: [B, 1, T, T_pad+T] additive mask built outside graph.
        states: 2*num_hidden_layers KV + 17 conv + 4 transconv.
        Returns: (wav [B, 7680], *new_states).
        """
        B = codes.shape[0]
        device = codes.device
        dtype = codes.dtype if codes.dtype.is_floating_point else torch.get_default_dtype()

        # Unpack states
        kv_list = [(states[2 * i], states[2 * i + 1]) for i in range(self.num_layers)]
        off = self._num_kv_state_tensors
        conv_states = list(states[off : off + NUM_CONV])
        transconv_states = list(states[off + NUM_CONV :])

        # 1. Quantizer (stateless)
        hidden = self.decoder.quantizer.decode(codes)  # [B, codebook_dim, 4]

        # 2. Pre-conv streaming
        hidden, conv_states[0] = streaming_causal_conv(
            self.decoder.pre_conv, hidden, conv_states[0]
        )
        hidden = hidden.transpose(1, 2)  # [B, 4, latent_dim]

        # 3. Transformer with sliding-window KV cache
        cache = SlidingWindowKVCache(kv_list, self.window_size)

        hidden = self.decoder.pre_transformer.input_proj(hidden)  # [B, 4, hidden_size]
        # position_ids [1, 4] to match transformer's expectation (same as cache_position.unsqueeze(0))
        position_ids = cache_position.unsqueeze(0) if cache_position.dim() == 1 else cache_position
        if position_ids.shape[0] != B:
            position_ids = position_ids.expand(B, -1)
        position_embeddings = self.decoder.pre_transformer.rotary_emb(hidden, position_ids)

        causal_mask = c2w_attention_bias.to(device=device, dtype=dtype).contiguous()

        for layer in self.decoder.pre_transformer.layers:
            hidden = layer(
                hidden,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        hidden = self.decoder.pre_transformer.norm(hidden)
        hidden = self.decoder.pre_transformer.output_proj(hidden)  # [B, 4, latent_dim]
        hidden = hidden.permute(0, 2, 1)  # [B, latent_dim, 4]

        # 4. Upsample blocks (2x): transconv (no overlap) + ConvNeXt with streaming
        hidden = self._upsample_streaming(hidden, conv_states)

        # 5. Decoder: initial conv + 4 blocks + SnakeBeta + final conv
        hidden, conv_states[3] = streaming_causal_conv(
            self.decoder.decoder[0], hidden, conv_states[3]
        )
        for block_idx in range(4):
            hidden, conv_states, transconv_states = self._decoder_block_streaming(
                block_idx, hidden, conv_states, transconv_states
            )
        hidden = self.decoder.decoder[5](hidden)  # SnakeBeta
        wav, conv_states[16] = streaming_causal_conv(
            self.decoder.decoder[6], hidden, conv_states[16]
        )
        wav = wav.clamp(min=-1, max=1)

        # Pack new states
        new_kv = []
        for i in range(self.num_layers):
            k, v = cache.get_present(i)
            new_kv.append(k)
            new_kv.append(v)
        return (wav,) + tuple(new_kv) + tuple(conv_states) + tuple(transconv_states)

    def _upsample_streaming(
        self,
        hidden: torch.Tensor,
        conv_states: List[torch.Tensor],
    ) -> torch.Tensor:
        for i, blocks in enumerate(self.decoder.upsample):
            transconv, convnext = blocks[0], blocks[1]
            hidden = transconv(hidden)  # right_pad=0 for (2,2)
            inp = hidden
            hidden, conv_states[1 + i] = streaming_causal_conv(
                convnext.dwconv, hidden, conv_states[1 + i]
            )
            hidden = hidden.permute(0, 2, 1)
            hidden = convnext.norm(hidden)
            hidden = convnext.pwconv1(hidden)
            hidden = convnext.act(hidden)
            hidden = convnext.pwconv2(hidden)
            hidden = convnext.gamma * hidden
            hidden = hidden.permute(0, 2, 1)
            hidden = inp + hidden
        return hidden

    def _decoder_block_streaming(
        self,
        block_idx: int,
        hidden: torch.Tensor,
        conv_states: List[torch.Tensor],
        transconv_states: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        block = self.decoder.decoder[1 + block_idx].block
        hidden = block[0](hidden)  # SnakeBeta
        hidden, transconv_states[block_idx] = streaming_causal_transconv(
            block[1], hidden, transconv_states[block_idx]
        )
        for ru_idx in range(3):
            res_unit = block[2 + ru_idx]
            residual = hidden
            hidden = res_unit.act1(hidden)
            hidden, conv_states[4 + block_idx * 3 + ru_idx] = streaming_causal_conv(
                res_unit.conv1, hidden, conv_states[4 + block_idx * 3 + ru_idx]
            )
            hidden = res_unit.act2(hidden)
            hidden = res_unit.conv2(hidden)  # kernel 1, no state
            hidden = residual + hidden
        return hidden, conv_states, transconv_states

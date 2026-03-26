#!/usr/bin/env python3
"""
[Step 09] Export Talker Unified + Code2Wav (chunk_T=1) fused ONNX.

Single engine per decode step: talker (prefill+decode+CP+codec_sum) -> full_codec -> code2wav(1 frame) -> wav.
Inputs: input_embeds, position_ids, attention_bias, past_seq_lens, cache_position,
  c2w_attention_bias, code2wav state tensors
  (2 * decoder.num_hidden_layers KV + 17 conv + 4 transconv; often 37 when n_layers=8).
Outputs: wav, codec_sum, full_codec, hidden, logits, present_kv_*, new code2wav states.

Depends on: tokenizer (code2wav decoder) + TTS variant (talker). Run export_06 (code2wav) + export_08 (talker unified) for verification order.

Also writes ``triton_manifest.json`` (includes ``code2wav_fused`` I/O layout for BLS + assemble); Phase C requires this file.

**Precision policy** (see ``docs/architecture.md``):

- **ONNX graph float I/O** is exported as **FP32** (``utils.ONNX_EXPORT_DTYPE``). This is the single source of truth for
  TensorRT **network** input/output tensor types unless you intentionally re-export with another I/O dtype.
- ``triton_manifest.json`` field ``engine_dtype`` (e.g. ``bf16``) refers to **TensorRT builder compute** (``trtexec --bf16``),
  not necessarily the dtype of I/O bindings.
- ``triton_io_float_dtype`` must match the ONNX/TRT engine I/O for ``talker_code2wav_fused`` (default ``fp32``). BLS / Triton
  ``config.pbtxt`` use this so **runtime tensor dtypes** stay consistent with the deployed graph.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn

_scripts_export = Path(__file__).resolve().parent
sys.path.insert(0, str(_scripts_export))
sys.path.insert(0, str(_scripts_export.parent / "python"))

from code2wav_streaming import (
    COLD_START_DUMMY_PAST_LEN,
    Code2WavStreamingWrapper,
    get_initial_state_shapes,
    NUM_CONV,
    NUM_TRANSCONV,
    num_code2wav_hidden_layers,
)
from talker_unified_modules import build_talker_unified_fused_module
from triton_manifest_io import build_manifest_for_export

from utils import (
    setup_logging,
    resolve_tokenizer_path,
    resolve_model_path,
    ensure_output_dir,
    load_speech_tokenizer,
    load_tts_model,
    patch_decoder_transconv_for_trt,
    export_onnx,
    verify_onnx,
    to_numpy,
    resolve_device,
    add_common_args,
    MODEL_VARIANTS,
    ONNX_EXPORT_DTYPE,
)

logger = logging.getLogger("onnx_export")

# Fused decode uses one codec frame per step (matches plan).
FUSED_CHUNK_T = 1


class TalkerCode2WavFusedONNX(nn.Module):
    """TalkerUnifiedFusedONNX + Code2WavStreamingWrapper (T=1 per forward)."""

    def __init__(self, talker_fused: nn.Module, code2wav: Code2WavStreamingWrapper):
        super().__init__()
        self.talker_fused = talker_fused
        self.code2wav = code2wav
        self.num_layers = talker_fused.num_layers

    def forward(
        self,
        input_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        attention_bias: torch.Tensor,
        past_seq_lens: torch.Tensor,
        cache_position: torch.Tensor,
        c2w_attention_bias: torch.Tensor,
        *inputs: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        """
        inputs: past_kv_* (talker) then code2wav state tensors (layout from decoder).
        cache_position: [B, FUSED_CHUNK_T] int64 (absolute frame indices for this chunk).
        """
        n = self.num_layers
        past_kv = inputs[: 2 * n]
        code2wav_states = inputs[2 * n :]

        codec_sum, full_codec, hidden, logits, *present_kv = self.talker_fused(
            input_embeds, position_ids, attention_bias, past_seq_lens, *past_kv
        )
        # Vocoder quantizer tables are 2048 rows per codebook; talker may emit specials (e.g. EOS)
        # outside this range — clamp so Gather in decoder stays in-bounds (matches safe decode path).
        _cb = 2048
        fc = full_codec.long().clamp(0, _cb - 1)
        codes = fc.unsqueeze(-1)
        c2w_out = self.code2wav(codes, cache_position, c2w_attention_bias, *code2wav_states)
        wav = c2w_out[0]
        new_c2w = c2w_out[1:]
        return (wav, codec_sum, full_codec, hidden, logits, *present_kv, *new_c2w)


def _export_talker_code2wav_fused_onnx(
    model,
    variant: str,
    output_dir: Path,
    device: str = "cpu",
    opset_version: int = 18,
    engine_dtype: str = "bf16",
    triton_io_float_dtype: str = "bf16",
) -> str:
    tokenizer_path = resolve_tokenizer_path(None)
    tokenizer_model = load_speech_tokenizer(tokenizer_path, device=device, dtype=torch.float32)
    decoder = tokenizer_model.decoder.to(device).eval()
    patch_decoder_transconv_for_trt(decoder)
    code2wav = Code2WavStreamingWrapper(decoder).to(device).eval()

    talker_fused, num_layers, hidden_size, num_kv_heads, head_dim = build_talker_unified_fused_module(
        model, device=device
    )
    fused = TalkerCode2WavFusedONNX(talker_fused, code2wav).to(device).eval()

    # Semantics aligned with export_05: [B, 1, H] + past length S_past; position triple = S_past.
    # Layout (B, 3, 1) matches Triton orchestrator (not export_05's (3, B, 1) tensor order).
    B, one, S_past = 1, 1, 0
    dummy_embeds = torch.randn(B, one, hidden_size, device=device, dtype=ONNX_EXPORT_DTYPE)
    position_ids = torch.full((B, 3, one), S_past, device=device, dtype=torch.long)
    attention_bias = torch.zeros(B, 1, one, S_past + one, device=device, dtype=ONNX_EXPORT_DTYPE)
    past_seq_lens = torch.full((B,), S_past, device=device, dtype=torch.long)

    past_list = []
    for _ in range(num_layers):
        past_list.append(
            torch.zeros(B, num_kv_heads, S_past, head_dim, device=device, dtype=ONNX_EXPORT_DTYPE)
        )
        past_list.append(
            torch.zeros(B, num_kv_heads, S_past, head_dim, device=device, dtype=ONNX_EXPORT_DTYPE)
        )

    # First fused frame index in the vocoder stream (chunk_T=1).
    cache_position = torch.zeros(B, FUSED_CHUNK_T, device=device, dtype=torch.long)
    # Code2wav states: use past_kv_len>0 for trace only (export_06 pattern). Zero-length KV
    # tensors are often dropped from the ONNX graph; runtime still uses S_past=0 via dynamic axes.
    TRACE_C2W_PAST_LEN = 1
    c2w_attention_bias = torch.zeros(
        B, 1, FUSED_CHUNK_T, TRACE_C2W_PAST_LEN + FUSED_CHUNK_T, device=device, dtype=ONNX_EXPORT_DTYPE
    )
    state_shapes = get_initial_state_shapes(
        decoder, batch_size=B, past_kv_len=TRACE_C2W_PAST_LEN
    )
    state_shapes_cold = get_initial_state_shapes(
        decoder, batch_size=B, past_kv_len=COLD_START_DUMMY_PAST_LEN
    )
    state_tensors = [torch.randn(s, device=device, dtype=ONNX_EXPORT_DTYPE) for _, s in state_shapes]

    dummy_inputs = (
        dummy_embeds,
        position_ids,
        attention_bias,
        past_seq_lens,
        cache_position,
        c2w_attention_bias,
        *past_list,
        *state_tensors,
    )

    with torch.no_grad():
        out = fused(*dummy_inputs)

    input_names = [
        "input_embeds",
        "position_ids",
        "attention_bias",
        "past_seq_lens",
        "cache_position",
        "c2w_attention_bias",
    ]
    for i in range(num_layers):
        input_names.append(f"past_kv_{i}_k")
        input_names.append(f"past_kv_{i}_v")
    for name, _ in state_shapes_cold:
        input_names.append(f"c2w_{name}")

    output_names = ["wav", "codec_sum", "full_codec", "hidden", "logits"]
    for i in range(num_layers):
        output_names.append(f"present_kv_{i}_k")
        output_names.append(f"present_kv_{i}_v")
    n_c2w = num_code2wav_hidden_layers(decoder)
    for i in range(n_c2w):
        output_names.append(f"c2w_present_kv_{i}_k")
        output_names.append(f"c2w_present_kv_{i}_v")
    for i in range(NUM_CONV):
        output_names.append(f"c2w_new_conv_state_{i}")
    for i in range(NUM_TRANSCONV):
        output_names.append(f"c2w_new_transconv_overlap_{i}")

    dynamic_axes = {
        "input_embeds": {0: "batch", 1: "seq"},
        "position_ids": {0: "batch", 1: "three", 2: "seq"},
        "attention_bias": {0: "batch", 2: "seq", 3: "key_total"},
        "past_seq_lens": {0: "batch"},
        "cache_position": {0: "batch", 1: "chunk_t"},
        "c2w_attention_bias": {0: "batch", 2: "chunk_t", 3: "c2w_key_total"},
        "wav": {0: "batch"},
        "codec_sum": {0: "batch"},
        "full_codec": {0: "batch"},
        "hidden": {0: "batch", 1: "seq"},
        "logits": {0: "batch", 1: "seq"},
    }
    for i in range(num_layers):
        dynamic_axes[f"past_kv_{i}_k"] = {0: "batch", 2: "S_past"}
        dynamic_axes[f"past_kv_{i}_v"] = {0: "batch", 2: "S_past"}
        dynamic_axes[f"present_kv_{i}_k"] = {0: "batch", 2: "S_total"}
        dynamic_axes[f"present_kv_{i}_v"] = {0: "batch", 2: "S_total"}
    for name, _ in state_shapes_cold:
        key = f"c2w_{name}"
        if "past_kv" in name:
            dynamic_axes[key] = {0: "batch", 2: "past_len"}
    for i in range(n_c2w):
        dynamic_axes[f"c2w_present_kv_{i}_k"] = {0: "batch", 2: "total_len"}
        dynamic_axes[f"c2w_present_kv_{i}_v"] = {0: "batch", 2: "total_len"}

    onnx_path = str(output_dir / "talker_code2wav_fused.onnx")
    export_onnx(
        model=fused,
        dummy_inputs=dummy_inputs,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        onnx_path=onnx_path,
        opset_version=opset_version,
        simplify=False,
        do_constant_folding=False,
    )

    c2w_out = []
    for i in range(n_c2w):
        c2w_out.append(f"c2w_present_kv_{i}_k")
        c2w_out.append(f"c2w_present_kv_{i}_v")
    for i in range(NUM_CONV):
        c2w_out.append(f"c2w_new_conv_state_{i}")
    for i in range(NUM_TRANSCONV):
        c2w_out.append(f"c2w_new_transconv_overlap_{i}")
    layout = {
        "num_code2wav_hidden_layers": n_c2w,
        "c2w_state_input_names": [f"c2w_{name}" for name, _ in state_shapes_cold],
        "c2w_state_output_names": c2w_out,
        "initial_state_shapes": [list(shape) for _, shape in state_shapes_cold],
    }

    weights_cfg_path = output_dir / "weights" / "config.json"
    weights_cfg: dict = {}
    if weights_cfg_path.is_file():
        with open(weights_cfg_path, encoding="utf-8") as f:
            weights_cfg = json.load(f)
    else:
        logger.warning(
            f"No {weights_cfg_path} — triton_manifest.json talker section uses export defaults"
        )
    manifest = build_manifest_for_export(
        variant,
        weights_cfg,
        layout,
        engine_mode="trt",
        engine_dtype=engine_dtype,
        triton_io_float_dtype=triton_io_float_dtype,
    )
    manifest_path = output_dir / "triton_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Wrote {manifest_path}")

    cpu_embeds = dummy_embeds.cpu()
    cpu_pos = position_ids.cpu()
    cpu_bias = attention_bias.cpu()
    cpu_past_seq_lens = past_seq_lens.cpu()
    cpu_cache = cache_position.cpu()
    cpu_c2w_bias = c2w_attention_bias.cpu()
    cpu_past = [t.cpu() for t in past_list]
    cpu_states = [t.cpu() for t in state_tensors]
    cpu_fused = fused.cpu().eval()
    with torch.no_grad():
        cpu_out = cpu_fused(
            cpu_embeds,
            cpu_pos,
            cpu_bias,
            cpu_past_seq_lens,
            cpu_cache,
            cpu_c2w_bias,
            *cpu_past,
            *cpu_states,
        )
    test_inputs = {
        "input_embeds": to_numpy(cpu_embeds),
        "position_ids": cpu_pos.numpy(),
        "attention_bias": to_numpy(cpu_bias),
        "past_seq_lens": cpu_past_seq_lens.numpy(),
        "cache_position": cpu_cache.numpy(),
        "c2w_attention_bias": to_numpy(cpu_c2w_bias),
    }
    off = 6
    for i, t in enumerate(cpu_past):
        test_inputs[input_names[off + i]] = to_numpy(t)
    off += len(cpu_past)
    for i, t in enumerate(cpu_states):
        test_inputs[input_names[off + i]] = to_numpy(t)

    torch_outputs = {output_names[i]: to_numpy(cpu_out[i]) for i in range(len(output_names))}
    ok = verify_onnx(onnx_path, test_inputs, torch_outputs, atol=2e-3, rtol=1e-2)

    # Case-2: alternate prefill/decode setting (B=1, non-zero logical past with explicit bias)
    B2, S_past2 = 1, 4
    c2w_past2 = 5
    cpu_embeds2 = torch.randn(B2, one, hidden_size, dtype=ONNX_EXPORT_DTYPE)
    cpu_pos2 = torch.full((B2, 3, one), S_past2, dtype=torch.long)
    cpu_bias2 = torch.zeros(B2, 1, one, S_past2 + one, dtype=ONNX_EXPORT_DTYPE)
    cpu_bias2[0, :, :, :2] = -1.0e4
    cpu_past_seq_lens2 = torch.tensor([3], dtype=torch.long)
    cpu_cache2 = torch.zeros(B2, FUSED_CHUNK_T, dtype=torch.long)
    cpu_c2w_bias2 = torch.zeros(B2, 1, FUSED_CHUNK_T, c2w_past2 + FUSED_CHUNK_T, dtype=ONNX_EXPORT_DTYPE)
    cpu_c2w_bias2[0, :, :, :3] = -1.0e4
    cpu_past2 = []
    for _ in range(num_layers):
        cpu_past2.append(torch.randn(B2, num_kv_heads, S_past2, head_dim, dtype=ONNX_EXPORT_DTYPE))
        cpu_past2.append(torch.randn(B2, num_kv_heads, S_past2, head_dim, dtype=ONNX_EXPORT_DTYPE))
    state_shapes2 = get_initial_state_shapes(decoder, batch_size=B2, past_kv_len=c2w_past2)
    cpu_states2 = [torch.randn(s, dtype=ONNX_EXPORT_DTYPE) for _, s in state_shapes2]
    with torch.no_grad():
        cpu_out2 = cpu_fused(
            cpu_embeds2,
            cpu_pos2,
            cpu_bias2,
            cpu_past_seq_lens2,
            cpu_cache2,
            cpu_c2w_bias2,
            *cpu_past2,
            *cpu_states2,
        )
    test_inputs2 = {
        "input_embeds": to_numpy(cpu_embeds2),
        "position_ids": cpu_pos2.numpy(),
        "attention_bias": to_numpy(cpu_bias2),
        "past_seq_lens": cpu_past_seq_lens2.numpy(),
        "cache_position": cpu_cache2.numpy(),
        "c2w_attention_bias": to_numpy(cpu_c2w_bias2),
    }
    off = 6
    for i, t in enumerate(cpu_past2):
        test_inputs2[input_names[off + i]] = to_numpy(t)
    off += len(cpu_past2)
    for i, t in enumerate(cpu_states2):
        test_inputs2[input_names[off + i]] = to_numpy(t)
    torch_outputs2 = {output_names[i]: to_numpy(cpu_out2[i]) for i in range(len(output_names))}
    ok2 = verify_onnx(onnx_path, test_inputs2, torch_outputs2, atol=2e-3, rtol=1e-2)

    if ok and ok2:
        logger.info("Talker+Code2Wav fused ONNX verification PASSED (base + alternate-prefill cases)")
    else:
        logger.warning("Talker+Code2Wav fused ONNX verification had differences")
    fused.to(device)

    del tokenizer_model
    if device != "cpu":
        torch.cuda.empty_cache()
    return onnx_path


def export_talker_code2wav_fused(
    variant: str,
    models_dir: str = None,
    output_dir: str = None,
    device: str = "cpu",
    engine_dtype: str = "bf16",
    triton_io_float_dtype: str = "bf16",
) -> dict:
    model_path = resolve_model_path(variant, models_dir)
    out_dir = ensure_output_dir(output_dir, variant)
    logger.info(f"Loading model: {variant} from {model_path}")
    model = load_tts_model(model_path, device="cpu", dtype=torch.float32)
    onnx_path = _export_talker_code2wav_fused_onnx(
        model,
        variant,
        out_dir,
        device=device,
        engine_dtype=engine_dtype,
        triton_io_float_dtype=triton_io_float_dtype,
    )
    del model
    if device != "cpu":
        torch.cuda.empty_cache()
    return {"onnx": onnx_path}


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Export Talker + Code2Wav fused ONNX")
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument(
        "--engine-dtype",
        type=str,
        default="bf16",
        choices=("bf16", "fp16", "fp32", "fp8"),
        help="TensorRT builder precision written to triton_manifest.json (trtexec --bf16/--fp16/--fp8)",
    )
    parser.add_argument(
        "--triton-io-float-dtype",
        type=str,
        default="bf16",
        choices=("bf16", "fp16", "fp32"),
        help="Float I/O binding + Triton TYPE_* (must match Phase B trtexec --inputIOFormats/--outputFormats)",
    )
    add_common_args(parser)
    args = parser.parse_args()
    device = resolve_device(args.device)
    variants = [args.variant] if args.variant else list(MODEL_VARIANTS.keys())
    for variant in variants:
        if variant not in MODEL_VARIANTS:
            logger.error(f"Unknown variant: {variant}")
            continue
        try:
            results = export_talker_code2wav_fused(
                variant,
                args.models_dir,
                args.output_dir,
                device,
                engine_dtype=args.engine_dtype,
                triton_io_float_dtype=args.triton_io_float_dtype,
            )
            for k, v in results.items():
                logger.info(f"[{variant}] {k}: {v}")
        except FileNotFoundError as e:
            logger.warning(f"[{variant}] Skipped: {e}")
        except Exception as e:
            logger.error(f"[{variant}] Failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()

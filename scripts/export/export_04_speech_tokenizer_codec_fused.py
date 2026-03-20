#!/usr/bin/env python3
"""
[Step 04] Export Speech Tokenizer Encoder + ICL ref codec sum (3D gather) to ONNX.

Fuses: waveform -> audio_codes (Mimi encoder) -> ref_codec_sum_vec [B, 1, H]
Matches PrefillBuilder ICL path: codec_embeddings_3d[g_idx, ref_codes, :].sum(dim=(0,1), keepdim=True).

Per-variant: stacked weights come from export_01 (codec_embeddings_3d.pt).
Shared: speech tokenizer from tokenizer checkpoint.

Usage: voice_clone_icl; replaces speech_tokenizer_encoder BLS + in-process 3D lookup.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn

from export_03_speech_tokenizer_encoder import SpeechTokenizerEncoderWrapper
from utils import (
    setup_logging,
    resolve_tokenizer_path,
    resolve_model_path,
    ensure_output_dir,
    load_speech_tokenizer,
    export_onnx,
    verify_onnx,
    to_numpy,
    resolve_device,
    resolve_dtype,
    add_common_args,
    ONNX_EXPORT_DTYPE,
)

logger = logging.getLogger("onnx_export")


class RefCodecSumFromAudioCodes(nn.Module):
    """Sum over (time, codebook) of stacked_3d[g, token_id, :]; output [B, 1, H]."""

    def __init__(self, stacked_3d: torch.Tensor):
        super().__init__()
        self.register_buffer("stacked_3d", stacked_3d)

    def forward(self, audio_codes: torch.Tensor) -> torch.Tensor:
        """
        audio_codes: [B, 16, T] int64
        Returns: [B, 1, H] same dtype as stacked_3d
        """
        B, G, Tlen = audio_codes.shape
        rc = audio_codes.permute(0, 2, 1).contiguous()
        g_idx = (
            torch.arange(G, device=audio_codes.device, dtype=torch.long)
            .view(1, 1, G)
            .expand(B, Tlen, G)
        )
        gathered = self.stacked_3d[g_idx, rc, :]
        return gathered.sum(dim=(1, 2), keepdim=True)


class SpeechTokenizerCodecFusedONNX(nn.Module):
    def __init__(self, speech_wrapper: SpeechTokenizerEncoderWrapper, stacked_3d: torch.Tensor):
        super().__init__()
        self.speech = speech_wrapper
        self.ref_sum = RefCodecSumFromAudioCodes(stacked_3d)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        codes = self.speech(waveform)
        return self.ref_sum(codes.long())


def _load_stacked_3d_for_variant(variant: str, models_dir: str, output_dir: str, device: str) -> torch.Tensor:
    """Load codec_embeddings_3d.pt produced by export_01 for this variant."""
    weights_dir = ensure_output_dir(output_dir, variant) / "weights"
    path = weights_dir / "codec_embeddings_3d.pt"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run export_01_embeddings.py for variant {variant} first."
        )
    t = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"Expected tensor in {path}, got {type(t)}")
    return t.to(device=device, dtype=torch.float32)


def export_speech_tokenizer_codec_fused(
    variant: str,
    models_dir: str = None,
    output_dir: str = None,
    device: str = "cpu",
    dtype: torch.dtype = ONNX_EXPORT_DTYPE,
) -> str:
    resolve_model_path(variant, models_dir)
    stacked = _load_stacked_3d_for_variant(variant, models_dir, output_dir, device)
    out_dir = ensure_output_dir(output_dir, variant)

    tokenizer_path = resolve_tokenizer_path(models_dir)
    logger.info(f"Loading speech tokenizer from {tokenizer_path}")
    tokenizer_model = load_speech_tokenizer(tokenizer_path, device=device, dtype=torch.float32)
    encoder = tokenizer_model.encoder.to(device).eval()
    speech_wrapper = SpeechTokenizerEncoderWrapper(encoder).to(device).eval()

    fused = SpeechTokenizerCodecFusedONNX(speech_wrapper, stacked).to(device).eval()

    B = 1
    sample_rate = 24000
    duration_sec = 3.0
    samples = int(sample_rate * duration_sec)
    dummy_wav = torch.randn(B, 1, samples, device=device, dtype=torch.float32)

    with torch.no_grad():
        ref_out = fused(dummy_wav)
    logger.info(f"Fused output shape: {ref_out.shape} (expected [1, 1, H])")

    onnx_path = str(out_dir / "speech_tokenizer_codec_fused.onnx")
    export_onnx(
        model=fused,
        dummy_inputs=(dummy_wav,),
        input_names=["waveform"],
        output_names=["ref_codec_sum_vec"],
        dynamic_axes={
            "waveform": {0: "batch", 2: "samples"},
            "ref_codec_sum_vec": {0: "batch"},
        },
        onnx_path=onnx_path,
        opset_version=18,
        simplify=True,
    )

    cpu_wav = dummy_wav.cpu()
    cpu_fused = fused.cpu().eval()
    with torch.no_grad():
        cpu_ref = cpu_fused(cpu_wav)
    fused.to(device)

    ok = verify_onnx(
        onnx_path,
        {"waveform": to_numpy(cpu_wav)},
        {"ref_codec_sum_vec": to_numpy(cpu_ref)},
        atol=1e-2,
        rtol=1e-2,
    )
    if ok:
        logger.info("Speech Tokenizer + Codec fused ONNX verification PASSED")
    else:
        logger.warning("Speech Tokenizer + Codec fused ONNX verification had differences")

    del tokenizer_model
    if device != "cpu":
        torch.cuda.empty_cache()
    return onnx_path


def main():
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Export Speech Tokenizer + ICL ref codec sum (3D) fused ONNX"
    )
    parser.add_argument("--variant", type=str, required=True, help="Model variant (base-* for ICL)")
    add_common_args(parser)
    args = parser.parse_args()

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)
    path = export_speech_tokenizer_codec_fused(
        args.variant, args.models_dir, args.output_dir, device, dtype
    )
    logger.info(f"Exported: {path}")


if __name__ == "__main__":
    main()

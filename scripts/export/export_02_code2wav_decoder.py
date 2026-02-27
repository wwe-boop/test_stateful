#!/usr/bin/env python3
"""
[Step 02] Export Code2Wav Decoder to ONNX.

Component: Code2Wav Decoder
Architecture: RVQ Dequant + Transformer(8L, sliding window) + BigVGAN ConvNet
Input:  codes [B, num_quantizers(16), T_codes]
Output: wav [B, T_codes * 1920]
Engine: ONNX Runtime
Usage: Called once per chunk (async, overlapped with decode loop)

The decoder is shared across all model variants (comes from the tokenizer checkpoint).
Supports chunked decoding with left_context overlap for seamless audio.
"""

import argparse
import logging

import torch

from utils import (
    setup_logging,
    resolve_tokenizer_path,
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


class Code2WavDecoderWrapper(torch.nn.Module):
    """Wraps the Qwen3TTSTokenizerV2Decoder for clean ONNX export.

    Directly calls the decoder forward which takes codes [B, Q, T]
    and returns wav [B, 1, T_audio].
    """

    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, codes: torch.LongTensor) -> torch.Tensor:
        """
        Args:
            codes: [B, num_quantizers, T_codes] - codec tokens (int64)
        Returns:
            wav: [B, T_audio] - audio waveform, clamped to [-1, 1]
        """
        wav = self.decoder(codes)
        return wav.squeeze(1)


def export_code2wav_decoder(
    models_dir: str = None,
    output_dir: str = None,
    device: str = "cpu",
    dtype: torch.dtype = ONNX_EXPORT_DTYPE,
) -> str:
    tokenizer_path = resolve_tokenizer_path(models_dir)
    out_dir = ensure_output_dir(output_dir, "tokenizer")

    logger.info(f"Loading speech tokenizer from {tokenizer_path} (fp32 for ONNX export)")
    tokenizer_model = load_speech_tokenizer(tokenizer_path, device=device, dtype=torch.float32)

    decoder = tokenizer_model.decoder.to(device).eval()
    wrapper = Code2WavDecoderWrapper(decoder).to(device).eval()

    B = 1
    num_quantizers = 16
    T_codes = 25
    dummy_codes = torch.randint(0, 2048, (B, num_quantizers, T_codes), device=device)

    with torch.no_grad():
        ref_output = wrapper(dummy_codes)

    logger.info(f"Decoder output shape: {ref_output.shape}")

    onnx_path = str(out_dir / "code2wav_decoder.onnx")
    export_onnx(
        model=wrapper,
        dummy_inputs=(dummy_codes,),
        input_names=["codes"],
        output_names=["wav"],
        dynamic_axes={
            "codes": {0: "batch", 2: "time"},
            "wav": {0: "batch", 1: "audio_length"},
        },
        onnx_path=onnx_path,
    )

    cpu_codes = dummy_codes.cpu()
    cpu_wrapper = wrapper.cpu().eval()
    with torch.no_grad():
        cpu_ref = cpu_wrapper(cpu_codes)
    wrapper.to(device)

    test_inputs = {"codes": to_numpy(cpu_codes)}
    torch_outputs = {"wav": to_numpy(cpu_ref)}
    ok = verify_onnx(onnx_path, test_inputs, torch_outputs, atol=1e-3)
    if ok:
        logger.info("Code2Wav Decoder ONNX verification PASSED")
    else:
        logger.warning("Code2Wav Decoder ONNX verification FAILED")

    del tokenizer_model
    if device != "cpu":
        torch.cuda.empty_cache()
    return onnx_path


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Export Code2Wav Decoder to ONNX")
    add_common_args(parser)
    args = parser.parse_args()

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)

    path = export_code2wav_decoder(args.models_dir, args.output_dir, device, dtype)
    logger.info(f"Code2Wav Decoder exported: {path}")


if __name__ == "__main__":
    main()

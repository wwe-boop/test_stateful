#!/usr/bin/env python3
"""
[Step 02] Export Code2Wav Decoder to ONNX.

Component: Code2Wav Decoder (stateful streaming, chunk_T=4)
Architecture: RVQ Dequant + Transformer(8L, sliding window) + BigVGAN ConvNet
Input:  codes [B, 16, 4], cache_position [4], 37 state tensors
Output: wav [B, 7680], 37 updated state tensors
Engine: ONNX Runtime / TensorRT
Usage: Incremental decode per 4 codec frames; states kept on GPU for batching.

The decoder is shared across all model variants (comes from the tokenizer checkpoint).
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

from code2wav_streaming import (
    Code2WavStreamingWrapper,
    create_initial_states,
    get_initial_state_shapes,
    NUM_KV,
    NUM_CONV,
    NUM_TRANSCONV,
    CHUNK_T,
)

logger = logging.getLogger("onnx_export")


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
    wrapper = Code2WavStreamingWrapper(decoder).to(device).eval()

    B = 1
    # Use past_kv_len > 0 for export so KV inputs survive ONNX trace & simplification.
    # Zero-length tensors would be constant-folded away, removing KV inputs entirely.
    EXPORT_PAST_LEN = CHUNK_T
    dummy_codes = torch.randint(0, 2048, (B, 16, CHUNK_T), device=device, dtype=torch.long)
    cache_position = torch.arange(EXPORT_PAST_LEN, EXPORT_PAST_LEN + CHUNK_T, device=device, dtype=torch.long).unsqueeze(0)  # [1, CHUNK_T]
    state_shapes = get_initial_state_shapes(decoder, batch_size=B, past_kv_len=EXPORT_PAST_LEN)
    state_tensors = [torch.randn(s, device=device, dtype=torch.float32) for _, s in state_shapes]

    with torch.no_grad():
        out = wrapper(dummy_codes, cache_position, *state_tensors)

    wav_ref = out[0]
    logger.info(f"Streaming decoder output wav shape: {wav_ref.shape} (expected [1, 7680])")

    input_names = ["codes", "cache_position"]
    output_names = ["wav"]
    for name, _ in get_initial_state_shapes(decoder, batch_size=B, past_kv_len=0):
        input_names.append(name)
    for i in range(8):
        output_names.append(f"present_kv_{i}_k")
        output_names.append(f"present_kv_{i}_v")
    for i in range(NUM_CONV):
        output_names.append(f"new_conv_state_{i}")
    for i in range(NUM_TRANSCONV):
        output_names.append(f"new_transconv_overlap_{i}")

    dynamic_axes = {
        "codes": {0: "batch"},
        "cache_position": {0: "batch"},
        "wav": {0: "batch"},
    }
    for i in range(8):
        dynamic_axes[f"past_kv_{i}_k"] = {0: "batch", 2: "past_len"}
        dynamic_axes[f"past_kv_{i}_v"] = {0: "batch", 2: "past_len"}
        dynamic_axes[f"present_kv_{i}_k"] = {0: "batch", 2: "total_len"}
        dynamic_axes[f"present_kv_{i}_v"] = {0: "batch", 2: "total_len"}

    dummy_inputs = (dummy_codes, cache_position, *state_tensors)
    onnx_path = str(out_dir / "code2wav_decoder.onnx")
    export_onnx(
        model=wrapper,
        dummy_inputs=dummy_inputs,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        onnx_path=onnx_path,
        opset_version=18,
        simplify=True,
        do_constant_folding=False,
    )

    # Verify with the same non-zero-length KV states used for export
    cpu_codes = dummy_codes.cpu()
    cpu_cache_pos = cache_position.cpu()
    cpu_states = [s.cpu() for s in state_tensors]
    cpu_wrapper = wrapper.cpu().eval()
    with torch.no_grad():
        cpu_out = cpu_wrapper(cpu_codes, cpu_cache_pos, *cpu_states)
    wrapper.to(device)

    test_inputs = {"codes": to_numpy(cpu_codes), "cache_position": to_numpy(cpu_cache_pos)}
    for i, t in enumerate(cpu_states):
        test_inputs[input_names[2 + i]] = to_numpy(t)
    torch_outputs = {output_names[i]: to_numpy(cpu_out[i]) for i in range(len(output_names))}
    ok = verify_onnx(onnx_path, test_inputs, torch_outputs, atol=1e-3)
    if ok:
        logger.info("Code2Wav Streaming ONNX verification PASSED")
    else:
        logger.warning("Code2Wav Streaming ONNX verification had differences")

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

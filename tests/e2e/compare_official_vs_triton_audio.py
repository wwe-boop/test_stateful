#!/usr/bin/env python3
"""
Listenability A/B: official Python API vs Triton tts_orchestrator (fused talker+code2wav).

This does NOT assert numerical parity — it writes two WAV files for you to listen:

  - proto.wav       — `Qwen3TTSModel` high-level API (same as upstream usage).
  - triton.wav      — streaming audio from Triton (production path; uses fused engine when deployed).

Prerequisites:
  - conda env with qwen3-tts (or project venv) and weights under workspace/models.
  - Triton running with assembled repo, e.g.:
      bash scripts/bash/build_triton.sh assemble --engine-mode onnx --variant custom-1.7b
      bash scripts/bash/build_triton.sh run

Usage:
  conda activate qwen3-tts
  python tests/e2e/compare_official_vs_triton_audio.py --variant custom-1.7b \\
      --text "你好，这是一段测试。" --speaker serena --triton-url localhost:8001

  # VoiceDesign variant:
  python tests/e2e/compare_official_vs_triton_audio.py --variant design-1.7b \\
      --text "Hello" --instruct "Speak calmly." --triton-url localhost:8001
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "export"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "third_party" / "Qwen3-TTS"))

from utils import (
    setup_logging,
    resolve_model_path,
    resolve_device,
    has_model_weights,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("compare_official_triton")


def _non_streaming_for_variant(variant: str) -> bool:
    return "design" in variant.lower()


def _build_triton_request(
    variant: str,
    text: str,
    language: str,
    speaker: str,
    instruct: str,
) -> dict:
    v = variant.lower()
    if v.startswith("design"):
        return {
            "text": text,
            "task_type": "voice_design",
            "language": language,
            "instruct": instruct,
        }
    if v.startswith("custom"):
        return {
            "text": text,
            "task_type": "custom_voice",
            "language": language,
            "speaker": speaker,
        }
    raise ValueError(
        f"Variant '{variant}' not supported here (add voice_clone + ref_audio separately). "
        "Use custom-* or design-*."
    )


def _stream_tts_triton(client, req_dict: dict, timeout: float = 120.0):
    import tritonclient.grpc as grpcclient

    req_json = json.dumps(req_dict)
    req_input = grpcclient.InferInput("request", [1], "BYTES")
    req_input.set_data_from_numpy(np.array([req_json], dtype=object))
    audio_out = grpcclient.InferRequestedOutput("audio_chunk")
    final_out = grpcclient.InferRequestedOutput("is_final")

    chunks = []
    errors: list[str] = []
    done = False

    def callback(result, error):
        nonlocal done
        if error:
            errors.append(str(error))
            done = True
            return
        audio = result.as_numpy("audio_chunk")
        if audio is not None and audio.size:
            chunks.append(audio.flatten().copy())
        fin = result.as_numpy("is_final")
        if fin is not None and fin.size and bool(fin.flatten()[0]):
            done = True

    t0 = time.perf_counter()
    client.start_stream(callback=callback)
    client.async_stream_infer(
        model_name="tts_orchestrator",
        inputs=[req_input],
        outputs=[audio_out, final_out],
    )
    while not done and (time.perf_counter() - t0) < timeout:
        time.sleep(0.05)
    client.stop_stream()
    if errors:
        raise RuntimeError(errors[0])
    if not chunks:
        raise RuntimeError("No audio from Triton")
    return np.concatenate(chunks).astype(np.float32), time.perf_counter() - t0


def main():
    setup_logging()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", default="custom-1.7b")
    p.add_argument("--text", default="你好，这是一段用于听感对比的测试语音。")
    p.add_argument("--language", default="Chinese")
    p.add_argument("--speaker", default="serena", help="custom-* only")
    p.add_argument("--instruct", default="", help="design-* only")
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--triton-url", default="localhost:8001")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--models-dir", default=None)
    p.add_argument(
        "--greedy",
        action="store_true",
        help="do_sample=False for official API (slightly more repeatable vs sampling)",
    )
    args = p.parse_args()

    device = resolve_device(args.device)
    path = resolve_model_path(args.variant, args.models_dir)
    if not has_model_weights(path):
        logger.error("No weights for variant %s at %s", args.variant, path)
        sys.exit(1)

    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "workspace" / "audio_compare"
    out_dir.mkdir(parents=True, exist_ok=True)

    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel as TTSModelWrapper

    logger.info("Loading official Qwen3TTSModel from %s ...", path)
    wrapper = TTSModelWrapper.from_pretrained(
        str(path), device_map=str(device), dtype=torch.float32
    )

    do_sample = not args.greedy
    gen_kwargs = dict(do_sample=do_sample, max_new_tokens=args.max_steps)
    if not do_sample:
        gen_kwargs["repetition_penalty"] = 1.0
        gen_kwargs["subtalker_dosample"] = False

    non_streaming = _non_streaming_for_variant(args.variant)
    with torch.no_grad():
        if "design" in args.variant.lower():
            wavs, sr = wrapper.generate_voice_design(
                text=args.text,
                instruct=args.instruct,
                language=args.language,
                non_streaming_mode=non_streaming,
                **gen_kwargs,
            )
        else:
            wavs, sr = wrapper.generate_custom_voice(
                text=args.text,
                speaker=args.speaker,
                language=args.language,
                non_streaming_mode=non_streaming,
                **gen_kwargs,
            )

    wav_proto = wavs[0]
    sf.write(str(out_dir / "proto.wav"), wav_proto, sr)
    logger.info("Wrote %s (official API, %d samples, %.2fs)", out_dir / "proto.wav", len(wav_proto), len(wav_proto) / sr)

    # Triton (fused path in production)
    try:
        import tritonclient.grpc as grpcclient
    except ImportError:
        logger.error("Install: pip install tritonclient[grpc]")
        sys.exit(1)

    req = _build_triton_request(
        args.variant, args.text, args.language, args.speaker, args.instruct
    )
    logger.info("Triton request: %s", json.dumps(req, ensure_ascii=False))
    client = grpcclient.InferenceServerClient(url=args.triton_url)
    if not client.is_server_ready():
        logger.error("Triton not ready at %s", args.triton_url)
        sys.exit(1)

    wav_triton, elapsed = _stream_tts_triton(client, req)
    # Orchestrator outputs float32 mono at model sample rate (typically 24kHz)
    sf.write(str(out_dir / "triton.wav"), wav_triton, sr)
    logger.info(
        "Wrote %s (Triton orchestrator, %d samples, %.2fs, elapsed=%.2fs)",
        out_dir / "triton.wav",
        len(wav_triton),
        len(wav_triton) / sr,
        elapsed,
    )

    readme = out_dir / "LISTEN_README.txt"
    readme.write_text(
        "\n".join(
            [
                "Official API vs Triton (listenability)",
                "========================================",
                f"variant: {args.variant}",
                f"text: {args.text[:80]}",
                "",
                "proto.wav  — Qwen3TTSModel official high-level API (this repo: qwen_tts.inference).",
                "triton.wav — tts_orchestrator gRPC stream (deployed fused talker_code2wav_fused path).",
                "",
                "Sampling: official uses do_sample=%s; Triton uses orchestrator decode settings."
                % do_sample,
                "",
                "These two are not expected to be bit-identical; compare by ear.",
            ]
        ),
        encoding="utf-8",
    )
    logger.info("Done. Read %s and A/B listen to proto.wav vs triton.wav", readme)


if __name__ == "__main__":
    main()

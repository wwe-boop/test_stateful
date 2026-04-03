#!/usr/bin/env python3
"""
Standalone Code Predictor: ONNX Runtime vs TensorRT engine token parity.

Isolates the fused-graph issue documented in docs/progress: if ORT matches PyTorch
and standalone TRT matches ORT for the same (past_hidden, codec_token_0), the remaining
gap in talker_code2wav_fused is likely due to graph fusion / cross-subgraph numerics,
not a generic CP TRT bug.

Requirements:
  - Host: onnxruntime + numpy + PyTorch (conda env qwen3-tts).
  - TRT: pre-built engine next to ONNX, or pass --trtexec-docker-image to build/run
    trtexec inside the NGC Triton image (same as build_engines.sh).

Example (engine already built):
  conda activate qwen3-tts
  python tests/integration/verify_code_predictor_trt.py \\
    --variant-dir workspace/exported/custom-1.7b \\
    --engine workspace/exported/custom-1.7b/code_predictor_unrolled_bf16.engine \\
    --seed 42 --codec-token0 1500

Example (build BF16 engine via Docker, then compare):
  docker run --rm --gpus all -v \"$PWD/workspace/exported/custom-1.7b:/mnt/model\" \\
    nvcr.io/nvidia/tritonserver:26.02-py3 \\
    /usr/src/tensorrt/bin/trtexec \\
    --onnx=/mnt/model/code_predictor_unrolled.onnx \\
    --saveEngine=/mnt/model/code_predictor_unrolled_bf16.engine \\
    --bf16 --inputIOFormats=fp32:chw,int64:chw --outputIOFormats=int64:chw \\
    --minShapes=past_hidden:1x1x2048,codec_token_0:1 \\
    --optShapes=past_hidden:1x1x2048,codec_token_0:1 \\
    --maxShapes=past_hidden:8x1x2048,codec_token_0:8 \\
    --memPoolSize=workspace:8192

Notes:
  - trtexec enables TF32 by default; for strict FP32 parity with some references,
    rebuild with --noTF32 (trtexec build flag).
  - NGC tritonserver image has trtexec but not Python tensorrt bindings; this script
    shells out to trtexec for inference when --engine is set.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("verify_cp_trt")


def _parse_trtexec_codec_tokens(stdout: str) -> np.ndarray | None:
    """Parse the '59 294 1398 ...' line after codec_tokens shape dump."""
    after = False
    for line in stdout.splitlines():
        if "codec_tokens:" in line and "x" in line:
            after = True
            continue
        if not after:
            continue
        m = re.search(r"\[I\]\s+([\d\s]+)\s*$", line)
        if m:
            parts = m.group(1).split()
            if parts and all(p.isdigit() for p in parts):
                return np.array([int(x) for x in parts], dtype=np.int64)
        # Some builds log without [I] prefix on the number line
        s = line.strip()
        if re.match(r"^[\d\s]+$", s) and len(s) > 4:
            parts = s.split()
            if len(parts) >= 8:
                return np.array([int(x) for x in parts], dtype=np.int64)
    return None


def _run_trtexec_infer(
    engine: Path,
    past_hidden: np.ndarray,
    codec_token_0: np.ndarray,
    docker_image: str | None,
) -> np.ndarray:
    """Run TRT engine via trtexec --loadInputs; returns codec_tokens [num_stages]."""
    past_hidden = np.ascontiguousarray(past_hidden, dtype=np.float32)
    codec_token_0 = np.ascontiguousarray(codec_token_0, dtype=np.int64)
    with tempfile.TemporaryDirectory() as tmp:
        ph_path = Path(tmp) / "past_hidden.bin"
        t0_path = Path(tmp) / "codec_token_0.bin"
        past_hidden.tofile(ph_path)
        codec_token_0.tofile(t0_path)

        load_inputs = f"past_hidden:{ph_path},codec_token_0:{t0_path}"
        trtexec = os.environ.get("TRTEXEC", "/usr/src/tensorrt/bin/trtexec")

        if docker_image:
            mount_engine = engine.parent.resolve()
            cmd = [
                "docker",
                "run",
                "--rm",
                "--gpus",
                "all",
                "-v",
                f"{mount_engine}:/mnt_model",
                "-v",
                f"{tmp}:/in",
                docker_image,
                trtexec,
                f"--loadEngine=/mnt_model/{engine.name}",
                f"--loadInputs=past_hidden:/in/past_hidden.bin,codec_token_0:/in/codec_token_0.bin",
                "--dumpOutput",
                "--iterations=1",
                "--warmUp=0",
                "--duration=0",
            ]
        else:
            cmd = [
                trtexec,
                f"--loadEngine={engine.resolve()}",
                f"--loadInputs=past_hidden:{ph_path},codec_token_0:{t0_path}",
                "--dumpOutput",
                "--iterations=1",
                "--warmUp=0",
                "--duration=0",
            ]

        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            logger.error("trtexec failed:\n%s", r.stderr[-4000:] or r.stdout[-4000:])
            sys.exit(1)
        out = _parse_trtexec_codec_tokens(r.stdout)
        if out is None:
            logger.error("Could not parse trtexec output:\n%s", r.stdout[-2000:])
            sys.exit(1)
        return out


def _run_ort(onnx_path: Path, past_hidden: np.ndarray, codec_token_0: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    feeds = {
        "past_hidden": past_hidden.astype(np.float32),
        "codec_token_0": codec_token_0.astype(np.int64),
    }
    return sess.run(None, feeds)[0].reshape(-1).astype(np.int64)


def _optional_pytorch_ref(
    variant_dir: Path,
    past_hidden: np.ndarray,
    codec_token_0: np.ndarray,
) -> np.ndarray | None:
    """If export env is available, compare ORT to PyTorch CodePredictorUnrolled."""
    repo_root = Path(__file__).resolve().parents[2]
    wdir = variant_dir / "weights"
    cfg = wdir / "config.json"
    if not cfg.is_file():
        return None
    sys.path.insert(0, str(repo_root / "scripts" / "export"))
    sys.path.insert(0, str(repo_root / "third_party" / "Qwen3-TTS"))
    try:
        import torch
        from utils import CodePredictorUnrolled, load_tts_model, resolve_model_path

        variant_name = variant_dir.name
        model_path = resolve_model_path(variant_name)
        device = "cpu"
        model = load_tts_model(str(model_path), device=device, dtype=torch.float32)
        talker = model.talker
        cp = talker.code_predictor
        talker_codec_emb = talker.model.codec_embedding
        wrapper = CodePredictorUnrolled(cp, talker_codec_emb).to(device).eval()

        with torch.no_grad():
            ph = torch.from_numpy(past_hidden).to(device)
            t0 = torch.from_numpy(codec_token_0).to(device)
            pt = wrapper(ph, t0).cpu().numpy().reshape(-1).astype(np.int64)
        return pt
    except Exception as e:
        logger.warning("PyTorch reference skipped: %s", e)
        return None
    finally:
        for p in list(sys.path):
            if "export" in p or "Qwen3-TTS" in p:
                try:
                    sys.path.remove(p)
                except ValueError:
                    pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant-dir",
        type=Path,
        default=Path("workspace/exported/custom-1.7b"),
        help="Directory containing code_predictor_unrolled.onnx",
    )
    parser.add_argument(
        "--engine",
        type=Path,
        default=None,
        help="Serialized TensorRT engine (e.g. code_predictor_unrolled_bf16.engine)",
    )
    parser.add_argument(
        "--trtexec-docker-image",
        default=os.environ.get("NGC_IMAGE", "nvcr.io/nvidia/tritonserver:26.02-py3"),
        help="Run trtexec inside this image; use empty string for host TRTEXEC (default: NGC_IMAGE or 26.02-py3)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--codec-token0", type=int, default=1500)
    parser.add_argument("--trials", type=int, default=1, help="Random trials (seed, seed+1, ...)")
    parser.add_argument(
        "--pytorch-ref",
        action="store_true",
        help="Load full TTS weights and compare ORT to PyTorch (slow; optional sanity check)",
    )
    args = parser.parse_args()

    variant_dir = args.variant_dir
    onnx_path = variant_dir / "code_predictor_unrolled.onnx"
    if not onnx_path.is_file():
        logger.error("Missing %s", onnx_path)
        return 1

    engine_arg = args.engine
    if engine_arg is None:
        engine = None
        for cand in (
            variant_dir / "code_predictor_unrolled_bf16.engine",
            variant_dir / "code_predictor_unrolled_fp32.engine",
            variant_dir / "code_predictor_unrolled.engine",
        ):
            if cand.is_file():
                engine = cand
                break
    else:
        engine = Path(engine_arg).expanduser()
        if not engine.is_file():
            engine = variant_dir / Path(engine_arg).name
    if engine is None or not engine.is_file():
        logger.error(
            "No engine found. Build one (see docstring) or pass --engine path "
            "(expected next to ONNX in %s)",
            variant_dir,
        )
        return 1
    engine = engine.resolve()

    docker_image = getattr(args, "trtexec_docker_image", None)
    if isinstance(docker_image, str) and docker_image.strip() == "":
        docker_image = None

    mismatches = 0
    for t in range(args.trials):
        seed = args.seed + t
        # Match legacy NumPy used in export_05 / manual trtexec tests (not default_rng).
        rng = np.random.RandomState(seed)
        past_hidden = rng.randn(1, 1, 2048).astype(np.float32)
        if t == 0 and args.trials == 1:
            codec_token_0 = np.array([args.codec_token0], dtype=np.int64)
        else:
            codec_token_0 = rng.randint(0, 3072, size=(1,), dtype=np.int64)

        ort_tokens = _run_ort(onnx_path, past_hidden, codec_token_0)
        pt_tokens = (
            _optional_pytorch_ref(variant_dir, past_hidden, codec_token_0)
            if args.pytorch_ref
            else None
        )
        if pt_tokens is not None and not np.array_equal(ort_tokens, pt_tokens):
            logger.warning(
                "ORT vs PyTorch mismatch (unexpected): seed=%s ort=%s pt=%s",
                seed,
                ort_tokens,
                pt_tokens,
            )

        trt_tokens = _run_trtexec_infer(engine, past_hidden, codec_token_0, docker_image)
        match = np.array_equal(ort_tokens, trt_tokens)
        if not match:
            mismatches += 1
            logger.error(
                "ORT vs TRT mismatch seed=%s codec_token_0=%s\n  ORT: %s\n  TRT: %s",
                seed,
                codec_token_0,
                ort_tokens,
                trt_tokens,
            )
        else:
            logger.info(
                "OK seed=%s stages=%d engine=%s",
                seed,
                len(ort_tokens),
                engine.name,
            )

    if mismatches:
        logger.error("Total mismatches: %s / %s", mismatches, args.trials)
        return 1
    logger.info("All %s trial(s) matched ORT vs TRT.", args.trials)
    return 0


if __name__ == "__main__":
    sys.exit(main())

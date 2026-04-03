"""
Cross-export consistency (optional artifacts).

Requires ONNX files from Phase A export (see export_all.py steps 03–04).
Skips individual tests when paths are missing — no GPU required for ICL fused vs decomposed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPORTED = REPO_ROOT / "workspace" / "exported"
VARIANT = "base-1.7b"


def _ref_codec_sum_vec_torch(stacked_3d: torch.Tensor, audio_codes: torch.Tensor) -> torch.Tensor:
    """Match export_04 RefCodecSumFromAudioCodes (CPU reference)."""
    ac = audio_codes.long()
    B, G, Tlen = ac.shape
    rc = ac.permute(0, 2, 1).contiguous()
    g_idx = (
        torch.arange(G, dtype=torch.long).view(1, 1, G).expand(B, Tlen, G)
    )
    gathered = stacked_3d[g_idx, rc, :]
    return gathered.sum(dim=(1, 2), keepdim=True)


def test_speech_tokenizer_codec_fused_matches_encoder_plus_3d():
    """Fused ONNX ref_codec_sum_vec vs speech_tokenizer_encoder + 3D gather (ORT CPU)."""
    pytest.importorskip("onnxruntime")
    import onnxruntime as ort

    fused_onnx = EXPORTED / VARIANT / "speech_tokenizer_codec_fused.onnx"
    enc_onnx = EXPORTED / "tokenizer" / "speech_tokenizer_encoder.onnx"
    weights_3d = EXPORTED / VARIANT / "weights" / "codec_embeddings_3d.pt"
    for p in (fused_onnx, enc_onnx, weights_3d):
        if not p.is_file():
            pytest.skip(f"Missing {p} (run export_03 + export_04 + export_01 for {VARIANT})")

    stacked = torch.load(weights_3d, map_location="cpu", weights_only=True)
    if not isinstance(stacked, torch.Tensor):
        pytest.skip("codec_embeddings_3d.pt is not a tensor")

    prov = ["CPUExecutionProvider"]
    sess_enc = ort.InferenceSession(str(enc_onnx), providers=prov)
    sess_fused = ort.InferenceSession(str(fused_onnx), providers=prov)

    rng = np.random.default_rng(42)
    wav = rng.standard_normal((1, 1, 72_000), dtype=np.float32).astype(np.float32)

    codes = sess_enc.run(None, {"waveform": wav})[0]
    codes_t = torch.from_numpy(np.asarray(codes, dtype=np.int64))
    ref = _ref_codec_sum_vec_torch(stacked, codes_t).numpy()

    fused_out = sess_fused.run(None, {"waveform": wav})[0]
    np.testing.assert_allclose(
        fused_out,
        ref,
        rtol=1e-3,
        atol=5e-3,
        err_msg="speech_tokenizer_codec_fused vs encoder+3d mismatch",
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_talker_backbone_onnx_exists_and_loadable():
    """If talker_backbone.onnx exists, ORT CUDA session can be created."""
    pytest.importorskip("onnxruntime")
    import onnxruntime as ort

    onnx_p = EXPORTED / VARIANT / "talker_backbone.onnx"
    if not onnx_p.is_file():
        pytest.skip(f"Missing {onnx_p} (run export_07 --skip-verification off)")

    so = ort.SessionOptions()
    so.log_severity_level = 3
    ort.InferenceSession(
        str(onnx_p),
        sess_options=so,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_talker_unified_onnx_exists_and_loadable():
    """If talker_unified.onnx exists, ORT CUDA session can be created."""
    pytest.importorskip("onnxruntime")
    import onnxruntime as ort

    onnx_p = EXPORTED / VARIANT / "talker_unified.onnx"
    if not onnx_p.is_file():
        pytest.skip(f"Missing {onnx_p} (run export_08)")

    so = ort.SessionOptions()
    so.log_severity_level = 3
    ort.InferenceSession(
        str(onnx_p),
        sess_options=so,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_talker_code2wav_fused_onnx_exists_and_loadable():
    """If talker_code2wav_fused.onnx exists, ORT CUDA session can be created."""
    pytest.importorskip("onnxruntime")
    import onnxruntime as ort

    onnx_p = EXPORTED / VARIANT / "talker_code2wav_fused.onnx"
    if not onnx_p.is_file():
        pytest.skip(f"Missing {onnx_p} (run export_09)")

    so = ort.SessionOptions()
    so.log_severity_level = 3
    ort.InferenceSession(
        str(onnx_p),
        sess_options=so,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

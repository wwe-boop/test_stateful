"""
L1 unit tests: audio_utils (T1.2).
Run from repo root: pytest tests/test_audio_utils.py -v
"""
import base64
import io
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ORCH_1 = REPO_ROOT / "model_repository" / "tts_orchestrator" / "1"
if str(ORCH_1) not in sys.path:
    sys.path.insert(0, str(ORCH_1))


@pytest.fixture(scope="module")
def ref_audio_16k_b64():
    """Synthetic 1s 440Hz sine at 16kHz mono, WAV base64 (stdlib-only WAV)."""
    sr = 16000
    dur = 1.0
    samples = np.sin(2 * np.pi * 440 * np.arange(int(sr * dur)) / sr).astype(np.float32)
    # Build WAV bytes with stdlib (no soundfile/scipy required)
    import struct
    n = len(samples)
    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + n * 2))  # file size - 8
    buf.write(b"WAVEfmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16))  # fmt chunk
    buf.write(b"data")
    buf.write(struct.pack("<I", n * 2))
    for s in samples:
        buf.write(struct.pack("<h", int(max(-32768, min(32767, s * 32767)))))
    return base64.b64encode(buf.getvalue()).decode()


@pytest.fixture(scope="module")
def ref_audio_with_data_uri(ref_audio_16k_b64):
    """Same audio with data:audio/wav;base64, prefix."""
    return "data:audio/wav;base64," + ref_audio_16k_b64


def test_decode_audio_from_base64(ref_audio_16k_b64):
    """Decode raw base64 WAV -> (waveform, sr)."""
    from audio_utils import decode_audio_from_base64
    audio, orig_sr = decode_audio_from_base64(ref_audio_16k_b64)
    assert orig_sr == 16000
    assert audio.dtype == np.float32
    assert audio.ndim == 1
    assert len(audio) == 16000  # 1s


def test_decode_audio_with_data_uri_prefix(ref_audio_with_data_uri):
    """Decode data:audio/...;base64, prefix."""
    from audio_utils import decode_audio_from_base64
    audio, sr = decode_audio_from_base64(ref_audio_with_data_uri)
    assert sr == 16000
    assert audio.ndim == 1 and len(audio) == 16000


def test_decode_audio_empty_raises():
    """Empty base64 string raises ValueError."""
    from audio_utils import decode_audio_from_base64
    with pytest.raises(ValueError, match="Empty"):
        decode_audio_from_base64("")
    with pytest.raises(ValueError, match="Empty"):
        decode_audio_from_base64("   ")


def test_resample_to_24k(ref_audio_16k_b64):
    """Resample 16kHz -> 24kHz."""
    from audio_utils import decode_audio_from_base64, resample_to_24k
    audio, sr = decode_audio_from_base64(ref_audio_16k_b64)
    audio_24k = resample_to_24k(audio, sr)
    assert audio_24k.dtype == np.float32
    assert len(audio_24k) == int(24000 * 1.0)  # 1s at 24k


def test_resample_to_24k_same_sr():
    """When orig_sr is 24k, return as-is."""
    from audio_utils import resample_to_24k
    x = np.random.randn(2400).astype(np.float32)
    out = resample_to_24k(x, 24000)
    np.testing.assert_array_almost_equal(out, x)


def test_compute_mel_spectrogram(ref_audio_16k_b64):
    """Mel shape [1, T, 128] (batch, time, mel)."""
    from audio_utils import (
        decode_audio_from_base64,
        resample_to_24k,
        compute_mel_spectrogram,
    )
    audio, sr = decode_audio_from_base64(ref_audio_16k_b64)
    audio_24k = resample_to_24k(audio, sr)
    mel = compute_mel_spectrogram(audio_24k)
    assert mel.ndim == 3
    assert mel.shape[0] == 1
    assert mel.shape[2] == 128
    assert mel.dtype == np.float32


def test_prepare_waveform_tensor():
    """Waveform tensor shape [1, 1, samples] float32."""
    from audio_utils import prepare_waveform_tensor
    audio_24k = np.random.randn(2400).astype(np.float32)
    waveform = prepare_waveform_tensor(audio_24k)
    assert waveform.shape == (1, 1, 2400)
    assert waveform.dtype == np.float32


def test_prepare_waveform_tensor_1d_already():
    """1D input -> [1, 1, samples]."""
    from audio_utils import prepare_waveform_tensor
    x = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    out = prepare_waveform_tensor(x)
    assert out.shape == (1, 1, 3)

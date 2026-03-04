"""
Audio preprocessing utilities for TTS Orchestrator BLS.

Used for Speaker Encoder (mel spectrogram) and Speech Tokenizer Encoder (waveform).
Parameters match Qwen3-TTS: extract_speaker_embedding / tokenizer input.
No torch dependency: numpy + scipy only.
"""

import base64
import io
import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger("audio_utils")

# Qwen3-TTS Speaker Encoder mel params (modeling_qwen3_tts.py extract_speaker_embedding)
TARGET_SR = 24000
N_FFT = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024
N_MELS = 128
FMIN = 0
FMAX = 12000

try:
    from librosa.filters import mel as librosa_mel_fn
    _HAS_LIBROSA = True
except ImportError:
    _HAS_LIBROSA = False


def decode_audio_from_base64(b64_str: str) -> Tuple[np.ndarray, int]:
    """
    Decode base64 audio string to (waveform, sample_rate).

    Supports data:audio/...;base64, prefix (strip it) and raw base64.
    Audio format: WAV/FLAC preferred; soundfile is used for reading.

    Returns:
        (audio_np, sr): float32 mono waveform, shape (samples,); sample rate.
    """
    if not b64_str or not b64_str.strip():
        raise ValueError("Empty base64 audio string")
    s = b64_str.strip()
    if "," in s and s.startswith("data:"):
        s = s.split(",", 1)[1]
    raw = base64.b64decode(s)
    buf = io.BytesIO(raw)
    try:
        import soundfile as sf
        audio, sr = sf.read(buf, dtype="float32", always_2d=False)
    except Exception as e:
        logger.warning(f"soundfile read failed: {e}, trying scipy.io.wavfile")
        try:
            import scipy.io.wavfile as wav_io
            buf.seek(0)
            sr, audio = wav_io.read(buf)
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
            if audio.ndim > 1:
                audio = np.mean(audio, axis=-1)
        except Exception as e2:
            raise RuntimeError(f"Failed to decode audio from base64: {e}, {e2}") from e2
    if audio.ndim > 1:
        audio = np.mean(audio, axis=-1)
    return audio.astype(np.float32), int(sr)


def resample_to_24k(audio: np.ndarray, orig_sr: int) -> np.ndarray:
    """Resample waveform to 24 kHz. Uses scipy.signal.resample_poly."""
    if orig_sr == TARGET_SR:
        return audio
    try:
        from scipy.signal import resample_poly
        num = TARGET_SR
        den = orig_sr
        g = np.gcd(num, den)
        num, den = num // g, den // g
        resampled = resample_poly(audio, num, den)
        return resampled.astype(np.float32)
    except ImportError:
        duration = len(audio) / orig_sr
        new_len = int(duration * TARGET_SR)
        x_old = np.linspace(0, 1, len(audio), dtype=np.float32)
        x_new = np.linspace(0, 1, new_len, dtype=np.float32)
        resampled = np.interp(x_new, x_old, audio)
        return resampled.astype(np.float32)


def _mel_basis_numpy(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    """Build mel filter bank (HTK-style). Used when librosa is not available."""
    n_fft_2 = n_fft // 2 + 1
    freq_bins = np.linspace(0, sr / 2, n_fft_2, dtype=np.float32)
    mel_low = 1127 * np.log(1 + fmin / 700)
    mel_high = 1127 * np.log(1 + (fmax or (sr / 2)) / 700)
    mel_pts = np.linspace(mel_low, mel_high, n_mels + 2)
    hz_pts = 700 * (np.exp(mel_pts / 1127) - 1)
    weights = np.zeros((n_mels, n_fft_2), dtype=np.float32)
    for i in range(n_mels):
        left, center, right = hz_pts[i], hz_pts[i + 1], hz_pts[i + 2]
        left_idx = np.searchsorted(freq_bins, left)
        center_idx = np.searchsorted(freq_bins, center)
        right_idx = np.searchsorted(freq_bins, right)
        if left_idx < center_idx:
            weights[i, left_idx:center_idx] = (
                (freq_bins[left_idx:center_idx] - left) / (center - left)
            )
        if center_idx < right_idx:
            weights[i, center_idx:right_idx] = (
                (right - freq_bins[center_idx:right_idx]) / (right - center)
            )
    return weights


def compute_mel_spectrogram(
    audio_np: np.ndarray,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    win_length: int = WIN_LENGTH,
    n_mels: int = N_MELS,
    fmin: int = FMIN,
    fmax: int = FMAX,
    sampling_rate: int = TARGET_SR,
) -> np.ndarray:
    """
    Compute mel spectrogram matching Qwen3-TTS extract_speaker_embedding.

    Input: 1D float32 waveform (samples,).
    Output: [1, T, 128] float32 numpy (batch, time, mel). No torch.
    """
    from scipy.signal import stft, get_window

    y = np.asarray(audio_np, dtype=np.float32)
    if y.ndim == 1:
        y = y.reshape(1, -1)
    padding = (n_fft - hop_length) // 2
    y = np.pad(y, ((0, 0), (padding, padding)), mode="reflect")
    window = get_window("hann", win_length, fftbins=True).astype(np.float32)
    _, _, z = stft(
        y,
        fs=sampling_rate,
        nperseg=win_length,
        noverlap=win_length - hop_length,
        window=window,
    )
    # z: (1, n_fft/2+1, T)
    magnitude = np.sqrt(np.real(z) ** 2 + np.imag(z) ** 2 + 1e-9)
    if _HAS_LIBROSA:
        mel_basis = librosa_mel_fn(
            sr=sampling_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax
        ).astype(np.float32)
    else:
        mel_basis = _mel_basis_numpy(
            sampling_rate, n_fft, n_mels, float(fmin), float(fmax)
        )
    mel_spec = np.dot(mel_basis, magnitude.squeeze(0))
    mel_spec = np.log(np.maximum(mel_spec, 1e-5))
    return mel_spec.T[np.newaxis, :, :].astype(np.float32)


def prepare_waveform_tensor(audio_np: np.ndarray) -> np.ndarray:
    """
    Prepare waveform for Speech Tokenizer Encoder input.

    Input: 1D float32 waveform (samples,).
    Output: [1, 1, samples] float32 numpy for ONNX (B, 1, samples).
    """
    audio = np.asarray(audio_np, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio[np.newaxis, np.newaxis, :]
    elif audio.ndim == 2:
        audio = audio[np.newaxis, :, :] if audio.shape[0] != 1 else audio
    return audio

"""Reference-audio preprocessing for standalone Base voice clone.

The Base ICL path needs two request-level tensors before the normal talker
prefill can run:

- ``spk_embedding`` from ``speaker_encoder.engine``
- temporal ``ref_codec_sum_vec`` from ``speech_tokenizer_codec_fused.engine``
- optional Code2Wav warm states from ``code2wav_decoder.engine``

Both are small CPU tensors when they leave this module.  The engine thread moves
them to the prefill device later, keeping the asyncio side free of GPU tensors.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import logging
import re
import threading
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


_REF_SAMPLE_RATE = 24000


@dataclass
class ReferenceAudioSupport:
    available: bool
    reason: str = ""
    speaker_encoder_path: Optional[Path] = None
    speech_tokenizer_encoder_path: Optional[Path] = None
    speech_tokenizer_codec_fused_path: Optional[Path] = None
    code2wav_decoder_path: Optional[Path] = None


@dataclass
class ReferenceAudioFeatures:
    spk_embedding: object
    ref_codec_sum_vec: Optional[object] = None
    ref_audio_codes: Optional[object] = None
    ref_c2w_kv: Optional[object] = None
    ref_c2w_conv_states: Optional[list[object]] = None
    ref_c2w_transconv_states: Optional[list[object]] = None
    ref_c2w_frame_idx: int = 0
    sample_rate: int = _REF_SAMPLE_RATE
    duration_sec: float = 0.0


class ReferenceAudioProcessor:
    """Prepare Base voice-clone reference features with TensorRT engines."""

    def __init__(self, engine_dir: str, variant: str, device_id: int = 0):
        self._engine_dir = Path(engine_dir) if engine_dir else Path()
        self._variant = variant or ""
        self._device_id = int(device_id)
        self._support = self._probe()
        self._lock = threading.Lock()
        self._speaker_engine = None
        self._codec_engine = None
        self._c2w_engine = None
        self._stream = None
        self._cache: dict[tuple[str, bool], ReferenceAudioFeatures] = {}

    @property
    def support(self) -> ReferenceAudioSupport:
        return self._support

    def process(self, ref_audio: bytes, *, require_ref_codec: bool) -> ReferenceAudioFeatures:
        if not self._support.available:
            raise RuntimeError(self._support.reason or "reference-audio preprocessing is unavailable")
        if require_ref_codec and self._support.speech_tokenizer_codec_fused_path is None:
            raise RuntimeError("speech_tokenizer_codec_fused engine is required for Base ICL mode")

        cache_key = (hashlib.sha256(ref_audio).hexdigest(), bool(require_ref_codec))
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        with self._lock:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

            self._ensure_engines(require_ref_codec=require_ref_codec)
            wav, sr = _decode_wav_bytes(ref_audio)
            wav_24k = _resample_linear(wav, sr, _REF_SAMPLE_RATE)
            duration_sec = float(wav_24k.shape[0]) / float(_REF_SAMPLE_RATE)

            spk_embedding = self._run_speaker_encoder(wav_24k)
            ref_codec_sum_vec = None
            ref_audio_codes = None
            ref_c2w_kv = None
            ref_c2w_conv_states = None
            ref_c2w_transconv_states = None
            ref_c2w_frame_idx = 0
            if require_ref_codec:
                ref_codec_sum_vec, ref_audio_codes = self._run_speech_tokenizer_codec(wav_24k)
                (
                    ref_c2w_kv,
                    ref_c2w_conv_states,
                    ref_c2w_transconv_states,
                    ref_c2w_frame_idx,
                ) = self._run_code2wav_warmup(ref_audio_codes)
            features = ReferenceAudioFeatures(
                spk_embedding=spk_embedding,
                ref_codec_sum_vec=ref_codec_sum_vec,
                ref_audio_codes=ref_audio_codes,
                ref_c2w_kv=ref_c2w_kv,
                ref_c2w_conv_states=ref_c2w_conv_states,
                ref_c2w_transconv_states=ref_c2w_transconv_states,
                ref_c2w_frame_idx=ref_c2w_frame_idx,
                duration_sec=duration_sec,
            )
            self._cache[cache_key] = features
            return features

    def _probe(self) -> ReferenceAudioSupport:
        if not self._variant.startswith("base-"):
            return ReferenceAudioSupport(
                available=False,
                reason=f"variant '{self._variant}' is not a base model; voice_clone is unsupported",
            )

        engine_dir = self._engine_dir
        speaker_encoder = _first_existing(
            engine_dir / "speaker_encoder.engine",
            engine_dir / "speaker_encoder" / "model.plan",
            engine_dir / "speaker_encoder" / "speaker_encoder.engine",
            engine_dir / "speaker_encoder.onnx",
        )
        speech_tokenizer_codec_fused = _first_existing(
            engine_dir / "speech_tokenizer_codec_fused.engine",
            engine_dir / "speech_tokenizer_codec_fused" / "model.plan",
            engine_dir / "speech_tokenizer_codec_fused" / "speech_tokenizer_codec_fused.engine",
            engine_dir / "speech_tokenizer_codec_fused.onnx",
        )
        code2wav_decoder = _first_existing(
            engine_dir.parent / "tokenizer" / "code2wav_decoder.engine",
            Path("workspace/exported/tokenizer/code2wav_decoder.engine"),
        )
        speech_tokenizer_encoder = _first_existing(
            engine_dir / "speech_tokenizer_encoder.onnx",
            Path("workspace/exported/tokenizer/speech_tokenizer_encoder.onnx"),
        ) or (engine_dir / "speech_tokenizer_encoder.onnx")

        if speaker_encoder is None:
            return ReferenceAudioSupport(
                available=False,
                reason=f"missing speaker encoder TensorRT engine under: {engine_dir}",
                speaker_encoder_path=engine_dir / "speaker_encoder.engine",
                speech_tokenizer_encoder_path=speech_tokenizer_encoder,
                speech_tokenizer_codec_fused_path=speech_tokenizer_codec_fused,
                code2wav_decoder_path=code2wav_decoder,
            )

        if not importlib.util.find_spec("torch"):
            return ReferenceAudioSupport(
                available=False,
                reason="torch is required for standalone ref_audio preprocessing",
                speaker_encoder_path=speaker_encoder,
                speech_tokenizer_encoder_path=speech_tokenizer_encoder,
                speech_tokenizer_codec_fused_path=speech_tokenizer_codec_fused,
                code2wav_decoder_path=code2wav_decoder,
            )

        if not importlib.util.find_spec("tensorrt"):
            return ReferenceAudioSupport(
                available=False,
                reason="tensorrt is required for standalone ref_audio preprocessing",
                speaker_encoder_path=speaker_encoder,
                speech_tokenizer_encoder_path=speech_tokenizer_encoder,
                speech_tokenizer_codec_fused_path=speech_tokenizer_codec_fused,
                code2wav_decoder_path=code2wav_decoder,
            )

        return ReferenceAudioSupport(
            available=True,
            speaker_encoder_path=speaker_encoder,
            speech_tokenizer_encoder_path=speech_tokenizer_encoder if speech_tokenizer_encoder.is_file() else None,
            speech_tokenizer_codec_fused_path=speech_tokenizer_codec_fused,
            code2wav_decoder_path=code2wav_decoder,
        )

    def _ensure_engines(self, *, require_ref_codec: bool) -> None:
        import torch

        from .executor import TRTEngine

        device = torch.device("cuda", self._device_id)
        if self._stream is None:
            self._stream = torch.cuda.Stream(device=device)
        if self._speaker_engine is None:
            self._speaker_engine = TRTEngine(str(self._support.speaker_encoder_path), device)
            self._speaker_engine.load()
        if require_ref_codec and self._codec_engine is None:
            if self._support.speech_tokenizer_codec_fused_path is None:
                raise RuntimeError("missing speech_tokenizer_codec_fused TensorRT engine")
            self._codec_engine = TRTEngine(
                str(self._support.speech_tokenizer_codec_fused_path),
                device,
            )
            self._codec_engine.load()
        if (
            require_ref_codec
            and self._support.code2wav_decoder_path is not None
            and self._c2w_engine is None
        ):
            self._c2w_engine = TRTEngine(
                str(self._support.code2wav_decoder_path),
                device,
            )
            self._c2w_engine.load()

    def _run_speaker_encoder(self, wav_24k: np.ndarray):
        import torch

        device = torch.device("cuda", self._device_id)
        mel = _mel_spectrogram_24k(wav_24k, device=device).transpose(1, 2).contiguous()
        with torch.cuda.stream(self._stream):
            out = self._speaker_engine.infer(
                {"mel": mel},
                ["speaker_embedding"],
                self._stream,
            )
        self._stream.synchronize()
        return out["speaker_embedding"].detach().float().cpu().contiguous()

    def _run_speech_tokenizer_codec(self, wav_24k: np.ndarray):
        import torch

        device = torch.device("cuda", self._device_id)
        waveform = torch.from_numpy(wav_24k).to(device=device, dtype=torch.float32)
        waveform = waveform.reshape(1, 1, -1).contiguous()
        _, available_outputs = self._codec_engine.get_io_names()
        output_names = ["ref_codec_sum_vec"]
        if "ref_audio_codes" in available_outputs:
            output_names.append("ref_audio_codes")
        with torch.cuda.stream(self._stream):
            out = self._codec_engine.infer(
                {"waveform": waveform},
                output_names,
                self._stream,
            )
        self._stream.synchronize()
        ref = out["ref_codec_sum_vec"].detach().float().cpu().contiguous()
        if ref.dim() == 4 and ref.shape[2] == 1:
            ref = ref.squeeze(2).contiguous()
        if ref.dim() != 3:
            raise RuntimeError(
                f"speech_tokenizer_codec_fused returned invalid shape {tuple(ref.shape)}; "
                "expected [B, T, H]"
            )
        if wav_24k.shape[0] >= _REF_SAMPLE_RATE and ref.shape[1] <= 1:
            raise RuntimeError(
                "speech_tokenizer_codec_fused returned a collapsed ICL ref codec sequence; "
                "rebuild the ONNX/TRT engine with temporal ref codec output"
            )
        codes = out.get("ref_audio_codes")
        if codes is None:
            logger.warning(
                "speech_tokenizer_codec_fused.engine does not expose ref_audio_codes; "
                "Code2Wav reference warmup is disabled"
            )
            return ref, None
        codes = codes.detach().to(dtype=torch.int64).cpu().contiguous()
        if codes.dim() != 3:
            raise RuntimeError(
                f"speech_tokenizer_codec_fused returned invalid code shape {tuple(codes.shape)}; "
                "expected [B, 16, T]"
            )
        return ref, codes

    def _run_code2wav_warmup(self, ref_audio_codes):
        if ref_audio_codes is None or self._c2w_engine is None:
            return None, None, None, 0

        import torch

        device = torch.device("cuda", self._device_id)
        codes = ref_audio_codes.to(device=device, dtype=torch.int64)
        if codes.dim() != 3 or codes.shape[0] != 1 or codes.shape[1] != 16:
            raise RuntimeError(
                f"ref_audio_codes must be [1, 16, T], got {tuple(codes.shape)}"
            )

        total_frames = int(codes.shape[2])
        if total_frames <= 0:
            return None, None, None, 0
        remainder = total_frames % 4
        if remainder:
            pad_frames = 4 - remainder
            codes = torch.cat(
                [codes, codes[:, :, -1:].expand(1, 16, pad_frames)],
                dim=2,
            ).contiguous()
        warm_frames = int(codes.shape[2])

        input_names, output_names = self._c2w_engine.get_io_names()
        n_layers = _count_indexed(input_names, "past_kv_", "_k")
        if n_layers <= 0:
            raise RuntimeError("code2wav_decoder.engine has no past_kv inputs")

        past_k_names = [f"past_kv_{i}_k" for i in range(n_layers)]
        past_v_names = [f"past_kv_{i}_v" for i in range(n_layers)]
        present_names = []
        for i in range(n_layers):
            present_names.extend([f"present_kv_{i}_k", f"present_kv_{i}_v"])
        conv_names = _indexed_names(input_names, "conv_state_")
        trans_names = _indexed_names(input_names, "transconv_overlap_")
        conv_out_names = _indexed_names(output_names, "new_conv_state_")
        trans_out_names = _indexed_names(output_names, "new_transconv_overlap_")
        if len(conv_names) != len(conv_out_names) or len(trans_names) != len(trans_out_names):
            raise RuntimeError(
                "code2wav_decoder.engine state input/output layout is inconsistent: "
                f"conv {len(conv_names)}->{len(conv_out_names)}, "
                f"transconv {len(trans_names)}->{len(trans_out_names)}"
            )

        state_dtype = self._c2w_engine._output_dtypes.get(
            "present_kv_0_k",
            self._c2w_engine._output_dtypes.get("wav", torch.bfloat16),
        )
        kv_k = [
            torch.zeros(
                _profile_shape(self._c2w_engine, name, past_len=1),
                device=device,
                dtype=state_dtype,
            )
            for name in past_k_names
        ]
        kv_v = [
            torch.zeros(
                _profile_shape(self._c2w_engine, name, past_len=1),
                device=device,
                dtype=state_dtype,
            )
            for name in past_v_names
        ]
        conv_states = [
            torch.zeros(
                _profile_shape(self._c2w_engine, name),
                device=device,
                dtype=state_dtype,
            )
            for name in conv_names
        ]
        trans_states = [
            torch.zeros(
                _profile_shape(self._c2w_engine, name),
                device=device,
                dtype=state_dtype,
            )
            for name in trans_names
        ]

        actual_past = 0
        for start in range(0, warm_frames, 4):
            chunk = codes[:, :, start:start + 4].contiguous()
            # The standalone decoder consumes 4 codec frames at a time and
            # keeps a 72-frame sliding attention window, so past+chunk must
            # not exceed 72 for the causal mask shape.
            if kv_k[0].shape[2] > 68:
                kv_k = [t[:, :, -68:, :].contiguous() for t in kv_k]
                kv_v = [t[:, :, -68:, :].contiguous() for t in kv_v]
            past_len = int(kv_k[0].shape[2])
            attn = torch.zeros(
                1, 1, 4, past_len + 4,
                device=device,
                dtype=state_dtype,
            )
            if actual_past == 0:
                attn[:, :, :, :past_len] = float("-inf")
            cache_position = torch.arange(
                start, start + 4,
                device=device,
                dtype=torch.float32,
            ).reshape(1, 4)

            inputs = {
                "codes": chunk,
                "cache_position": cache_position,
                "c2w_attention_bias": attn.contiguous(),
            }
            for i, name in enumerate(past_k_names):
                inputs[name] = kv_k[i].contiguous()
            for i, name in enumerate(past_v_names):
                inputs[name] = kv_v[i].contiguous()
            for i, name in enumerate(conv_names):
                inputs[name] = conv_states[i].contiguous()
            for i, name in enumerate(trans_names):
                inputs[name] = trans_states[i].contiguous()

            with torch.cuda.stream(self._stream):
                out = self._c2w_engine.infer(
                    inputs,
                    ["wav"] + present_names + conv_out_names + trans_out_names,
                    self._stream,
                )
            self._stream.synchronize()

            new_k = [out[f"present_kv_{i}_k"].detach().clone().contiguous() for i in range(n_layers)]
            new_v = [out[f"present_kv_{i}_v"].detach().clone().contiguous() for i in range(n_layers)]
            if actual_past == 0 and new_k[0].shape[2] > 4:
                new_k = [t[:, :, 1:, :].contiguous() for t in new_k]
                new_v = [t[:, :, 1:, :].contiguous() for t in new_v]
            if new_k[0].shape[2] > 72:
                new_k = [t[:, :, -72:, :].contiguous() for t in new_k]
                new_v = [t[:, :, -72:, :].contiguous() for t in new_v]
            kv_k, kv_v = new_k, new_v
            conv_states = [out[name].detach().clone().contiguous() for name in conv_out_names]
            trans_states = [out[name].detach().clone().contiguous() for name in trans_out_names]
            actual_past = min(actual_past + 4, 72)

        # The fused step has chunk_T=1, so its max C2W past is sliding_window - 1.
        if kv_k[0].shape[2] > 71:
            kv_k = [t[:, :, -71:, :].contiguous() for t in kv_k]
            kv_v = [t[:, :, -71:, :].contiguous() for t in kv_v]
        packed = torch.stack(
            [item for pair in zip(kv_k, kv_v) for item in pair],
            dim=1,
        ).detach().cpu().contiguous()
        return (
            packed,
            [t.detach().cpu().contiguous() for t in conv_states],
            [t.detach().cpu().contiguous() for t in trans_states],
            warm_frames,
        )


def _first_existing(*paths: Path) -> Optional[Path]:
    for path in paths:
        if path.is_file():
            return path
    return None


def _indexed_names(names: list[str], prefix: str) -> list[str]:
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    indexed = []
    for name in names:
        match = pattern.match(name)
        if match:
            indexed.append((int(match.group(1)), name))
    return [name for _, name in sorted(indexed)]


def _count_indexed(names: list[str], prefix: str, suffix: str) -> int:
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+){re.escape(suffix)}$")
    indices = []
    for name in names:
        match = pattern.match(name)
        if match:
            indices.append(int(match.group(1)))
    if not indices:
        return 0
    return max(indices) + 1


def _profile_shape(engine, name: str, *, past_len: Optional[int] = None) -> tuple[int, ...]:
    shape = list(engine._engine.get_tensor_shape(name))
    shape = [1 if int(dim) < 0 else int(dim) for dim in shape]
    if past_len is not None and len(shape) >= 3:
        shape[2] = int(past_len)
    if shape:
        shape[0] = 1
    return tuple(shape)


def _decode_wav_bytes(data: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(data), "rb") as wf:
        channels = int(wf.getnchannels())
        sample_width = int(wf.getsampwidth())
        sample_rate = int(wf.getframerate())
        frames = int(wf.getnframes())
        compression = wf.getcomptype()
        if compression != "NONE":
            raise ValueError(f"unsupported WAV compression: {compression}")
        raw = wf.readframes(frames)

    if sample_width == 1:
        audio = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 2:
        audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 3:
        audio = _pcm24_to_float32(raw)
    elif sample_width == 4:
        audio = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported WAV sample width: {sample_width}")

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return np.ascontiguousarray(np.clip(audio, -1.0, 1.0), dtype=np.float32), sample_rate


def _pcm24_to_float32(raw: bytes) -> np.ndarray:
    u8 = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
    values = u8[:, 0] | (u8[:, 1] << 8) | (u8[:, 2] << 16)
    sign = values & 0x800000
    values = values - (sign << 1)
    return values.astype(np.float32) / 8388608.0


def _resample_linear(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if audio.size == 0:
        raise ValueError("reference audio is empty")
    if src_sr == dst_sr:
        return np.ascontiguousarray(audio, dtype=np.float32)
    duration = audio.shape[0] / float(src_sr)
    dst_len = max(1, int(round(duration * dst_sr)))
    src_x = np.linspace(0.0, duration, num=audio.shape[0], endpoint=False)
    dst_x = np.linspace(0.0, duration, num=dst_len, endpoint=False)
    return np.ascontiguousarray(np.interp(dst_x, src_x, audio).astype(np.float32))


def _mel_spectrogram_24k(audio: np.ndarray, *, device):
    import torch

    y = torch.from_numpy(audio).to(device=device, dtype=torch.float32).unsqueeze(0)
    n_fft = 1024
    hop_size = 256
    win_size = 1024
    padding = (n_fft - hop_size) // 2
    y = torch.nn.functional.pad(y.unsqueeze(1), (padding, padding), mode="reflect").squeeze(1)
    window = torch.hann_window(win_size, device=device)
    spec = torch.stft(
        y,
        n_fft=n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=window,
        center=False,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    ).abs()
    mel_basis = torch.from_numpy(
        _librosa_slaney_mel(
            sr=_REF_SAMPLE_RATE,
            n_fft=n_fft,
            n_mels=128,
            fmin=0.0,
            fmax=12000.0,
        )
    ).to(device=device, dtype=torch.float32)
    mel = torch.matmul(mel_basis, spec)
    return torch.log(torch.clamp(mel, min=1e-5))


def _librosa_slaney_mel(
    *,
    sr: int,
    n_fft: int,
    n_mels: int,
    fmin: float,
    fmax: float,
) -> np.ndarray:
    fftfreqs = np.fft.rfftfreq(n=n_fft, d=1.0 / sr)
    mel_f = _mel_frequencies(n_mels + 2, fmin=fmin, fmax=fmax)
    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)
    weights = np.zeros((n_mels, int(1 + n_fft // 2)), dtype=np.float32)
    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0.0, np.minimum(lower, upper))
    weights *= (2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels]))[:, np.newaxis]
    return weights


def _mel_frequencies(n_mels: int, *, fmin: float, fmax: float) -> np.ndarray:
    min_mel = _hz_to_mel(fmin)
    max_mel = _hz_to_mel(fmax)
    return _mel_to_hz(np.linspace(min_mel, max_mel, n_mels))


def _hz_to_mel(frequencies) -> np.ndarray:
    frequencies = np.asanyarray(frequencies, dtype=float)
    f_sp = 200.0 / 3
    mels = frequencies / f_sp
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    log_t = frequencies >= min_log_hz
    log_mels = min_log_mel + np.log(np.maximum(frequencies, min_log_hz) / min_log_hz) / logstep
    return np.asarray(np.where(log_t, log_mels, mels))


def _mel_to_hz(mels) -> np.ndarray:
    mels = np.asanyarray(mels, dtype=float)
    f_sp = 200.0 / 3
    freqs = f_sp * mels
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    log_t = mels >= min_log_mel
    log_freqs = min_log_hz * np.exp(logstep * (mels - min_log_mel))
    return np.asarray(np.where(log_t, log_freqs, freqs))

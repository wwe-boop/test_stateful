"""Reference-audio capability probe for standalone voice-clone support.

This module intentionally keeps the first step lightweight:

- detect whether the current standalone deployment *can* process ref_audio
- provide a precise reason when it cannot

Actual ref-audio preprocessing can be added on top of this once the required
runtime dependencies and exported models are present in the standalone path.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ReferenceAudioSupport:
    available: bool
    reason: str = ""
    speaker_encoder_path: Optional[Path] = None
    speech_tokenizer_encoder_path: Optional[Path] = None
    speech_tokenizer_codec_fused_path: Optional[Path] = None


class ReferenceAudioProcessor:
    """Detect standalone voice-clone preprocessing capability."""

    def __init__(self, engine_dir: str, variant: str):
        self._engine_dir = Path(engine_dir) if engine_dir else Path()
        self._variant = variant or ""
        self._support = self._probe()

    @property
    def support(self) -> ReferenceAudioSupport:
        return self._support

    def _probe(self) -> ReferenceAudioSupport:
        if not self._variant.startswith("base-"):
            return ReferenceAudioSupport(
                available=False,
                reason=f"variant '{self._variant}' is not a base model; voice_clone is unsupported",
            )

        engine_dir = self._engine_dir
        speaker_encoder = engine_dir / "speaker_encoder.onnx"
        speech_tokenizer_codec_fused = engine_dir / "speech_tokenizer_codec_fused.onnx"
        speech_tokenizer_encoder = engine_dir / "speech_tokenizer_encoder.onnx"

        if not speaker_encoder.is_file():
            return ReferenceAudioSupport(
                available=False,
                reason=f"missing speaker encoder export: {speaker_encoder}",
                speaker_encoder_path=speaker_encoder,
                speech_tokenizer_encoder_path=speech_tokenizer_encoder,
                speech_tokenizer_codec_fused_path=speech_tokenizer_codec_fused,
            )

        if not importlib.util.find_spec("onnxruntime"):
            return ReferenceAudioSupport(
                available=False,
                reason="onnxruntime is not installed for standalone ref_audio preprocessing",
                speaker_encoder_path=speaker_encoder,
                speech_tokenizer_encoder_path=speech_tokenizer_encoder,
                speech_tokenizer_codec_fused_path=speech_tokenizer_codec_fused,
            )

        if not importlib.util.find_spec("numpy"):
            return ReferenceAudioSupport(
                available=False,
                reason="numpy is required for standalone ref_audio preprocessing",
                speaker_encoder_path=speaker_encoder,
                speech_tokenizer_encoder_path=speech_tokenizer_encoder,
                speech_tokenizer_codec_fused_path=speech_tokenizer_codec_fused,
            )

        return ReferenceAudioSupport(
            available=True,
            speaker_encoder_path=speaker_encoder,
            speech_tokenizer_encoder_path=speech_tokenizer_encoder if speech_tokenizer_encoder.is_file() else None,
            speech_tokenizer_codec_fused_path=speech_tokenizer_codec_fused if speech_tokenizer_codec_fused.is_file() else None,
        )

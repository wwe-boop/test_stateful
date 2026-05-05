from __future__ import annotations

import os
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .schemas import (
    RunMetrics,
    RunResult,
    TraceEvent,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY = REPO_ROOT / "third_party" / "Qwen3-TTS"
DEFAULT_MODEL_DIR = REPO_ROOT / "workspace" / "models" / "Qwen3-TTS-12Hz-1.7B-CustomVoice"


class OfficialPyTorchUnavailable(RuntimeError):
    pass


def is_official_streaming_info_warning(message: str) -> bool:
    return message.startswith(
        "Official upstream generate_custom_voice(non_streaming_mode=False) only simulates streaming text input"
    )


class _OfficialGenerationProbe:
    """Measure official PyTorch approximate TTFT without changing generated audio."""

    def __init__(self, torch_module: Any) -> None:
        self._torch = torch_module
        self.started_at: float | None = None
        self.decode_first_codec0_ms: float | None = None

    def start(self) -> None:
        self.started_at = time.perf_counter()

    def _mark(self, field: str) -> None:
        if self.started_at is None or getattr(self, field) is not None:
            return
        if self._torch.cuda.is_available():
            self._torch.cuda.synchronize()
        setattr(self, field, (time.perf_counter() - self.started_at) * 1000.0)

    def streamer(self):
        probe = self

        class FirstDecodeCode0Streamer:
            def __init__(self) -> None:
                self._seen_initial_put = False

            def put(self, token_ids) -> None:
                if not self._seen_initial_put:
                    self._seen_initial_put = True
                    return
                probe._mark("decode_first_codec0_ms")

            def end(self) -> None:
                return None

        return FirstDecodeCode0Streamer()

    @contextmanager
    def instrument(self, tts: Any) -> Iterator[None]:
        talker = tts.model.talker
        orig_generate = talker.generate
        streamer = self.streamer()

        def generate_wrapper(*args, **kwargs):
            kwargs.setdefault("streamer", streamer)
            return orig_generate(*args, **kwargs)

        talker.generate = generate_wrapper
        try:
            yield
        finally:
            talker.generate = orig_generate


class OfficialPyTorchRunner:
    def __init__(self) -> None:
        self._tts = None
        self._torch = None

    def enabled(self) -> bool:
        return os.environ.get("QWEN_DEMO_ENABLE_OFFICIAL_LIVE", "0").lower() in {"1", "true", "yes", "on"}

    def synthesize(self, *, text: str, speaker: str, language: str, streaming_mode: bool) -> RunResult:
        if not self.enabled():
            raise OfficialPyTorchUnavailable("set QWEN_DEMO_ENABLE_OFFICIAL_LIVE=1 to run official PyTorch baselines")
        tts, torch = self._load()
        run_id = f"official-{'stream' if streaming_mode else 'offline'}-{uuid.uuid4().hex[:10]}"
        backend = "official_pytorch_streaming" if streaming_mode else "official_pytorch_offline"
        label = "Official PyTorch Online Text" if streaming_mode else "Official PyTorch Offline"
        mode = "official PyTorch non_streaming_mode=false; public API returns full waveform" if streaming_mode else "official PyTorch full waveform"

        probe = _OfficialGenerationProbe(torch)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        with probe.instrument(tts):
            probe.start()
            wavs, sr = tts.generate_custom_voice(
                text=text,
                language=language if language else "auto",
                speaker=speaker,
                non_streaming_mode=not streaming_mode,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            assert probe.started_at is not None
            total_ms = (time.perf_counter() - probe.started_at) * 1000.0

        wav = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
        raw_audio = wav.tobytes()
        audio_duration_ms = wav.size / float(sr) * 1000.0
        approx_ttft_ms = probe.decode_first_codec0_ms
        events = [TraceEvent(run_id=run_id, backend=backend, type="request_started", t_ms=0.0)]
        if approx_ttft_ms is not None:
            events.append(
                TraceEvent(
                    run_id=run_id,
                    backend=backend,
                    type="official_approx_ttft",
                    t_ms=approx_ttft_ms,
                    meta={
                        "official_high_level_api": True,
                        "marker_semantics": "first generated code0 observed by talker.generate streamer after request start",
                    },
                )
            )
        events.extend(
            [
                TraceEvent(
                    run_id=run_id,
                    backend=backend,
                    type="full_audio_ready",
                    t_ms=total_ms,
                    meta={
                        "official_high_level_api": True,
                        "non_streaming_mode": not streaming_mode,
                        "audio_chunk_ttft_exposed": False,
                    },
                ),
                TraceEvent(run_id=run_id, backend=backend, type="done", t_ms=total_ms),
            ]
        )
        warnings = [
            "Official PyTorch TTFT is approximated as first decode-stage code0 timestamp minus request start; the public high-level API still returns a complete waveform."
        ]
        metrics = RunMetrics(
            first_playable_ms=approx_ttft_ms if approx_ttft_ms is not None else total_ms,
            official_approx_ttft_ms=approx_ttft_ms,
            full_audio_ready_ms=total_ms,
            total_ms=total_ms,
            audio_duration_ms=audio_duration_ms,
            chunks=1,
            cache_hit=None,
        )
        if approx_ttft_ms is None:
            warnings.append(
                "Official approximate TTFT probe did not fire; WebUI uses public full-wav ready time for this official row."
            )
        if streaming_mode:
            warnings.append(
                "Official upstream generate_custom_voice(non_streaming_mode=False) only simulates streaming text input and still returns a complete waveform; true audio chunk TTFT is not exposed by the public high-level API."
            )
        return RunResult(
            run_id=run_id,
            backend=backend,
            label=label,
            mode=mode,
            source="live_official_pytorch",
            metrics=metrics,
            events=events,
            warnings=warnings,
            audio_format={"encoding": "pcm_f32", "sample_rate": int(sr), "channels": 1},
            raw_audio=raw_audio,
        )

    def _load(self):
        if self._tts is not None and self._torch is not None:
            return self._tts, self._torch
        if str(THIRD_PARTY) not in sys.path:
            sys.path.insert(0, str(THIRD_PARTY))
        try:
            import torch
            from qwen_tts import Qwen3TTSModel
        except Exception as exc:
            raise OfficialPyTorchUnavailable(f"official Qwen3-TTS imports failed: {exc}") from exc

        model_dir = Path(os.environ.get("QWEN_DEMO_OFFICIAL_MODEL_DIR", str(DEFAULT_MODEL_DIR)))
        if not model_dir.exists():
            raise OfficialPyTorchUnavailable(f"official model directory not found: {model_dir}")
        dtype_name = os.environ.get("QWEN_DEMO_OFFICIAL_DTYPE", "bf16").lower()
        dtype = torch.bfloat16 if dtype_name in {"bf16", "bfloat16"} else torch.float16
        self._tts = Qwen3TTSModel.from_pretrained(
            str(model_dir),
            device_map=os.environ.get("QWEN_DEMO_OFFICIAL_DEVICE", "cuda:0"),
            dtype=dtype,
            attn_implementation=os.environ.get("QWEN_DEMO_OFFICIAL_ATTN", "eager"),
        )
        self._torch = torch
        return self._tts, self._torch

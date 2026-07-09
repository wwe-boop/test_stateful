#!/usr/bin/env python3
"""Run timed Table 2 probes for available SteadyStream variants.

This runner is deliberately conservative: it measures only variants that are
callable through the current engine API and keeps C1/C2/C3/C4 rows separate
until implementation switches exist.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import grpc
import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine.gateway import tts_pb2, tts_pb2_grpc
from eval.boundary_metrics import (
    boundary_positions_from_concat,
    boundary_positions_proportional,
    measure_boundary_metrics,
    summarize_boundary_metrics,
)
from eval.pause_metrics import measure_pause_metrics, summarize_pause_metrics


EXPECTED_PAUSE_MS = {
    "comma": 220.0,
    "period": 460.0,
    "question": 480.0,
    "exclamation": 430.0,
    "semicolon": 300.0,
    "colon": 260.0,
}

C2_DIAGNOSTIC_VARIANTS = [
    (
        "full_current_silence",
        "full_current_silence.wav",
        {
            "steadystream_variant": "full_steadystream",
            "kv_tail_tokens": "384",
            "kv_reprefill_token_history": "true",
            "kv_reprefill_token_history_full_current": "true",
            "kv_terminal_drop_mode": "silence",
        },
        False,
    ),
    (
        "full_current_eos_only",
        "full_current_eos_only.wav",
        {
            "steadystream_variant": "full_steadystream",
            "kv_tail_tokens": "384",
            "kv_reprefill_token_history": "true",
            "kv_reprefill_token_history_full_current": "true",
            "kv_terminal_drop_mode": "eos_only",
        },
        False,
    ),
    (
        "full_current_codes_only_silence",
        "full_current_codes_only_silence.wav",
        {
            "steadystream_variant": "full_steadystream",
            "kv_tail_tokens": "384",
            "kv_reprefill_token_history": "true",
            "kv_reprefill_token_history_full_current": "true",
            "kv_terminal_drop_mode": "silence",
            "kv_reprefill_token_history_drop_history_text": "true",
        },
        False,
    ),
]

def c2_diagnostic_variants(
    kv_tail_tokens: int | None = None,
) -> list[tuple[str, str, dict[str, str], bool]]:
    tail = 384 if kv_tail_tokens is None else int(kv_tail_tokens)
    suffix = "" if kv_tail_tokens is None else f"_tail{tail}"
    variants: list[tuple[str, str, dict[str, str], bool]] = []
    for key, wav_name, experimental, pause_recovery in C2_DIAGNOSTIC_VARIANTS:
        config = dict(experimental)
        config["kv_tail_tokens"] = str(tail)
        variants.append(
            (
                f"{key}{suffix}",
                wav_name.replace(".wav", f"{suffix}.wav"),
                config,
                pause_recovery,
            )
        )
    return variants


EXPERIMENTAL_VARIANTS = [
    (
        "acoustic_tail_only",
        "acoustic_tail_only.wav",
        {"steadystream_variant": "acoustic_tail_only"},
        False,
    ),
    (
        "kv_tail_only",
        "kv_tail_only.wav",
        {"steadystream_variant": "kv_tail_only", "kv_tail_tokens": "384"},
        False,
    ),
    (
        "tail_kv_pause_recovery",
        "tail_kv_pause_recovery.wav",
        {"steadystream_variant": "tail_kv_pause_recovery", "kv_tail_tokens": "384"},
        True,
    ),
    (
        "full_steadystream",
        "full_steadystream.wav",
        {"steadystream_variant": "full_steadystream", "kv_tail_tokens": "384"},
        True,
    ),
]

REQUIRED_AUDIO_BLOCKS = [
    ("stateless_once", "stateless_once.wav"),
    ("stateful_stream", "stateful_stream.wav"),
    ("offline_full", "offline_full.wav"),
    *[(key, wav_name) for key, wav_name, _, _ in EXPERIMENTAL_VARIANTS],
]


def has_audio_block(result: dict[str, Any], out_dir: Path, key: str, wav_name: str) -> bool:
    return key in result and (out_dir / wav_name).is_file()


def audio_blocks_for_run(
    include_c2_diagnostics: bool = False,
    c2_diagnostic_kv_tail_tokens: int | None = None,
) -> list[tuple[str, str]]:
    blocks = list(REQUIRED_AUDIO_BLOCKS)
    if include_c2_diagnostics:
        blocks.extend(
            (key, wav_name)
            for key, wav_name, _, _ in c2_diagnostic_variants(
                c2_diagnostic_kv_tail_tokens
            )
        )
    return blocks


def sample_complete(
    sample_dir: Path,
    include_c2_diagnostics: bool = False,
    c2_diagnostic_kv_tail_tokens: int | None = None,
) -> bool:
    result_path = sample_dir / "results.json"
    if not result_path.is_file():
        return False
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return all(
        has_audio_block(result, sample_dir, key, wav_name)
        for key, wav_name in audio_blocks_for_run(
            include_c2_diagnostics,
            c2_diagnostic_kv_tail_tokens,
        )
    )


def input_mode_to_proto(name: str) -> int:
    return {
        "token": tts_pb2.INPUT_MODE_TOKEN,
        "clause": tts_pb2.INPUT_MODE_CLAUSE,
        "long_segment": tts_pb2.INPUT_MODE_LONG_SEGMENT,
        "full_text": tts_pb2.INPUT_MODE_FULL_TEXT,
    }[name]


def group_policy_to_proto(name: str) -> int:
    return {
        "none": tts_pb2.GROUP_POLICY_NONE,
        "auto": tts_pb2.GROUP_POLICY_AUTO,
    }[name]


def make_session_config(
    *,
    task_type: str,
    speaker: str,
    language: str,
    instruct: str,
    input_mode: str,
    group_policy: str,
    sample_rate: int,
    experimental: dict[str, str] | None = None,
) -> Any:
    cfg = tts_pb2.SessionConfig(
        task_type=task_type,
        speaker=speaker,
        language=language,
        instruct=instruct,
        input_mode=input_mode_to_proto(input_mode),
        group_policy=group_policy_to_proto(group_policy),
        audio=tts_pb2.AudioFormat(
            encoding=tts_pb2.AUDIO_ENCODING_PCM_F32,
            sample_rate=sample_rate,
            channels=1,
        ),
    )
    if experimental:
        cfg.experimental.update({str(k): str(v) for k, v in experimental.items()})
    return cfg


def stream_experimental_config(
    base: dict[str, str] | None,
    *,
    force_text_chunk_boundary: bool,
) -> dict[str, str]:
    cfg = {str(k): str(v) for k, v in (base or {}).items()}
    if force_text_chunk_boundary:
        cfg.setdefault("force_text_chunk_boundary", "true")
    return cfg


def synthesize_once_timed(
    stub: Any,
    *,
    endpoint_timeout: float,
    session_id: str,
    text: str,
    task_type: str,
    speaker: str,
    language: str,
    instruct: str,
    sample_rate: int,
    input_mode: str = "full_text",
) -> tuple[np.ndarray, int, dict[str, Any]]:
    request = tts_pb2.SynthesizeOnceRequest(
        session_id=session_id,
        text=text,
        config=make_session_config(
            task_type=task_type,
            speaker=speaker,
            language=language,
            instruct=instruct,
            input_mode=input_mode,
            group_policy="none",
            sample_rate=sample_rate,
        ),
    )
    t0 = time.perf_counter()
    first_audio_ms = None
    chunks: list[np.ndarray] = []
    events: list[dict[str, Any]] = []
    sr = sample_rate
    for response in stub.SynthesizeOnce(request, timeout=endpoint_timeout):
        now = time.perf_counter()
        which = response.WhichOneof("response")
        if which == "audio":
            if first_audio_ms is None:
                first_audio_ms = (now - t0) * 1000.0
            sr = int(response.audio.sample_rate or sr)
            chunks.append(np.frombuffer(response.audio.pcm_data, dtype=np.float32))
        elif which == "event":
            events.append(
                {
                    "type": response.event.type,
                    "segment_id": response.event.segment_id,
                    "text": response.event.text,
                    "message": response.event.message,
                    "dt_ms": round((now - t0) * 1000.0, 2),
                    "meta": dict(response.event.meta),
                }
            )
            if response.event.type == "error":
                raise RuntimeError(response.event.message or "engine oneshot error")
    wav = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    total_ms = (time.perf_counter() - t0) * 1000.0
    return wav, sr, {
        "session_id": session_id,
        "first_audio_ms": round(first_audio_ms, 2) if first_audio_ms is not None else None,
        "total_ms": round(total_ms, 2),
        "events": events,
        "audio_chunks": len(chunks),
    }


def synthesize_stream_timed(
    stub: Any,
    *,
    endpoint_timeout: float,
    session_id: str,
    segments: list[str],
    task_type: str,
    speaker: str,
    language: str,
    instruct: str,
    sample_rate: int,
    input_mode: str = "clause",
    group_policy: str = "none",
    experimental: dict[str, str] | None = None,
) -> tuple[np.ndarray, int, list[int], dict[str, Any]]:
    text_send_times: list[dict[str, Any]] = []

    def request_gen():
        yield tts_pb2.SynthesizeRequest(
            start=tts_pb2.StartRequest(
                session_id=session_id,
                config=make_session_config(
                    task_type=task_type,
                    speaker=speaker,
                    language=language,
                    instruct=instruct,
                    input_mode=input_mode,
                    group_policy=group_policy,
                    sample_rate=sample_rate,
                    experimental=experimental,
                ),
            )
        )
        for idx, text in enumerate(segments):
            text_send_times.append(
                {"segment_index": idx, "text": text, "t": time.perf_counter()}
            )
            yield tts_pb2.SynthesizeRequest(text=tts_pb2.TextChunk(text=text))
        yield tts_pb2.SynthesizeRequest(end=tts_pb2.EndRequest())

    t0 = time.perf_counter()
    first_audio_ms = None
    chunks: list[np.ndarray] = []
    exact_boundaries: list[int] = []
    audio_samples_seen = 0
    events: list[dict[str, Any]] = []
    sr = sample_rate
    for response in stub.SynthesizeStream(request_gen(), timeout=endpoint_timeout):
        now = time.perf_counter()
        which = response.WhichOneof("response")
        if which == "audio":
            if first_audio_ms is None:
                first_audio_ms = (now - t0) * 1000.0
            sr = int(response.audio.sample_rate or sr)
            chunk = np.frombuffer(response.audio.pcm_data, dtype=np.float32)
            chunks.append(chunk)
            audio_samples_seen += len(chunk)
        elif which == "event":
            events.append(
                {
                    "type": response.event.type,
                    "segment_id": response.event.segment_id,
                    "text": response.event.text,
                    "message": response.event.message,
                    "dt_ms": round((now - t0) * 1000.0, 2),
                    "audio_samples_seen": audio_samples_seen,
                    "meta": dict(response.event.meta),
                }
            )
            if response.event.type == "text_boundary_commit":
                exact_boundaries.append(audio_samples_seen)
            if response.event.type == "error":
                raise RuntimeError(response.event.message or "engine stream error")
    wav = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    total_ms = (time.perf_counter() - t0) * 1000.0
    if len(exact_boundaries) > max(0, len(segments) - 1):
        exact_boundaries = exact_boundaries[: len(segments) - 1]
    return wav, sr, exact_boundaries, {
        "session_id": session_id,
        "first_audio_ms": round(first_audio_ms, 2) if first_audio_ms is not None else None,
        "total_ms": round(total_ms, 2),
        "events": events,
        "text_send_times": [
            {
                "segment_index": item["segment_index"],
                "text": item["text"],
                "dt_ms": round((item["t"] - t0) * 1000.0, 2),
            }
            for item in text_send_times
        ],
        "audio_chunks": len(chunks),
        "exact_boundary_count": len(exact_boundaries),
    }


def pause_summary_for(
    audio: np.ndarray,
    sample_rate: int,
    boundaries: list[int],
    puncts: list[str],
) -> dict[str, Any]:
    expected = [
        EXPECTED_PAUSE_MS.get(puncts[i], 460.0)
        for i in range(min(len(boundaries), len(puncts)))
    ]
    boundaries = boundaries[: len(expected)]
    if not boundaries:
        return {"pause_deviation_mean_ms": None, "pause_coverage": 0.0}
    items = measure_pause_metrics(audio, sample_rate, boundaries, expected)
    return {
        **summarize_pause_metrics(items),
        "pause_metrics": items,
    }


def silence_span_around_boundary(
    audio: np.ndarray,
    sample_rate: int,
    boundary_idx: int,
    *,
    search_radius_ms: float = 700.0,
    energy_threshold_db: float = -40.0,
) -> tuple[int, int] | None:
    radius = int(search_radius_ms * sample_rate / 1000.0)
    start = max(0, boundary_idx - radius)
    end = min(len(audio), boundary_idx + radius)
    window = audio[start:end]
    if len(window) < 512:
        return None
    frame_length = 512
    hop_length = 256
    frames = np.lib.stride_tricks.sliding_window_view(window, frame_length)[::hop_length]
    if frames.size == 0:
        return None
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    energy_db = 20 * np.log10(rms + 1e-12)
    silent = energy_db < energy_threshold_db
    boundary_frame = min(max((boundary_idx - start) // hop_length, 0), len(silent) - 1)
    if not bool(silent[boundary_frame]):
        return None
    left = boundary_frame
    while left > 0 and bool(silent[left - 1]):
        left -= 1
    right = boundary_frame
    while right < len(silent) - 1 and bool(silent[right + 1]):
        right += 1
    span_start = start + left * hop_length
    span_end = min(start + right * hop_length + frame_length, len(audio))
    if span_end <= span_start:
        return None
    return span_start, span_end


def apply_pause_recovery(
    audio: np.ndarray,
    sample_rate: int,
    boundaries: list[int],
    puncts: list[str],
    *,
    tolerance_ms: float = 80.0,
) -> tuple[np.ndarray, list[int], list[dict[str, Any]]]:
    """Inference-side C3 pause recovery for table probes.

    It replaces the detected silent span around each boundary with the
    punctuation-conditioned target silence. If no silence is detected, it
    inserts the target silence at the boundary.
    """
    if not boundaries:
        return audio, boundaries, []
    out = audio.astype(np.float32, copy=True)
    adjusted_boundaries: list[int] = []
    meta: list[dict[str, Any]] = []
    offset = 0
    for idx, raw_boundary in enumerate(boundaries):
        punct = puncts[idx] if idx < len(puncts) else "period"
        target_ms = EXPECTED_PAUSE_MS.get(punct, 460.0)
        target = int(round(target_ms * sample_rate / 1000.0))
        boundary = max(0, min(len(out), int(raw_boundary + offset)))
        span = silence_span_around_boundary(out, sample_rate, boundary)
        if span is None:
            left = right = boundary
            current = 0
        else:
            left, right = span
            current = right - left
        current_ms = current * 1000.0 / sample_rate
        if abs(current_ms - target_ms) <= tolerance_ms:
            new_boundary = boundary
            delta = 0
            action = "keep"
        else:
            replacement = np.zeros(target, dtype=np.float32)
            out = np.concatenate([out[:left], replacement, out[right:]])
            new_boundary = left + target // 2
            delta = target - current
            offset += delta
            action = "replace" if current > 0 else "insert"
        adjusted_boundaries.append(new_boundary)
        meta.append(
            {
                "boundary": idx + 1,
                "punct": punct,
                "target_ms": round(target_ms, 1),
                "before_ms": round(current_ms, 1),
                "action": action,
                "delta_samples": int(delta),
            }
        )
    return out, adjusted_boundaries, meta


def stream_variant_block(
    stub: Any,
    *,
    variant_key: str,
    wav_name: str,
    endpoint_timeout: float,
    session_id: str,
    segments: list[str],
    task_type: str,
    speaker: str,
    language: str,
    instruct: str,
    sample_rate: int,
    puncts: list[str],
    out_dir: Path,
    stream_input_mode: str = "token",
    stream_group_policy: str = "none",
    require_exact_boundaries: bool = True,
    force_text_chunk_boundary: bool = False,
    experimental: dict[str, str] | None = None,
    pause_recovery: bool = False,
) -> dict[str, Any]:
    stream_experimental = stream_experimental_config(
        experimental,
        force_text_chunk_boundary=force_text_chunk_boundary,
    )
    stream_audio, stream_sr, exact_boundaries, stream_timing = synthesize_stream_timed(
        stub,
        endpoint_timeout=endpoint_timeout,
        session_id=session_id,
        segments=segments,
        task_type=task_type,
        speaker=speaker,
        language=language,
        instruct=instruct,
        sample_rate=sample_rate,
        input_mode=stream_input_mode,
        group_policy=stream_group_policy,
        experimental=stream_experimental,
    )
    expected_boundary_count = max(0, len(segments) - 1)
    boundary_valid = len(exact_boundaries) == expected_boundary_count
    if require_exact_boundaries and not boundary_valid:
        raise RuntimeError(
            f"{session_id}: expected {expected_boundary_count} exact boundaries, "
            f"got {len(exact_boundaries)}; refusing proxy-proportional Table 2 metrics"
        )
    stream_boundaries = (
        exact_boundaries
        if boundary_valid
        else boundary_positions_proportional(stream_audio, segments)
    )
    stream_source = (
        "exact_event"
        if boundary_valid
        else "proxy_proportional"
    )
    pause_recovery_meta: list[dict[str, Any]] = []
    if pause_recovery:
        stream_audio, stream_boundaries, pause_recovery_meta = apply_pause_recovery(
            stream_audio,
            stream_sr,
            stream_boundaries,
            puncts,
        )
        stream_source = f"{stream_source}+pause_recovery"
    stream_metrics = measure_boundary_metrics(
        stream_audio,
        stream_sr,
        stream_boundaries,
        boundary_source=stream_source,
    )
    sf.write(out_dir / wav_name, stream_audio, stream_sr)
    stream_pause = pause_summary_for(stream_audio, stream_sr, stream_boundaries, puncts)
    block = {
        "audio_sec": round(len(stream_audio) / stream_sr, 3) if stream_sr else 0.0,
        "boundary_metrics": stream_metrics,
        "boundary_summary": summarize_boundary_metrics(stream_metrics),
        "pause_summary": {k: v for k, v in stream_pause.items() if k != "pause_metrics"},
        "pause_metrics": stream_pause.get("pause_metrics", []),
        "exact_boundary_count": len(exact_boundaries),
        "boundary_expected_count": expected_boundary_count,
        "boundary_valid": boundary_valid,
        "boundary_source_used": stream_source,
        "stream_input_mode": stream_input_mode,
        "stream_group_policy": stream_group_policy,
        "experimental": stream_experimental,
        "timing": {
            "fasl_first_audio_ms": stream_timing.get("first_audio_ms"),
            "total_ms": stream_timing.get("total_ms"),
            "audio_chunks": stream_timing.get("audio_chunks"),
        },
        "events": stream_timing.get("events", []),
    }
    if pause_recovery_meta:
        block["pause_recovery"] = pause_recovery_meta
    return block


def run_sample(
    stub: Any,
    row: dict[str, Any],
    *,
    seed: int,
    out_dir: Path,
    endpoint_timeout: float,
    sample_rate: int,
    stream_input_mode: str = "token",
    stream_group_policy: str = "none",
    require_exact_boundaries: bool = True,
    force_text_chunk_boundary: bool = False,
    resume: bool = False,
    include_c2_diagnostics: bool = False,
    c2_diagnostic_kv_tail_tokens: int | None = None,
    override_language: str | None = None,
    override_instruct: str | None = None,
) -> dict[str, Any]:
    sample_id = row["sample_id"]
    task_type = "custom_voice"
    speaker = row.get("speaker", "Vivian")
    language = override_language if override_language is not None else row.get("language", "Chinese")
    instruct = override_instruct if override_instruct is not None else row.get("instruct", "")
    segments = [seg["text"] for seg in row["segments"]]
    puncts = row.get("boundary_punct_classes", [])
    full_text = "".join(segments)
    out_dir.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {
        "sample_id": sample_id,
        "seed": seed,
        "speaker": speaker,
        "language": language,
        "instruct": instruct,
        "segments": segments,
        "boundary_punct_classes": puncts,
    }
    result_path = out_dir / "results.json"
    if resume and result_path.exists():
        try:
            existing = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
        for key, value in existing.items():
            if key not in result:
                result[key] = value
        for key, _wav_name in REQUIRED_AUDIO_BLOCKS:
            if key in existing:
                result[key] = existing[key]

    # Row 1: stateless per-segment oneshot.
    sr = sample_rate
    if not has_audio_block(result, out_dir, "stateless_once", "stateless_once.wav"):
        stateless_parts: list[np.ndarray] = []
        stateless_timings: list[dict[str, Any]] = []
        for idx, seg in enumerate(segments):
            wav, sr, timing = synthesize_once_timed(
                stub,
                endpoint_timeout=endpoint_timeout,
                session_id=f"t2-{seed}-{sample_id}-stateless-{idx}",
                text=seg,
                task_type=task_type,
                speaker=speaker,
                language=language,
                instruct=instruct,
                sample_rate=sample_rate,
            )
            stateless_parts.append(wav)
            stateless_timings.append(timing)
        stateless_audio = (
            np.concatenate(stateless_parts) if stateless_parts else np.zeros(0, dtype=np.float32)
        )
        stateless_boundaries = boundary_positions_from_concat(stateless_parts)
        stateless_metrics = measure_boundary_metrics(
            stateless_audio,
            sr,
            stateless_boundaries,
            boundary_source="exact_concat",
        )
        sf.write(out_dir / "stateless_once.wav", stateless_audio, sr)
        stateless_pause = pause_summary_for(stateless_audio, sr, stateless_boundaries, puncts)
        result["stateless_once"] = {
            "audio_sec": round(len(stateless_audio) / sr, 3) if sr else 0.0,
            "boundary_metrics": stateless_metrics,
            "boundary_summary": summarize_boundary_metrics(stateless_metrics),
            "pause_summary": {k: v for k, v in stateless_pause.items() if k != "pause_metrics"},
            "pause_metrics": stateless_pause.get("pause_metrics", []),
            "timing": {
                "segment_first_audio_ms": [
                    t.get("first_audio_ms") for t in stateless_timings
                ],
                "fasl_mean_ms": round(
                    float(np.mean([t["first_audio_ms"] for t in stateless_timings if t.get("first_audio_ms") is not None])),
                    2,
                ) if any(t.get("first_audio_ms") is not None for t in stateless_timings) else None,
                "total_ms": round(float(sum(t.get("total_ms") or 0.0 for t in stateless_timings)), 2),
            },
        }

    # Row 2: current stateful stream. Table 2 SteadyStream measurements require
    # token-mode input so each designed clause can become an engine segment.
    if not has_audio_block(result, out_dir, "stateful_stream", "stateful_stream.wav"):
        stream_experimental = stream_experimental_config(
            None,
            force_text_chunk_boundary=force_text_chunk_boundary,
        )
        stream_audio, stream_sr, stream_exact_boundaries, stream_timing = synthesize_stream_timed(
            stub,
            endpoint_timeout=endpoint_timeout,
            session_id=f"t2-{seed}-{sample_id}-stateful",
            segments=segments,
            task_type=task_type,
            speaker=speaker,
            language=language,
            instruct=instruct,
            sample_rate=sample_rate,
            input_mode=stream_input_mode,
            group_policy=stream_group_policy,
            experimental=stream_experimental,
        )
        expected_boundary_count = max(0, len(segments) - 1)
        boundary_valid = len(stream_exact_boundaries) == expected_boundary_count
        if require_exact_boundaries and not boundary_valid:
            raise RuntimeError(
                f"t2-{seed}-{sample_id}-stateful: expected {expected_boundary_count} "
                f"exact boundaries, got {len(stream_exact_boundaries)}; "
                "refusing proxy-proportional Table 2 metrics"
            )
        stream_boundaries = (
            stream_exact_boundaries
            if boundary_valid
            else boundary_positions_proportional(stream_audio, segments)
        )
        stream_source = (
            "exact_event"
            if boundary_valid
            else "proxy_proportional"
        )
        stream_metrics = measure_boundary_metrics(
            stream_audio,
            stream_sr,
            stream_boundaries,
            boundary_source=stream_source,
        )
        sf.write(out_dir / "stateful_stream.wav", stream_audio, stream_sr)
        stream_pause = pause_summary_for(stream_audio, stream_sr, stream_boundaries, puncts)
        result["stateful_stream"] = {
            "audio_sec": round(len(stream_audio) / stream_sr, 3) if stream_sr else 0.0,
            "boundary_metrics": stream_metrics,
            "boundary_summary": summarize_boundary_metrics(stream_metrics),
            "pause_summary": {k: v for k, v in stream_pause.items() if k != "pause_metrics"},
            "pause_metrics": stream_pause.get("pause_metrics", []),
            "exact_boundary_count": len(stream_exact_boundaries),
            "boundary_expected_count": expected_boundary_count,
            "boundary_valid": boundary_valid,
            "boundary_source_used": stream_source,
            "stream_input_mode": stream_input_mode,
            "stream_group_policy": stream_group_policy,
            "experimental": stream_experimental,
            "timing": {
                "fasl_first_audio_ms": stream_timing.get("first_audio_ms"),
                "total_ms": stream_timing.get("total_ms"),
                "audio_chunks": stream_timing.get("audio_chunks"),
            },
            "events": stream_timing.get("events", []),
        }

    variants_to_run = list(EXPERIMENTAL_VARIANTS)
    if include_c2_diagnostics:
        variants_to_run.extend(c2_diagnostic_variants(c2_diagnostic_kv_tail_tokens))
    for variant_key, wav_name, experimental, pause_recovery in variants_to_run:
        if has_audio_block(result, out_dir, variant_key, wav_name):
            continue
        result[variant_key] = stream_variant_block(
            stub,
            variant_key=variant_key,
            wav_name=wav_name,
            endpoint_timeout=endpoint_timeout,
            session_id=f"t2-{seed}-{sample_id}-{variant_key}",
            segments=segments,
            task_type=task_type,
            speaker=speaker,
            language=language,
            instruct=instruct,
            sample_rate=sample_rate,
            puncts=puncts,
            out_dir=out_dir,
            stream_input_mode=stream_input_mode,
            stream_group_policy=stream_group_policy,
            require_exact_boundaries=require_exact_boundaries,
            force_text_chunk_boundary=force_text_chunk_boundary,
            experimental=experimental,
            pause_recovery=pause_recovery,
        )

    # Offline full reference for SIM/topline.
    if not has_audio_block(result, out_dir, "offline_full", "offline_full.wav"):
        offline_audio, offline_sr, offline_timing = synthesize_once_timed(
            stub,
            endpoint_timeout=endpoint_timeout,
            session_id=f"t2-{seed}-{sample_id}-offline",
            text=full_text,
            task_type=task_type,
            speaker=speaker,
            language=language,
            instruct=instruct,
            sample_rate=sample_rate,
        )
        offline_boundaries = boundary_positions_proportional(offline_audio, segments)
        offline_metrics = measure_boundary_metrics(
            offline_audio,
            offline_sr,
            offline_boundaries,
            boundary_source="proxy_proportional",
        )
        sf.write(out_dir / "offline_full.wav", offline_audio, offline_sr)
        offline_pause = pause_summary_for(offline_audio, offline_sr, offline_boundaries, puncts)
        result["offline_full"] = {
            "audio_sec": round(len(offline_audio) / offline_sr, 3) if offline_sr else 0.0,
            "boundary_metrics_proxy": offline_metrics,
            "boundary_summary_proxy": summarize_boundary_metrics(offline_metrics),
            "pause_summary": {k: v for k, v in offline_pause.items() if k != "pause_metrics"},
            "pause_metrics": offline_pause.get("pause_metrics", []),
            "timing": {
                "first_audio_ms": offline_timing.get("first_audio_ms"),
                "total_ms": offline_timing.get("total_ms"),
            },
        }

    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="workspace/datasets/test-prosody-mini.jsonl")
    parser.add_argument("--out-root", default="workspace/table2_runs")
    parser.add_argument("--endpoint", default="127.0.0.1:50051")
    parser.add_argument("--seeds", default="42,123,456")
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--stream-input-mode",
        choices=["token", "clause", "long_segment"],
        default="token",
        help="Streaming mode for stateful/SteadyStream rows. Table 2 diagnostics default to token.",
    )
    parser.add_argument(
        "--stream-group-policy",
        choices=["none", "auto"],
        default="none",
    )
    parser.add_argument(
        "--allow-proxy-boundaries",
        action="store_true",
        help="Allow proxy_proportional boundary metrics when exact text_boundary_commit events are missing.",
    )
    parser.add_argument(
        "--disable-force-text-chunk-boundary",
        action="store_true",
        help="Do not force each input TextChunk to become one engine segment in token-mode diagnostics.",
    )
    parser.add_argument(
        "--include-c2-diagnostics",
        action="store_true",
        help="Also run token-history full-current C2 diagnostic variants.",
    )
    parser.add_argument(
        "--c2-diagnostic-kv-tail-tokens",
        type=int,
        default=None,
        help="Override kv_tail_tokens for C2 diagnostic variants and suffix their output keys.",
    )
    parser.add_argument(
        "--override-language",
        default=None,
        help="Override dataset language for serving-prefix diagnostics, e.g. auto.",
    )
    parser.add_argument(
        "--override-instruct",
        default=None,
        help="Override dataset instruct for serving-prefix diagnostics; pass an empty string to disable instruct.",
    )
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in Path(args.dataset).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit > 0:
        rows = rows[: args.limit]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    force_text_chunk_boundary = (
        args.stream_input_mode == "token"
        and not args.disable_force_text_chunk_boundary
    )

    channel = grpc.insecure_channel(args.endpoint)
    grpc.channel_ready_future(channel).result(timeout=10.0)
    stub = tts_pb2_grpc.TTSServiceStub(channel)

    progress: list[dict[str, Any]] = []
    for seed in seeds:
        for idx, row in enumerate(rows, start=1):
            sample_id = row["sample_id"]
            sample_dir = out_root / f"seed_{seed}" / sample_id
            if args.resume and sample_complete(
                sample_dir,
                args.include_c2_diagnostics,
                args.c2_diagnostic_kv_tail_tokens,
            ):
                print(f"[skip] seed={seed} {idx:03d}/{len(rows)} {sample_id}", flush=True)
                progress.append({"seed": seed, "sample_id": sample_id, "status": "skipped"})
                continue
            print(f"[run] seed={seed} {idx:03d}/{len(rows)} {sample_id}", flush=True)
            t0 = time.perf_counter()
            try:
                run_sample(
                    stub,
                    row,
                    seed=seed,
                    out_dir=sample_dir,
                    endpoint_timeout=args.timeout,
                    sample_rate=args.sample_rate,
                    stream_input_mode=args.stream_input_mode,
                    stream_group_policy=args.stream_group_policy,
                    require_exact_boundaries=not args.allow_proxy_boundaries,
                    force_text_chunk_boundary=force_text_chunk_boundary,
                    resume=args.resume,
                    include_c2_diagnostics=args.include_c2_diagnostics,
                    c2_diagnostic_kv_tail_tokens=args.c2_diagnostic_kv_tail_tokens,
                    override_language=args.override_language,
                    override_instruct=args.override_instruct,
                )
                elapsed = time.perf_counter() - t0
                print(f"[ok] seed={seed} {sample_id} elapsed={elapsed:.1f}s", flush=True)
                progress.append(
                    {
                        "seed": seed,
                        "sample_id": sample_id,
                        "status": "ok",
                        "elapsed_sec": round(elapsed, 2),
                    }
                )
            except Exception as exc:
                elapsed = time.perf_counter() - t0
                print(f"[failed] seed={seed} {sample_id}: {exc}", flush=True)
                progress.append(
                    {
                        "seed": seed,
                        "sample_id": sample_id,
                        "status": "failed",
                        "elapsed_sec": round(elapsed, 2),
                        "error": str(exc),
                    }
                )
                (out_root / "progress.json").write_text(
                    json.dumps(progress, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                raise
            (out_root / "progress.json").write_text(
                json.dumps(progress, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
    print(f"[done] wrote runs -> {out_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

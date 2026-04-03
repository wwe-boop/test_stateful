"""Generate WAV audio samples via Triton TTS for listening evaluation.

Usage:
    python tests/gen_audio.py

Requires:
    - Triton server running: bash scripts/bash/deploy.sh run --gateway triton
    - tritonclient[grpc]: pip install tritonclient[grpc]

Outputs WAV files to workspace/audio_samples/
"""
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "workspace" / "audio_samples"
SAMPLE_RATE = 24000

GRPC_HOST = "localhost"
GRPC_PORT = 8001

SAMPLES = [
    {
        "name": "01_greeting",
        "text": "你好，欢迎使用Qwen3-TTS语音合成系统。",
        "speaker": "zhitian",
    },
    {
        "name": "02_weather",
        "text": "今天天气晴朗，万里无云，非常适合外出活动。",
        "speaker": "zhitian",
    },
    {
        "name": "03_story",
        "text": "从前有座山，山上有座庙，庙里有个老和尚在给小和尚讲故事。",
        "speaker": "zhitian",
    },
    {
        "name": "04_english",
        "text": "Hello, this is a test of the Qwen3 text to speech system. How does it sound?",
        "speaker": "zhitian",
    },
    {
        "name": "05_mixed",
        "text": "深度学习领域的Transformer架构，自2017年提出以来，已经彻底改变了自然语言处理的格局。",
        "speaker": "zhitian",
    },
]


def make_wav(samples_f32: np.ndarray, sr: int = SAMPLE_RATE) -> bytes:
    """Convert float32 samples to 16-bit PCM WAV bytes."""
    pcm16 = np.clip(samples_f32 * 32767, -32768, 32767).astype(np.int16)
    n = pcm16.size
    buf = bytearray()
    buf += b"RIFF"
    buf += struct.pack("<I", 36 + n * 2)
    buf += b"WAVEfmt "
    buf += struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16)
    buf += b"data"
    buf += struct.pack("<I", n * 2)
    buf += pcm16.tobytes()
    return bytes(buf)


def stream_tts(client, text: str, speaker: str, timeout: float = 60.0):
    """Send TTS request and collect all audio chunks."""
    grpcclient = sys.modules["tritonclient.grpc"]
    req_dict = {
        "text": text,
        "task_type": "custom_voice",
        "speaker": speaker,
    }
    req_json = json.dumps(req_dict)
    req_input = grpcclient.InferInput("request", [1], "BYTES")
    req_input.set_data_from_numpy(np.array([req_json], dtype=object))
    audio_out = grpcclient.InferRequestedOutput("audio_chunk")
    final_out = grpcclient.InferRequestedOutput("is_final")

    chunks = []
    errors = []
    done = False
    first_time = None

    def callback(result, error):
        nonlocal done, first_time
        if error:
            errors.append(str(error))
            done = True
            return
        if first_time is None:
            first_time = time.perf_counter()
        audio = result.as_numpy("audio_chunk")
        is_final = result.as_numpy("is_final")
        chunks.append(audio.flatten())
        if is_final is not None and is_final.size and is_final.flatten()[0]:
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

    total = time.perf_counter() - t0
    first_sec = (first_time - t0) if first_time else None
    err = errors[0] if errors else None
    return chunks, first_sec, total, err


def main():
    try:
        import tritonclient.grpc as grpcclient
    except ImportError:
        print("ERROR: tritonclient[grpc] not installed.")
        print("  pip install tritonclient[grpc]")
        sys.exit(1)

    client = grpcclient.InferenceServerClient(url=f"{GRPC_HOST}:{GRPC_PORT}")
    if not client.is_server_ready():
        print(f"ERROR: Triton server not ready at {GRPC_HOST}:{GRPC_PORT}")
        print("  bash scripts/bash/deploy.sh run --gateway triton")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Generating {len(SAMPLES)} audio samples → {OUTPUT_DIR}/\n")

    for sample in SAMPLES:
        name = sample["name"]
        text = sample["text"]
        speaker = sample["speaker"]

        print(f"  [{name}] \"{text[:40]}...\"")
        chunks, first_sec, total_sec, err = stream_tts(client, text, speaker)

        if err:
            print(f"    ERROR: {err}")
            continue

        if not chunks:
            print(f"    WARNING: no audio chunks received")
            continue

        audio = np.concatenate(chunks)
        duration = audio.size / SAMPLE_RATE
        wav_path = OUTPUT_DIR / f"{name}.wav"
        wav_path.write_bytes(make_wav(audio))

        print(f"    → {wav_path.name}  "
              f"duration={duration:.2f}s  "
              f"first_chunk={first_sec*1000:.0f}ms  "
              f"total={total_sec*1000:.0f}ms  "
              f"chunks={len(chunks)}")

    print(f"\nDone. Files in: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()

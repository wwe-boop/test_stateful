"""
Quick integration test for the TTS Orchestrator on Triton.

Sends a VoiceDesign request and collects streaming audio chunks.
Saves the result as a WAV file for playback verification.

Usage:
    python tests/e2e/test_triton_tts.py [--text "..."] [--output output.wav]
"""

import argparse
import json
import struct
import sys
import time
import wave

import numpy as np


def test_http_non_streaming(url: str, text: str, output_path: str):
    """Test via HTTP (non-streaming, will get first response only)."""
    import requests

    req_json = json.dumps({
        "task_type": "voice_design",
        "text": text,
        "language": "auto",
    })

    payload = {
        "inputs": [
            {
                "name": "request",
                "shape": [1],
                "datatype": "BYTES",
                "data": [req_json],
            }
        ],
        "outputs": [
            {"name": "audio_chunk"},
            {"name": "is_final"},
        ],
    }

    print(f"Sending request to {url} ...")
    print(f"  Text: {text}")
    t0 = time.time()

    resp = requests.post(
        f"{url}/v2/models/tts_orchestrator/versions/1/infer",
        json=payload,
        timeout=120,
    )

    elapsed = time.time() - t0
    print(f"  Response status: {resp.status_code} ({elapsed:.2f}s)")

    if resp.status_code != 200:
        print(f"  Error: {resp.text}")
        return False

    result = resp.json()
    print(f"  Response keys: {list(result.keys())}")

    for out in result.get("outputs", []):
        name = out["name"]
        shape = out.get("shape", [])
        print(f"  Output '{name}': shape={shape}")

    return True


def test_grpc_streaming(host: str, port: int, text: str, output_path: str):
    """Test via gRPC streaming (decoupled model)."""
    try:
        import tritonclient.grpc as grpcclient
    except ImportError:
        print("tritonclient not available, installing ...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "tritonclient[grpc]", "-q"])
        import tritonclient.grpc as grpcclient

    client = grpcclient.InferenceServerClient(url=f"{host}:{port}")

    if not client.is_server_ready():
        print("ERROR: Triton server not ready")
        return False

    print(f"Server ready. Sending TTS request ...")
    print(f"  Text: {text}")

    req_json = json.dumps({
        "task_type": "voice_design",
        "text": text,
        "language": "auto",
    })

    req_input = grpcclient.InferInput("request", [1], "BYTES")
    req_input.set_data_from_numpy(np.array([req_json], dtype=object))

    audio_output = grpcclient.InferRequestedOutput("audio_chunk")
    final_output = grpcclient.InferRequestedOutput("is_final")

    audio_chunks = []
    errors = []
    done = False

    def callback(result, error):
        nonlocal done
        if error:
            errors.append(str(error))
            done = True
            return

        audio = result.as_numpy("audio_chunk")
        is_final = result.as_numpy("is_final")
        audio_chunks.append(audio.flatten())
        print(f"  Chunk {len(audio_chunks)}: {audio.shape} samples, "
              f"final={is_final.flatten()[0]}")
        if is_final.flatten()[0]:
            done = True

    t0 = time.time()
    client.start_stream(callback=callback)
    client.async_stream_infer(
        model_name="tts_orchestrator",
        inputs=[req_input],
        outputs=[audio_output, final_output],
    )

    timeout = 120
    while not done and (time.time() - t0) < timeout:
        time.sleep(0.1)

    client.stop_stream()
    elapsed = time.time() - t0

    if errors:
        print(f"\n  Errors: {errors}")
        return False

    if not audio_chunks:
        print(f"\n  No audio chunks received ({elapsed:.2f}s)")
        return False

    all_audio = np.concatenate(audio_chunks)
    print(f"\n  Total audio: {len(all_audio)} samples ({len(all_audio)/24000:.2f}s at 24kHz)")
    print(f"  Latency: {elapsed:.2f}s")
    print(f"  Audio range: [{all_audio.min():.4f}, {all_audio.max():.4f}]")

    if np.all(all_audio == 0):
        print("  WARNING: Audio is all zeros!")

    save_wav(all_audio, output_path, sample_rate=24000)
    print(f"  Saved: {output_path}")
    return True


def save_wav(audio: np.ndarray, path: str, sample_rate: int = 24000):
    """Save float32 audio as 16-bit WAV."""
    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * 32767).astype(np.int16)

    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())


def main():
    parser = argparse.ArgumentParser(description="Test TTS Orchestrator on Triton")
    parser.add_argument("--text", default="今天天气真好，我们一起出去玩吧。",
                        help="Text to synthesize")
    parser.add_argument("--output", default="workspace/test_tts_output.wav",
                        help="Output WAV file path")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--grpc-port", type=int, default=8001)
    parser.add_argument("--http-port", type=int, default=8000)
    parser.add_argument("--mode", choices=["grpc", "http"], default="grpc",
                        help="Client mode")
    args = parser.parse_args()

    print("=" * 60)
    print("  TTS Orchestrator Integration Test")
    print("=" * 60)

    if args.mode == "grpc":
        ok = test_grpc_streaming(args.host, args.grpc_port, args.text, args.output)
    else:
        ok = test_http_non_streaming(
            f"http://{args.host}:{args.http_port}", args.text, args.output)

    print()
    if ok:
        print("  RESULT: PASS")
    else:
        print("  RESULT: FAIL")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

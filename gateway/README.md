# TTS Gateway (gRPC ↔ Triton)

Bridges custom gRPC `TTSService` (see proto 12.3) to Triton Inference Server `tts_orchestrator`.

## Generate Python stubs

From this directory:

```bash
pip install -r requirements.txt
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. proto/tts_service.proto
```

## Run Gateway

1. Start Triton with `tts_orchestrator` loaded (see repo root `scripts/bash/build_triton.sh run`).
2. Run gateway (default: Triton at localhost:8001, gateway at 50051):

```bash
python server.py
# Or: python server.py --triton localhost:8001 --port 50051
```

## Protocol

- Client opens `StreamingSynthesize(stream TTSRequest)`.
- Send one `InitRequest` (task_type, language, ref_audio/speaker/instruct as needed).
- Send one or more `TextChunk` with text; then `TextComplete`.
- Server calls Triton and streams back `TTSResponse` (AudioChunk PCM16 LE 24kHz or TTSError).

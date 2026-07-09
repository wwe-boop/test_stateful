# C4 Synthetic Full-Passage API Notes

Date: 2026-07-10
Scope: Phase 1.1 probe for `steadystream_c4_execution_playbook_20260710.md`.

## Decision

Use 5090-Host `qwen3-engine-custom` gRPC `SynthesizeOnce` as the speaker `001` full-passage synthesis source.

Do not use `http://39.101.65.229:44083/` for final speaker001 C4 data because its `custom_voice` preset speaker list does not include `001`.

## HTTP Service Probe

Base URL: `http://39.101.65.229:44083/`

Observed endpoints:

| endpoint | result |
|---|---|
| `/` | 200 OK, Qwen3-TTS web console |
| `/docs` | 200 OK, FastAPI Swagger UI |
| `/openapi.json` | 200 OK |
| `/health` | 404 Not Found |
| `/api/status` | `{"loaded":true,"model_key":"base","device":"cuda:0"}` |
| `/api/config` | model keys: `base`, `custom_voice`, `voice_design` |

OpenAPI `GenerateBody` supports:

- `mode`: `voice_clone`, `custom_voice`, `voice_design`
- `text`
- `language`
- `speaker`
- `instruct`
- `ref_audio_path`, `ref_text`, `x_vector_only_mode` for clone mode

`/api/config` reported preset speakers:

`Vivian`, `Serena`, `Uncle_Fu`, `Dylan`, `Eric`, `Ryan`, `Aiden`, `Ono_Anna`, `Sohee`

There is no `001`, so this service is useful for generic UI/API reference only, not for the speaker001 C4 training data requested here.

## 5090 gRPC Probe

Container:

- host: `5090-Host`
- service/container: `qwen3-engine-custom`
- image: `qwen3-engine:25.10`
- gRPC endpoint from host/container: `127.0.0.1:50071`
- deployed checkpoint: `/home/train/tts/qwen3-tts/trained/zehan/0701_trained_model`

Capabilities probe returned:

```json
{
  "variant": "custom-1.7b",
  "loaded_model_type": "custom_voice",
  "tasks": ["custom_voice"],
  "formats": [
    {"encoding": 1, "sample_rate": 24000, "channels": 1},
    {"encoding": 1, "sample_rate": 16000, "channels": 1},
    {"encoding": 2, "sample_rate": 24000, "channels": 1},
    {"encoding": 2, "sample_rate": 16000, "channels": 1}
  ]
}
```

Speaker001 full-text smoke:

- request: `task_type=custom_voice`, `speaker=001`, `language=Chinese`, `instruct=""`, `input_mode=full_text`
- text: `今天我们先做一个整段合成探测，确认音色一号可以稳定输出。`
- result: 74 audio chunks, 142080 samples, 5.92 s audio, 0.842 s wall time
- no error event observed

Minimal request shape:

```python
tts_pb2.SynthesizeOnceRequest(
    session_id="c4-full-...",
    text=full_passage_text,
    config=tts_pb2.SessionConfig(
        task_type="custom_voice",
        speaker="001",
        language="Chinese",
        instruct="",
        input_mode=tts_pb2.INPUT_MODE_FULL_TEXT,
        group_policy=tts_pb2.GROUP_POLICY_NONE,
        audio=tts_pb2.AudioFormat(
            encoding=tts_pb2.AUDIO_ENCODING_PCM_F32,
            sample_rate=24000,
            channels=1,
        ),
    ),
)
```

## Next Implementation Constraint

The old `scripts/python/synthesize_c4_smoke_dataset.py` is not valid for final C4 data because it synthesizes each segment separately and then concatenates with artificial pauses. Phase 1.3 must synthesize the full paragraph once, then cut or timestamp that single wav into continuation samples.

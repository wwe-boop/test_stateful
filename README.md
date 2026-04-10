# Qwen3-TTS Triton

*Qwen3-TTS In The Wild: High-performance streaming TTS inference service built on **Triton Inference Server & TensorRT**.*

Takes the [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) model, splits it into 6 independent components, exports them as ONNX / TRT-LLM / TensorRT engines, and serves them through Triton with streaming audio output.

## Features

- **Streaming output** — frame-level audio streaming, low first-chunk latency (~76ms TTS portion)
- **High throughput** — ~4.1ms per decode step (vs ~20ms with vLLM-omni), GPU utilization >80%
- **Multi-user batching** — 8-10 concurrent sessions per GPU with continuous insert/complete
- **Smart launcher** — one command to set up, build, and deploy; interactive guided mode
- **Auto-detection** — GPU driver version, NGC container compatibility, China mirror fallback
- **Three-phase build** — environment setup, engine compilation, and deployment fully decoupled

## Prerequisites

- NVIDIA GPU (Compute Capability >= 7.0)
- NVIDIA Driver >= 550.54
- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- ~40GB disk space (models + engines)
- Linux (tested on Ubuntu 22.04/24.04)

## Quick Start

```bash
git clone --recursive https://github.com/user/Qwen3-TTS-Triton.git
cd Qwen3-TTS-Triton

# Full pipeline: setup → build engines → deploy Triton
bash scripts/bash/autorun.sh
```

The launcher will:
1. Detect your GPU and Docker environment
2. Ask which model variant to deploy
3. Run all three phases automatically

## Docker Compose Deployment

Docker deployment is now standardized on `compose.yaml` plus the wrapper `scripts/bash/compose.sh`.

There are now two supported deployment tracks:

- `engine`: pure `TTSEngine`, used for internal testing and protocol iteration.
- `triton`: Triton Server + thin Python adapter, used for ops-facing deployment.

For the standalone engine container:

```bash
# Build and start engine.server in Docker
bash scripts/bash/compose.sh up --gateway engine --variant custom-1.7b

# Logs / status / stop
bash scripts/bash/compose.sh logs --gateway engine --follow
bash scripts/bash/compose.sh ps
bash scripts/bash/compose.sh down --gateway engine
```

For Triton:

```bash
# Assemble workspace/model_repository from exported artifacts
bash scripts/bash/compose.sh prepare --gateway triton --variant custom-1.7b

# Build and start Triton with the mounted model repository
bash scripts/bash/compose.sh up --gateway triton --variant custom-1.7b

# Logs / status / stop
bash scripts/bash/compose.sh logs --gateway triton --follow
bash scripts/bash/compose.sh ps
bash scripts/bash/compose.sh down --gateway triton
```

The higher-level deploy entry point still works:

```bash
bash scripts/bash/deploy.sh run --gateway engine-docker --variant custom-1.7b
bash scripts/bash/deploy.sh run --gateway triton --variant custom-1.7b
```

Those Docker-based deploy modes now call the compose workflow internally.

## Streaming Protocol Notes

The streaming contract now carries two text-side events in parallel with audio:

- `text_token`: incremental committed text token, suitable for a token-player style UI.
- `text_boundary_commit`: a text boundary is now known and committed, while audio may still be streaming.

`segment_end` remains the audio-completion signal for a committed text segment.

### Specify a Model Variant

```bash
# Deploy a specific variant directly
bash scripts/bash/autorun.sh base-1.7b
```

Available variants:

| Variant | Model | Description |
|---------|-------|-------------|
| `base-1.7b` | Qwen3-TTS-12Hz-1.7B-Base | Voice clone, recommended for most use cases |
| `custom-1.7b` | Qwen3-TTS-12Hz-1.7B-CustomVoice | 9 premium built-in voices + instruction control |
| `design-1.7b` | Qwen3-TTS-12Hz-1.7B-VoiceDesign | Voice design via text description |
| `base-0.6b` | Qwen3-TTS-12Hz-0.6B-Base | Smaller voice clone model |
| `custom-0.6b` | Qwen3-TTS-12Hz-0.6B-CustomVoice | Smaller custom voice model |
| `all-1.7b` | All 1.7B variants | Download and export all three 1.7B models |
| `all` | Everything | All 5 model variants |

## Usage

### Subcommands

```bash
autorun.sh [command] [variant] [options]

# Run individual phases
bash scripts/bash/autorun.sh setup       # Phase A: environment + model export
bash scripts/bash/autorun.sh build       # Phase B: TRT-LLM engine compilation
bash scripts/bash/autorun.sh deploy      # Phase C: start Triton server

# Pipeline control
bash scripts/bash/autorun.sh all         # run all phases A → B → C
bash scripts/bash/autorun.sh status      # show pipeline status
bash scripts/bash/autorun.sh stop        # stop Triton server
```

### Options

```
Global:
  --variant, -m <name>    Model variant
  --yes, -y               Skip confirmations
  --dry-run               Show what would be done
  -h, --help              Show full help

Phase A (setup_env.sh):
  --python <version>      Python version (default: 3.10)
  --env-name <name>       Virtual env name (default: qwen3-tts)
  --source <source>       Download source (auto|hf|modelscope)
  --skip-models           Skip model download
  --skip-deps             Skip dependency installation
  --skip-export           Skip model export

Phase B (build_engines.sh):
  --max-batch-size <N>    Max batch size (default: 8)
  --image <uri>           Override NGC container image
  --dtype <type>          Engine precision (default: bfloat16)

Phase C (build_triton.sh):
  --grpc-port <port>      gRPC port (default: 8001)
  --http-port <port>      HTTP port (default: 8000)
```

### Examples

```bash
# Interactive guided setup (recommended for first time)
bash scripts/bash/autorun.sh

# Full pipeline for custom voice model
bash scripts/bash/autorun.sh custom-1.7b

# Phase A only, skip model download (models already present)
bash scripts/bash/autorun.sh setup --skip-models

# Phase B with custom batch size
bash scripts/bash/autorun.sh build --max-batch-size 4

# Preview what would be done without executing
bash scripts/bash/autorun.sh all -m base-1.7b --dry-run

# Check pipeline status
bash scripts/bash/autorun.sh status
```

### Standalone Scripts

Each phase can also be run directly:

```bash
bash scripts/bash/setup_env.sh           # Phase A
bash scripts/bash/build_engines.sh       # Phase B
bash scripts/bash/build_triton.sh run    # Phase C

# Utilities
bash scripts/bash/download_models.sh     # Download models only
bash scripts/bash/export_models.sh       # Export models only (requires venv)
```

When running standalone export/download scripts, activate the virtual environment first:

```bash
conda activate qwen3-tts
# or
source <venv-path>/bin/activate
```

## Three-Phase Build Pipeline

```
Phase A — setup_env.sh (host)
  PyTorch + qwen_tts → ONNX models + TRT-LLM checkpoints + .pt weights

Phase B — build_engines.sh (Docker)
  NGC TRT-LLM container → trtllm-build → .engine files

Phase C — build_triton.sh (Triton)
  Assemble model_repository → pull NGC image → start Triton server
```

- **Phase A** runs on the host with PyTorch + ONNX tools. No TRT-LLM needed.
- **Phase B** runs inside an NGC container via `docker run --gpus all`. No PyTorch needed. The container image is auto-selected based on your driver version.
- **Phase C** assembles exported artifacts into a Triton model repository and starts the server.
- Phases communicate through `workspace/exported/` — fully decoupled.

## Model Architecture

The Qwen3-TTS model is split into 6 independently served components:

| Component | Params | Engine | Called |
|-----------|--------|--------|--------|
| Text Embedder | ~312M | PyTorch weights (in Orchestrator) | Once per request |
| Speaker Encoder (ECAPA-TDNN) | ~6M | ONNX | Once per request |
| Speech Tokenizer Encoder (Mimi) | ~26M | ONNX | Once per request |
| **Talker Backbone** (Qwen3, 20L) | ~180M | **TRT-LLM** | Every decode step |
| **Code Predictor** (Qwen3, 5L + 15 heads) | ~200M | **TensorRT** | Every decode step |
| Code2Wav Decoder | ~60M | ONNX | Every chunk |

## Project Structure

```
Qwen3-TTS-Triton/
├── scripts/
│   ├── bash/
│   │   ├── autorun.sh             # Smart launcher (A→B→C, interactive)
│   │   ├── setup_env.sh           # Phase A: environment + export
│   │   ├── build_engines.sh       # Phase B: TRT-LLM engine build
│   │   ├── build_triton.sh        # Phase C: Triton deployment
│   │   ├── download_models.sh     # Standalone model downloader
│   │   ├── export_models.sh       # Standalone model exporter
│   │   ├── tools.sh               # Library aggregator
│   │   └── lib/                   # Modular function library
│   └── export/                    # Python export scripts (01-06)
├── docs/
│   └── architecture.md            # Detailed architecture design
├── model_repository/              # Triton model repo (generated)
├── third_party/Qwen3-TTS/         # Official repo (git submodule)
└── workspace/
    ├── models/                    # Downloaded weights (gitignored)
    └── exported/                  # Exported engines (gitignored)
```

## China Network Support

The build scripts automatically detect network conditions and configure mirrors:

- **GitHub** — auto-selects ghproxy mirror when direct access is slow
- **HuggingFace** — falls back to hf-mirror.com
- **pip/conda** — configures BFSU (TUNA) mirrors
- **Model download** — prefers ModelScope by default

Set `CONFIGURE_MIRRORS=china` to force China mirror configuration, or `CONFIGURE_MIRRORS=skip` to skip.

## Documentation

- [Architecture Design](docs/architecture.md) — comprehensive technical reference (2000+ lines)

## License

This project builds upon [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) by Alibaba Cloud. Refer to the upstream repository for model license terms.

---

<p align="center">
  <sub>Built with AI-assisted programming, powered by <a href="https://cursor.com">Cursor</a></sub>
</p>

#!/bin/bash

set -euo pipefail

variant="${MODEL_VARIANT:-}"
if [[ -z "$variant" ]]; then
    echo "MODEL_VARIANT is required" >&2
    exit 1
fi

case "$variant" in
    design-1.7b) tokenizer_subdir="Qwen3-TTS-12Hz-1.7B-VoiceDesign" ;;
    custom-1.7b) tokenizer_subdir="Qwen3-TTS-12Hz-1.7B-CustomVoice" ;;
    base-1.7b) tokenizer_subdir="Qwen3-TTS-12Hz-1.7B-Base" ;;
    custom-0.6b) tokenizer_subdir="Qwen3-TTS-12Hz-0.6B-CustomVoice" ;;
    base-0.6b) tokenizer_subdir="Qwen3-TTS-12Hz-0.6B-Base" ;;
    *)
        echo "Unsupported MODEL_VARIANT: $variant" >&2
        exit 1
        ;;
esac

model_root="${MODEL_ROOT:-/models}"
exported_root="${EXPORTED_ROOT:-/exported}"
config_path="${ENGINE_CONFIG:-/app/engine.yaml}"

tokenizer_dir="${TOKENIZER_DIR:-}"
weights_dir="${WEIGHTS_DIR:-}"
engine_dir="${ENGINE_RUNTIME_DIR:-}"

if [[ -z "$tokenizer_dir" ]]; then
    tokenizer_dir="${model_root}/${tokenizer_subdir}"
fi
if [[ -z "$weights_dir" ]]; then
    weights_dir="${exported_root}/${variant}/weights"
fi

if [[ -z "$engine_dir" ]]; then
    if [[ -f "${exported_root}/${variant}/engines/talker_code2wav_fused/model.plan" ]] || \
       [[ -f "${exported_root}/${variant}/engines/talker_code2wav_fused/talker_code2wav_fused.engine" ]]; then
        engine_dir="${exported_root}/${variant}/engines/talker_code2wav_fused"
    elif [[ -f "${exported_root}/${variant}/talker_code2wav_fused.engine" ]]; then
        engine_dir="${exported_root}/${variant}"
    fi
fi

if [[ ! -d "$tokenizer_dir" ]]; then
    echo "Tokenizer directory not found: $tokenizer_dir" >&2
    exit 1
fi
if [[ ! -d "$weights_dir" ]]; then
    echo "Weights directory not found: $weights_dir" >&2
    exit 1
fi
if [[ ! -f "$config_path" ]]; then
    echo "Config file not found: $config_path" >&2
    exit 1
fi

export ENGINE_SCHEDULER_MAX_BATCH_SIZE="${ENGINE_MAX_BATCH_SIZE:-48}"
export ENGINE_SERVER_WEBSOCKET_PORT="${ENGINE_WEBSOCKET_PORT:-50052}"
export ENGINE_SERVER_HEALTH_PORT="${ENGINE_HEALTH_PORT:-8080}"
if [[ -n "${ENGINE_MAX_SEQ_LEN:-}" ]]; then
    export ENGINE_SCHEDULER_MAX_SEQ_LEN="${ENGINE_MAX_SEQ_LEN}"
fi

cmd=(
    python3 -m engine.server
    --config "$config_path"
    --tokenizer-dir "$tokenizer_dir"
    --weights-dir "$weights_dir"
    --device "${ENGINE_DEVICE:-0}"
    --max-batch "${ENGINE_MAX_BATCH_SIZE:-48}"
    --max-sessions "${ENGINE_MAX_SESSIONS:-128}"
    --port "${ENGINE_GRPC_PORT:-50051}"
)
if [[ -n "$engine_dir" ]]; then
    cmd+=(--engine-dir "$engine_dir")
fi

echo "Starting engine for variant=${variant}" >&2
echo "  config=${config_path}" >&2
echo "  tokenizer=${tokenizer_dir}" >&2
echo "  weights=${weights_dir}" >&2
echo "  model_root=${model_root}" >&2
echo "  exported_root=${exported_root}" >&2
if [[ -n "$engine_dir" ]]; then
    echo "  engine_dir=${engine_dir}" >&2
else
    echo "  engine_dir=stub-mode" >&2
fi

exec "${cmd[@]}"

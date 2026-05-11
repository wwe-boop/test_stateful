#!/bin/bash

set -euo pipefail

config_path="${ENGINE_CONFIG:-/app/engine.yaml}"

model_repo="${ENGINE_MODEL_REPOSITORY:-/models}"
model_name="${ENGINE_MODEL_NAME:-tts_orchestrator}"
model_version="${ENGINE_MODEL_VERSION:-${MODEL_VERSION:-1}}"
model_package_dir="${ENGINE_MODEL_PACKAGE_DIR:-${model_repo}/${model_name}/${model_version}}"

resolved_paths="$(
    python3 - "$model_package_dir" <<'PY'
import sys
from engine.config import resolve_model_package_paths

p = resolve_model_package_paths(sys.argv[1])
print("\t".join([
    p.package_dir,
    p.engine_dir,
    p.weights_dir,
    p.tokenizer_dir,
    p.manifest_path,
    p.runtime_artifact_path,
    p.engine_mode,
]))
PY
)"
IFS=$'\t' read -r model_package_dir engine_dir weights_dir tokenizer_dir manifest_path runtime_artifact engine_mode <<< "$resolved_paths"

if [[ ! -d "$model_package_dir" ]]; then
    echo "Model package directory not found: $model_package_dir" >&2
    echo "Expected Triton-compatible model package: /models/tts_orchestrator/${model_version}/{runtime,weights,tokenizer}" >&2
    exit 1
fi
if [[ ! -d "$tokenizer_dir" ]]; then
    echo "Tokenizer directory not found: $tokenizer_dir" >&2
    exit 1
fi
if [[ ! -d "$weights_dir" ]]; then
    echo "Weights directory not found: $weights_dir" >&2
    exit 1
fi
if [[ ! -d "$engine_dir" ]]; then
    echo "Runtime directory not found: $engine_dir" >&2
    exit 1
fi
if [[ ! -f "$manifest_path" ]]; then
    echo "Model package manifest not found: $manifest_path" >&2
    exit 1
fi
if [[ "$engine_mode" != "trt" ]]; then
    echo "Engine Docker requires a TensorRT model package, got engine_mode=${engine_mode:-unknown}: $manifest_path" >&2
    echo "Re-assemble with: bash scripts/bash/compose.sh prepare --gateway engine --engine-mode trt --model-version ${model_version}" >&2
    exit 1
fi
if [[ ! -f "$runtime_artifact" ]]; then
    if [[ -f "${engine_dir}/model.onnx" ]]; then
        echo "Engine Docker requires a TensorRT model package, but found ONNX runtime only: ${engine_dir}/model.onnx" >&2
        echo "Re-assemble with: bash scripts/bash/compose.sh prepare --gateway engine --engine-mode trt --model-version ${model_version}" >&2
    else
        echo "TensorRT runtime artifact not found: ${runtime_artifact}" >&2
        echo "Run Phase B and assemble the shared model_repository in trt mode." >&2
    fi
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
    --model-package-dir "$model_package_dir"
    --device "${ENGINE_DEVICE:-0}"
    --max-batch "${ENGINE_MAX_BATCH_SIZE:-48}"
    --max-sessions "${ENGINE_MAX_SESSIONS:-128}"
    --port "${ENGINE_GRPC_PORT:-50051}"
)

echo "Starting engine from shared model package" >&2
echo "  config=${config_path}" >&2
echo "  model_repo=${model_repo}" >&2
echo "  model_package=${model_package_dir}" >&2
echo "  tokenizer=${tokenizer_dir}" >&2
echo "  weights=${weights_dir}" >&2
echo "  engine_dir=${engine_dir}" >&2
echo "  manifest=${manifest_path}" >&2
echo "  runtime_artifact=${runtime_artifact}" >&2
echo "  engine_mode=${engine_mode}" >&2

exec "${cmd[@]}"

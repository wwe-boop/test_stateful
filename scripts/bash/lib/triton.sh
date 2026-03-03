#!/bin/bash
# ===========================================================================
#  triton.sh — Triton Inference Server deployment helpers (Phase C)
#
#  Functions: assemble_model_repo, validate_model_repo,
#             build_triton_image, resolve_triton_deploy_image,
#             triton_run, triton_health_check
#  Depends:   lib/logging.sh, lib/utils.sh, lib/docker.sh
#
#  Assembles Phase A/B exported artifacts into a Triton model_repository
#  and manages the deployment container lifecycle.
# ===========================================================================

[[ -n "${_LIB_TRITON_LOADED:-}" ]] && return 0
_LIB_TRITON_LOADED=1

_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_LIB_DIR}/logging.sh"
source "${_LIB_DIR}/utils.sh"
source "${_LIB_DIR}/docker.sh"

# Triton model repository layout (all sub-models + orchestrator)
_TRITON_MODELS=(
    "speaker_encoder"
    "speech_tokenizer_encoder"
    "talker_unified"
    "code2wav"
    "tts_orchestrator"
)

# Default Triton gRPC / HTTP / metrics ports
TRITON_GRPC_PORT="${TRITON_GRPC_PORT:-8001}"
TRITON_HTTP_PORT="${TRITON_HTTP_PORT:-8000}"
TRITON_METRICS_PORT="${TRITON_METRICS_PORT:-8002}"

# ---------------------------------------------------------------------------
#  resolve_triton_deploy_image [driver_version]
#  Returns the Triton deployment image for Phase C.
#  Uses the full py3 image (onnxruntime + tensorrt + python backends).
#  Echoes the full image tag to stdout.
# ---------------------------------------------------------------------------
resolve_triton_deploy_image() {
    local ngc_tag
    ngc_tag=$(resolve_ngc_tag "$@") \
        || { log_error "Cannot determine NGC tag"; return 1; }
    local image="${_NGC_TRITON_BASE}:${ngc_tag}${_NGC_FULL_SUFFIX}"
    log_info "Using Triton full image: $image"
    echo "$image"
}

# ---------------------------------------------------------------------------
#  _link_or_copy <src> <dst>
#  Copies src to dst.  Always uses cp (not symlinks) because the assembled
#  model_repository is mounted into Docker containers where host-absolute
#  symlinks would be dangling.
# ---------------------------------------------------------------------------
_link_or_copy() {
    local src="$1" dst="$2"
    cp -a "$src" "$dst"
}

# ---------------------------------------------------------------------------
#  assemble_model_repo <exported_dir> <variant> <model_repo_dir> [engine_mode]
#
#  engine_mode: trt (default) | onnx
#  TRT mode:  copies .engine → model.plan, backend tensorrt
#  ONNX mode: copies .onnx → model.onnx, backend onnxruntime
#
#  Layout: speaker_encoder, speech_tokenizer_encoder, talker_unified,
#  code2wav as independent models; tts_orchestrator = Python BLS.
# ---------------------------------------------------------------------------
assemble_model_repo() {
    local exported_dir="$1"
    local variant="$2"
    local repo_dir="$3"
    local engine_mode="${4:-trt}"

    local variant_dir="$exported_dir/$variant"
    local tokenizer_dir="$exported_dir/tokenizer"

    log_step "Assembling Triton model repository (engine_mode=$engine_mode)"
    log_info "  Variant:    $variant"
    log_info "  Source:     $variant_dir"
    log_info "  Repository: $repo_dir"

    if [ ! -d "$variant_dir" ]; then
        log_error "Variant directory not found: $variant_dir"
        log_error "Run Phase A first: autorun.sh setup or export_all.py"
        return 1
    fi

    mkdir -p "$repo_dir"

    # Helper: copy model file and write config by engine_mode
    _place_model() {
        local name="$1"
        local src="$2"
        local model_dir="$repo_dir/$name/1"
        mkdir -p "$model_dir"
        if [ "$engine_mode" = "trt" ]; then
            _link_or_copy "$src" "$model_dir/model.plan"
            _write_trt_minimal_config "$repo_dir/$name" "$name"
        else
            _link_or_copy "$src" "$model_dir/model.onnx"
            _write_onnx_minimal_config "$repo_dir/$name" "$name"
        fi
    }

    # _resolve_model_src <base_path> selects .onnx or .engine based on engine_mode,
    # verifying that the chosen file actually exists.
    # Returns: 0 + prints path on success, 1 on not found.
    _resolve_model_src() {
        local base="$1"
        local ext; [ "$engine_mode" = "trt" ] && ext=".engine" || ext=".onnx"
        if [ -f "${base}${ext}" ]; then
            echo "${base}${ext}"; return 0
        fi
        return 1
    }

    # ── 1. Speaker Encoder (optional — only needed for voice clone) ──
    local spk_src
    if spk_src="$(_resolve_model_src "$variant_dir/speaker_encoder")"; then
        _place_model "speaker_encoder" "$spk_src"
        log_info "  speaker_encoder: OK"
    else
        log_warn "  speaker_encoder: SKIPPED (${engine_mode} file not found — only needed for voice clone)"
    fi

    # ── 2. Speech Tokenizer Encoder (optional — only needed for ICL mode) ──
    local stoken_src
    if stoken_src="$(_resolve_model_src "$tokenizer_dir/speech_tokenizer_encoder")"; then
        _place_model "speech_tokenizer_encoder" "$stoken_src"
        log_info "  speech_tokenizer_encoder: OK"
    else
        log_warn "  speech_tokenizer_encoder: SKIPPED (${engine_mode} file not found — only needed for ICL mode)"
    fi

    # ── 3. Talker Unified (required; single engine for prefill + decode) ──
    local talker_src
    if talker_src="$(_resolve_model_src "$variant_dir/talker_unified")"; then
        _place_model "talker_unified" "$talker_src"
        log_info "  talker_unified: OK"
    else
        log_error "  talker_unified: MISSING ${engine_mode} file (required). Run export_04_talker_unified.py + Phase B first."
        return 1
    fi

    # ── 4. Code2Wav Decoder (required) ──
    local c2w_src
    if c2w_src="$(_resolve_model_src "$tokenizer_dir/code2wav_decoder")"; then
        _place_model "code2wav" "$c2w_src"
        log_info "  code2wav: OK"
    else
        log_error "  code2wav: MISSING ${engine_mode} file (required). Run Phase A/B first."
        return 1
    fi

    # ── 5. TTS Orchestrator (Python BLS backend) ──
    local weights_dir="$variant_dir/weights"
    mkdir -p "$repo_dir/tts_orchestrator/1/weights"
    if [ -d "$weights_dir" ]; then
        for f in "$weights_dir"/*; do
            _link_or_copy "$f" "$repo_dir/tts_orchestrator/1/weights/$(basename "$f")"
        done
        log_info "  tts_orchestrator/weights: OK"
    else
        log_warn "  tts_orchestrator/weights: MISSING (no embedding weights found)"
    fi

    # Copy orchestrator Python source files (paths relative to repo root)
    local repo_root
    repo_root="$(cd "${_LIB_DIR}/../../.." && pwd)"
    local orch_py_dir="$repo_root/model_repository/tts_orchestrator/1"
    if [ -d "$orch_py_dir" ]; then
        for pyf in model.py prefill_builder.py; do
            if [ -f "$orch_py_dir/$pyf" ]; then
                cp "$orch_py_dir/$pyf" "$repo_dir/tts_orchestrator/1/$pyf"
            fi
        done
        local ces="$repo_root/scripts/python/codec_embedding_sum.py"
        if [ -f "$ces" ]; then
            cp "$ces" "$repo_dir/tts_orchestrator/1/codec_embedding_sum.py"
        fi
        log_info "  tts_orchestrator/python: OK (copied model.py + helpers)"
    else
        log_warn "  tts_orchestrator/python: source dir not found, using stub"
    fi

    # Copy text tokenizer files for orchestrator
    local model_base_dir
    model_base_dir="$(cd "$(dirname "$exported_dir")" && pwd)/models"
    local tok_dir=""
    case "$variant" in
        design-1.7b) tok_dir="$model_base_dir/Qwen3-TTS-12Hz-1.7B-VoiceDesign" ;;
        custom-1.7b) tok_dir="$model_base_dir/Qwen3-TTS-12Hz-1.7B-CustomVoice" ;;
        base-1.7b)   tok_dir="$model_base_dir/Qwen3-TTS-12Hz-1.7B-Base" ;;
        custom-0.6b) tok_dir="$model_base_dir/Qwen3-TTS-12Hz-0.6B-CustomVoice" ;;
        base-0.6b)   tok_dir="$model_base_dir/Qwen3-TTS-12Hz-0.6B-Base" ;;
    esac
    if [ -n "$tok_dir" ] && [ -d "$tok_dir" ]; then
        mkdir -p "$repo_dir/tts_orchestrator/1/tokenizer"
        for tf in tokenizer_config.json vocab.json merges.txt \
                  config.json generation_config.json; do
            [ -f "$tok_dir/$tf" ] && \
                _link_or_copy "$tok_dir/$tf" "$repo_dir/tts_orchestrator/1/tokenizer/$tf"
        done
        log_info "  tts_orchestrator/tokenizer: OK"
    else
        log_warn "  tts_orchestrator/tokenizer: SKIPPED (model dir not found)"
    fi

    _write_orchestrator_config "$repo_dir/tts_orchestrator" "$variant"

    echo ""
    log_info "Model repository assembled: $repo_dir"
    return 0
}

# ---------------------------------------------------------------------------
#  validate_model_repo <model_repo_dir>
#  Quick sanity check: each expected model has config.pbtxt + version dir.
#  Returns 0 if all critical models present, 1 otherwise.
# ---------------------------------------------------------------------------
validate_model_repo() {
    local repo_dir="$1"
    local missing=0
    local warned=0

    log_step "Validating model repository: $repo_dir"

    for model in "${_TRITON_MODELS[@]}"; do
        local model_dir="$repo_dir/$model"
        if [ ! -f "$model_dir/config.pbtxt" ]; then
            if [[ "$model" == "speaker_encoder" || "$model" == "speech_tokenizer_encoder" ]]; then
                log_warn "  $model: no config.pbtxt (optional)"
                warned=$((warned + 1))
            else
                log_error "  $model: no config.pbtxt (required)"
                missing=$((missing + 1))
            fi
            continue
        fi

        if [ ! -d "$model_dir/1" ]; then
            log_error "  $model: no version directory (1/)"
            missing=$((missing + 1))
            continue
        fi

        log_info "  $model: OK"
    done

    if [ "$missing" -gt 0 ]; then
        log_error "Validation failed: $missing required model(s) missing"
        return 1
    fi

    if [ "$warned" -gt 0 ]; then
        log_warn "Validation passed with $warned optional model(s) missing"
    else
        log_info "Validation passed: all models present"
    fi
    return 0
}

# ---------------------------------------------------------------------------
#  build_triton_image <repo_root> <image_tag> [base_image]
#  Builds a self-contained deployment image with all model artifacts baked in.
#  Uses the Dockerfile at <repo_root>/Dockerfile.triton.
#
#  For development, prefer triton_run() with volume mounts instead.
# ---------------------------------------------------------------------------
build_triton_image() {
    local repo_root="$1"
    local image_tag="$2"
    local base_image="${3:-}"

    if [ -z "$base_image" ]; then
        base_image=$(resolve_triton_deploy_image) \
            || { log_error "Cannot determine base image"; return 1; }
    fi

    local dockerfile="$repo_root/Dockerfile.triton"
    if [ ! -f "$dockerfile" ]; then
        log_error "Dockerfile not found: $dockerfile"
        log_error "Generate it first with: bash scripts/bash/build_triton.sh --generate-dockerfile"
        return 1
    fi

    log_step "Building Triton deployment image: $image_tag"
    log_info "  Base image:  $base_image"
    log_info "  Dockerfile:  $dockerfile"

    docker build \
        --build-arg "BASE_IMAGE=$base_image" \
        -t "$image_tag" \
        -f "$dockerfile" \
        "$repo_root" \
        || { log_error "Docker build failed"; return 1; }

    log_info "Image built: $image_tag"
}

# ---------------------------------------------------------------------------
#  triton_run <model_repo_dir> [image] [container_name] [extra_docker_args...]
#  Starts a Triton server container with the model repository volume-mounted.
#  Runs in detached mode; use triton_health_check to verify readiness.
# ---------------------------------------------------------------------------
triton_run() {
    local repo_dir="$1"
    local image="${2:-}"
    local container_name="${3:-qwen3-tts-triton}"
    shift 3 2>/dev/null || true

    if [ -z "$image" ]; then
        image=$(resolve_triton_deploy_image) \
            || { log_error "Cannot determine Triton image"; return 1; }
    fi

    repo_dir="$(cd "$repo_dir" && pwd)"

    # Stop existing container if running
    if docker ps -q --filter "name=$container_name" | grep -q .; then
        log_warn "Stopping existing container: $container_name"
        docker stop "$container_name" &>/dev/null || true
        docker rm "$container_name" &>/dev/null || true
    fi

    log_step "Starting Triton Inference Server"
    log_info "  Image:      $image"
    log_info "  Repository: $repo_dir"
    log_info "  Container:  $container_name"
    log_info "  Ports:      gRPC=$TRITON_GRPC_PORT HTTP=$TRITON_HTTP_PORT metrics=$TRITON_METRICS_PORT"

    local gpu_device="${TRITON_GPU_DEVICE:-0}"
    docker run -d --gpus "\"device=${gpu_device}\"" \
        --name "$container_name" \
        --shm-size=1g \
        --ulimit memlock=-1 \
        -p "${TRITON_HTTP_PORT}:8000" \
        -p "${TRITON_GRPC_PORT}:8001" \
        -p "${TRITON_METRICS_PORT}:8002" \
        -v "$repo_dir:/models" \
        "$@" \
        "$image" \
        tritonserver \
            --model-repository=/models \
            --log-verbose=1 \
            --strict-model-config=false \
        || { log_error "Failed to start Triton container"; return 1; }

    log_info "Container started: $container_name"
    log_info "Waiting for server to be ready ..."
}

# ---------------------------------------------------------------------------
#  triton_health_check [host] [port] [timeout_sec]
#  Polls the Triton health endpoint until ready or timeout.
# ---------------------------------------------------------------------------
triton_health_check() {
    local host="${1:-localhost}"
    local port="${2:-$TRITON_HTTP_PORT}"
    local timeout="${3:-60}"

    local url="http://${host}:${port}/v2/health/ready"
    local elapsed=0
    local interval=3

    while [ "$elapsed" -lt "$timeout" ]; do
        if curl -sf "$url" &>/dev/null; then
            log_info "Triton server ready at ${host}:${port}"
            return 0
        fi
        sleep "$interval"
        elapsed=$((elapsed + interval))
    done

    log_error "Triton health check timed out after ${timeout}s"
    log_error "Check logs: docker logs <container_name>"
    return 1
}

# ---------------------------------------------------------------------------
#  triton_stop [container_name]
#  Stops and removes a running Triton container.
# ---------------------------------------------------------------------------
triton_stop() {
    local container_name="${1:-qwen3-tts-triton}"

    if docker ps -q --filter "name=$container_name" | grep -q .; then
        log_info "Stopping Triton container: $container_name"
        docker stop "$container_name" &>/dev/null
        docker rm "$container_name" &>/dev/null
        log_info "Container stopped and removed: $container_name"
    else
        log_info "Container not running: $container_name"
    fi
}

# ===========================================================================
#  Config file generators (internal)
# ===========================================================================

# _write_onnx_minimal_config <model_dir> <model_name>
# Generates a minimal config.pbtxt (backend + max_batch_size only).
# With --strict-model-config=false, Triton auto-detects I/O from ONNX.
_write_onnx_minimal_config() {
    local model_dir="$1"
    local model_name="$2"

    cat > "$model_dir/config.pbtxt" << EOF
name: "${model_name}"
backend: "onnxruntime"
max_batch_size: 0

instance_group [
  {
    count: 1
    kind: KIND_GPU
    gpus: [ 0 ]
  }
]
EOF
}

# _write_trt_minimal_config <model_dir> <model_name>
# For TensorRT backend; model.plan must exist in <model_dir>/1/
_write_trt_minimal_config() {
    local model_dir="$1"
    local model_name="$2"

    cat > "$model_dir/config.pbtxt" << EOF
name: "${model_name}"
backend: "tensorrt"
max_batch_size: 0

instance_group [
  {
    count: 1
    kind: KIND_GPU
    gpus: [ 0 ]
  }
]
EOF
}

# _write_onnx_config <model_dir> <model_name> <inputs_spec> <outputs_spec>
#
# inputs_spec / outputs_spec: comma-separated "name:dtype:shape" entries
# shape uses -1 for dynamic dims
_write_onnx_config() {
    local model_dir="$1"
    local model_name="$2"
    local inputs_spec="$3"
    local outputs_spec="$4"

    local config_file="$model_dir/config.pbtxt"

    cat > "$config_file" << EOF
name: "${model_name}"
backend: "onnxruntime"
max_batch_size: 0

EOF

    # Write input specs
    IFS=',' read -ra input_entries <<< "$inputs_spec"
    for entry in "${input_entries[@]}"; do
        IFS=':' read -r io_name io_dtype io_shape <<< "$entry"
        local triton_dtype
        triton_dtype=$(_to_triton_dtype "$io_dtype")
        cat >> "$config_file" << EOF
input [
  {
    name: "${io_name}"
    data_type: ${triton_dtype}
    dims: [ $(echo "$io_shape" | tr 'x' ', ') ]
  }
]

EOF
    done

    # Write output specs
    IFS=',' read -ra output_entries <<< "$outputs_spec"
    for entry in "${output_entries[@]}"; do
        IFS=':' read -r io_name io_dtype io_shape <<< "$entry"
        local triton_dtype
        triton_dtype=$(_to_triton_dtype "$io_dtype")
        cat >> "$config_file" << EOF
output [
  {
    name: "${io_name}"
    data_type: ${triton_dtype}
    dims: [ $(echo "$io_shape" | tr 'x' ', ') ]
  }
]

EOF
    done

    # Instance group: single GPU
    cat >> "$config_file" << EOF
instance_group [
  {
    count: 1
    kind: KIND_GPU
    gpus: [ 0 ]
  }
]
EOF
}

# _write_trtllm_config <model_dir>
_write_trtllm_config() {
    local model_dir="$1"

    cat > "$model_dir/config.pbtxt" << 'EOF'
name: "talker_backbone"
backend: "tensorrtllm"
max_batch_size: 8

input [
  {
    name: "inputs_embeds"
    data_type: TYPE_BF16
    dims: [ -1, 1024 ]
  }
]

output [
  {
    name: "hidden_states"
    data_type: TYPE_BF16
    dims: [ -1, 1024 ]
  },
  {
    name: "logits"
    data_type: TYPE_BF16
    dims: [ -1, 3072 ]
  }
]

instance_group [
  {
    count: 1
    kind: KIND_GPU
    gpus: [ 0 ]
  }
]

parameters: {
  key: "gpt_attention_plugin"
  value: { string_value: "bfloat16" }
}
parameters: {
  key: "paged_kv_cache"
  value: { string_value: "disable" }
}
EOF
}

# _write_orchestrator_config <model_dir> <variant>
_write_orchestrator_config() {
    local model_dir="$1"
    local variant="$2"

    cat > "$model_dir/config.pbtxt" << EOF
name: "tts_orchestrator"
backend: "python"
max_batch_size: 0

model_transaction_policy {
  decoupled: true
}

input [
  {
    name: "request"
    data_type: TYPE_STRING
    dims: [ 1 ]
  }
]

output [
  {
    name: "audio_chunk"
    data_type: TYPE_FP32
    dims: [ -1 ]
  },
  {
    name: "is_final"
    data_type: TYPE_BOOL
    dims: [ 1 ]
  }
]

instance_group [
  {
    count: 1
    kind: KIND_GPU
    gpus: [ 0 ]
  }
]

parameters: {
  key: "model_variant"
  value: { string_value: "${variant}" }
}
parameters: {
  key: "weights_dir"
  value: { string_value: "/models/tts_orchestrator/1/weights" }
}
parameters: {
  key: "tokenizer_dir"
  value: { string_value: "/models/tts_orchestrator/1/tokenizer" }
}
parameters: {
  key: "max_decode_steps"
  value: { string_value: "4096" }
}
parameters: {
  key: "audio_chunk_frames"
  value: { string_value: "25" }
}
EOF

    # Stub model.py only if the real orchestrator was not copied
    local model_py="$model_dir/1/model.py"
    if [ ! -f "$model_py" ]; then
        cat > "$model_py" << 'PYEOF'
import triton_python_backend_utils as pb_utils
import numpy as np
import json

class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])
        print("[TTS Orchestrator] Stub initialized")

    def execute(self, requests):
        responses = []
        for request in requests:
            audio = np.zeros(1, dtype=np.float32)
            is_final = np.array([True], dtype=bool)
            response = pb_utils.InferenceResponse(
                output_tensors=[pb_utils.Tensor("audio_chunk", audio),
                                pb_utils.Tensor("is_final", is_final)])
            responses.append(response)
        return responses

    def finalize(self):
        print("[TTS Orchestrator] Finalized")
PYEOF
        log_info "  tts_orchestrator/model.py: stub created (no source found)"
    fi
}

# _to_triton_dtype <short_dtype>
_to_triton_dtype() {
    case "$1" in
        float|float32|fp32)  echo "TYPE_FP32" ;;
        float16|fp16)        echo "TYPE_FP16" ;;
        bfloat16|bf16)       echo "TYPE_BF16" ;;
        int32)               echo "TYPE_INT32" ;;
        int64)               echo "TYPE_INT64" ;;
        string)              echo "TYPE_STRING" ;;
        bool)                echo "TYPE_BOOL" ;;
        *)                   echo "TYPE_FP32" ;;
    esac
}

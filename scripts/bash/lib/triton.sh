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

# Default production layout (optional: ASSEMBLE_VERIFICATION_MODELS=1 adds tokenizer/talker_unified/code2wav)
_TRITON_MODELS=(
    "speaker_encoder"
    "speech_tokenizer_codec_fused"
    "talker_code2wav_fused"
    "tts_orchestrator"
)

# Default Triton gRPC / HTTP / metrics ports
TRITON_GRPC_PORT="${TRITON_GRPC_PORT:-8001}"
TRITON_HTTP_PORT="${TRITON_HTTP_PORT:-8000}"
TRITON_METRICS_PORT="${TRITON_METRICS_PORT:-8002}"

# ---------------------------------------------------------------------------
#  resolve_triton_deploy_image [driver_version]
#  Returns the Triton deployment image for Phase C.
#  Uses the deploy image (py3 + torch/tokenizers) built by
#  build_triton_deploy_image().  Falls back to raw -py3 if deploy
#  image is not yet built.
#  Echoes the full image tag to stdout.
# ---------------------------------------------------------------------------
resolve_triton_deploy_image() {
    local ngc_tag
    ngc_tag=$(resolve_ngc_tag "$@") \
        || { log_error "Cannot determine NGC tag"; return 1; }

    local deploy_tag="${_DEPLOY_IMAGE_NAME}:${ngc_tag}"
    if docker image inspect "$deploy_tag" &>/dev/null; then
        log_info "Using Triton deploy image: $deploy_tag"
        echo "$deploy_tag"
        return 0
    fi

    local image="${_NGC_TRITON_BASE}:${ngc_tag}${_NGC_PY3_SUFFIX}"
    log_warn "Deploy image not found ($deploy_tag), falling back to: $image"
    log_warn "Build it first: bash scripts/bash/build_triton.sh build-image"
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
#  Default layout: speaker_encoder, speech_tokenizer_codec_fused (base ICL),
#  talker_code2wav_fused, tts_orchestrator. ASSEMBLE_VERIFICATION_MODELS=1 adds
#  speech_tokenizer_encoder, talker_unified, code2wav.
# ---------------------------------------------------------------------------
assemble_model_repo() {
    local exported_dir="$1"
    local variant="$2"
    local repo_dir="$3"
    local engine_mode="${4:-trt}"

    local variant_dir="$exported_dir/$variant"
    local tokenizer_dir="$exported_dir/tokenizer"
    # Read engine dtype from build_engines.sh output; default bf16
    local engine_dtype="bf16"
    if [ -f "$exported_dir/.engine_dtype" ]; then
        engine_dtype=$(cat "$exported_dir/.engine_dtype" 2>/dev/null | tr -d '\n' || echo "bf16")
    fi
    engine_dtype="${engine_dtype:-bf16}"

    log_step "Assembling Triton model repository (engine_mode=$engine_mode, dtype=$engine_dtype)"
    log_info "  Variant:    $variant"
    log_info "  Source:     $variant_dir"
    log_info "  Repository: $repo_dir"

    if [ ! -d "$variant_dir" ]; then
        log_error "Variant directory not found: $variant_dir"
        log_error "Run Phase A first: autorun.sh setup or export_all.py"
        return 1
    fi
    if [ ! -f "$variant_dir/triton_manifest.json" ]; then
        log_error "Missing $variant_dir/triton_manifest.json — run export_09 (writes manifest + fused ONNX)"
        return 1
    fi

    mkdir -p "$repo_dir"
    rm -rf \
        "$repo_dir/speaker_encoder" \
        "$repo_dir/speech_tokenizer_codec_fused" \
        "$repo_dir/talker_code2wav_fused" \
        "$repo_dir/speech_tokenizer_encoder" \
        "$repo_dir/talker_unified" \
        "$repo_dir/code2wav" \
        "$repo_dir/tts_orchestrator" \
        "$repo_dir/triton_manifest.json"

    # Helper: copy engine/onnx only; config.pbtxt comes from generate_triton_configs.py
    _place_model() {
        local name="$1"
        local src="$2"
        local model_dir="$repo_dir/$name/1"
        mkdir -p "$model_dir"
        if [ "$engine_mode" = "trt" ]; then
            _link_or_copy "$src" "$model_dir/model.plan"
        else
            _link_or_copy "$src" "$model_dir/model.onnx"
            # Copy ONNX external data file if present (large models use .onnx.data).
            # Keep the original filename because the ONNX proto references it internally.
            if [ -f "${src}.data" ]; then
                _link_or_copy "${src}.data" "$model_dir/$(basename "${src}.data")"
                log_info "    + copied external data: $(basename "${src}.data")"
            fi
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

    # ── 2. Speech Tokenizer + Codec 3D fused (optional — base ICL) ──
    local stcodec_src
    if stcodec_src="$(_resolve_model_src "$variant_dir/speech_tokenizer_codec_fused")"; then
        _place_model "speech_tokenizer_codec_fused" "$stcodec_src"
        log_info "  speech_tokenizer_codec_fused: OK"
    else
        log_warn "  speech_tokenizer_codec_fused: SKIPPED (not required for non-base / non-ICL)"
    fi

    # ── 3. Talker + Code2Wav fused (required — production single engine) ──
    local fused_src
    if fused_src="$(_resolve_model_src "$variant_dir/talker_code2wav_fused")"; then
        _place_model "talker_code2wav_fused" "$fused_src"
        log_info "  talker_code2wav_fused: OK"
    else
        log_error "  talker_code2wav_fused: MISSING ${engine_mode} file (required). Run export_09 + Phase B."
        return 1
    fi

    # ── 4. Verification-only models (optional) ──
    if [ "${ASSEMBLE_VERIFICATION_MODELS:-0}" = "1" ]; then
        local stoken_src
        if stoken_src="$(_resolve_model_src "$tokenizer_dir/speech_tokenizer_encoder")"; then
            _place_model "speech_tokenizer_encoder" "$stoken_src"
            log_info "  speech_tokenizer_encoder: OK (verification)"
        fi
        local talker_src
        if talker_src="$(_resolve_model_src "$variant_dir/talker_unified")"; then
            _place_model "talker_unified" "$talker_src"
            log_info "  talker_unified: OK (verification)"
        fi
        local c2w_src
        if c2w_src="$(_resolve_model_src "$tokenizer_dir/code2wav_decoder")"; then
            _place_model "code2wav" "$c2w_src"
            log_info "  code2wav: OK (verification)"
        fi
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

    # Copy orchestrator adapter + new engine package (paths relative to repo root)
    local repo_root
    repo_root="$(cd "${_LIB_DIR}/../../.." && pwd)"
    local orch_py_dir="$repo_root/model_repository/tts_orchestrator/1"
    if [ -d "$orch_py_dir" ]; then
        cp "$orch_py_dir/model.py" "$repo_dir/tts_orchestrator/1/model.py"

        rm -rf "$repo_dir/tts_orchestrator/1/engine"
        cp -R "$repo_root/engine" "$repo_dir/tts_orchestrator/1/engine"

        log_info "  tts_orchestrator/python: OK (copied model.py + engine/ package)"
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
        for tf in tokenizer.json tokenizer_config.json vocab.json merges.txt \
                  config.json generation_config.json; do
            [ -f "$tok_dir/$tf" ] && \
                _link_or_copy "$tok_dir/$tf" "$repo_dir/tts_orchestrator/1/tokenizer/$tf"
        done
        log_info "  tts_orchestrator/tokenizer: OK"
    else
        log_warn "  tts_orchestrator/tokenizer: SKIPPED (model dir not found)"
    fi

    _write_tts_orchestrator_stub_if_missing "$repo_dir"

    _link_or_copy "$variant_dir/triton_manifest.json" "$repo_dir/triton_manifest.json"
    mkdir -p "$repo_dir/tts_orchestrator/1"
    _link_or_copy "$variant_dir/triton_manifest.json" \
        "$repo_dir/tts_orchestrator/1/triton_manifest.json"
    log_info "  triton_manifest.json: copied (repo root + tts_orchestrator/1)"

    # Keep the Triton Python backend payload minimal.  Only the new adapter,
    # engine package, tokenizer / weights, and manifest should enter the container.
    find "$repo_dir/tts_orchestrator/1" -mindepth 1 -maxdepth 1 \
        ! -name "model.py" \
        ! -name "engine" \
        ! -name "tokenizer" \
        ! -name "weights" \
        ! -name "triton_manifest.json" \
        -exec rm -rf {} +
    find "$repo_dir/tts_orchestrator/1" -type d -name "__pycache__" -prune -exec rm -rf {} +
    log_info "  tts_orchestrator/python: pruned legacy payload"

    if ! python3 "$repo_root/scripts/python/generate_triton_configs.py" \
        --manifest "$repo_dir/triton_manifest.json" \
        --output-repo "$repo_dir" \
        --engine-mode "$engine_mode" \
        --engine-dtype "$engine_dtype"; then
        log_error "  generate_triton_configs.py failed"
        return 1
    fi
    log_info "  Triton config.pbtxt: generated from triton_manifest.json"

    echo ""
    log_info "Model repository assembled: $repo_dir"
    return 0
}

# ---------------------------------------------------------------------------
#  sync_trt_configs <repo_dir> <engine_mode> <exported_dir>
#  When engine_mode=trt, ensure optional models (speaker_encoder,
#  speech_tokenizer_encoder) that exist in the repo have full TRT config
#  (explicit input/output). Fixes "failed to specify dimensions" when using
#  an existing repo assembled for a different variant (e.g. base had speaker_encoder).
# ---------------------------------------------------------------------------
sync_trt_configs() {
    local repo_dir="$1"
    local engine_mode="$2"
    local exported_dir="${3:-}"
    [ "$engine_mode" != "trt" ] && return 0

    local engine_dtype="bf16"
    if [ -n "$exported_dir" ] && [ -f "$exported_dir/.engine_dtype" ]; then
        engine_dtype=$(cat "$exported_dir/.engine_dtype" 2>/dev/null | tr -d '\n' || echo "bf16")
    fi
    engine_dtype="${engine_dtype:-bf16}"

    if [ ! -f "$repo_dir/triton_manifest.json" ]; then
        log_warn "  sync_trt_configs: no triton_manifest.json in repo (skip)"
        return 0
    fi
    local repo_root
    repo_root="$(cd "${_LIB_DIR}/../../.." && pwd)"
    if python3 "$repo_root/scripts/python/generate_triton_configs.py" \
        --manifest "$repo_dir/triton_manifest.json" \
        --output-repo "$repo_dir" \
        --engine-mode "$engine_mode" \
        --engine-dtype "$engine_dtype"; then
        log_info "  Triton configs: regenerated from triton_manifest.json (sync_trt_configs)"
        return 0
    fi
    log_error "  generate_triton_configs.py failed"
    return 1
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
            if [[ "$model" == "speaker_encoder" || "$model" == "speech_tokenizer_codec_fused" ]]; then
                log_warn "  $model: no config.pbtxt (optional)"
                warned=$((warned + 1))
            else
                log_error "  $model: no config.pbtxt (required)"
                missing=$((missing + 1))
            fi
            continue
        fi

        if [ ! -d "$model_dir/1" ]; then
            if [[ "$model" == "speaker_encoder" || "$model" == "speech_tokenizer_codec_fused" ]]; then
                log_warn "  $model: no version directory (optional)"
                warned=$((warned + 1))
            else
                log_error "  $model: no version directory (1/)"
                missing=$((missing + 1))
            fi
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
    local trt_python_version="${TRITON_TENSORRT_PIP_VERSION:-10.15.1.29}"

    if [ -z "$base_image" ]; then
        base_image=$(resolve_triton_deploy_image) \
            || { log_error "Cannot determine base image"; return 1; }
    fi

    local model_repo="$repo_root/workspace/model_repository"
    local fused_artifact=""
    if [ -f "$model_repo/talker_code2wav_fused/1/model.plan" ]; then
        fused_artifact="$model_repo/talker_code2wav_fused/1/model.plan"
    elif [ -f "$model_repo/talker_code2wav_fused/1/model.onnx" ]; then
        fused_artifact="$model_repo/talker_code2wav_fused/1/model.onnx"
    fi

    if [ ! -f "$model_repo/tts_orchestrator/1/model.py" ] \
        || [ ! -d "$model_repo/tts_orchestrator/1/engine" ] \
        || [ -z "$fused_artifact" ]; then
        log_error "Assembled model repository is incomplete for image build"
        log_error "Expected:"
        log_error "  $model_repo/tts_orchestrator/1/model.py"
        log_error "  $model_repo/tts_orchestrator/1/engine/"
        log_error "  $model_repo/talker_code2wav_fused/1/model.plan or model.onnx"
        return 1
    fi

    local build_dir
    build_dir=$(mktemp -d)
    local dockerfile="$build_dir/Dockerfile"
    cat > "$dockerfile" <<DOCKERFILE
ARG BASE_IMAGE=${base_image}
ARG TENSORRT_PYTHON_VERSION=${trt_python_version}
FROM \${BASE_IMAGE}
ARG TENSORRT_PYTHON_VERSION

RUN python3 -m pip install --no-cache-dir \
    -i https://mirrors.bfsu.edu.cn/pypi/web/simple \
    --trusted-host mirrors.bfsu.edu.cn \
    --extra-index-url https://download.pytorch.org/whl/cu130 \
    torch \
    tokenizers \
    scipy \
    soundfile \
    "tensorrt==\${TENSORRT_PYTHON_VERSION}"

COPY workspace/model_repository /models

HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=6 \
    CMD curl -f http://localhost:8000/v2/health/ready || exit 1

EXPOSE 8000 8001 8002

ENTRYPOINT ["tritonserver"]
CMD ["--model-repository=/models", "--strict-model-config=false", "--disable-auto-complete-config", "--log-verbose=1"]
DOCKERFILE

    log_step "Building Triton deployment image: $image_tag"
    log_info "  Base image:  $base_image"
    log_info "  Runtime:     /models/tts_orchestrator/1/model.py + engine/ + $(basename "$fused_artifact")"

    docker build \
        --build-arg "BASE_IMAGE=$base_image" \
        --build-arg "TENSORRT_PYTHON_VERSION=$trt_python_version" \
        -t "$image_tag" \
        -f "$dockerfile" \
        "$repo_root" \
        || { rm -rf "$build_dir"; log_error "Docker build failed"; return 1; }

    rm -rf "$build_dir"

    log_info "Image built: $image_tag"
}

# ---------------------------------------------------------------------------
#  triton_resolve_container_name <model_repo_dir> <container_name_arg> [variant_cli]
#  Must stay in sync with triton_run naming: default base name + model_variant
#  from assembled tts_orchestrator/config.pbtxt, or explicit --variant on stop.
#  Precedence: non-default container_name_arg; else variant_cli; else pbtxt.
# ---------------------------------------------------------------------------
triton_resolve_container_name() {
    local repo_dir="$1"
    local cname="${2:-qwen3-tts-triton}"
    local variant_cli="${3:-}"

    if [ "$cname" != "qwen3-tts-triton" ]; then
        printf '%s\n' "$cname"
        return 0
    fi
    if [ -n "$variant_cli" ]; then
        printf '%s\n' "qwen3-tts-triton-${variant_cli}"
        return 0
    fi
    local config_file="$repo_dir/tts_orchestrator/config.pbtxt"
    if [ -f "$config_file" ]; then
        local var_val
        var_val=$(grep -A 1 'key: "model_variant"' "$config_file" | grep 'string_value' | cut -d'"' -f2 || true)
        if [ -n "$var_val" ]; then
            printf '%s\n' "qwen3-tts-triton-${var_val}"
            return 0
        fi
    fi
    printf '%s\n' "qwen3-tts-triton"
}

# ---------------------------------------------------------------------------
#  triton_run <model_repo_dir> [image] [container_name] [extra_docker_args...]
#  Starts a Triton server container with the model repository volume-mounted.
#  Runs in detached mode; use triton_health_check to verify readiness.
# ---------------------------------------------------------------------------
triton_run() {
    local repo_dir="$1"
    local image="${2:-}"
    local cname_arg="${3:-qwen3-tts-triton}"
    shift 3 2>/dev/null || true

    if [ -z "$image" ]; then
        image=$(resolve_triton_deploy_image) \
            || { log_error "Cannot determine Triton image"; return 1; }
    fi

    repo_dir="$(cd "$repo_dir" && pwd)"

    local container_name
    container_name=$(triton_resolve_container_name "$repo_dir" "$cname_arg" "")

    # Remove existing container (running or exited) so we can start fresh
    if docker container inspect "$container_name" &>/dev/null; then
        log_warn "Removing existing container: $container_name"
        docker rm -f "$container_name" &>/dev/null || true
    fi

    log_step "Starting Triton Inference Server"
    log_info "  Image:      $image"
    log_info "  Repository: $repo_dir"
    log_info "  Container:  $container_name"
    log_info "  Ports:      gRPC=$TRITON_GRPC_PORT HTTP=$TRITON_HTTP_PORT metrics=$TRITON_METRICS_PORT"

    local gpu_device="${TRITON_GPU_DEVICE:-0}"
    local variant_label=""
    local type_label=""
    
    # Try to extract variant from config.pbtxt if available
    local config_file="$repo_dir/tts_orchestrator/config.pbtxt"
    if [ -f "$config_file" ]; then
        local var_val=$(grep -A 1 'key: "model_variant"' "$config_file" | grep 'string_value' | cut -d'"' -f2 || true)
        local type_val=$(grep -A 1 'key: "tts_model_type"' "$config_file" | grep 'string_value' | cut -d'"' -f2 || true)
        if [ -n "$var_val" ]; then variant_label="--label tts.variant=${var_val}"; fi
        if [ -n "$type_val" ]; then type_label="--label tts.model_type=${type_val}"; fi
    fi

    # When code2wav uses TensorRT (BF16), pass override so orchestrator sends correct dtype
    local code2wav_bf16_env=""
    if [ -f "$repo_dir/code2wav/config.pbtxt" ] && grep -q "tensorrt" "$repo_dir/code2wav/config.pbtxt" 2>/dev/null; then
        code2wav_bf16_env="-e OVERRIDE_CODE2WAV_BF16=1"
    fi

    docker run -d --gpus "\"device=${gpu_device}\"" \
        --name "$container_name" \
        --shm-size=1g \
        --ulimit memlock=-1 \
        $variant_label $type_label \
        $code2wav_bf16_env \
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
            --disable-auto-complete-config \
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

    # Exact name only (docker ps --filter name= is substring match and caused false positives)
    if ! docker container inspect "$container_name" &>/dev/null; then
        log_info "Container not found: $container_name"
        return 0
    fi
    log_info "Stopping and removing Triton container: $container_name"
    docker rm -f "$container_name" \
        || { log_error "Failed to remove container: $container_name"; return 1; }
    log_info "Container removed: $container_name"
}


# ---------------------------------------------------------------------------
#  _write_tts_orchestrator_stub_if_missing <repo_dir>
#  Minimal Python stub when model_repository sources were not copied.
# ---------------------------------------------------------------------------
_write_tts_orchestrator_stub_if_missing() {
    local orch_dir="$1/tts_orchestrator/1"
    mkdir -p "$orch_dir"
    if [ -f "$orch_dir/model.py" ]; then
        return 0
    fi
    cat > "$orch_dir/model.py" << 'PYEOF'
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
            audio = np.array([b""], dtype=object)
            is_final = np.array([True], dtype=bool)
            response = pb_utils.InferenceResponse(
                output_tensors=[pb_utils.Tensor("audio_chunk", audio),
                                pb_utils.Tensor("event_type", np.array(["error"], dtype=object)),
                                pb_utils.Tensor("event_json", np.array([json.dumps({"type":"error","message":"stub"})], dtype=object)),
                                pb_utils.Tensor("is_final", is_final)])
            responses.append(response)
        return responses

    def finalize(self):
        print("[TTS Orchestrator] Finalized")
PYEOF
    log_warn "  tts_orchestrator/model.py: stub created (no source tree)"
}

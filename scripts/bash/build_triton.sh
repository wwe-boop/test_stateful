#!/bin/bash
# ===========================================================================
#  build_triton.sh — Phase C: Triton deployment container setup
#
#  Assembles the Triton model_repository from Phase A/B artifacts, pulls
#  the NGC Triton container, and optionally starts the server.
#
#  Modes of operation:
#    1. assemble    — Build model_repository/ from workspace/exported/
#    2. pull        — Pull the NGC Triton container image
#    3. run         — Assemble + run Triton server (uses full py3 image)
#    4. build       — Build a self-contained Docker image with baked-in models
#    5. stop        — Stop a running Triton container
#
#  Prerequisites:
#    - Phase A: workspace/exported/<variant>/ with ONNX + weights
#    - Phase B (optional): .engine files for --engine-mode trt
#
#  Usage:
#    bash scripts/bash/build_triton.sh assemble                          # assemble (ONNX)
#    bash scripts/bash/build_triton.sh assemble --engine-mode trt         # use TensorRT engines
#    bash scripts/bash/build_triton.sh run --engine-mode onnx|trt
#    bash scripts/bash/build_triton.sh stop
#    bash scripts/bash/build_triton.sh --generate-dockerfile
#
#  Environment variables:
#    TRITON_IMAGE          Docker image override (default: auto from driver)
#    TRITON_GRPC_PORT      gRPC port   (default: 8001)
#    TRITON_HTTP_PORT      HTTP port   (default: 8000)
#    TRITON_METRICS_PORT   Metrics port (default: 8002)
#    CONTAINER_NAME        Container name (default: qwen3-tts-triton)
#    MODEL_REPO_DIR        Override model_repository path
#    TRITON_GPU_DEVICE     Runtime GPU device (auto | N | cuda:N)
#    TRITON_MAX_BATCH_SLOTS Runtime active decode slots
#    TRITON_MAX_SEQ_LEN    Runtime max sequence length
# ===========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
source "${SCRIPT_DIR}/tools.sh"

_NGC_VERIFY_MANIFEST=1

# ── Defaults ──
EXPORTED_DIR="${REPO_ROOT}/workspace/exported"
MODEL_REPO_DIR="${MODEL_REPO_DIR:-${REPO_ROOT}/workspace/model_repository}"
CONTAINER_NAME="${CONTAINER_NAME:-qwen3-tts-triton}"
VARIANT=""
ENGINE_MODE="${ENGINE_MODE:-trt}"
USER_IMAGE="${TRITON_IMAGE:-}"
BUILD_TAG=""
HEALTH_TIMEOUT=120
TRITON_GPU_DEVICE="${TRITON_GPU_DEVICE:-${RUNTIME_GPU_DEVICE:-auto}}"
TRITON_MAX_BATCH_SLOTS="${TRITON_MAX_BATCH_SLOTS:-${RUNTIME_MAX_BATCH_SIZE:-}}"
TRITON_MAX_SEQ_LEN="${TRITON_MAX_SEQ_LEN:-${RUNTIME_MAX_SEQ_LEN:-}}"

# ── Subcommand functions ──

usage() {
    cat << 'EOF'
Usage: build_triton.sh <command> [options]

Commands:
  assemble               Assemble model_repository/ from exported artifacts
  pull                   Pull the NGC Triton container image
  build-image            Build deploy image (NGC base + torch/tokenizers, no model repo)
  run                    Assemble (if needed) + start Triton server
  build                  Build a self-contained deployment Docker image
  stop                   Stop the running Triton container
  status                 Show container status and health

Options:
  --variant <name>       Target model variant (default: auto-discover first)
  --engine-mode onnx|trt Use ONNX or TensorRT engines (default: trt)
  --image <uri>          Override NGC container image
  --repo-dir <path>      Override model_repository output path
  --container <name>     Container name (default: qwen3-tts-triton)
  --device <N|auto>      Runtime GPU device for Triton
  --max-batch <N>        Runtime active decode slots
  --max-seq-len <N>      Runtime max sequence length
  --tag <image:tag>      Docker image tag (for 'build' command)
  --no-health-check      Skip health check after 'run'
  --generate-dockerfile  Generate Dockerfile.triton and exit
  --dry-run              Show what would be done
  -h, --help             Show this help
EOF
}

# ── Generate Dockerfile.triton ──
# Defined early because --generate-dockerfile triggers it before subcommand parsing.
generate_dockerfile() {
    local dockerfile="$REPO_ROOT/Dockerfile.triton"

    log_step "Generating Dockerfile.triton"

    cat > "$dockerfile" << 'DOCKERFILE'
# ===========================================================================
#  Dockerfile.triton — Self-contained Qwen3-TTS Triton deployment image
#
#  Base: NVIDIA Triton full py3 image (onnxruntime + tensorrt + python).
#  Build: bash scripts/bash/build_triton.sh build --tag qwen3-tts-triton:latest
#  Run:   docker run --gpus all -p 8000:8000 -p 8001:8001 -p 8002:8002 <tag>
# ===========================================================================

ARG BASE_IMAGE=nvcr.io/nvidia/tritonserver:25.05-py3
FROM ${BASE_IMAGE}

LABEL maintainer="Qwen3-TTS-Triton"
LABEL description="Qwen3-TTS streaming TTS inference with Triton"

# Model repository
COPY workspace/model_repository /models

# Health check
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=6 \
    CMD curl -f http://localhost:8000/v2/health/ready || exit 1

EXPOSE 8000 8001 8002

ENTRYPOINT ["tritonserver"]
CMD ["--model-repository=/models", "--strict-model-config=false", "--log-verbose=1"]
DOCKERFILE

    log_info "Generated: $dockerfile"
    log_info "Build with: bash scripts/bash/build_triton.sh build --tag qwen3-tts-triton:latest"
}

# ── Argument parsing ──
COMMAND=""
DRY_RUN=false
NO_HEALTH_CHECK=false
GENERATE_DOCKERFILE=false

# Handle --generate-dockerfile before subcommand parsing
for arg in "$@"; do
    if [[ "$arg" == "--generate-dockerfile" ]]; then
        GENERATE_DOCKERFILE=true
    fi
done

if $GENERATE_DOCKERFILE; then
    generate_dockerfile
    exit 0
fi

if [[ $# -eq 0 ]]; then
    usage
    exit 1
fi

COMMAND="$1"
shift

while [[ $# -gt 0 ]]; do
    case "$1" in
        --variant)        VARIANT="$2"; shift 2 ;;
        --engine-mode)    ENGINE_MODE="$2"; shift 2 ;;
        --image)          USER_IMAGE="$2"; shift 2 ;;
        --repo-dir)       MODEL_REPO_DIR="$2"; shift 2 ;;
        --container)      CONTAINER_NAME="$2"; shift 2 ;;
        --device|--triton-device) TRITON_GPU_DEVICE="$2"; shift 2 ;;
        --max-batch|--runtime-max-batch-size|--runtime-max-batch) TRITON_MAX_BATCH_SLOTS="$2"; shift 2 ;;
        --max-seq-len|--runtime-max-seq-len|--runtime-max-seq) TRITON_MAX_SEQ_LEN="$2"; shift 2 ;;
        --tag)            BUILD_TAG="$2"; shift 2 ;;
        --no-health-check) NO_HEALTH_CHECK=true; shift ;;
        --dry-run)        DRY_RUN=true; shift ;;
        --generate-dockerfile) shift ;;  # already handled
        --help|-h)        usage; exit 0 ;;
        *) log_error "Unknown option: $1"; usage; exit 1 ;;
    esac
done

# ── Discover first available variant ──
discover_first_variant() {
    for vdir in "$EXPORTED_DIR"/*/; do
        local vname
        vname=$(basename "$vdir")
        [[ "$vname" == "tokenizer" ]] && continue
        if [ -d "$vdir/weights" ]; then
            echo "$vname"
            return 0
        fi
    done
    return 1
}

resolve_variant() {
    if [ -n "$VARIANT" ]; then
        if [ ! -d "$EXPORTED_DIR/$VARIANT" ]; then
            log_error "Variant not found: $EXPORTED_DIR/$VARIANT"
            exit 1
        fi
        return 0
    fi

    VARIANT=$(discover_first_variant) \
        || { log_error "No exported variants found in $EXPORTED_DIR/"; exit 1; }
    log_info "Auto-discovered variant: $VARIANT"
}

# ── Commands ──

cmd_assemble() {
    resolve_variant

    if $DRY_RUN; then
        log_info "[DRY RUN] Would assemble model repo:"
        log_info "  Source:  $EXPORTED_DIR/$VARIANT"
        log_info "  Target:  $MODEL_REPO_DIR"
        log_info "  Engine:  $ENGINE_MODE"
        return 0
    fi

    assemble_model_repo "$EXPORTED_DIR" "$VARIANT" "$MODEL_REPO_DIR" "$ENGINE_MODE" \
        || { log_error "Assembly failed"; exit 1; }

    echo ""
    validate_model_repo "$MODEL_REPO_DIR"
    local status=$?

    echo ""
    if [ $status -eq 0 ]; then
        log_info "Next steps:"
        log_info "  1. Pull container:  bash scripts/bash/build_triton.sh pull"
        log_info "  2. Start server:    bash scripts/bash/build_triton.sh run"
    fi

    return $status
}

cmd_pull() {
    check_docker_gpu_ready || exit 1

    if $DRY_RUN; then
        log_info "[DRY RUN] Would pull Triton full image (py3)"
        return 0
    fi

    # Sync NGC compatibility matrix from NVIDIA website (best-effort, 25s timeout)
    if [[ -z "${NGC_SKIP_MATRIX_UPDATE:-}" ]]; then
        log_info "[1/3] Syncing NGC matrix from NVIDIA website (timeout 25s)..."
        if command -v timeout &>/dev/null; then
            timeout 25 bash -c "source '${SCRIPT_DIR}/lib/ngc_updater.sh' 2>/dev/null && update_ngc_matrix '${SCRIPT_DIR}/ngc_matrix.conf'" 2>/dev/null || true
        else
            source "${SCRIPT_DIR}/lib/ngc_updater.sh" 2>/dev/null || true
            update_ngc_matrix "${SCRIPT_DIR}/ngc_matrix.conf" 2>/dev/null || true
        fi
        log_info "[1/3] Done (or skipped)"
    fi

    log_info "[2/3] Resolving NGC image for your driver (checking registry)..."
    TRITON_IMAGE=$(resolve_triton_deploy_image) || exit 1
    log_info "[2/3] Using: $TRITON_IMAGE"
    log_info "[3/3] Pulling image (15-30 GB, may take several minutes)..."
    ensure_ngc_image "$TRITON_IMAGE" || exit 1
    log_info "Image ready: $TRITON_IMAGE"
}

cmd_run() {
    resolve_variant

    # Assemble if needed, or re-assemble if source engines are newer than assembled files
    local need_assemble=false
    if [ ! -d "$MODEL_REPO_DIR" ] || [ -z "$(ls -A "$MODEL_REPO_DIR" 2>/dev/null)" ]; then
        need_assemble=true
        log_info "Model repository not found, will assemble (engine_mode=$ENGINE_MODE) ..."
    else
        local stale=false
        local target_artifact
        if [ "$ENGINE_MODE" = "trt" ]; then
            target_artifact="$MODEL_REPO_DIR/tts_orchestrator/1/runtime/model.plan"
            local src_engine="$EXPORTED_DIR/$VARIANT/talker_code2wav_fused.engine"
            if [ -f "$src_engine" ] && { [ ! -f "$target_artifact" ] || [ ! -s "$target_artifact" ] || [ "$src_engine" -nt "$target_artifact" ]; }; then
                stale=true
                log_warn "Runtime fused TRT engine is missing or stale"
            fi
        else
            target_artifact="$MODEL_REPO_DIR/tts_orchestrator/1/runtime/model.onnx"
            local src_engine="$EXPORTED_DIR/$VARIANT/talker_code2wav_fused.onnx"
            if [ -f "$src_engine" ] && { [ ! -f "$target_artifact" ] || [ ! -s "$target_artifact" ] || [ "$src_engine" -nt "$target_artifact" ]; }; then
                stale=true
                log_warn "Runtime fused ONNX is missing or stale"
            fi
        fi
        local src_runtime_onnx
        local dst_runtime_onnx
        for src_runtime_onnx in \
            "$EXPORTED_DIR/$VARIANT/speaker_encoder.onnx" \
            "$EXPORTED_DIR/$VARIANT/speech_tokenizer_codec_fused.onnx"; do
            [ -f "$src_runtime_onnx" ] || continue
            dst_runtime_onnx="$MODEL_REPO_DIR/tts_orchestrator/1/runtime/$(basename "$src_runtime_onnx")"
            if [ ! -f "$dst_runtime_onnx" ] || [ "$src_runtime_onnx" -nt "$dst_runtime_onnx" ]; then
                stale=true
                log_warn "Runtime ONNX asset is missing or stale: $(basename "$src_runtime_onnx")"
                break
            fi
        done
        if [ -f "$EXPORTED_DIR/$VARIANT/triton_manifest.json" ] && \
            { [ ! -f "$MODEL_REPO_DIR/tts_orchestrator/1/runtime/triton_manifest.json" ] || \
              [ "$EXPORTED_DIR/$VARIANT/triton_manifest.json" -nt "$MODEL_REPO_DIR/tts_orchestrator/1/runtime/triton_manifest.json" ]; }; then
            stale=true
            log_warn "Runtime manifest is missing or stale"
        fi
        # Also re-assemble when orchestrator Python source (e.g. model.py) is newer
        local orch_src="$REPO_ROOT/model_repository/tts_orchestrator/1/model.py"
        local orch_dst="$MODEL_REPO_DIR/tts_orchestrator/1/model.py"
        if [ -f "$orch_src" ] && [ -f "$orch_dst" ] && [ "$orch_src" -nt "$orch_dst" ]; then
            stale=true
            log_warn "Orchestrator Python source (model.py) is newer than assembled copy"
        fi
        # Re-assemble if the new TTSEngine payload is missing from an older assemble.
        if [ ! -f "$MODEL_REPO_DIR/tts_orchestrator/1/engine/server.py" ]; then
            stale=true
            log_warn "TTSEngine package missing in model repo: tts_orchestrator/1/engine/server.py"
        fi
        local legacy_payload
        for legacy_payload in \
            "__pycache__" \
            "greedy_tokenizer.py" \
            "batch_decode_scheduler.py" \
            "text_segmenter.py" \
            "prefill_builder.py" \
            "audio_utils.py" \
            "lightweight_tokenizer.py" \
            "session_manager.py" \
            "decode_fsm.py" \
            "ratio_tracker.py" \
            "mlfq_scheduler.py"; do
            if [ -e "$MODEL_REPO_DIR/tts_orchestrator/1/$legacy_payload" ]; then
                stale=true
                log_warn "Legacy BLS payload still present in model repo: tts_orchestrator/1/$legacy_payload"
                break
            fi
        done
        if $stale; then
            need_assemble=true
            log_warn "Model repository is stale (source engines or orchestrator newer), re-assembling ..."
        else
            log_info "Using existing model repository: $MODEL_REPO_DIR"
        fi
    fi
    if $need_assemble; then
        assemble_model_repo "$EXPORTED_DIR" "$VARIANT" "$MODEL_REPO_DIR" "$ENGINE_MODE" \
            || { log_error "Assembly failed"; exit 1; }
    fi

    # Sync TRT configs for optional models (speaker_encoder, speech_tokenizer_encoder)
    # when they exist from a prior run but current variant didn't place them.
    sync_trt_configs "$MODEL_REPO_DIR" "$ENGINE_MODE" "$EXPORTED_DIR"

    validate_model_repo "$MODEL_REPO_DIR" || exit 1

    check_docker_gpu_ready || exit 1

    if [ -z "$USER_IMAGE" ]; then
        TRITON_IMAGE=$(resolve_triton_deploy_image) \
            || { log_error "Failed to resolve Triton image"; exit 1; }
    else
        TRITON_IMAGE="$USER_IMAGE"
        log_info "Using user-specified image: $TRITON_IMAGE"
    fi

    local resolved_cname
    resolved_cname=$(triton_resolve_container_name "$MODEL_REPO_DIR" "$CONTAINER_NAME" "")

    if $DRY_RUN; then
        log_info "[DRY RUN] Would start Triton:"
        log_info "  Image:      $TRITON_IMAGE"
        log_info "  Repository: $MODEL_REPO_DIR"
        log_info "  Container:  $resolved_cname"
        log_info "  Device:     $TRITON_GPU_DEVICE"
        log_info "  Max batch:  ${TRITON_MAX_BATCH_SLOTS:-manifest/default}"
        log_info "  Max seq:    ${TRITON_MAX_SEQ_LEN:-manifest/default}"
        log_info "  Ports:      gRPC=$TRITON_GRPC_PORT HTTP=$TRITON_HTTP_PORT metrics=$TRITON_METRICS_PORT"
        return 0
    fi

    local compose_args=(
        up
        --gateway triton
        --variant "$VARIANT"
        --repo-dir "$MODEL_REPO_DIR"
        --image "$TRITON_IMAGE"
        --device "$TRITON_GPU_DEVICE"
    )
    if [[ -n "$TRITON_MAX_BATCH_SLOTS" ]]; then
        compose_args+=(--max-batch "$TRITON_MAX_BATCH_SLOTS")
    fi
    if [[ -n "$TRITON_MAX_SEQ_LEN" ]]; then
        compose_args+=(--max-seq-len "$TRITON_MAX_SEQ_LEN")
    fi
    if [[ -n "${CONTAINER_NAME:-}" ]]; then
        compose_args+=(--container "$CONTAINER_NAME")
    fi
    if $NO_HEALTH_CHECK; then
        compose_args+=(--no-health-check)
    fi
    bash "${SCRIPT_DIR}/compose.sh" "${compose_args[@]}" || exit 1

    echo ""
    log_step "Triton Server Running"
    log_info "  gRPC endpoint:    localhost:${TRITON_GRPC_PORT}"
    log_info "  HTTP endpoint:    localhost:${TRITON_HTTP_PORT}"
    log_info "  Metrics endpoint: localhost:${TRITON_METRICS_PORT}/metrics"
    log_info "  Container:        ${CONTAINER_NAME:-qwen3-tts-triton}"
    echo ""
    log_info "Stop server: bash scripts/bash/build_triton.sh stop"
}

cmd_build_image() {
    log_step "Building Triton deploy image (Triton base + torch + tokenizers)"
    log_info "Can run in parallel with engine build (build_engines.sh)"
    check_docker_gpu_ready || exit 1
    ensure_triton_deploy_image || exit 1
    log_info "Deploy image ready. Run 'build' or 'run' after engines are ready."
}

cmd_build() {
    if [ -z "$BUILD_TAG" ]; then
        BUILD_TAG="qwen3-tts-triton:latest"
        log_info "Using default image tag: $BUILD_TAG"
    fi

    check_docker_gpu_ready || exit 1

    if [ -z "$USER_IMAGE" ]; then
        TRITON_IMAGE=$(resolve_triton_deploy_image) \
            || { log_error "Failed to resolve Triton image"; exit 1; }
    else
        TRITON_IMAGE="$USER_IMAGE"
    fi

    if [ ! -d "$MODEL_REPO_DIR" ] || [ -z "$(ls -A "$MODEL_REPO_DIR" 2>/dev/null)" ]; then
        resolve_variant
        assemble_model_repo "$EXPORTED_DIR" "$VARIANT" "$MODEL_REPO_DIR" "$ENGINE_MODE" \
            || { log_error "Assembly failed"; exit 1; }
    fi
    validate_model_repo "$MODEL_REPO_DIR" || exit 1

    # Generate Dockerfile if missing
    if [ ! -f "$REPO_ROOT/Dockerfile.triton" ]; then
        generate_dockerfile
    fi

    if $DRY_RUN; then
        log_info "[DRY RUN] Would build: $BUILD_TAG (base: $TRITON_IMAGE)"
        return 0
    fi

    build_triton_image "$REPO_ROOT" "$BUILD_TAG" "$TRITON_IMAGE" || exit 1

    echo ""
    log_info "Run with:"
    log_info "  docker run --gpus all -p 8000:8000 -p 8001:8001 -p 8002:8002 $BUILD_TAG"
}

cmd_stop() {
    local compose_args=(down --gateway triton --repo-dir "$MODEL_REPO_DIR")
    if [[ -n "${CONTAINER_NAME:-}" ]]; then
        compose_args+=(--container "$CONTAINER_NAME")
    fi
    bash "${SCRIPT_DIR}/compose.sh" "${compose_args[@]}"
}

cmd_status() {
    log_step "Triton Container Status"
    local compose_args=(ps --gateway triton --repo-dir "$MODEL_REPO_DIR")
    if [[ -n "${CONTAINER_NAME:-}" ]]; then
        compose_args+=(--container "$CONTAINER_NAME")
    fi
    bash "${SCRIPT_DIR}/compose.sh" "${compose_args[@]}"
}

# ── Main dispatch ──
case "$COMMAND" in
    assemble)     cmd_assemble ;;
    pull)        cmd_pull ;;
    build-image) cmd_build_image ;;
    run)         cmd_run ;;
    build)       cmd_build ;;
    stop)        cmd_stop ;;
    status)      cmd_status ;;
    -h|--help)   usage ;;
    *)
        log_error "Unknown command: $COMMAND"
        usage
        exit 1
        ;;
esac

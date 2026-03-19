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

# ── Subcommand functions ──

usage() {
    cat << 'EOF'
Usage: build_triton.sh <command> [options]

Commands:
  assemble               Assemble model_repository/ from exported artifacts
  pull                   Pull the NGC Triton container image
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
        local ext; [ "$ENGINE_MODE" = "trt" ] && ext="engine" || ext="onnx"
        for src_engine in "$EXPORTED_DIR/$VARIANT"/*."$ext" "$EXPORTED_DIR/tokenizer"/*."$ext"; do
            [ -f "$src_engine" ] || continue
            local base_name
            base_name=$(basename "$src_engine" ".$ext")
            local repo_name="$base_name"
            [ "$repo_name" = "code2wav_decoder" ] && repo_name="code2wav"
            local target; [ "$ENGINE_MODE" = "trt" ] && target="model.plan" || target="model.onnx"
            local dst="$MODEL_REPO_DIR/$repo_name/1/$target"
            if [ ! -f "$dst" ] || [ ! -s "$dst" ] || [ "$src_engine" -nt "$dst" ]; then
                stale=true
                break
            fi
        done
        # Also re-assemble when orchestrator Python source (e.g. model.py) is newer
        local orch_src="$REPO_ROOT/model_repository/tts_orchestrator/1/model.py"
        local orch_dst="$MODEL_REPO_DIR/tts_orchestrator/1/model.py"
        if [ -f "$orch_src" ] && [ -f "$orch_dst" ] && [ "$orch_src" -nt "$orch_dst" ]; then
            stale=true
            log_warn "Orchestrator Python source (model.py) is newer than assembled copy"
        fi
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

    validate_model_repo "$MODEL_REPO_DIR" || exit 1

    check_docker_gpu_ready || exit 1

    if [ -z "$USER_IMAGE" ]; then
        TRITON_IMAGE=$(resolve_triton_deploy_image) \
            || { log_error "Failed to resolve Triton image"; exit 1; }
    else
        TRITON_IMAGE="$USER_IMAGE"
        log_info "Using user-specified image: $TRITON_IMAGE"
    fi

    if $DRY_RUN; then
        log_info "[DRY RUN] Would start Triton:"
        log_info "  Image:      $TRITON_IMAGE"
        log_info "  Repository: $MODEL_REPO_DIR"
        log_info "  Container:  $CONTAINER_NAME"
        log_info "  Ports:      gRPC=$TRITON_GRPC_PORT HTTP=$TRITON_HTTP_PORT metrics=$TRITON_METRICS_PORT"
        return 0
    fi

    triton_run "$MODEL_REPO_DIR" "$TRITON_IMAGE" "$CONTAINER_NAME" \
        || exit 1

    if ! $NO_HEALTH_CHECK; then
        if triton_health_check "localhost" "$TRITON_HTTP_PORT" "$HEALTH_TIMEOUT"; then
            echo ""
            log_step "Triton Server Running"
            log_info "  gRPC endpoint:    localhost:${TRITON_GRPC_PORT}"
            log_info "  HTTP endpoint:    localhost:${TRITON_HTTP_PORT}"
            log_info "  Metrics endpoint: localhost:${TRITON_METRICS_PORT}/metrics"
            log_info "  Container:        $CONTAINER_NAME"
            echo ""
            log_info "Loaded models:"
            curl -s "http://localhost:${TRITON_HTTP_PORT}/v2/models" 2>/dev/null \
                | python3 -m json.tool 2>/dev/null \
                || log_warn "Could not query model list (server may still be loading)"
            echo ""
            log_info "Stop server: bash scripts/bash/build_triton.sh stop"
        else
            log_error "Server failed to become ready within ${HEALTH_TIMEOUT}s"
            log_info "Check logs: docker logs $CONTAINER_NAME"
            exit 1
        fi
    else
        log_info "Health check skipped. Check manually:"
        log_info "  curl http://localhost:${TRITON_HTTP_PORT}/v2/health/ready"
    fi
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
    triton_stop "$CONTAINER_NAME"
}

cmd_status() {
    log_step "Triton Container Status"

    if docker ps --filter "name=$CONTAINER_NAME" --format "{{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -q .; then
        log_info "Container running:"
        docker ps --filter "name=$CONTAINER_NAME" --format "  Name:    {{.Names}}\n  Status:  {{.Status}}\n  Ports:   {{.Ports}}"
        echo ""
        if curl -sf "http://localhost:${TRITON_HTTP_PORT}/v2/health/ready" &>/dev/null; then
            log_info "Health: READY"
            echo ""
            log_info "Loaded models:"
            curl -s "http://localhost:${TRITON_HTTP_PORT}/v2/models" 2>/dev/null \
                | python3 -m json.tool 2>/dev/null \
                || true
        else
            log_warn "Health: NOT READY (still loading or unhealthy)"
        fi
    else
        log_info "Container '$CONTAINER_NAME' is not running"
    fi
}

# ── Main dispatch ──
case "$COMMAND" in
    assemble)    cmd_assemble ;;
    pull)        cmd_pull ;;
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

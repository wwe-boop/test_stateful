#!/bin/bash
# ===========================================================================
#  deploy.sh — Phase C: Deploy TTS service (standalone engine or Triton)
#
#  Unified entry point for deploying the TTS service. Supports three gateway modes:
#
#  A) Standalone gRPC Server (--gateway standalone)
#     Runs `python -m engine.server` on the host (see ENGINE_PYTHON / conda).
#     Best for: development, single-model production, low-latency.
#
#  B) Triton Backend (--gateway triton)
#     Assembles model_repository + starts Triton via docker compose.
#     Best for: multi-model serving, K8s, enterprise infrastructure.
#
#  C) Engine Docker (--gateway engine-docker)
#     Builds (if missing) Dockerfile.engine and runs engine.server via docker compose;
#     mounts workspace/ read-only. No host PyTorch required.
#     Best for: portable deploy, matching TRT base with Phase B.
#
#  Usage:
#    bash scripts/bash/deploy.sh run                               # standalone (default)
#    bash scripts/bash/deploy.sh run --gateway triton               # Triton mode
#    bash scripts/bash/deploy.sh run --gateway engine-docker      # Engine image + container
#    bash scripts/bash/deploy.sh run --foreground                   # don't daemonize
#    bash scripts/bash/deploy.sh stop                               # stop service
#    bash scripts/bash/deploy.sh status                             # show status
#
#  Triton-specific commands (forwarded to build_triton.sh):
#    bash scripts/bash/deploy.sh assemble [--engine-mode onnx|trt]
#    bash scripts/bash/deploy.sh pull
#    bash scripts/bash/deploy.sh build-image
#    bash scripts/bash/deploy.sh build [--tag <image:tag>]
#
#  Environment variables:
#    GATEWAY_MODE            Override gateway: standalone | triton | engine-docker
#    ENGINE_GRPC_PORT        Standalone gRPC port (default: 50051)
#    ENGINE_PYTHON           Python binary for standalone engine (default: conda env qwen3-tts, else PATH)
#    QWEN3_TTS_ENV_NAME      Conda env name for auto-resolve (default: qwen3-tts)
#    TRITON_GRPC_PORT        Triton gRPC port (default: 8001)
# ===========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
source "${SCRIPT_DIR}/tools.sh"

# ── Defaults ──
EXPORTED_DIR="${REPO_ROOT}/workspace/exported"
GATEWAY_MODE="${GATEWAY_MODE:-standalone}"
VARIANT=""
DRY_RUN=false

# Standalone options
ENGINE_PORT="${ENGINE_GRPC_PORT:-50051}"
GPU_DEVICE=0
MAX_BATCH=48
MAX_SESSIONS=128
FOREGROUND=false

# Triton forwarding
TRITON_ARGS=()

# engine-docker: ENGINE_IMAGE is read by lib/engine.sh (default qwen3-engine:26.02).

# ── Help ──

usage() {
    cat << 'EOF'
Usage: deploy.sh <command> [options]

Commands:
  run                    Start the TTS service
  stop                   Stop the TTS service
  status                 Show service status

  Triton-specific (forwarded to build_triton.sh):
    assemble             Assemble Triton model_repository
    pull                 Pull NGC Triton container image
    build-image          Build deploy image
    build                Build self-contained Docker image

Options:
  --gateway <mode>       Gateway: standalone | triton | engine-docker (default: standalone)
  --engine-image <tag>   Image tag for engine-docker (default: qwen3-engine:26.02)
  --variant <name>       Model variant (default: auto-discover)
  --dry-run              Show what would be done

  Standalone options:
    --port <N>           gRPC port (default: 50051)
    --device <N>         GPU device (default: 0)
    --max-batch <N>      Max batch size (default: 48)
    --max-sessions <N>   Max concurrent sessions (default: 128)
    --foreground         Run in foreground (don't daemonize)

  Triton options (forwarded to build_triton.sh):
    --engine-mode <mode> onnx | trt (default: trt)
    --image <uri>        Override NGC container image
    --container <name>   Container name
    --no-health-check    Skip health check

Examples:
  deploy.sh run                                  # standalone, auto-discover variant
  deploy.sh run --variant custom-1.7b            # standalone, specific variant
  deploy.sh run --gateway triton                 # Triton mode
  deploy.sh run --gateway engine-docker          # Engine Dockerfile + container
  deploy.sh run --foreground                     # standalone, foreground
  deploy.sh stop                                 # stop whatever is running
  deploy.sh status                               # show status for both modes
  deploy.sh assemble --engine-mode trt           # Triton: assemble model repo
EOF
}

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

# ── Argument parsing ──
COMMAND=""

if [[ $# -eq 0 ]]; then
    usage
    exit 1
fi

COMMAND="$1"
shift

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gateway)        GATEWAY_MODE="$2"; shift 2 ;;
        --variant)        VARIANT="$2"; shift 2 ;;
        --dry-run)        DRY_RUN=true; shift ;;
        --engine-image)   ENGINE_IMAGE="$2"; shift 2 ;;
        --help|-h)        usage; exit 0 ;;

        # Standalone options
        --port)           ENGINE_PORT="$2"; shift 2 ;;
        --device)         GPU_DEVICE="$2"; shift 2 ;;
        --max-batch)      MAX_BATCH="$2"; shift 2 ;;
        --max-sessions)   MAX_SESSIONS="$2"; shift 2 ;;
        --foreground)     FOREGROUND=true; shift ;;

        # Everything else is forwarded to build_triton.sh
        *)
            TRITON_ARGS+=("$1")
            # Options with arguments: consume next arg too
            case "$1" in
                --engine-mode|--image|--repo-dir|--container|--tag)
                    if [[ $# -gt 1 ]]; then
                        TRITON_ARGS+=("$2")
                        shift
                    fi
                    ;;
            esac
            shift
            ;;
    esac
done

# Validate gateway mode
case "$GATEWAY_MODE" in
    standalone|triton|engine-docker) ;;
    *)
        log_error "Unknown gateway mode: $GATEWAY_MODE (expected: standalone | triton | engine-docker)"
        exit 1
        ;;
esac

# ── Commands ──

cmd_run() {
    resolve_variant

    case "$GATEWAY_MODE" in
        standalone)
            cmd_run_standalone
            ;;
        triton)
            cmd_run_triton
            ;;
        engine-docker)
            cmd_run_engine_docker
            ;;
    esac
}

cmd_run_standalone() {
    if $DRY_RUN; then
        log_info "[DRY RUN] Would start standalone engine:"
        log_info "  Variant:    $VARIANT"
        log_info "  Port:       $ENGINE_PORT"
        log_info "  Device:     $GPU_DEVICE"
        log_info "  Max Batch:  $MAX_BATCH"
        log_info "  Max Sess:   $MAX_SESSIONS"
        log_info "  Foreground: $FOREGROUND"
        return 0
    fi

    local start_args=(
        --port "$ENGINE_PORT"
        --device "$GPU_DEVICE"
        --max-batch "$MAX_BATCH"
        --max-sessions "$MAX_SESSIONS"
    )
    if $FOREGROUND; then
        start_args+=(--foreground)
    fi

    engine_start "$REPO_ROOT" "$VARIANT" "${start_args[@]}" || exit 1

    if ! $FOREGROUND; then
        echo ""
        if engine_health_check "$ENGINE_PORT" 30; then
            echo ""
            log_step "Standalone TTS Engine Running"
            log_info "  gRPC endpoint:  localhost:${ENGINE_PORT}"
            log_info "  Variant:        $VARIANT"
            log_info "  Log file:       $(engine_log_file "$REPO_ROOT")"
            echo ""
            log_info "Stop: bash scripts/bash/deploy.sh stop"
        else
            log_warn "Engine started but port not yet reachable"
            log_info "Check logs: tail -f $(engine_log_file "$REPO_ROOT")"
        fi
    fi
}

cmd_run_triton() {
    local compose_args=(
        up
        --gateway triton
    )
    [ -n "$VARIANT" ] && compose_args+=(--variant "$VARIANT")
    $DRY_RUN && compose_args+=(--dry-run)
    compose_args+=("${TRITON_ARGS[@]}")

    bash "${SCRIPT_DIR}/compose.sh" "${compose_args[@]}"
}

cmd_run_engine_docker() {
    if ! command -v docker &>/dev/null; then
        log_error "Docker is required for --gateway engine-docker"
        exit 1
    fi

    local img="${ENGINE_IMAGE:-qwen3-engine:26.02}"

    if $DRY_RUN; then
        log_info "[DRY RUN] Would start engine Docker container (Dockerfile.engine)"
        log_info "  Variant:     $VARIANT"
        log_info "  Image:       $img"
        log_info "  Port:        $ENGINE_PORT"
        log_info "  Device:      $GPU_DEVICE"
        log_info "  Max batch:   $MAX_BATCH"
        log_info "  Max sess:    $MAX_SESSIONS"
        return 0
    fi

    local need_build=false
    if ! docker image inspect "$img" &>/dev/null; then
        need_build=true
    elif ! engine_docker_image_has_app "$img"; then
        log_warn "镜像 $img 存在但未包含 /app 下的 engine 包（常见于把 TensorRT 基础镜像误打成同名 tag）。"
        log_info "将按 Dockerfile.engine 重新构建..."
        need_build=true
    fi
    if $need_build; then
        bash "${SCRIPT_DIR}/compose.sh" build --gateway engine --image "$img" || exit 1
    fi

    local compose_args=(
        up
        --gateway engine
        --variant "$VARIANT"
        --image "$img"
        --port "$ENGINE_PORT"
        --device "$GPU_DEVICE"
        --max-batch "$MAX_BATCH"
        --max-sessions "$MAX_SESSIONS"
    )
    $DRY_RUN && compose_args+=(--dry-run)

    bash "${SCRIPT_DIR}/compose.sh" "${compose_args[@]}" || exit 1

    echo ""
    if engine_health_check "$ENGINE_PORT" 90; then
        echo ""
        log_step "Engine Docker Running"
        log_info "  gRPC endpoint:  localhost:${ENGINE_PORT}"
        log_info "  Variant:        $VARIANT"
        log_info "  Image:          $img"
        log_info "  Container:      ${ENGINE_CONTAINER_NAME:-qwen3-engine}"
        echo ""
        log_info "Logs: bash scripts/bash/compose.sh logs --gateway engine --follow"
        log_info "Stop: bash scripts/bash/deploy.sh stop"
    else
        log_warn "Container started but port not yet reachable"
        log_info "Check: bash scripts/bash/compose.sh logs --gateway engine"
    fi
}

cmd_stop() {
    local stopped=false

    # Stop standalone engine (if running)
    local engine_st
    engine_st=$(engine_status "$REPO_ROOT")
    if [ "$engine_st" != "none" ]; then
        engine_stop "$REPO_ROOT"
        stopped=true
    fi

    # Stop compose-managed Docker services
    if command -v docker &>/dev/null && docker compose version &>/dev/null 2>&1; then
        if docker compose -f "${REPO_ROOT}/compose.yaml" ps -q 2>/dev/null | grep -q .; then
            bash "${SCRIPT_DIR}/compose.sh" down --gateway all
            stopped=true
        fi
    fi

    if ! $stopped; then
        log_info "No running TTS service found"
    fi
}

cmd_status() {
    log_step "TTS Service Status"
    echo ""

    # Standalone engine status
    engine_show_status "$REPO_ROOT"
    echo ""

    if command -v docker &>/dev/null && docker compose version &>/dev/null 2>&1; then
        bash "${SCRIPT_DIR}/compose.sh" ps || true
    else
        log_info "Docker Compose: not available"
    fi
}

# ── Triton-specific commands: forward to build_triton.sh ──

cmd_forward_triton() {
    local subcmd="$1"
    local forward_args=()
    [ -n "$VARIANT" ] && forward_args+=(--variant "$VARIANT")
    $DRY_RUN && forward_args+=(--dry-run)
    forward_args+=("${TRITON_ARGS[@]}")

    bash "${SCRIPT_DIR}/build_triton.sh" "$subcmd" "${forward_args[@]}"
}

# ── Main dispatch ──

case "$COMMAND" in
    run)         cmd_run ;;
    stop)        cmd_stop ;;
    status)      cmd_status ;;
    assemble)    cmd_forward_triton "assemble" ;;
    pull)        cmd_forward_triton "pull" ;;
    build-image) cmd_forward_triton "build-image" ;;
    build)       cmd_forward_triton "build" ;;
    -h|--help)   usage ;;
    *)
        log_error "Unknown command: $COMMAND"
        usage
        exit 1
        ;;
esac

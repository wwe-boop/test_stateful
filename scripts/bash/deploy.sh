#!/bin/bash
# ===========================================================================
#  deploy.sh — Phase C: Deploy TTS service (standalone engine or Triton)
#
#  Unified entry point for deploying the TTS service. Supports two gateway
#  modes as described in docs/engine_architecture_decision.md:
#
#  A) Standalone gRPC Server (--gateway standalone)
#     Runs `python -m engine.server` directly. Zero external dependency.
#     Best for: development, single-model production, low-latency.
#
#  B) Triton Backend (--gateway triton)
#     Assembles model_repository + starts Triton container.
#     Best for: multi-model serving, K8s, enterprise infrastructure.
#
#  Usage:
#    bash scripts/bash/deploy.sh run                               # standalone (default)
#    bash scripts/bash/deploy.sh run --gateway triton               # Triton mode
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
#    GATEWAY_MODE            Override default gateway (standalone|triton)
#    ENGINE_GRPC_PORT        Standalone gRPC port (default: 50051)
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
  --gateway <mode>       Gateway mode: standalone | triton (default: standalone)
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
    standalone|triton) ;;
    *)
        log_error "Unknown gateway mode: $GATEWAY_MODE (expected: standalone | triton)"
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
    local triton_cmd_args=()
    [ -n "$VARIANT" ] && triton_cmd_args+=(--variant "$VARIANT")
    $DRY_RUN && triton_cmd_args+=(--dry-run)
    triton_cmd_args+=("${TRITON_ARGS[@]}")

    bash "${SCRIPT_DIR}/build_triton.sh" run "${triton_cmd_args[@]}"
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

    # Stop Triton container (if running)
    if command -v docker &>/dev/null; then
        local repo="${MODEL_REPO_DIR:-${REPO_ROOT}/workspace/model_repository}"
        local container_name="${CONTAINER_NAME:-qwen3-tts-triton}"
        local cname
        cname=$(triton_resolve_container_name "$repo" "$container_name" "${VARIANT:-}" 2>/dev/null || echo "$container_name")
        if docker container inspect "$cname" &>/dev/null 2>&1; then
            triton_stop "$cname"
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

    # Triton container status (reuse existing logic but don't fail if docker missing)
    if command -v docker &>/dev/null; then
        local repo="${MODEL_REPO_DIR:-${REPO_ROOT}/workspace/model_repository}"
        local container_name="${CONTAINER_NAME:-qwen3-tts-triton}"
        local cname
        cname=$(triton_resolve_container_name "$repo" "$container_name" "${VARIANT:-}" 2>/dev/null || echo "$container_name")

        if docker ps --format "{{.Names}}" 2>/dev/null | grep -Fxq "$cname"; then
            log_info "Triton container: RUNNING ($cname)"
            local http_port="${TRITON_HTTP_PORT:-8000}"
            if curl -sf "http://localhost:${http_port}/v2/health/ready" &>/dev/null; then
                log_info "  Health: READY"
            else
                log_warn "  Health: NOT READY"
            fi
        else
            log_info "Triton container: NOT RUNNING"
        fi
    else
        log_info "Triton container: Docker not available"
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

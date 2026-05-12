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
#     mounts the shared model_repository read-only. No host PyTorch required.
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
#    MODEL_REPO_DIR          Shared model_repository path (default: workspace/model_repository)
#    ENGINE_GRPC_PORT        Standalone gRPC port (default: 50051)
#    ENGINE_WEBSOCKET_PORT   Standalone WebSocket port (default: 50052)
#    RUNTIME_GPU_DEVICE      Runtime GPU device (auto | N | cuda:N; default: auto)
#    RUNTIME_MAX_BATCH_SIZE  Runtime scheduler batch limit (default: manifest profile)
#    RUNTIME_MAX_SEQ_LEN     Runtime scheduler seq limit (default: manifest profile)
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
MODEL_REPO_DIR="${MODEL_REPO_DIR:-${REPO_ROOT}/workspace/model_repository}"
GATEWAY_MODE="${GATEWAY_MODE:-standalone}"
VARIANT=""
MODEL_VERSION="${MODEL_VERSION:-${ENGINE_MODEL_VERSION:-1}}"
DRY_RUN=false

# Standalone options
ENGINE_PORT="${ENGINE_GRPC_PORT:-50051}"
ENGINE_WS_PORT="${ENGINE_WEBSOCKET_PORT:-50052}"
GPU_DEVICE="${RUNTIME_GPU_DEVICE:-auto}"
MAX_BATCH="${RUNTIME_MAX_BATCH_SIZE:-}"
MAX_SESSIONS=128
MAX_SEQ_LEN="${RUNTIME_MAX_SEQ_LEN:-}"
FOREGROUND=false

# Triton forwarding
TRITON_ARGS=()

# engine-docker: use --engine-image for an explicit image.  Without it, Phase C
# derives qwen3-engine:<tag> from the model manifest's Phase B builder_image.
ENGINE_IMAGE_EXPLICIT=false

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
  --engine-image <tag>   Image tag for engine-docker (default: Phase B NGC tag)
  --variant <name>       Model variant (default: auto-discover)
  --model-version <N>    Triton model version directory (default: 1)
  --dry-run              Show what would be done

  Standalone options:
    --port <N>           gRPC port (default: 50051)
    --ws-port <N>        WebSocket port (default: 50052)
    --device <N|auto>    Runtime GPU device (default: auto)
    --max-batch <N>      Runtime max batch size (default: manifest profile, else 128)
    --max-seq-len <N>    Runtime max sequence length (default: manifest profile, else 512)
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
  deploy.sh run --model-version 2                # use tts_orchestrator/2 package
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
        --model-version)  MODEL_VERSION="$2"; shift 2 ;;
        --dry-run)        DRY_RUN=true; shift ;;
        --engine-image)   ENGINE_IMAGE="$2"; ENGINE_IMAGE_EXPLICIT=true; shift 2 ;;
        --help|-h)        usage; exit 0 ;;

        # Standalone options
        --port)           ENGINE_PORT="$2"; shift 2 ;;
        --ws-port)        ENGINE_WS_PORT="$2"; shift 2 ;;
        --device|--runtime-device) GPU_DEVICE="$2"; shift 2 ;;
        --max-batch|--runtime-max-batch-size|--runtime-max-batch) MAX_BATCH="$2"; shift 2 ;;
        --max-seq-len|--runtime-max-seq-len|--runtime-max-seq) MAX_SEQ_LEN="$2"; shift 2 ;;
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

MODEL_VERSION=$(resolve_model_version "$MODEL_VERSION") || exit 1
export MODEL_VERSION
export ENGINE_MODEL_VERSION="$MODEL_VERSION"

# Validate gateway mode
case "$GATEWAY_MODE" in
    standalone|triton|engine-docker) ;;
    *)
        log_error "Unknown gateway mode: $GATEWAY_MODE (expected: standalone | triton | engine-docker)"
        exit 1
        ;;
esac

# ── Runtime defaults ──

_manifest_profile_value() {
    local key="$1"
    local manifest="$EXPORTED_DIR/$VARIANT/triton_manifest.json"
    if [ ! -f "$manifest" ]; then
        return 0
    fi
    python3 - "$manifest" "$key" <<'PY' 2>/dev/null || true
import json
import sys

path, key = sys.argv[1], sys.argv[2]
with open(path, encoding="utf-8") as f:
    data = json.load(f)
value = data.get("engine_profile", {}).get(key, "")
if value not in ("", None):
    print(value)
PY
}

resolve_runtime_controls() {
    local raw_device="$GPU_DEVICE"
    GPU_DEVICE=$(resolve_gpu_device_index "$GPU_DEVICE") || exit 1
    if [ "$raw_device" = "auto" ] || [ -z "$raw_device" ]; then
        log_gpu_selection "Runtime" "$GPU_DEVICE"
    fi

    if [ -z "$MAX_BATCH" ]; then
        MAX_BATCH=$(_manifest_profile_value max_batch_size)
        MAX_BATCH="${MAX_BATCH:-128}"
    fi
    if [ -z "$MAX_SEQ_LEN" ]; then
        MAX_SEQ_LEN=$(_manifest_profile_value max_seq_len)
        MAX_SEQ_LEN="${MAX_SEQ_LEN:-512}"
    fi
    if ! [[ "$MAX_BATCH" =~ ^[1-9][0-9]*$ ]]; then
        log_error "Runtime max batch must be a positive integer, got: $MAX_BATCH"
        exit 1
    fi
    if ! [[ "$MAX_SEQ_LEN" =~ ^[1-9][0-9]*$ ]]; then
        log_error "Runtime max seq len must be a positive integer, got: $MAX_SEQ_LEN"
        exit 1
    fi

    export RUNTIME_GPU_DEVICE="$GPU_DEVICE"
    export RUNTIME_MAX_BATCH_SIZE="$MAX_BATCH"
    export RUNTIME_MAX_SEQ_LEN="$MAX_SEQ_LEN"
}

# ── Commands ──

cmd_run() {
    resolve_variant
    resolve_runtime_controls

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
        log_info "  Model repo: $MODEL_REPO_DIR"
        log_info "  Package:    $MODEL_REPO_DIR/tts_orchestrator/$MODEL_VERSION"
        log_info "  Port:       $ENGINE_PORT"
        log_info "  WS Port:    $ENGINE_WS_PORT"
        log_info "  Device:     $GPU_DEVICE"
        log_info "  Max Batch:  $MAX_BATCH"
        log_info "  Max Seq:    ${MAX_SEQ_LEN:-auto}"
        log_info "  Max Sess:   $MAX_SESSIONS"
        log_info "  Foreground: $FOREGROUND"
        return 0
    fi

    MODEL_REPO_DIR="$MODEL_REPO_DIR" bash "${SCRIPT_DIR}/compose.sh" \
        prepare \
        --gateway engine \
        --variant "$VARIANT" \
        --engine-mode trt \
        --repo-dir "$MODEL_REPO_DIR" \
        --model-version "$MODEL_VERSION"

    ENGINE_MODEL_PACKAGE_DIR="$MODEL_REPO_DIR/tts_orchestrator/$MODEL_VERSION"

    local start_args=(
        --port "$ENGINE_PORT"
        --ws-port "$ENGINE_WS_PORT"
        --device "$GPU_DEVICE"
        --max-batch "$MAX_BATCH"
        --max-sessions "$MAX_SESSIONS"
    )
    if [ -n "$MAX_SEQ_LEN" ]; then
        start_args+=(--max-seq-len "$MAX_SEQ_LEN")
    fi
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
            log_info "  WebSocket:      ws://localhost:${ENGINE_WS_PORT}/v1/ws"
            log_info "  Variant:        $VARIANT"
            log_info "  Model package:  $MODEL_REPO_DIR/tts_orchestrator/$MODEL_VERSION"
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
    local has_image_override=false
    local arg
    for arg in "${TRITON_ARGS[@]}"; do
        if [ "$arg" = "--image" ]; then
            has_image_override=true
            break
        fi
    done

    local compose_args=(
        up
        --gateway triton
        --prepare
        --device "$GPU_DEVICE"
        --max-batch "$MAX_BATCH"
        --max-seq-len "$MAX_SEQ_LEN"
        --model-version "$MODEL_VERSION"
    )
    if ! $has_image_override; then
        local triton_image="${TRITON_IMAGE:-}"
        if [ -z "$triton_image" ]; then
            if [ -n "${NGC_TAG:-}" ]; then
                triton_image=$(resolve_triton_deploy_image) || exit 1
            fi
        fi
        if [ -z "$triton_image" ]; then
            local manifest_tag
            manifest_tag=$(resolve_manifest_ngc_tag "$EXPORTED_DIR/$VARIANT/triton_manifest.json" 2>/dev/null || true)
            if [ -z "$manifest_tag" ]; then
                manifest_tag=$(resolve_manifest_ngc_tag "$MODEL_REPO_DIR" "$MODEL_VERSION" 2>/dev/null || true)
            fi
            if [ -n "$manifest_tag" ]; then
                triton_image="qwen3-tts-triton:${manifest_tag}"
                log_info "Using Triton image from Phase B manifest: $triton_image"
            fi
        fi
        if [ -z "$triton_image" ]; then
            triton_image=$(resolve_triton_deploy_image) || exit 1
        fi
        compose_args+=(--image "$triton_image")
    fi
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
    local expected_release=""
    local manifest_tag=""
    if ! $ENGINE_IMAGE_EXPLICIT && [[ "$img" == qwen3-engine:* ]]; then
        if [[ "$img" =~ ^qwen3-engine:([0-9]+\.[0-9]+)$ ]]; then
            expected_release="${BASH_REMATCH[1]}"
        fi
        if [ -z "$expected_release" ] || [ "$expected_release" = "26.02" ]; then
            manifest_tag="${NGC_TAG:-}"
            local tag_source="manifest"
            if [ -n "$manifest_tag" ]; then
                tag_source="ngc_tag"
            fi
            if [ -z "$manifest_tag" ]; then
                manifest_tag=$(resolve_manifest_ngc_tag "$EXPORTED_DIR/$VARIANT/triton_manifest.json" 2>/dev/null || true)
            fi
            if [ -z "$manifest_tag" ]; then
                manifest_tag=$(resolve_manifest_ngc_tag "$MODEL_REPO_DIR" "$MODEL_VERSION" 2>/dev/null || true)
            fi
            if [ -z "$manifest_tag" ]; then
                tag_source="driver"
                manifest_tag=$(resolve_ngc_tag 2>/dev/null || true)
            fi
            if [ -n "$manifest_tag" ]; then
                img="qwen3-engine:${manifest_tag}"
                expected_release="$manifest_tag"
                if [ "$tag_source" = "manifest" ]; then
                    log_info "Using engine Docker image from Phase B manifest: $img"
                elif [ "$tag_source" = "ngc_tag" ]; then
                    log_info "Using engine Docker image from NGC_TAG: $img"
                else
                    log_info "Using engine Docker image from driver-compatible NGC tag: $img"
                fi
            fi
        fi
    fi

    if [ -n "$expected_release" ] && [ -z "${ENGINE_BASE_IMAGE:-}" ]; then
        export ENGINE_BASE_IMAGE="nvcr.io/nvidia/tensorrt:${expected_release}-py3"
    fi
    if [ -n "$expected_release" ] && [ -z "${ENGINE_PYTORCH_CUDA_TAG:-${PYTORCH_CUDA_TAG:-}}" ]; then
        local torch_cuda_tag
        torch_cuda_tag=$(resolve_ngc_torch_index_tag "$expected_release" 2>/dev/null || true)
        if [ -n "$torch_cuda_tag" ]; then
            export ENGINE_PYTORCH_CUDA_TAG="$torch_cuda_tag"
        fi
    fi

    if $DRY_RUN; then
        log_info "[DRY RUN] Would start engine Docker container (Dockerfile.engine)"
        log_info "  Variant:     $VARIANT"
        log_info "  Image:       $img"
        log_info "  Port:        $ENGINE_PORT"
        log_info "  WS Port:     $ENGINE_WS_PORT"
        log_info "  Device:      $GPU_DEVICE"
        log_info "  Max batch:   $MAX_BATCH"
        log_info "  Max seq len: ${MAX_SEQ_LEN:-auto}"
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
    elif [ -n "$expected_release" ] && ! engine_docker_image_matches_release "$img" "$expected_release"; then
        local actual_release
        actual_release=$(engine_docker_image_tensorrt_release "$img" || true)
        log_warn "镜像 $img 的 TensorRT 版本是 ${actual_release:-unknown}，但 Phase B manifest 对应 $expected_release。"
        log_info "将按 Dockerfile.engine 使用 ENGINE_BASE_IMAGE=$ENGINE_BASE_IMAGE 重新构建..."
        need_build=true
    fi
    if $need_build; then
        bash "${SCRIPT_DIR}/compose.sh" build --gateway engine --image "$img" || exit 1
    fi

    local compose_args=(
        up
        --gateway engine
        --prepare
        --variant "$VARIANT"
        --image "$img"
        --port "$ENGINE_PORT"
        --ws-port "$ENGINE_WS_PORT"
        --device "$GPU_DEVICE"
        --max-batch "$MAX_BATCH"
        --max-sessions "$MAX_SESSIONS"
        --model-version "$MODEL_VERSION"
    )
    if [ -n "$MAX_SEQ_LEN" ]; then
        compose_args+=(--max-seq-len "$MAX_SEQ_LEN")
    fi
    $DRY_RUN && compose_args+=(--dry-run)

    bash "${SCRIPT_DIR}/compose.sh" "${compose_args[@]}" || exit 1

    echo ""
    log_step "Engine Docker Running"
    log_info "  gRPC endpoint:  localhost:${ENGINE_PORT}"
    log_info "  WebSocket:      ws://localhost:${ENGINE_WS_PORT}/v1/ws"
    log_info "  Variant:        $VARIANT"
    log_info "  Image:          $img"
    log_info "  Container:      ${ENGINE_CONTAINER_NAME:-qwen3-engine}"
    echo ""
    log_info "Logs: bash scripts/bash/compose.sh logs --gateway engine --follow"
    log_info "Stop: bash scripts/bash/deploy.sh stop"
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
    forward_args+=(--model-version "$MODEL_VERSION")
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

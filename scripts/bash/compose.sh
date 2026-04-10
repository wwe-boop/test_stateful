#!/bin/bash
# ===========================================================================
#  compose.sh — Unified Docker Compose workflow for engine / Triton deploy
#
#  Wraps compose.yaml with project-specific defaults:
#    - engine: standalone engine.server in Docker
#    - triton: Triton Inference Server with mounted model_repository
# ===========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
source "${SCRIPT_DIR}/tools.sh"

COMPOSE_FILE="${REPO_ROOT}/compose.yaml"
COMPOSE_DEV_FILE="${REPO_ROOT}/compose.dev.yaml"
EXPORTED_DIR="${REPO_ROOT}/workspace/exported"
ENGINE_MODELS_DIR="${ENGINE_MODELS_DIR:-${REPO_ROOT}/workspace/models}"
ENGINE_EXPORTED_DIR="${ENGINE_EXPORTED_DIR:-${REPO_ROOT}/workspace/exported}"
MODEL_REPO_DIR="${MODEL_REPO_DIR:-${REPO_ROOT}/workspace/model_repository}"

COMMAND=""
GATEWAY="all"
VARIANT=""
ENGINE_MODE="${ENGINE_MODE:-trt}"
DRY_RUN=false
NO_HEALTH_CHECK=false
FORCE_PREPARE=false
FOLLOW=false
BUILD_BEFORE_UP=false
USE_DEV_OVERLAY=false
WATCH_NO_UP=false
WATCH_QUIET=false
WATCH_NO_PRUNE=false
IMAGE_OVERRIDE=""
CONTAINER_OVERRIDE=""

ENGINE_PORT="${ENGINE_GRPC_PORT:-50051}"
ENGINE_HEALTH="${ENGINE_HEALTH_PORT:-8080}"
ENGINE_DEVICE="${ENGINE_DEVICE:-0}"
ENGINE_MAX_BATCH="${ENGINE_MAX_BATCH_SIZE:-128}"
ENGINE_MAX_SESSIONS="${ENGINE_MAX_SESSIONS:-128}"
ENGINE_MAX_SEQ_LEN="${ENGINE_MAX_SEQ_LEN:-}"

TRITON_HTTP="${TRITON_HTTP_PORT:-8000}"
TRITON_GRPC="${TRITON_GRPC_PORT:-8001}"
TRITON_METRICS="${TRITON_METRICS_PORT:-8002}"

usage() {
    cat <<'EOF'
Usage: compose.sh <command> [options]

Commands:
  build                  Build compose image(s)
  prepare                Assemble Triton model_repository
  up                     Start compose service(s)
  watch                  Watch files and auto refresh/rebuild service(s)
  down                   Stop/remove compose service(s)
  logs                   Show service logs
  ps                     Show compose status
  config                 Render resolved compose config

Options:
  --gateway <mode>       engine | triton | all (default: all)
  --variant <name>       Model variant (auto-discover when possible)
  --engine-mode <mode>   trt | onnx for Triton repo assembly (default: trt)
  --repo-dir <path>      Triton model repository path
  --image <tag>          Override service image tag for selected gateway
  --container <name>     Override container name for selected gateway
  --port <N>             Engine gRPC port
  --health-port <N>      Engine health port
  --grpc-port <N>        Triton gRPC port
  --http-port <N>        Triton HTTP port
  --metrics-port <N>     Triton metrics port
  --device <N>           Engine CUDA device
  --max-batch <N>        Engine max batch size
  --max-sessions <N>     Engine max sessions
  --max-seq-len <N>      Optional engine scheduler max seq len override
  --build                Build before `up`
  --dev                  Enable compose.dev.yaml source bind mounts
  --prepare              Force Triton repo assembly before `up`
  --no-up                For `watch`, do not start services before watching
  --quiet                For `watch`, hide build output
  --no-prune             For `watch`, keep dangling images after rebuild
  --no-health-check      Skip readiness wait after `up`
  --follow               Follow logs (for `logs`)
  --dry-run              Print the compose command instead of executing
  -h, --help             Show this help
EOF
}

discover_first_variant() {
    local vdir
    for vdir in "$EXPORTED_DIR"/*/; do
        local vname
        vname=$(basename "$vdir")
        [[ "$vname" == "tokenizer" ]] && continue
        if [[ -d "$vdir/weights" ]]; then
            printf '%s\n' "$vname"
            return 0
        fi
    done
    return 1
}

resolve_variant_if_needed() {
    if [[ -n "$VARIANT" ]]; then
        if [[ ! -d "$EXPORTED_DIR/$VARIANT" ]]; then
            log_error "Variant not found: $EXPORTED_DIR/$VARIANT"
            exit 1
        fi
        return 0
    fi
    VARIANT=$(discover_first_variant) || {
        log_error "No exported variants found in $EXPORTED_DIR"
        exit 1
    }
    log_info "Auto-discovered variant: $VARIANT"
}

require_docker_compose() {
    if ! command -v docker &>/dev/null; then
        log_error "Docker is required"
        exit 1
    fi
    if ! docker compose version &>/dev/null; then
        log_error "docker compose is required"
        exit 1
    fi
}

compose_cmd() {
    local profile_args=()
    case "$GATEWAY" in
        engine) profile_args=(--profile engine) ;;
        triton) profile_args=(--profile triton) ;;
        all) profile_args=(--profile engine --profile triton) ;;
        *)
            log_error "Unknown gateway: $GATEWAY"
            exit 1
            ;;
    esac

    local compose_files=(-f "$COMPOSE_FILE")
    if $USE_DEV_OVERLAY && [[ -f "$COMPOSE_DEV_FILE" ]]; then
        compose_files+=(-f "$COMPOSE_DEV_FILE")
    fi

    local cmd=(docker compose "${compose_files[@]}" "${profile_args[@]}")
    cmd+=("$@")

    if $DRY_RUN; then
        printf '[DRY RUN] '
        printf '%q ' "${cmd[@]}"
        printf '\n'
        return 0
    fi

    "${cmd[@]}"
}

export_compose_env() {
    export COMPOSE_PROJECT_NAME="${COMPOSE_PROJECT_NAME:-qwen3-tts}"
    export MODEL_VARIANT="$VARIANT"
    export ENGINE_MODELS_DIR="$ENGINE_MODELS_DIR"
    export ENGINE_EXPORTED_DIR="$ENGINE_EXPORTED_DIR"
    export TRITON_MODEL_REPO_DIR="$MODEL_REPO_DIR"

    export ENGINE_GRPC_PORT="$ENGINE_PORT"
    export ENGINE_HEALTH_PORT="$ENGINE_HEALTH"
    export ENGINE_DEVICE="$ENGINE_DEVICE"
    export ENGINE_MAX_BATCH_SIZE="$ENGINE_MAX_BATCH"
    export ENGINE_MAX_SESSIONS="$ENGINE_MAX_SESSIONS"
    export ENGINE_MAX_SEQ_LEN="$ENGINE_MAX_SEQ_LEN"

    export TRITON_HTTP_PORT="$TRITON_HTTP"
    export TRITON_GRPC_PORT="$TRITON_GRPC"
    export TRITON_METRICS_PORT="$TRITON_METRICS"
}

prepare_triton_repo() {
    resolve_variant_if_needed
    assemble_model_repo "$EXPORTED_DIR" "$VARIANT" "$MODEL_REPO_DIR" "$ENGINE_MODE"
    validate_model_repo "$MODEL_REPO_DIR"
}

ensure_triton_repo() {
    if $FORCE_PREPARE; then
        prepare_triton_repo
        return 0
    fi
    if [[ ! -d "$MODEL_REPO_DIR" ]] || [[ -z "$(ls -A "$MODEL_REPO_DIR" 2>/dev/null)" ]]; then
        log_info "Model repository missing, assembling it first"
        prepare_triton_repo
    fi
}

cmd_build() {
    require_docker_compose
    export_compose_env
    case "$GATEWAY" in
        engine) compose_cmd build engine ;;
        triton) compose_cmd build triton ;;
        all) compose_cmd build engine triton ;;
    esac
}

cmd_prepare() {
    if [[ "$GATEWAY" != "triton" && "$GATEWAY" != "all" ]]; then
        log_error "`prepare` is only valid for Triton"
        exit 1
    fi
    prepare_triton_repo
}

cmd_up() {
    require_docker_compose

    if $DRY_RUN; then
        case "$GATEWAY" in
            engine)
                resolve_variant_if_needed
                export_compose_env
                if $BUILD_BEFORE_UP; then
                    compose_cmd up --build -d engine
                else
                    compose_cmd up -d engine
                fi
                ;;
            triton)
                export_compose_env
                if $BUILD_BEFORE_UP; then
                    compose_cmd up --build -d triton
                else
                    compose_cmd up -d triton
                fi
                ;;
            all)
                resolve_variant_if_needed
                export_compose_env
                if $BUILD_BEFORE_UP; then
                    compose_cmd up --build -d engine triton
                else
                    compose_cmd up -d engine triton
                fi
                ;;
        esac
        return 0
    fi

    export_compose_env

    case "$GATEWAY" in
        engine)
            resolve_variant_if_needed
            export_compose_env
            if $BUILD_BEFORE_UP; then
                compose_cmd up --build -d engine
            else
                compose_cmd up -d engine
            fi
            if ! $NO_HEALTH_CHECK; then
                engine_health_check "$ENGINE_PORT" 90
            fi
            ;;
        triton)
            ensure_triton_repo
            export_compose_env
            if $BUILD_BEFORE_UP; then
                compose_cmd up --build -d triton
            else
                compose_cmd up -d triton
            fi
            if ! $NO_HEALTH_CHECK; then
                triton_health_check "localhost" "$TRITON_HTTP" 120
            fi
            ;;
        all)
            ensure_triton_repo
            resolve_variant_if_needed
            export_compose_env
            if $BUILD_BEFORE_UP; then
                compose_cmd up --build -d engine triton
            else
                compose_cmd up -d engine triton
            fi
            if ! $NO_HEALTH_CHECK; then
                engine_health_check "$ENGINE_PORT" 90
                triton_health_check "localhost" "$TRITON_HTTP" 120
            fi
            ;;
    esac
}

cmd_watch() {
    require_docker_compose

    if $USE_DEV_OVERLAY; then
        log_error "`watch` cannot be combined with --dev; use either bind mounts or compose watch"
        exit 1
    fi

    case "$GATEWAY" in
        engine)
            resolve_variant_if_needed
            ;;
        triton)
            ensure_triton_repo
            ;;
        all)
            resolve_variant_if_needed
            ensure_triton_repo
            ;;
    esac

    export_compose_env

    local args=(watch)
    $WATCH_NO_UP && args+=(--no-up)
    $WATCH_QUIET && args+=(--quiet)
    $WATCH_NO_PRUNE && args+=(--prune=false)

    case "$GATEWAY" in
        engine) args+=(engine) ;;
        triton) args+=(triton) ;;
        all) args+=(engine triton) ;;
    esac

    compose_cmd "${args[@]}"
}

cmd_down() {
    require_docker_compose
    export_compose_env
    case "$GATEWAY" in
        engine)
            compose_cmd stop engine
            compose_cmd rm -sf engine
            ;;
        triton)
            compose_cmd stop triton
            compose_cmd rm -sf triton
            ;;
        all)
            compose_cmd down --remove-orphans
            ;;
    esac
}

cmd_logs() {
    require_docker_compose
    export_compose_env
    local args=(logs)
    $FOLLOW && args+=(-f)
    case "$GATEWAY" in
        engine) args+=(engine) ;;
        triton) args+=(triton) ;;
        all) ;;
    esac
    compose_cmd "${args[@]}"
}

cmd_ps() {
    require_docker_compose
    export_compose_env
    compose_cmd ps
}

cmd_config() {
    require_docker_compose
    export_compose_env
    compose_cmd config
}

if [[ $# -eq 0 ]]; then
    usage
    exit 1
fi

COMMAND="$1"
shift

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gateway) GATEWAY="$2"; shift 2 ;;
        --variant) VARIANT="$2"; shift 2 ;;
        --engine-mode) ENGINE_MODE="$2"; shift 2 ;;
        --repo-dir) MODEL_REPO_DIR="$2"; shift 2 ;;
        --image) IMAGE_OVERRIDE="$2"; shift 2 ;;
        --container) CONTAINER_OVERRIDE="$2"; shift 2 ;;
        --port) ENGINE_PORT="$2"; shift 2 ;;
        --health-port) ENGINE_HEALTH="$2"; shift 2 ;;
        --grpc-port) TRITON_GRPC="$2"; shift 2 ;;
        --http-port) TRITON_HTTP="$2"; shift 2 ;;
        --metrics-port) TRITON_METRICS="$2"; shift 2 ;;
        --device) ENGINE_DEVICE="$2"; shift 2 ;;
        --max-batch) ENGINE_MAX_BATCH="$2"; shift 2 ;;
        --max-sessions) ENGINE_MAX_SESSIONS="$2"; shift 2 ;;
        --max-seq-len) ENGINE_MAX_SEQ_LEN="$2"; shift 2 ;;
        --build) BUILD_BEFORE_UP=true; shift ;;
        --dev) USE_DEV_OVERLAY=true; shift ;;
        --prepare) FORCE_PREPARE=true; shift ;;
        --no-up) WATCH_NO_UP=true; shift ;;
        --quiet) WATCH_QUIET=true; shift ;;
        --no-prune) WATCH_NO_PRUNE=true; shift ;;
        --no-health-check) NO_HEALTH_CHECK=true; shift ;;
        --follow) FOLLOW=true; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1
        ;;
    esac
done

if [[ -n "$IMAGE_OVERRIDE" ]]; then
    case "$GATEWAY" in
        engine) export ENGINE_IMAGE="$IMAGE_OVERRIDE" ;;
        triton) export TRITON_IMAGE="$IMAGE_OVERRIDE" ;;
        *)
            log_error "--image requires --gateway engine or --gateway triton"
            exit 1
            ;;
    esac
fi

if [[ -n "$CONTAINER_OVERRIDE" ]]; then
    case "$GATEWAY" in
        engine) export ENGINE_CONTAINER_NAME="$CONTAINER_OVERRIDE" ;;
        triton) export TRITON_CONTAINER_NAME="$CONTAINER_OVERRIDE" ;;
        *)
            log_error "--container requires --gateway engine or --gateway triton"
            exit 1
            ;;
    esac
fi

case "$COMMAND" in
    build) cmd_build ;;
    prepare) cmd_prepare ;;
    up) cmd_up ;;
    watch) cmd_watch ;;
    down) cmd_down ;;
    logs) cmd_logs ;;
    ps) cmd_ps ;;
    config) cmd_config ;;
    *)
        log_error "Unknown command: $COMMAND"
        usage
        exit 1
        ;;
esac

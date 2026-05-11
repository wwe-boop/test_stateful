#!/bin/bash
# ===========================================================================
#  autorun.sh — Intelligent launcher for Qwen3-TTS Triton pipeline
#
#  Smart entry point that orchestrates all three build phases:
#    Phase A (setup):  Environment + model export    → setup_env.sh
#    Phase B (build):  TensorRT engine compilation    → build_engines.sh
#    Phase C (deploy): TTS service deployment          → deploy.sh
#
#  Usage:
#    bash scripts/bash/autorun.sh                    # interactive mode
#    bash scripts/bash/autorun.sh base-1.7b          # full pipeline for variant
#    bash scripts/bash/autorun.sh all [options]       # full pipeline (explicit)
#    bash scripts/bash/autorun.sh setup [options]     # Phase A only
#    bash scripts/bash/autorun.sh build [options]     # Phase B only
#    bash scripts/bash/autorun.sh deploy [options]    # Phase C only
#    bash scripts/bash/autorun.sh status              # show pipeline status
#    bash scripts/bash/autorun.sh stop                # stop TTS service
#    bash scripts/bash/autorun.sh update-matrix       # update NGC compat matrix
#
#  Options:
#    --variant, -m <name>    Model variant (base-1.7b, custom-1.7b, ...)
#    --model-version <N>     Triton model version directory (default: 1)
#    --target-driver <ver>   Target NVIDIA driver for NGC container selection
#    --yes, -y               Skip confirmations (non-interactive)
#    --dry-run               Show what would be done
#    -h, --help              Show full help
#
#    Phase A (forwarded to setup_env.sh):
#      --python <ver>        Python version (default: from NGC matrix)
#      --env-name <name>     Virtual env name (default: qwen3-tts)
#      --source <src>        Model source (auto|hf|modelscope)
#      --skip-models         Skip model download
#      --skip-deps           Skip dependency installation
#      --skip-export         Skip model export
#
#    Phase B (forwarded to build_engines.sh):
#      --max-batch-size <N>  TRT max batch (default: suggested by build GPU memory)
#      --max-input-len <N>   TRT max prefill/input token length
#      --max-seq-len <N>     TRT max KV sequence length
#      --image <uri>         Override NGC container image
#      --dtype <type>        Engine precision: bf16|fp16|fp32 (default: bf16)
#      --build-device <dev>  GPU for TensorRT build (auto|all|N|cuda:N)
#
#    Phase C (forwarded to deploy.sh):
#      --gateway <mode>      Gateway: standalone | triton | engine-docker
#      --engine-mode <mode>  Triton assemble mode: trt | onnx
#      --runtime-max-batch-size <N> Runtime scheduler max batch
#      --runtime-max-seq-len <N> Runtime scheduler max sequence length
#      --runtime-device <N>  GPU for serving runtime (auto|N|cuda:N)
#      --engine-image <tag>  engine-docker image tag (optional)
#      --port <port>         Standalone gRPC port (default: 50051)
#      --grpc-port <port>    Triton gRPC port (default: 8001)
#      --http-port <port>    Triton HTTP port (default: 8000)
# ===========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
source "${SCRIPT_DIR}/tools.sh"

# ── Known model variant names (for auto-detection of positional args) ──
_KNOWN_VARIANTS="base-1.7b custom-1.7b design-1.7b base-0.6b custom-0.6b all-1.7b all"
_KNOWN_COMMANDS="all setup build deploy status stop update-matrix help"

_is_variant() { [[ " $_KNOWN_VARIANTS " == *" $1 "* ]]; }
_is_command() { [[ " $_KNOWN_COMMANDS " == *" $1 "* ]]; }
_is_enabled() { [[ "${1:-}" == "true" || "${1:-}" == "1" ]]; }
_env_or_empty() { printenv "$1" 2>/dev/null || true; }

# ── Defaults ──
COMMAND=""
VARIANT="${MODEL_VARIANT:-}"
MODEL_VERSION="${MODEL_VERSION:-${ENGINE_MODEL_VERSION:-1}}"
DRY_RUN=false
YES_MODE=false
TARGET_DRIVER="${TARGET_DRIVER:-}"
GLOBAL_DEVICE="${QWEN3_TTS_GPU_DEVICE:-${GPU_DEVICE:-}}"
EXPORT_DEVICE="${EXPORT_DEVICE:-}"
BUILD_GPU_DEVICE="${BUILD_GPU_DEVICE:-}"
RUNTIME_GPU_DEVICE="${RUNTIME_GPU_DEVICE:-}"

# Phase A forwarding
SETUP_ARGS=()
PYTHON_VERSION="${PYTHON_VERSION:-}"
ENV_NAME="${ENV_NAME:-}"
MODEL_SOURCE="${MODEL_SOURCE:-}"
SKIP_MODELS=false
SKIP_DEPS=false
SKIP_EXPORT=false

# Phase B forwarding
BUILD_ARGS=()
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-}"
MAX_INPUT_LEN="${MAX_INPUT_LEN:-}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-}"
BUILD_IMAGE="${NGC_IMAGE:-}"
ENGINE_DTYPE="${ENGINE_DTYPE:-}"
TRITON_IO_FLOAT_DTYPE="${TRITON_IO_FLOAT_DTYPE:-}"

# Phase C forwarding
DEPLOY_ARGS=()
GATEWAY_MODE="${GATEWAY_MODE:-}"
ENGINE_MODE="${ENGINE_MODE:-}"
ENGINE_PORT="$(_env_or_empty ENGINE_GRPC_PORT)"
ENGINE_DOCKER_IMAGE="$(_env_or_empty ENGINE_IMAGE)"
GRPC_PORT="$(_env_or_empty TRITON_GRPC_PORT)"
HTTP_PORT="$(_env_or_empty TRITON_HTTP_PORT)"
RUNTIME_MAX_BATCH_SIZE="${RUNTIME_MAX_BATCH_SIZE:-}"
RUNTIME_MAX_SEQ_LEN="${RUNTIME_MAX_SEQ_LEN:-}"

# ── Help ──

usage() {
    cat << 'EOF'
Usage: autorun.sh [command] [model_variant] [options]

Commands:
  all               Full pipeline: setup → build → deploy (default)
  setup             Phase A only (environment + model export)
  build             Phase B only (TensorRT engine compilation)
  deploy            Phase C only (TTS service deployment)
  status            Show pipeline status
  stop              Stop TTS service
  update-matrix     Update NGC compatibility matrix from NVIDIA website
  help              Show this help

If no command is given, launches interactive mode (Phase C: standalone | triton | engine-docker).
If a model variant name is given without a command, runs the full pipeline.

Options:
  --variant, -m <name>    Model variant (base-1.7b, custom-1.7b, design-1.7b,
                          base-0.6b, custom-0.6b, all-1.7b, all)
  --model-version <N>     Triton model version directory (default: 1)
  --target-driver <ver>   Target NVIDIA driver version for NGC container
                          selection (e.g. 575.57 for production machines).
                          Overrides local driver detection.
  --device <dev>          Use this GPU for export, build, and runtime
                          (auto | N | cuda:N; default: auto).
  --export-device <dev>   Override Phase A export device only (auto | cpu | N | cuda:N).
  --build-device <dev>    Override Phase B TensorRT build GPU only (auto | all | N | cuda:N).
  --runtime-device <dev>  Override Phase C serving GPU only (auto | N | cuda:N).
  --yes, -y               Skip confirmations (non-interactive)
  --dry-run               Show what would be done without executing
  -h, --help              Show this help

Phase A options (forwarded to setup_env.sh):
  --python <version>      Python version (default: from NGC matrix)
  --env-name <name>       Virtual env name (default: qwen3-tts)
  --source <source>       Model download source (auto|hf|modelscope)
  --skip-models           Skip model download
  --skip-deps             Skip dependency installation
  --skip-export           Skip model export

Phase B options (forwarded to build_engines.sh):
  --max-batch-size <N>    TensorRT profile max batch (default: suggested by build GPU memory)
  --max-input-len <N>     TensorRT profile max input/prefill length
  --max-seq-len <N>       TensorRT profile max sequence/KV length
  --image <uri>           Override NGC container image
  --dtype <type>          Engine precision: bf16|fp16|fp32|fp8 (default: bf16)
  --triton-io-float-dtype <type>
                          Float I/O dtype for generated TRT/Triton configs
                          Use fp32 if Triton reports dtype mismatch (e.g. TYPE_FP32 vs TYPE_BF16).

Phase C options (forwarded to deploy.sh):
  --gateway <mode>        Gateway: standalone | triton | engine-docker (default: standalone)
  --engine-mode <mode>    Triton assemble mode: trt | onnx (default: trt)
  --runtime-max-batch-size <N>
                          Runtime scheduler max batch size
  --runtime-max-seq-len <N>
                          Runtime scheduler max sequence length
  --engine-image <tag>    Image for engine-docker (default: qwen3-engine:26.02)
  --port <port>           Standalone / engine-docker gRPC port (default: 50051)
  --grpc-port <port>      Triton gRPC port (default: 8001)
  --http-port <port>      Triton HTTP port (default: 8000)

Examples:
  autorun.sh                          # interactive guided setup
  autorun.sh base-1.7b                # full pipeline for base-1.7b
  autorun.sh all -m custom-1.7b       # full pipeline for custom-1.7b
  autorun.sh all -m custom-1.7b --device auto
  autorun.sh setup --skip-export      # Phase A without export
  autorun.sh custom-1.7b --dtype fp32 # Full pipeline, FP32 engines (fixes dtype mismatch)
  autorun.sh build -m custom-1.7b --dtype fp32    # Phase B with fp32
  autorun.sh build -m custom-1.7b --dtype fp16    # Phase B with fp16
  autorun.sh build -m custom-1.7b --max-batch-size 64 --max-input-len 128 --max-seq-len 512
  autorun.sh build -m custom-1.7b --build-device 1
  autorun.sh build --target-driver 575.57   # build for production driver
  autorun.sh deploy                   # Phase C (standalone engine)
  autorun.sh deploy --gateway triton         # Phase C (Triton)
  autorun.sh deploy --gateway engine-docker  # Phase C (engine container image)
  autorun.sh deploy --runtime-max-batch-size 32 --runtime-max-seq-len 512
  autorun.sh status                   # show pipeline status
  autorun.sh update-matrix            # fetch latest NGC compat data
EOF
}

# ── Argument parsing ──
# Handles: autorun.sh [command] [variant] [--options...]
# Smart detection: first positional arg can be a command or variant name.

parse_args() {
    # First pass: detect command and variant from positional args
    local positionals=()

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --variant|-m)       VARIANT="$2"; shift 2 ;;
            --model-version)    MODEL_VERSION="$2"; shift 2 ;;
            --yes|-y)           YES_MODE=true; shift ;;
            --dry-run)          DRY_RUN=true; shift ;;
            --target-driver)    TARGET_DRIVER="$2"; shift 2 ;;
            --device)           GLOBAL_DEVICE="$2"; shift 2 ;;
            --export-device)    EXPORT_DEVICE="$2"; shift 2 ;;
            --build-device)     BUILD_GPU_DEVICE="$2"; shift 2 ;;
            --runtime-device)   RUNTIME_GPU_DEVICE="$2"; shift 2 ;;
            --help|-h)          usage; exit 0 ;;

            # Phase A
            --python)           PYTHON_VERSION="$2"; shift 2 ;;
            --env-name)         ENV_NAME="$2"; shift 2 ;;
            --source)           MODEL_SOURCE="$2"; shift 2 ;;
            --skip-models)      SKIP_MODELS=true; shift ;;
            --skip-deps)        SKIP_DEPS=true; shift ;;
            --skip-export)      SKIP_EXPORT=true; shift ;;

            # Phase B
            --max-batch-size)   MAX_BATCH_SIZE="$2"; shift 2 ;;
            --max-input-len)    MAX_INPUT_LEN="$2"; shift 2 ;;
            --max-seq-len)      MAX_SEQ_LEN="$2"; shift 2 ;;
            --image)            BUILD_IMAGE="$2"; shift 2 ;;
            --dtype)            ENGINE_DTYPE="$2"; shift 2 ;;
            --triton-io-float-dtype) TRITON_IO_FLOAT_DTYPE="$2"; shift 2 ;;

            # Phase C
            --gateway)          GATEWAY_MODE="$2"; shift 2 ;;
            --engine-mode)      ENGINE_MODE="$2"; shift 2 ;;
            --runtime-max-batch-size|--runtime-max-batch) RUNTIME_MAX_BATCH_SIZE="$2"; shift 2 ;;
            --runtime-max-seq-len|--runtime-max-seq) RUNTIME_MAX_SEQ_LEN="$2"; shift 2 ;;
            --engine-image)      ENGINE_DOCKER_IMAGE="$2"; shift 2 ;;
            --port)             ENGINE_PORT="$2"; shift 2 ;;
            --grpc-port)        GRPC_PORT="$2"; shift 2 ;;
            --http-port)        HTTP_PORT="$2"; shift 2 ;;

            -*)
                log_error "Unknown option: $1"
                usage
                exit 1
                ;;
            *)
                positionals+=("$1")
                shift
                ;;
        esac
    done

    # Resolve positional arguments
    for pos in "${positionals[@]}"; do
        if [ -z "$COMMAND" ] && _is_command "$pos"; then
            COMMAND="$pos"
        elif [ -z "$VARIANT" ] && _is_variant "$pos"; then
            VARIANT="$pos"
            # Bare variant without command implies "all"
            [ -z "$COMMAND" ] && COMMAND="all"
        else
            log_error "Unexpected argument: $pos"
            usage
            exit 1
        fi
    done
}

# ── Build forwarding arg arrays ──

build_forward_args() {
    # Target driver override (propagated via env var to all phases)
    if [ -n "$TARGET_DRIVER" ]; then
        export TARGET_DRIVER
    fi
    MODEL_VERSION=$(resolve_model_version "$MODEL_VERSION") || exit 1

    # Phase A args
    SETUP_ARGS=()
    if [ -n "$VARIANT" ]; then
        # Pass variant via env var (setup_env.sh reads MODEL_VARIANT)
        export MODEL_VARIANT="$VARIANT"
    fi
    export MODEL_VERSION
    export ENGINE_MODEL_VERSION="$MODEL_VERSION"
    if [ -n "$PYTHON_VERSION" ]; then
        export PYTHON_VERSION
    fi
    if [ -n "$MODEL_SOURCE" ]; then
        export MODEL_SOURCE="$MODEL_SOURCE"
    fi
    local effective_export_device="${EXPORT_DEVICE:-$GLOBAL_DEVICE}"
    local effective_build_device="${BUILD_GPU_DEVICE:-$GLOBAL_DEVICE}"
    local effective_runtime_device="${RUNTIME_GPU_DEVICE:-$GLOBAL_DEVICE}"

    if [ -n "$effective_export_device" ]; then
        export EXPORT_DEVICE="$effective_export_device"
    fi
    if $SKIP_MODELS; then export SKIP_MODELS=1; fi
    if $SKIP_DEPS; then export SKIP_DEPS=1; fi
    if $SKIP_EXPORT; then export SKIP_EXPORT=1; fi
    if [ -n "$ENV_NAME" ]; then export ENV_NAME="$ENV_NAME"; fi

    # Phase B args — "all"/"all-1.7b" are meta-variants; omit --variant so
    # build_engines.sh discovers all exported checkpoints automatically.
    BUILD_ARGS=()
    if [ -n "$VARIANT" ] && [[ "$VARIANT" != all* ]]; then
        BUILD_ARGS+=(--variant "$VARIANT")
    fi
    if [ -n "$TARGET_DRIVER" ]; then BUILD_ARGS+=(--target-driver "$TARGET_DRIVER"); fi
    if [ -n "$effective_build_device" ]; then BUILD_ARGS+=(--device "$effective_build_device"); fi
    if [ -n "$MAX_BATCH_SIZE" ]; then BUILD_ARGS+=(--max-batch-size "$MAX_BATCH_SIZE"); fi
    if [ -n "$MAX_INPUT_LEN" ]; then BUILD_ARGS+=(--max-input-len "$MAX_INPUT_LEN"); fi
    if [ -n "$MAX_SEQ_LEN" ]; then BUILD_ARGS+=(--max-seq-len "$MAX_SEQ_LEN"); fi
    if [ -n "$BUILD_IMAGE" ]; then BUILD_ARGS+=(--image "$BUILD_IMAGE"); fi
    if [ -n "$ENGINE_DTYPE" ]; then BUILD_ARGS+=(--dtype "$ENGINE_DTYPE"); fi
    if [ -n "$TRITON_IO_FLOAT_DTYPE" ]; then BUILD_ARGS+=(--triton-io-float-dtype "$TRITON_IO_FLOAT_DTYPE"); fi
    if $DRY_RUN; then BUILD_ARGS+=(--dry-run); fi

    # Phase C args (forwarded to deploy.sh)
    DEPLOY_ARGS=()
    if [ -n "$VARIANT" ] && [[ "$VARIANT" != all* ]]; then
        DEPLOY_ARGS+=(--variant "$VARIANT")
    fi
    if [ -n "$GATEWAY_MODE" ]; then DEPLOY_ARGS+=(--gateway "$GATEWAY_MODE"); fi
    if [ -n "$ENGINE_MODE" ]; then DEPLOY_ARGS+=(--engine-mode "$ENGINE_MODE"); fi
    if [ -n "$effective_runtime_device" ]; then DEPLOY_ARGS+=(--device "$effective_runtime_device"); fi
    if [ -n "$RUNTIME_MAX_BATCH_SIZE" ]; then DEPLOY_ARGS+=(--max-batch "$RUNTIME_MAX_BATCH_SIZE"); fi
    if [ -n "$RUNTIME_MAX_SEQ_LEN" ]; then DEPLOY_ARGS+=(--max-seq-len "$RUNTIME_MAX_SEQ_LEN"); fi
    if [ -n "$ENGINE_DOCKER_IMAGE" ]; then DEPLOY_ARGS+=(--engine-image "$ENGINE_DOCKER_IMAGE"); fi
    if [ -n "$ENGINE_PORT" ]; then DEPLOY_ARGS+=(--port "$ENGINE_PORT"); fi
    if [ -n "$MODEL_VERSION" ]; then DEPLOY_ARGS+=(--model-version "$MODEL_VERSION"); fi
    if [ -n "$BUILD_IMAGE" ]; then DEPLOY_ARGS+=(--image "$BUILD_IMAGE"); fi
    if [ -n "$GRPC_PORT" ]; then
        export TRITON_GRPC_PORT="$GRPC_PORT"
    fi
    if [ -n "$HTTP_PORT" ]; then
        export TRITON_HTTP_PORT="$HTTP_PORT"
    fi
    if $DRY_RUN; then DEPLOY_ARGS+=(--dry-run); fi
}

# ── Phase runners ──

run_phase_a() {
    log_step "阶段 A: 环境配置与模型导出"
    echo ""

    if $DRY_RUN; then
        log_info "[DRY RUN] Would run: bash setup_env.sh"
        [ -n "$VARIANT" ] && log_info "  MODEL_VARIANT=$VARIANT"
        [ -n "$PYTHON_VERSION" ] && log_info "  PYTHON_VERSION=$PYTHON_VERSION"
        [ -n "$ENV_NAME" ] && log_info "  ENV_NAME=$ENV_NAME"
        [ -n "$MODEL_SOURCE" ] && log_info "  MODEL_SOURCE=$MODEL_SOURCE"
        [ -n "${EXPORT_DEVICE:-$GLOBAL_DEVICE}" ] && log_info "  EXPORT_DEVICE=${EXPORT_DEVICE:-$GLOBAL_DEVICE}"
        _is_enabled "$SKIP_MODELS" && log_info "  SKIP_MODELS=1"
        _is_enabled "$SKIP_DEPS" && log_info "  SKIP_DEPS=1"
        _is_enabled "$SKIP_EXPORT" && log_info "  SKIP_EXPORT=1"
        return 0
    fi

    bash "${SCRIPT_DIR}/setup_env.sh" "${SETUP_ARGS[@]}" || {
        log_error "Phase A 失败。"
        log_info  "请修正问题后重新执行: bash scripts/bash/autorun.sh setup"
        return 1
    }
}

run_phase_b() {
    log_step "阶段 B: TensorRT 引擎编译 (trtexec)"
    echo ""

    if $DRY_RUN; then
        log_info "[DRY RUN] Would run: bash build_engines.sh ${BUILD_ARGS[*]}"
        return 0
    fi

    bash "${SCRIPT_DIR}/build_engines.sh" "${BUILD_ARGS[@]}" || {
        log_error "Phase B 失败。"
        log_info  "请修正问题后重新执行: bash scripts/bash/autorun.sh build"
        return 1
    }
}

run_phase_c() {
    log_step "阶段 C: 部署 TTS 服务"
    echo ""

    if $DRY_RUN; then
        log_info "[DRY RUN] Would run: bash deploy.sh run ${DEPLOY_ARGS[*]}"
        return 0
    fi

    bash "${SCRIPT_DIR}/deploy.sh" run "${DEPLOY_ARGS[@]}" || {
        log_error "Phase C 失败。"
        log_info  "请修正问题后重新执行: bash scripts/bash/autorun.sh deploy"
        return 1
    }
}

# ── Command handlers ──

cmd_all() {
    show_run_banner "完整流程" "A → B → C"

    run_phase_a || exit 1
    echo ""
    run_phase_b || exit 1
    echo ""
    run_phase_c || exit 1

    echo ""
    echo -e "${_CLR_GREEN}全部阶段完成！TTS 服务已运行。${_CLR_RESET}"
    echo ""
}

cmd_setup() {
    show_run_banner "阶段 A" "环境 + 导出"
    run_phase_a || exit 1
}

cmd_build() {
    show_run_banner "阶段 B" "TensorRT 引擎"
    run_phase_b || exit 1
}

cmd_deploy() {
    show_run_banner "阶段 C" "部署 TTS 服务"
    run_phase_c || exit 1
}

cmd_status() {
    print_status_summary "$REPO_ROOT" "$VARIANT"
}

cmd_stop() {
    local stop_args=()
    [ -n "$VARIANT" ] && stop_args+=(--variant "$VARIANT")
    bash "${SCRIPT_DIR}/deploy.sh" stop "${stop_args[@]}"
}

cmd_update_matrix() {
    source "${SCRIPT_DIR}/lib/ngc_updater.sh"
    update_ngc_matrix "${SCRIPT_DIR}/ngc_matrix.conf"
}

# ── Banner ──

show_run_banner() {
    local phase_name="$1"
    local description="$2"

    echo ""
    echo -e "${_CLR_BLUE}╔══════════════════════════════════════════════════════════╗${_CLR_RESET}"
    echo -e "${_CLR_BLUE}║     Qwen3-TTS Triton — ${phase_name}${_CLR_RESET}"
    echo -e "${_CLR_BLUE}╚══════════════════════════════════════════════════════════╝${_CLR_RESET}"
    echo ""
    echo "  模式:      $description"
    [ -n "$VARIANT" ] && echo "  变体:      $VARIANT"
    [ -n "$MODEL_VERSION" ] && echo "  版本:      $MODEL_VERSION"
    [ -n "${GATEWAY_MODE:-}" ] && echo "  阶段 C:    $GATEWAY_MODE"
    [ -n "$ENGINE_DTYPE" ] && echo "  精度:      $ENGINE_DTYPE"
    [ -n "${EXPORT_DEVICE:-$GLOBAL_DEVICE}" ] && echo "  导出 GPU:  ${EXPORT_DEVICE:-$GLOBAL_DEVICE}"
    [ -n "${BUILD_GPU_DEVICE:-$GLOBAL_DEVICE}" ] && echo "  构建 GPU:  ${BUILD_GPU_DEVICE:-$GLOBAL_DEVICE}"
    [ -n "${RUNTIME_GPU_DEVICE:-$GLOBAL_DEVICE}" ] && echo "  运行 GPU:  ${RUNTIME_GPU_DEVICE:-$GLOBAL_DEVICE}"
    [ -n "$MAX_BATCH_SIZE" ] && echo "  构建 batch: $MAX_BATCH_SIZE"
    [ -n "$MAX_INPUT_LEN" ] && echo "  构建 input: $MAX_INPUT_LEN"
    [ -n "$MAX_SEQ_LEN" ] && echo "  构建 seq:   $MAX_SEQ_LEN"
    [ -n "$RUNTIME_MAX_BATCH_SIZE" ] && echo "  运行 batch: $RUNTIME_MAX_BATCH_SIZE"
    [ -n "$RUNTIME_MAX_SEQ_LEN" ] && echo "  运行 seq:   $RUNTIME_MAX_SEQ_LEN"
    [ -n "$TARGET_DRIVER" ] && echo "  目标驱动:  $TARGET_DRIVER"
    $DRY_RUN && echo "  预演:      是"
    echo ""
}

# ── Interactive mode ──

interactive_mode() {
    print_status_summary "$REPO_ROOT" ""

    # Determine where we are in the pipeline
    local resume_point
    resume_point=$(detect_resume_point "$REPO_ROOT" "")

    echo "  要执行什么操作？"
    echo ""
    echo "  [1] 完整流程  (setup → build → deploy)"
    echo "  [2] 环境配置  (阶段 A)"
    echo "  [3] 构建引擎  (阶段 B)"
    echo "  [4] 部署服务    (阶段 C)"

    if [ "$resume_point" != "setup" ] && [ "$resume_point" != "done" ]; then
        echo "  [5] 从 $resume_point 恢复"
    fi

    echo "  [6] 查看详细状态"
    echo "  [q] 退出"
    echo ""

    local choice
    if [ ! -t 0 ]; then
        log_info "非交互模式，默认执行完整流程"
        choice="1"
    else
        read -rp "  请选择 [1-6/q] (默认: 1, 30s 后自动执行完整流程): " -t 30 choice || true
        choice="${choice:-1}"
    fi

    # Ask for variant if not already set
    if [[ "$choice" =~ ^[1-5]$ ]] && [ -z "$VARIANT" ]; then
        echo ""
        VARIANT=$(select_model_variant)
        VARIANT="${VARIANT:-base-1.7b}"
        export MODEL_VARIANT="$VARIANT"
    fi

    local needs_setup=false
    case "${choice:-1}" in
        1|2) needs_setup=true ;;
        5) [[ "$resume_point" == "setup" ]] && needs_setup=true ;;
    esac

    # Ask for engine dtype and profile when Phase B will run (choices 1, 3, or 5→build)
    local needs_build=false
    case "${choice:-1}" in
        1) needs_build=true ;;
        3) needs_build=true ;;
        5) [[ "$resume_point" == "setup" || "$resume_point" == "build" ]] && needs_build=true ;;
    esac

    # Phase C gateway: standalone | Triton | engine Docker image
    local will_deploy=false
    case "${choice:-1}" in
        1|4|5) will_deploy=true ;;
    esac

    if [ -t 0 ] && { $needs_build || $will_deploy; }; then
        echo ""
        echo "  模型版本 (Triton model version)"
        local _mv=""
        read -rp "  版本号 [${MODEL_VERSION}] (30s 后自动): " -t 30 _mv || true
        if [ -n "$_mv" ]; then
            MODEL_VERSION="$_mv"
        fi
    fi

    if [ -t 0 ] && { $needs_setup || $needs_build || $will_deploy; } && \
        [ -z "$GLOBAL_DEVICE" ] && [ -z "$EXPORT_DEVICE" ] && [ -z "$BUILD_GPU_DEVICE" ] && [ -z "$RUNTIME_GPU_DEVICE" ]; then
        echo ""
        echo "  GPU 选择 (默认: auto，自动选择当前空闲显存最多的 GPU)"
        if command -v nvidia-smi &>/dev/null; then
            nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.free --format=csv,noheader,nounits 2>/dev/null \
                | sed 's/^/    /' || true
        fi
        local _gpu=""
        read -rp "  GPU [auto] (可填 0 / 1 / cuda:1，30s 后自动): " -t 30 _gpu || true
        if [ -n "$_gpu" ]; then
            GLOBAL_DEVICE="$_gpu"
        fi
    fi

    if $needs_build && [ -z "$ENGINE_DTYPE" ] && [ -t 0 ]; then
        echo ""
        echo "  引擎精度 (阶段 B): bf16 | fp16 | fp32 | fp8 (默认: bf16)"
        read -rp "  精度 [bf16] (30s 后自动选择默认): " -t 30 _dtyp || true
        if [ -n "$_dtyp" ]; then
            ENGINE_DTYPE="$_dtyp"
        fi
    fi

    if $needs_build && [ -t 0 ] && \
        { [ -z "$MAX_BATCH_SIZE" ] || [ -z "$MAX_INPUT_LEN" ] || [ -z "$MAX_SEQ_LEN" ]; }; then
        echo ""
        echo "  TensorRT 构建 profile (留空则 Phase B 按构建 GPU 显存给建议默认值)"
        local _mb="" _mi="" _ms=""
        [ -z "$MAX_BATCH_SIZE" ] && read -rp "  max-batch-size [auto]: " -t 30 _mb || true
        [ -z "$MAX_INPUT_LEN" ] && read -rp "  max-input-len [auto]: " -t 30 _mi || true
        [ -z "$MAX_SEQ_LEN" ] && read -rp "  max-seq-len [auto]: " -t 30 _ms || true
        [ -n "$_mb" ] && MAX_BATCH_SIZE="$_mb"
        [ -n "$_mi" ] && MAX_INPUT_LEN="$_mi"
        [ -n "$_ms" ] && MAX_SEQ_LEN="$_ms"
    fi

    if $will_deploy && [ -z "$GATEWAY_MODE" ] && [ -t 0 ]; then
        echo ""
        echo "  阶段 C — 部署方式:"
        echo "    [1] standalone    — 本机 Python 运行 engine（需 Phase A conda / torch）"
        echo "    [2] triton        — Docker 内 Triton Server + 组装 model_repository"
        echo "    [3] engine-docker — 独立引擎镜像，挂载同一份 model_repository，不依赖本机 PyTorch"
        echo ""
        local gwch=""
        read -rp "  请选择 [1-3] (默认: 1 standalone, 30s 后选默认): " -t 30 gwch || true
        gwch="${gwch:-1}"
        case "$gwch" in
            2) GATEWAY_MODE="triton" ;;
            3) GATEWAY_MODE="engine-docker" ;;
            *) GATEWAY_MODE="standalone" ;;
        esac
        log_info "已选择部署方式: $GATEWAY_MODE"
    fi
    if $will_deploy && [ -z "$GATEWAY_MODE" ]; then
        GATEWAY_MODE="standalone"
    fi

    if $will_deploy && [ -t 0 ] && \
        { [ -z "$RUNTIME_MAX_BATCH_SIZE" ] || [ -z "$RUNTIME_MAX_SEQ_LEN" ]; }; then
        echo ""
        echo "  Runtime 上限 (留空则读取 manifest engine_profile；不能超过 Phase B profile)"
        local _rb="" _rs=""
        [ -z "$RUNTIME_MAX_BATCH_SIZE" ] && read -rp "  runtime-max-batch-size [manifest]: " -t 30 _rb || true
        [ -z "$RUNTIME_MAX_SEQ_LEN" ] && read -rp "  runtime-max-seq-len [manifest]: " -t 30 _rs || true
        [ -n "$_rb" ] && RUNTIME_MAX_BATCH_SIZE="$_rb"
        [ -n "$_rs" ] && RUNTIME_MAX_SEQ_LEN="$_rs"
    fi

    build_forward_args

    case "${choice:-1}" in
        1) cmd_all ;;
        2) cmd_setup ;;
        3) cmd_build ;;
        4) cmd_deploy ;;
        5)
            case "$resume_point" in
                build)  cmd_build; echo ""; cmd_deploy ;;
                deploy) cmd_deploy ;;
                *)      cmd_all ;;
            esac
            ;;
        6)
            print_status_summary "$REPO_ROOT" "$VARIANT"
            ;;
        q|Q)
            echo "  已退出。"
            exit 0
            ;;
        *)
            log_error "无效选择: $choice"
            exit 1
            ;;
    esac
}

# ── Main ──

main() {
    parse_args "$@"
    MODEL_VERSION=$(resolve_model_version "$MODEL_VERSION") || exit 1
    export MODEL_VERSION
    export ENGINE_MODEL_VERSION="$MODEL_VERSION"

    # No command → interactive mode
    if [ -z "$COMMAND" ]; then
        interactive_mode
        return
    fi

    # Resolve variant interactively if needed for phases that require it
    if [ -z "$VARIANT" ] && [[ "$COMMAND" =~ ^(all|setup)$ ]] && ! $YES_MODE; then
        if [ -t 0 ]; then
            VARIANT=$(select_model_variant)
            VARIANT="${VARIANT:-base-1.7b}"
        else
            VARIANT="base-1.7b"
        fi
    fi

    build_forward_args

    case "$COMMAND" in
        all)            cmd_all ;;
        setup)          cmd_setup ;;
        build)          cmd_build ;;
        deploy)         cmd_deploy ;;
        status)         cmd_status ;;
        stop)           cmd_stop ;;
        update-matrix)  cmd_update_matrix ;;
        help)           usage ;;
        *)              log_error "Unknown command: $COMMAND"; usage; exit 1 ;;
    esac
}

main "$@"

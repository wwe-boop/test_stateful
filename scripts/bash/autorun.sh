#!/bin/bash
# ===========================================================================
#  autorun.sh — Intelligent launcher for Qwen3-TTS Triton pipeline
#
#  Smart entry point that orchestrates all three build phases:
#    Phase A (setup):  Environment + model export    → setup_env.sh
#    Phase B (build):  TensorRT engine compilation    → build_engines.sh
#    Phase C (deploy): Triton server deployment       → build_triton.sh
#
#  Usage:
#    bash scripts/bash/autorun.sh                    # interactive mode
#    bash scripts/bash/autorun.sh base-1.7b          # full pipeline for variant
#    bash scripts/bash/autorun.sh all [options]       # full pipeline (explicit)
#    bash scripts/bash/autorun.sh setup [options]     # Phase A only
#    bash scripts/bash/autorun.sh build [options]     # Phase B only
#    bash scripts/bash/autorun.sh deploy [options]    # Phase C only
#    bash scripts/bash/autorun.sh status              # show pipeline status
#    bash scripts/bash/autorun.sh stop                # stop Triton server
#    bash scripts/bash/autorun.sh update-matrix       # update NGC compat matrix
#
#  Options:
#    --variant, -m <name>    Model variant (base-1.7b, custom-1.7b, ...)
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
#      --max-batch-size <N>  TRT max batch (default: 8)
#      --image <uri>         Override NGC container image
#      --dtype <type>        Engine precision: bf16|fp16|fp32 (default: bf16)
#
#    Phase C (forwarded to build_triton.sh):
#      --grpc-port <port>    gRPC port (default: 8001)
#      --http-port <port>    HTTP port (default: 8000)
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

# ── Defaults ──
COMMAND=""
VARIANT=""
DRY_RUN=false
YES_MODE=false
TARGET_DRIVER="${TARGET_DRIVER:-}"

# Phase A forwarding
SETUP_ARGS=()
PYTHON_VERSION=""
ENV_NAME=""
MODEL_SOURCE=""
SKIP_MODELS=false
SKIP_DEPS=false
SKIP_EXPORT=false

# Phase B forwarding
BUILD_ARGS=()
MAX_BATCH_SIZE=""
BUILD_IMAGE=""
ENGINE_DTYPE=""

# Phase C forwarding
DEPLOY_ARGS=()
GRPC_PORT=""
HTTP_PORT=""

# ── Help ──

usage() {
    cat << 'EOF'
Usage: autorun.sh [command] [model_variant] [options]

Commands:
  all               Full pipeline: setup → build → deploy (default)
  setup             Phase A only (environment + model export)
  build             Phase B only (TensorRT engine compilation)
  deploy            Phase C only (Triton server deployment)
  status            Show pipeline status
  stop              Stop Triton server
  update-matrix     Update NGC compatibility matrix from NVIDIA website
  help              Show this help

If no command is given, launches interactive mode.
If a model variant name is given without a command, runs the full pipeline.

Options:
  --variant, -m <name>    Model variant (base-1.7b, custom-1.7b, design-1.7b,
                          base-0.6b, custom-0.6b, all-1.7b, all)
  --target-driver <ver>   Target NVIDIA driver version for NGC container
                          selection (e.g. 575.57 for production machines).
                          Overrides local driver detection.
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
  --max-batch-size <N>    Max batch size (default: 8)
  --image <uri>           Override NGC container image
  --dtype <type>          Engine precision: bf16|fp16|fp32 (default: bf16)
                          Use fp32 if Triton reports dtype mismatch (e.g. TYPE_FP32 vs TYPE_BF16).

Phase C options (forwarded to build_triton.sh):
  --grpc-port <port>      gRPC port (default: 8001)
  --http-port <port>      HTTP port (default: 8000)

Examples:
  autorun.sh                          # interactive guided setup
  autorun.sh base-1.7b                # full pipeline for base-1.7b
  autorun.sh all -m custom-1.7b       # full pipeline for custom-1.7b
  autorun.sh setup --skip-export      # Phase A without export
  autorun.sh custom-1.7b --dtype fp32 # Full pipeline, FP32 engines (fixes dtype mismatch)
  autorun.sh build -m custom-1.7b --dtype fp32    # Phase B with fp32
  autorun.sh build -m custom-1.7b --dtype fp16    # Phase B with fp16
  autorun.sh build --target-driver 575.57   # build for production driver
  autorun.sh deploy                   # Phase C (start Triton)
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
            --yes|-y)           YES_MODE=true; shift ;;
            --dry-run)          DRY_RUN=true; shift ;;
            --target-driver)    TARGET_DRIVER="$2"; shift 2 ;;
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
            --image)            BUILD_IMAGE="$2"; shift 2 ;;
            --dtype)            ENGINE_DTYPE="$2"; shift 2 ;;

            # Phase C
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

    # Phase A args
    SETUP_ARGS=()
    if [ -n "$VARIANT" ]; then
        # Pass variant via env var (setup_env.sh reads MODEL_VARIANT)
        export MODEL_VARIANT="$VARIANT"
    fi
    if [ -n "$PYTHON_VERSION" ]; then
        SETUP_ARGS+=('' '' '')  # placeholder positional args
        SETUP_ARGS[2]="$PYTHON_VERSION"
    fi
    if [ -n "$MODEL_SOURCE" ]; then
        export MODEL_SOURCE="$MODEL_SOURCE"
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
    if [ -n "$MAX_BATCH_SIZE" ]; then BUILD_ARGS+=(--max-batch-size "$MAX_BATCH_SIZE"); fi
    if [ -n "$BUILD_IMAGE" ]; then BUILD_ARGS+=(--image "$BUILD_IMAGE"); fi
    if [ -n "$ENGINE_DTYPE" ]; then BUILD_ARGS+=(--dtype "$ENGINE_DTYPE"); fi
    if $DRY_RUN; then BUILD_ARGS+=(--dry-run); fi

    # Phase C args
    DEPLOY_ARGS=()
    if [ -n "$VARIANT" ] && [[ "$VARIANT" != all* ]]; then
        DEPLOY_ARGS+=(--variant "$VARIANT")
    fi
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
        return 0
    fi

    bash "${SCRIPT_DIR}/setup_env.sh" || {
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
    log_step "阶段 C: Triton 部署"
    echo ""

    if $DRY_RUN; then
        log_info "[DRY RUN] Would run: bash build_triton.sh run ${DEPLOY_ARGS[*]}"
        return 0
    fi

    bash "${SCRIPT_DIR}/build_triton.sh" run "${DEPLOY_ARGS[@]}" || {
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
    echo -e "${_CLR_GREEN}全部阶段完成！Triton 服务已运行。${_CLR_RESET}"
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
    show_run_banner "阶段 C" "Triton 部署"
    run_phase_c || exit 1
}

cmd_status() {
    print_status_summary "$REPO_ROOT" "$VARIANT"
}

cmd_stop() {
    local container="${CONTAINER_NAME:-qwen3-tts-triton}"
    triton_stop "$container"
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
    [ -n "$ENGINE_DTYPE" ] && echo "  精度:      $ENGINE_DTYPE"
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
    echo "  [4] 部署 Triton  (阶段 C)"

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

    # Ask for engine dtype when Phase B will run (choices 1, 3, or 5→build)
    local needs_build=false
    case "${choice:-1}" in
        1) needs_build=true ;;
        3) needs_build=true ;;
        5) [[ "$resume_point" == "build" ]] && needs_build=true ;;
    esac
    if $needs_build && [ -z "$ENGINE_DTYPE" ] && [ -t 0 ]; then
        echo ""
        echo "  引擎精度 (阶段 B): bf16 | fp16 | fp32 (默认: bf16)"
        read -rp "  精度 [bf16] (30s 后自动选择默认): " -t 30 _dtyp || true
        if [ -n "$_dtyp" ]; then
            ENGINE_DTYPE="$_dtyp"
        fi
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

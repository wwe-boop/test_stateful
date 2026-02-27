#!/bin/bash
# ===========================================================================
#  setup_env.sh — Phase A: Build environment setup for Qwen3-TTS Triton
#
#  Sets up the environment needed to export model components to ONNX / TRT
#  and prepare everything for the final Triton deployment image.
#
#  This is the BUILD phase — the final Triton image does NOT need PyTorch
#  or the qwen-tts package; it only ships the exported engines.
#
#  Steps: prerequisites → submodule → workspace → mirrors → venv → deps
#         → models → export deps → export
#
#  Can be called directly or through the autorun.sh smart launcher.
#
#  Usage:
#    bash scripts/bash/setup_env.sh [workdir] [env_name] [python_version] [model_variant]
#
#  Environment variables (override any positional arg):
#    WORKDIR          workspace directory       (default: <repo>/workspace)
#    ENV_NAME         Python env name           (default: qwen3-tts)
#    PYTHON_VERSION   Python version            (default: 3.10)
#    MODEL_VARIANT    Model variant to download (default: base-1.7b)
#    MODEL_SOURCE     Download source           (auto | hf | modelscope)
#    SKIP_MODELS      Set to 1 to skip model download
#    SKIP_DEPS        Set to 1 to skip dependency installation
#    SKIP_EXPORT      Set to 1 to skip ONNX model export
#    CONFIGURE_MIRRORS  Mirror config (auto | china | skip)  (default: auto)
# ===========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
source "${SCRIPT_DIR}/tools.sh"

WORKDIR="${WORKDIR:-${1:-${REPO_ROOT}/workspace}}"
ENV_NAME="${ENV_NAME:-${2:-qwen3-tts}}"
PYTHON_VERSION="${PYTHON_VERSION:-${3:-3.10}}"
MODEL_VARIANT="${MODEL_VARIANT:-${4:-}}"
MODEL_SOURCE="${MODEL_SOURCE:-auto}"
SKIP_MODELS="${SKIP_MODELS:-0}"
SKIP_DEPS="${SKIP_DEPS:-0}"
SKIP_EXPORT="${SKIP_EXPORT:-0}"

SUBMODULE_PATH="third_party/Qwen3-TTS"
MODEL_DIR="${WORKDIR}/models"

# ---- Banner ---------------------------------------------------------------

show_banner() {
    echo ""
    echo -e "${_CLR_BLUE}╔══════════════════════════════════════════════════════════╗${_CLR_RESET}"
    echo -e "${_CLR_BLUE}║     Qwen3-TTS Triton — Phase A: Environment Setup      ║${_CLR_RESET}"
    echo -e "${_CLR_BLUE}╚══════════════════════════════════════════════════════════╝${_CLR_RESET}"
    echo ""
    echo "  Workspace:       $WORKDIR"
    echo "  Python env:      $ENV_NAME (Python $PYTHON_VERSION)"
    echo "  Model variant:   $MODEL_VARIANT"
    echo "  Model source:    $MODEL_SOURCE"
    echo ""
}

# ---- Step 1: Submodule ---------------------------------------------------

init_qwen3_tts() {
    local abs_path="${REPO_ROOT}/${SUBMODULE_PATH}"

    if [ -d "${abs_path}/.git" ] || [ -f "${abs_path}/.git" ]; then
        log_info "Qwen3-TTS submodule already initialised, skipping"
        return 0
    fi

    log_step "Initialising Qwen3-TTS submodule"

    local ORIGINAL_URL="https://github.com/QwenLM/Qwen3-TTS.git"
    local RESOLVED_URL
    RESOLVED_URL=$(github_url "$ORIGINAL_URL") || return 1

    if [ "$RESOLVED_URL" != "$ORIGINAL_URL" ]; then
        log_info "Using mirror URL for submodule: $RESOLVED_URL"
        git -C "$REPO_ROOT" config "submodule.${SUBMODULE_PATH}.url" "$RESOLVED_URL"
    fi

    GIT_HTTP_LOW_SPEED_LIMIT=1000 GIT_HTTP_LOW_SPEED_TIME=30 \
        git -C "$REPO_ROOT" submodule update --init --depth 1 -- "$SUBMODULE_PATH" \
        || { log_error "Failed to initialise Qwen3-TTS submodule"; return 1; }

    log_info "Qwen3-TTS submodule ready at: ${abs_path}"
}

# ---- Step 2: Symlink -----------------------------------------------------

link_into_workdir() {
    local src="${REPO_ROOT}/${SUBMODULE_PATH}"
    local dst="${WORKDIR}/Qwen3-TTS"

    if [ -L "$dst" ]; then
        log_info "Symlink already exists: $dst"
        return 0
    fi
    if [ -e "$dst" ]; then
        log_warn "$dst exists but is not a symlink, skipping"
        return 0
    fi

    ln -s "$src" "$dst"
    log_info "Linked: $dst -> $src"
}

# ---- Step 3: Dependencies ------------------------------------------------

install_dependencies() {
    if [ "$SKIP_DEPS" = "1" ]; then
        log_info "SKIP_DEPS=1, skipping dependency installation"
        return 0
    fi

    log_step "Installing dependencies..."

    # setuptools>=82 removed pkg_resources, which modelscope still needs
    python3 -m pip install --upgrade pip "setuptools<81" wheel -q \
        || log_warn "pip/setuptools upgrade failed (non-fatal)"

    # PyTorch with CUDA
    install_torch_cuda

    # flash-attn: fused attention kernels used by Qwen3-TTS Talker backbone
    install_flash_attn \
        || log_warn "flash-attn install failed (non-fatal, will fall back to manual attention)"

    # modelscope CLI (official recommended download tool)
    pip_install modelscope

    # Qwen3-TTS: the model architecture (Qwen3TTSForConditionalGeneration) lives
    # in this package — required by the export scripts to extract sub-modules
    # (Talker, Code Predictor, etc.) and convert them to ONNX / TRT.
    # NOT needed in the final Triton image.
    install_qwen3_tts "${REPO_ROOT}/${SUBMODULE_PATH}"
}

# ---- Step 4: Model download ----------------------------------------------

download_models() {
    if [ "$SKIP_MODELS" = "1" ]; then
        log_info "SKIP_MODELS=1, skipping model download"
        return 0
    fi

    log_step "Downloading model weights (variant: $MODEL_VARIANT)..."
    download_qwen3_tts_models "$MODEL_DIR" "$MODEL_VARIANT" "$MODEL_SOURCE"
}

# ---- Step 5: Export dependencies -----------------------------------------

install_export_deps() {
    log_step "Installing export dependencies ..."
    install_onnx_export_deps
    install_safetensors
}

# ---- Step 6: Model export ------------------------------------------------

export_models() {
    if [ "$SKIP_EXPORT" = "1" ]; then
        log_info "SKIP_EXPORT=1, skipping model export"
        return 0
    fi
    if [ "$SKIP_MODELS" = "1" ]; then
        log_info "SKIP_MODELS=1, no models to export"
        return 0
    fi

    log_step "Exporting models (ONNX / TRT-LLM checkpoints / PyTorch weights)..."

    local export_dir="${REPO_ROOT}/scripts/export"
    local export_args=()

    if [ "$MODEL_VARIANT" != "all" ] && [ "$MODEL_VARIANT" != "all-1.7b" ]; then
        export_args+=(--variant "$MODEL_VARIANT")
    fi

    cd "$export_dir"
    python3 export_all.py "${export_args[@]}" \
        || { log_error "Model export failed"; return 1; }

    log_info "Models exported to: ${REPO_ROOT}/workspace/exported/"
}

# ---- Step 7: Validation --------------------------------------------------

validate_setup() {
    log_step "Validating setup..."

    validate_python_env || return 1

    # Check model directory
    if [ "$SKIP_MODELS" != "1" ]; then
        if [ -d "$MODEL_DIR" ] && [ "$(ls -A "$MODEL_DIR" 2>/dev/null)" ]; then
            log_info "Model directory: $MODEL_DIR"
            ls -1 "$MODEL_DIR"/
        else
            log_warn "Model directory is empty: $MODEL_DIR"
        fi
    fi

    # Check exported models
    local exported_dir="${REPO_ROOT}/workspace/exported"
    if [ -d "$exported_dir" ] && [ "$(ls -A "$exported_dir" 2>/dev/null)" ]; then
        log_info "Exported models: $exported_dir"
        find "$exported_dir" \( -name "*.onnx" -o -name "*.pt" -o -name "*.safetensors" \) 2>/dev/null \
            | while read -r f; do
                local size
                size=$(du -h "$f" | cut -f1)
                log_info "  ${f#$exported_dir/}  ($size)"
            done
        # TRT-LLM checkpoints (engine compiled separately by build_engines.sh)
        find "$exported_dir" -path "*/trtllm_checkpoint/config.json" 2>/dev/null \
            | while read -r f; do
                log_info "  ${f#$exported_dir/}  (TRT-LLM checkpoint — run build_engines.sh to compile engine)"
            done
    fi
}

# ---- Main -----------------------------------------------------------------

main() {
    # If model variant not explicitly specified and download is not skipped,
    # ask the user which model(s) to download before proceeding.
    if [ -z "$MODEL_VARIANT" ] && [ "$SKIP_MODELS" != "1" ]; then
        MODEL_VARIANT=$(select_model_variant)
    fi
    MODEL_VARIANT="${MODEL_VARIANT:-base-1.7b}"

    show_banner

    log_step "[1/9] Checking prerequisites..."
    check_prerequisites || exit 1

    log_step "[2/9] Initialising Qwen3-TTS submodule..."
    init_qwen3_tts

    log_step "[3/9] Preparing workspace..."
    ensure_workdir "$WORKDIR"
    link_into_workdir

    log_step "[4/9] Configuring package mirrors..."
    configure_mirrors

    log_step "[5/9] Setting up Python environment..."
    ensure_venv "$ENV_NAME" "$PYTHON_VERSION"

    log_step "[6/9] Installing dependencies..."
    install_dependencies

    log_step "[7/9] Downloading model weights..."
    download_models

    log_step "[8/9] Installing export dependencies ..."
    install_export_deps

    log_step "[9/9] Exporting models..."
    export_models

    echo ""
    echo -e "${_CLR_BLUE}──────────────────────────────────────────────────${_CLR_RESET}"
    validate_setup
    echo -e "${_CLR_BLUE}──────────────────────────────────────────────────${_CLR_RESET}"
    echo ""
    echo -e "${_CLR_GREEN}Phase A complete!${_CLR_RESET}"
    echo ""
    echo "  Workspace:   $WORKDIR"
    echo "  Models:      $MODEL_DIR"
    echo "  Exported:    ${REPO_ROOT}/workspace/exported/"
    echo "  Activate:    conda activate $ENV_NAME  (or source .venv/bin/activate)"
    echo ""
    echo "  Next steps:"
    echo "    1. Build TRT-LLM engines:  bash scripts/bash/build_engines.sh  (auto-selects NGC container)"
    echo "    2. Deploy Triton:           bash scripts/bash/build_triton.sh run"
    echo "    3. Or run full pipeline:    bash scripts/bash/autorun.sh all"
    echo ""
}

main

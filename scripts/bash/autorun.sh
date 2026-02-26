#!/bin/bash
# ===========================================================================
#  autorun.sh — One-command BUILD environment setup for Qwen3-TTS Triton
#
#  Sets up the environment needed to export model components to ONNX / TRT
#  and prepare everything for the final Triton deployment image.
#
#  This is the BUILD phase — the final Triton image does NOT need PyTorch
#  or the qwen-tts package; it only ships the exported engines.
#
#  Usage:
#    bash scripts/bash/autorun.sh [workdir] [env_name] [python_version] [model_variant]
#
#  Environment variables (override any positional arg):
#    WORKDIR          workspace directory       (default: <repo>/workspace)
#    ENV_NAME         Python env name           (default: qwen3-tts)
#    PYTHON_VERSION   Python version            (default: 3.10)
#    MODEL_VARIANT    Model variant to download (default: base-1.7b)
#    MODEL_SOURCE     Download source           (auto | hf | modelscope)
#    SKIP_MODELS      Set to 1 to skip model download
#    SKIP_DEPS        Set to 1 to skip dependency installation
# ===========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
source "${SCRIPT_DIR}/tools.sh"

WORKDIR="${WORKDIR:-${1:-${REPO_ROOT}/workspace}}"
ENV_NAME="${ENV_NAME:-${2:-qwen3-tts}}"
PYTHON_VERSION="${PYTHON_VERSION:-${3:-3.10}}"
MODEL_VARIANT="${MODEL_VARIANT:-${4:-base-1.7b}}"
MODEL_SOURCE="${MODEL_SOURCE:-auto}"
SKIP_MODELS="${SKIP_MODELS:-0}"
SKIP_DEPS="${SKIP_DEPS:-0}"

SUBMODULE_PATH="third_party/Qwen3-TTS"
MODEL_DIR="${WORKDIR}/models"

# ---- Banner ---------------------------------------------------------------

show_banner() {
    echo ""
    echo -e "${_CLR_BLUE}╔══════════════════════════════════════════════════════════╗${_CLR_RESET}"
    echo -e "${_CLR_BLUE}║     Qwen3-TTS Triton — Build Environment Setup         ║${_CLR_RESET}"
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

    python3 -m pip install --upgrade pip setuptools wheel -q \
        || log_warn "pip/setuptools upgrade failed (non-fatal)"

    # PyTorch with CUDA
    install_torch_cuda

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

# ---- Step 5: Validation --------------------------------------------------

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
}

# ---- Main -----------------------------------------------------------------

main() {
    show_banner

    log_step "[1/6] Checking prerequisites..."
    check_prerequisites || exit 1

    log_step "[2/6] Initialising Qwen3-TTS submodule..."
    init_qwen3_tts

    log_step "[3/6] Preparing workspace..."
    ensure_workdir "$WORKDIR"
    link_into_workdir

    log_step "[4/6] Setting up Python environment..."
    ensure_venv "$ENV_NAME" "$PYTHON_VERSION"

    log_step "[5/6] Installing dependencies..."
    install_dependencies

    log_step "[6/6] Downloading model weights..."
    download_models

    echo ""
    echo -e "${_CLR_BLUE}──────────────────────────────────────────────────${_CLR_RESET}"
    validate_setup
    echo -e "${_CLR_BLUE}──────────────────────────────────────────────────${_CLR_RESET}"
    echo ""
    echo -e "${_CLR_GREEN}Setup complete!${_CLR_RESET}"
    echo ""
    echo "  Workspace:   $WORKDIR"
    echo "  Models:      $MODEL_DIR"
    echo "  Activate:    conda activate $ENV_NAME  (or source .venv/bin/activate)"
    echo ""
    echo "  Next steps:"
    echo "    1. Activate the environment"
    echo "    2. Run model export:   python scripts/export/export_*.py"
    echo "    3. Build TRT engines:  bash scripts/build/build_trt_engines.sh"
    echo ""
}

main

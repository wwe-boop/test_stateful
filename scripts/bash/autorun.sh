#!/bin/bash
# ===========================================================================
#  01_create_env.sh — Clone Qwen3-TTS repo and prepare Python environment
#
#  Usage:  bash 01_create_env.sh [workdir]
# ===========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/tools.sh"

WORKDIR="${1:-$(pwd)/workspace}"
ENV_NAME="${2:-qwen3-tts}"
PYTHON_VERSION="${3:-3.10}"
COMMIT_ID="${4:-main}"

# ---- Clone ----------------------------------------------------------------

clone_qwen3_tts() {
    local CLONE_URL
    CLONE_URL=$(github_url "https://github.com/QwenLM/Qwen3-TTS.git") || return 1

    if [ -d "Qwen3-TTS/.git" ]; then
        log_info "Qwen3-TTS repo already cloned, skipping"
        return 0
    fi

    log_step "Cloning from: $CLONE_URL"
    GIT_HTTP_LOW_SPEED_LIMIT=1000 GIT_HTTP_LOW_SPEED_TIME=30 \
        git clone --depth 1 --branch "$COMMIT_ID" "$CLONE_URL" \
        || { log_error "git clone failed"; return 1; }
    log_info "Clone completed"
}

# ---- Main -----------------------------------------------------------------

main() {
    ensure_workdir "$WORKDIR";
    clone_qwen3_tts;
    ensure_venv "$ENV_NAME" "$PYTHON_VERSION";
    log_info "Done. Workspace ready at: $(pwd)";
}

main

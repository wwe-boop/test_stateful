#!/bin/bash
# ===========================================================================
#  network.sh — Network / mirror helpers
#
#  Functions: github_url, hf_url, download_hf_model
#  Depends:   lib/logging.sh, lib/utils.sh
# ===========================================================================

[[ -n "${_LIB_NETWORK_LOADED:-}" ]] && return 0
_LIB_NETWORK_LOADED=1

_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_LIB_DIR}/logging.sh"
source "${_LIB_DIR}/utils.sh"

# ---- GitHub ---------------------------------------------------------------

GITHUB_MIRROR="${GITHUB_MIRROR:-https://v6.gh-proxy.org}"

# ---------------------------------------------------------------------------
#  github_url <repo_url>
#  Echoes the original URL if github.com is reachable, otherwise tries the
#  mirror.  Returns 1 if neither is reachable.
# ---------------------------------------------------------------------------
github_url() {
    local REPO_URL="$1"
    local HTTP_CODE

    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 5 --max-time 10 \
        https://github.com 2>/dev/null)
    if [ "$HTTP_CODE" = "200" ]; then
        echo "$REPO_URL"
        return 0
    fi
    log_warn "GitHub unreachable (HTTP: $HTTP_CODE), trying mirror(${GITHUB_MIRROR})..."

    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 5 --max-time 10 \
        "${GITHUB_MIRROR}" 2>/dev/null)
    if [ "$HTTP_CODE" = "200" ] || [ "$HTTP_CODE" = "301" ] || [ "$HTTP_CODE" = "302" ]; then
        echo "${GITHUB_MIRROR}/${REPO_URL}"
        return 0
    fi
    log_error "GitHub mirror(${GITHUB_MIRROR}) also unreachable (HTTP: $HTTP_CODE)"
    return 1
}

# ---- HuggingFace ----------------------------------------------------------

HF_MIRROR="${HF_MIRROR:-https://hf-mirror.com}"

# ---------------------------------------------------------------------------
#  hf_url <original_url>
#  Rewrites a huggingface.co URL to use the mirror when HF is unreachable.
# ---------------------------------------------------------------------------
hf_url() {
    local ORIGINAL_URL="$1"
    local HTTP_CODE
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 5 --max-time 10 \
        https://huggingface.co 2>/dev/null)
    if [ "$HTTP_CODE" = "200" ]; then
        echo "$ORIGINAL_URL"
        return 0
    fi
    log_warn "HuggingFace unreachable (HTTP: $HTTP_CODE), using mirror(${HF_MIRROR})..."
    echo "$ORIGINAL_URL" | sed "s|https://huggingface.co|${HF_MIRROR}|g"
}

# ---------------------------------------------------------------------------
#  download_hf_model <model_id> [target_dir]
#  Shallow-clones a HuggingFace repo (idempotent).
# ---------------------------------------------------------------------------
download_hf_model() {
    local MODEL_ID="$1"
    local TARGET_DIR="${2:-.}"

    if [ -z "$MODEL_ID" ]; then
        log_error "Usage: download_hf_model <model_id> [target_dir]"
        return 1
    fi

    require_cmd git || return 1
    mkdir -p "$TARGET_DIR"

    local MODEL_NAME
    MODEL_NAME=$(basename "$MODEL_ID")
    local DEST="$TARGET_DIR/$MODEL_NAME"

    if [ -d "$DEST" ] && [ -d "$DEST/.git" ]; then
        log_info "Model '$MODEL_NAME' already exists at $DEST, skipping"
        return 0
    fi

    local REPO_URL
    REPO_URL=$(hf_url "https://huggingface.co/${MODEL_ID}")

    log_step "Downloading model: $MODEL_ID"
    log_info "From: $REPO_URL → $DEST"

    GIT_LFS_SKIP_SMUDGE=0 \
    GIT_HTTP_LOW_SPEED_LIMIT=1000 GIT_HTTP_LOW_SPEED_TIME=60 \
        git clone --depth 1 "$REPO_URL" "$DEST" \
        || { log_error "Failed to clone model $MODEL_ID"; return 1; }

    log_info "Model downloaded to: $DEST"
}

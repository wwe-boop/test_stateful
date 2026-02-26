#!/bin/bash
# ===========================================================================
#  network.sh — Network / mirror helpers
#
#  Functions: github_url, hf_url, download_hf_model,
#             download_model, download_qwen3_tts_models
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
#  _hf_reachable
#  Returns 0 if huggingface.co is directly reachable, 1 otherwise.
#  Caches the result for the lifetime of the script.
# ---------------------------------------------------------------------------
_hf_reachable() {
    if [ -z "${_HF_REACHABLE_CACHE:-}" ]; then
        local code
        code=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 5 --max-time 10 \
            https://huggingface.co 2>/dev/null)
        if [ "$code" = "200" ]; then
            _HF_REACHABLE_CACHE="yes"
        else
            _HF_REACHABLE_CACHE="no"
            log_warn "HuggingFace unreachable (HTTP: $code), using mirror(${HF_MIRROR})"
        fi
    fi
    [ "$_HF_REACHABLE_CACHE" = "yes" ]
}

# ---------------------------------------------------------------------------
#  hf_url <original_url>
#  Rewrites a huggingface.co URL to use the mirror when HF is unreachable.
# ---------------------------------------------------------------------------
hf_url() {
    local ORIGINAL_URL="$1"
    if _hf_reachable; then
        echo "$ORIGINAL_URL"
    else
        echo "$ORIGINAL_URL" | sed "s|https://huggingface.co|${HF_MIRROR}|g"
    fi
}

# ---------------------------------------------------------------------------
#  _hf_endpoint
#  Returns the HF API endpoint (respects HF_ENDPOINT env, falls back to
#  mirror when HuggingFace is unreachable).
# ---------------------------------------------------------------------------
_hf_endpoint() {
    if [ -n "${HF_ENDPOINT:-}" ]; then
        echo "$HF_ENDPOINT"
    elif _hf_reachable; then
        echo "https://huggingface.co"
    else
        echo "$HF_MIRROR"
    fi
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

# ---- High-level model download -------------------------------------------

# ---------------------------------------------------------------------------
#  download_model <model_id> <target_dir> [source]
#  Downloads a model using the best available method.
#
#  source: "auto" (default) | "hf" | "modelscope"
#
#  Strategy (source=auto):
#    1. huggingface-cli  (resume-capable, progress bar)
#    2. modelscope download  (for China users)
#    3. git clone (fallback)
# ---------------------------------------------------------------------------
download_model() {
    local MODEL_ID="$1"
    local TARGET_DIR="$2"
    local SOURCE="${3:-auto}"

    if [ -z "$MODEL_ID" ] || [ -z "$TARGET_DIR" ]; then
        log_error "Usage: download_model <model_id> <target_dir> [hf|modelscope|auto]"
        return 1
    fi

    local MODEL_NAME
    MODEL_NAME=$(basename "$MODEL_ID")
    local DEST="$TARGET_DIR/$MODEL_NAME"
    mkdir -p "$TARGET_DIR"

    # Already downloaded?
    if _model_dir_valid "$DEST"; then
        log_info "Model '$MODEL_NAME' already at $DEST, skipping"
        return 0
    fi

    log_step "Downloading: $MODEL_ID → $DEST"

    case "$SOURCE" in
        modelscope|ms)
            _download_via_modelscope "$MODEL_ID" "$DEST" && return 0
            log_warn "ModelScope download failed, trying HuggingFace..."
            _download_via_hf "$MODEL_ID" "$DEST" && return 0
            ;;
        hf)
            _download_via_hf "$MODEL_ID" "$DEST" && return 0
            ;;
        auto|*)
            # ModelScope first (official recommendation), HF as fallback
            _download_via_modelscope "$MODEL_ID" "$DEST" && return 0
            log_warn "ModelScope failed, trying HuggingFace..."
            _download_via_hf "$MODEL_ID" "$DEST" && return 0
            ;;
    esac

    log_error "All download methods failed for $MODEL_ID"
    return 1
}

# Internal: check if a model directory looks valid
_model_dir_valid() {
    local dir="$1"
    [ -d "$dir" ] && {
        # Has config or model files
        [ -f "$dir/config.json" ] \
            || [ -f "$dir/tokenizer_config.json" ] \
            || [ -d "$dir/.git" ] \
            || ls "$dir"/*.safetensors &>/dev/null 2>&1 \
            || ls "$dir"/*.bin &>/dev/null 2>&1
    }
}

# Internal: download via huggingface-cli (preferred)
_download_via_hf() {
    local model_id="$1"
    local dest="$2"
    local endpoint
    endpoint=$(_hf_endpoint)

    # Try huggingface-cli first
    if command -v huggingface-cli &>/dev/null; then
        log_info "Using huggingface-cli (endpoint: $endpoint)"
        HF_ENDPOINT="$endpoint" \
            huggingface-cli download "$model_id" --local-dir "$dest" \
            && return 0
        log_warn "huggingface-cli failed"
    fi

    # Fall back to git clone
    log_info "Falling back to git clone..."
    local repo_url
    repo_url=$(hf_url "https://huggingface.co/${model_id}")

    rm -rf "$dest"
    GIT_LFS_SKIP_SMUDGE=0 \
    GIT_HTTP_LOW_SPEED_LIMIT=1000 GIT_HTTP_LOW_SPEED_TIME=60 \
        retry 2 10 git clone --depth 1 "$repo_url" "$dest"
}

# Internal: download via modelscope (official recommended method)
_download_via_modelscope() {
    local model_id="$1"
    local dest="$2"

    if ! command -v modelscope &>/dev/null; then
        if ! python3 -c "import modelscope" 2>/dev/null; then
            log_info "Installing modelscope CLI..."
            python3 -m pip install -q modelscope 2>/dev/null || {
                log_warn "Could not install modelscope"
                return 1
            }
        fi
    fi

    if ! command -v modelscope &>/dev/null; then
        log_warn "modelscope CLI not available after install"
        return 1
    fi

    log_info "Using ModelScope to download: $model_id"
    retry 2 10 modelscope download --model "$model_id" --local_dir "$dest"
}

# ---------------------------------------------------------------------------
#  download_qwen3_tts_models <target_dir> [variant] [source]
#
#  variant: base-1.7b (default) | custom-1.7b | design-1.7b
#           | base-0.6b | custom-0.6b | all-1.7b | all
#
#  Always downloads the Tokenizer; then downloads the TTS model(s) matching
#  the selected variant.
# ---------------------------------------------------------------------------

_QWEN3_TTS_TOKENIZER="Qwen/Qwen3-TTS-Tokenizer-12Hz"

declare -A _QWEN3_TTS_MODELS 2>/dev/null || true
_QWEN3_TTS_MODELS=(
    [base-1.7b]="Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    [custom-1.7b]="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    [design-1.7b]="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    [base-0.6b]="Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    [custom-0.6b]="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
)

list_model_variants() {
    echo "Available model variants:"
    echo "  base-1.7b     Qwen3-TTS-12Hz-1.7B-Base        (voice clone, recommended)"
    echo "  custom-1.7b   Qwen3-TTS-12Hz-1.7B-CustomVoice (9 premium voices + instructions)"
    echo "  design-1.7b   Qwen3-TTS-12Hz-1.7B-VoiceDesign (voice design from description)"
    echo "  base-0.6b     Qwen3-TTS-12Hz-0.6B-Base        (lightweight voice clone)"
    echo "  custom-0.6b   Qwen3-TTS-12Hz-0.6B-CustomVoice (lightweight, 9 premium voices)"
    echo "  all-1.7b      All 1.7B models"
    echo "  all           All models"
}

download_qwen3_tts_models() {
    local TARGET_DIR="$1"
    local VARIANT="${2:-base-1.7b}"
    local SOURCE="${3:-auto}"

    if [ -z "$TARGET_DIR" ]; then
        log_error "Usage: download_qwen3_tts_models <target_dir> [variant] [source]"
        return 1
    fi

    mkdir -p "$TARGET_DIR"

    # Tokenizer is always required
    log_step "Downloading Tokenizer: $_QWEN3_TTS_TOKENIZER"
    download_model "$_QWEN3_TTS_TOKENIZER" "$TARGET_DIR" "$SOURCE" \
        || { log_error "Tokenizer download failed"; return 1; }

    # Resolve model list from variant
    local -a models_to_download=()

    case "$VARIANT" in
        all-1.7b)
            models_to_download=(
                "${_QWEN3_TTS_MODELS[base-1.7b]}"
                "${_QWEN3_TTS_MODELS[custom-1.7b]}"
                "${_QWEN3_TTS_MODELS[design-1.7b]}"
            )
            ;;
        all)
            for key in "${!_QWEN3_TTS_MODELS[@]}"; do
                models_to_download+=("${_QWEN3_TTS_MODELS[$key]}")
            done
            ;;
        *)
            if [ -z "${_QWEN3_TTS_MODELS[$VARIANT]:-}" ]; then
                log_error "Unknown model variant: $VARIANT"
                list_model_variants >&2
                return 1
            fi
            models_to_download=("${_QWEN3_TTS_MODELS[$VARIANT]}")
            ;;
    esac

    local failed=0
    for model_id in "${models_to_download[@]}"; do
        download_model "$model_id" "$TARGET_DIR" "$SOURCE" \
            || { log_error "Failed to download $model_id"; failed=$((failed + 1)); }
    done

    if [ "$failed" -gt 0 ]; then
        log_error "$failed model(s) failed to download"
        return 1
    fi

    log_info "All models downloaded to: $TARGET_DIR"
    ls -1 "$TARGET_DIR"/
}

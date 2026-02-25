#!/bin/bash
# ===========================================================================
#  utils.sh — General-purpose shell utilities
#
#  Functions: retry, require_cmd, ensure_workdir, check_gpu
#  Depends:   lib/logging.sh
# ===========================================================================

[[ -n "${_LIB_UTILS_LOADED:-}" ]] && return 0
_LIB_UTILS_LOADED=1

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/logging.sh"

# ---------------------------------------------------------------------------
#  retry <max_attempts> <initial_delay_sec> <command...>
#  Retries a command with exponential back-off.
# ---------------------------------------------------------------------------
retry() {
    local max_attempts="${1:-3}"
    local delay="${2:-5}"
    shift 2
    local attempt=1
    while [ "$attempt" -le "$max_attempts" ]; do
        if "$@"; then
            return 0
        fi
        log_warn "Attempt $attempt/$max_attempts failed: $*"
        if [ "$attempt" -lt "$max_attempts" ]; then
            log_info "Retrying in ${delay}s..."
            sleep "$delay"
            delay=$((delay * 2))
        fi
        attempt=$((attempt + 1))
    done
    log_error "All $max_attempts attempts failed: $*"
    return 1
}

# ---------------------------------------------------------------------------
#  require_cmd <command>
#  Exits with error if the command is not available.
# ---------------------------------------------------------------------------
require_cmd() {
    local cmd="$1"
    if ! command -v "$cmd" &>/dev/null; then
        log_error "Required command not found: $cmd"
        return 1
    fi
}

# ---------------------------------------------------------------------------
#  ensure_workdir <path>
#  Creates the directory if needed, then cd into it.
# ---------------------------------------------------------------------------
ensure_workdir() {
    local WORKDIR="$1"
    if [ -z "$WORKDIR" ]; then
        log_error "Workdir is not set"
        return 1
    fi
    if [ ! -d "$WORKDIR" ]; then
        log_info "Workdir($WORKDIR) not found, creating..."
        mkdir -p "$WORKDIR" || { log_error "Failed to create $WORKDIR"; return 1; }
    fi
    cd "$WORKDIR" || { log_error "Failed to cd into $WORKDIR"; return 1; }
    log_info "Working directory: $(pwd)"
}

# ---------------------------------------------------------------------------
#  check_gpu
#  Detects NVIDIA GPUs via nvidia-smi; returns 1 if none found.
# ---------------------------------------------------------------------------
check_gpu() {
    if ! command -v nvidia-smi &>/dev/null; then
        log_warn "nvidia-smi not found — no NVIDIA GPU driver detected"
        return 1
    fi
    local gpu_count
    gpu_count=$(nvidia-smi -L 2>/dev/null | wc -l)
    if [ "$gpu_count" -eq 0 ]; then
        log_warn "No NVIDIA GPU detected"
        return 1
    fi
    log_info "Detected $gpu_count GPU(s):"
    nvidia-smi -L
}

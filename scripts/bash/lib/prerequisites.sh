#!/bin/bash
# ===========================================================================
#  prerequisites.sh — System prerequisites checking
#
#  Functions: check_prerequisites, ensure_git_lfs, detect_cuda_version,
#             check_disk_space, check_gpu_memory
#  Depends:   lib/logging.sh, lib/utils.sh
# ===========================================================================

[[ -n "${_LIB_PREREQUISITES_LOADED:-}" ]] && return 0
_LIB_PREREQUISITES_LOADED=1

_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_LIB_DIR}/logging.sh"
source "${_LIB_DIR}/utils.sh"

# ---------------------------------------------------------------------------
#  detect_cuda_version
#  Echoes the CUDA toolkit version (e.g. "12.6").  Returns 1 if not found.
# ---------------------------------------------------------------------------
detect_cuda_version() {
    local cuda_ver=""

    if command -v nvcc &>/dev/null; then
        cuda_ver=$(nvcc --version 2>/dev/null \
            | grep -oP 'release \K[0-9]+\.[0-9]+' | head -1)
    fi

    if [ -z "$cuda_ver" ] && command -v nvidia-smi &>/dev/null; then
        cuda_ver=$(nvidia-smi 2>/dev/null \
            | grep -oP 'CUDA Version:\s*\K[0-9]+\.[0-9]+' | head -1)
    fi

    if [ -z "$cuda_ver" ]; then
        return 1
    fi
    echo "$cuda_ver"
}

# ---------------------------------------------------------------------------
#  cuda_to_torch_tag <cuda_version>
#  Maps a CUDA version to the PyTorch index URL tag (e.g. "12.4" → "cu124").
#  Pins to the nearest PyTorch-supported CUDA build.
# ---------------------------------------------------------------------------
cuda_to_torch_tag() {
    local cuda_ver="$1"
    local major minor
    major=$(echo "$cuda_ver" | cut -d. -f1)
    minor=$(echo "$cuda_ver" | cut -d. -f2)

    if [ "$major" -le 11 ]; then
        echo "cu118"
    elif [ "$major" -eq 12 ]; then
        if   [ "$minor" -le 1 ]; then echo "cu121"
        elif [ "$minor" -le 4 ]; then echo "cu124"
        else                          echo "cu126"
        fi
    else
        echo "cu${major}${minor}"
    fi
}

# ---------------------------------------------------------------------------
#  ensure_git_lfs
#  Installs git-lfs if missing, then runs `git lfs install`.
# ---------------------------------------------------------------------------
ensure_git_lfs() {
    if command -v git-lfs &>/dev/null; then
        git lfs install --skip-repo &>/dev/null
        return 0
    fi

    log_step "Installing git-lfs..."

    if command -v apt-get &>/dev/null; then
        sudo apt-get update -qq && sudo apt-get install -y -qq git-lfs
    elif command -v yum &>/dev/null; then
        sudo yum install -y git-lfs
    elif command -v dnf &>/dev/null; then
        sudo dnf install -y git-lfs
    elif command -v pacman &>/dev/null; then
        sudo pacman -S --noconfirm git-lfs
    elif command -v brew &>/dev/null; then
        brew install git-lfs
    else
        log_error "Cannot auto-install git-lfs: no supported package manager found"
        log_error "Please install git-lfs manually: https://git-lfs.com"
        return 1
    fi

    if ! command -v git-lfs &>/dev/null; then
        log_error "git-lfs installation failed"
        return 1
    fi

    git lfs install --skip-repo &>/dev/null
    log_info "git-lfs installed"
}

# ---------------------------------------------------------------------------
#  check_disk_space <path> <required_gb>
#  Returns 1 if available space is below the threshold.
# ---------------------------------------------------------------------------
check_disk_space() {
    local path="${1:-.}"
    local required_gb="${2:-20}"

    local available_gb
    available_gb=$(df -BG "$path" 2>/dev/null \
        | awk 'NR==2 {gsub(/G/,"",$4); print $4}')

    if [ -z "$available_gb" ]; then
        log_warn "Could not determine disk space at $path"
        return 0
    fi

    if [ "$available_gb" -lt "$required_gb" ]; then
        log_error "Disk space insufficient: ${available_gb}GB available, need ${required_gb}GB (at $path)"
        return 1
    fi
    log_info "Disk space: ${available_gb}GB available (need ${required_gb}GB)"
}

# ---------------------------------------------------------------------------
#  check_gpu_memory [required_mb]
#  Warns if GPU memory is below the threshold (default 8 GB).
# ---------------------------------------------------------------------------
check_gpu_memory() {
    local required_mb="${1:-8000}"

    if ! command -v nvidia-smi &>/dev/null; then
        return 0
    fi

    local total_mb
    total_mb=$(nvidia-smi --query-gpu=memory.total \
        --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')

    if [ -z "$total_mb" ]; then
        log_warn "Could not determine GPU memory"
        return 0
    fi

    if [ "$total_mb" -lt "$required_mb" ]; then
        log_warn "GPU memory: ${total_mb}MB (recommended: ${required_mb}MB+)"
        log_warn "Small GPU may limit batch size and performance"
    else
        log_info "GPU memory: ${total_mb}MB"
    fi
}

# ---------------------------------------------------------------------------
#  check_prerequisites
#  Runs all prerequisite checks.  Returns 1 on hard failure.
# ---------------------------------------------------------------------------
check_prerequisites() {
    local errors=0

    log_step "Checking system prerequisites..."

    for cmd in git curl python3; do
        if command -v "$cmd" &>/dev/null; then
            log_info "$cmd: $(command -v "$cmd")"
        else
            log_error "$cmd not found (required)"
            errors=$((errors + 1))
        fi
    done

    if check_gpu; then
        check_gpu_memory 8000
        local cuda_ver
        if cuda_ver=$(detect_cuda_version); then
            log_info "CUDA version: $cuda_ver (PyTorch tag: $(cuda_to_torch_tag "$cuda_ver"))"
        else
            log_warn "CUDA toolkit not detected (nvcc not found)"
            log_warn "GPU inference requires CUDA — install the CUDA toolkit"
        fi
    else
        log_warn "No NVIDIA GPU detected — inference will be CPU-only"
    fi

    ensure_git_lfs || errors=$((errors + 1))

    check_disk_space "." 20 || errors=$((errors + 1))

    if [ "$errors" -gt 0 ]; then
        log_error "Prerequisites check failed ($errors error(s))"
        return 1
    fi

    log_info "All prerequisites satisfied"
}

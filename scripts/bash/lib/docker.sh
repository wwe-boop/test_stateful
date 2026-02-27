#!/bin/bash
# ===========================================================================
#  docker.sh — Docker / NGC container helpers
#
#  Functions: detect_driver_version, resolve_ngc_image, ensure_ngc_image,
#             check_docker_gpu_ready, print_container_recommendation
#  Depends:   lib/logging.sh, lib/utils.sh
#
#  NGC compatibility matrix sourced from:
#  https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/
#    docs/introduction/compatibility.html
# ===========================================================================

[[ -n "${_LIB_DOCKER_LOADED:-}" ]] && return 0
_LIB_DOCKER_LOADED=1

_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_LIB_DIR}/logging.sh"
source "${_LIB_DIR}/utils.sh"

# NGC trtllm-python-py3 images: tag  min_driver  trtllm_version  cuda_version
# Sorted newest-first so the first compatible match is the best.
# Update this table when new NGC releases come out.
_NGC_TRTLLM_MATRIX=(
    "25.08  575.51  0.21.0  12.9"
    "25.07  575.51  0.20.0  12.9"
    "25.06  575.51  0.20.0  12.9"
    "25.05  570.124 0.19.0  12.8"
    "25.04  570.124 0.18.2  12.8"
    "25.03  570.124 0.18.0  12.8"
    "25.02  570.86  0.17.0  12.8"
    "25.01  570.86  0.17.0  12.8"
    "24.12  560.35  0.16.0  12.6"
    "24.11  555.42  0.15.0  12.6"
    "24.10  555.42  0.14.0  12.5"
    "24.09  555.42  0.13.0  12.5"
    "24.08  555.42  0.12.0  12.5"
    "24.07  550.54  0.11.0  12.4"
)

# NGC Triton full image (for deployment, same driver requirements)
_NGC_TRITON_DEPLOY_SUFFIX="-trtllm-python-py3"
_NGC_TRITON_BASE="nvcr.io/nvidia/tritonserver"

# ---------------------------------------------------------------------------
#  detect_driver_version
#  Echoes the NVIDIA driver version (e.g. "570.86.10").  Returns 1 if N/A.
# ---------------------------------------------------------------------------
detect_driver_version() {
    local driver_ver=""

    if command -v nvidia-smi &>/dev/null; then
        driver_ver=$(nvidia-smi --query-gpu=driver_version \
            --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    fi

    if [ -z "$driver_ver" ]; then
        return 1
    fi
    echo "$driver_ver"
}

# ---------------------------------------------------------------------------
#  _driver_ge <installed_version> <required_version>
#  Returns 0 if the installed driver >= required driver.
#  Compares major.minor numerically (ignores patch, e.g. 570.86.10 → 570.86).
# ---------------------------------------------------------------------------
_driver_ge() {
    local installed="$1"
    local required="$2"

    local inst_major inst_minor req_major req_minor
    inst_major=$(echo "$installed" | cut -d. -f1)
    inst_minor=$(echo "$installed" | cut -d. -f2)
    req_major=$(echo "$required" | cut -d. -f1)
    req_minor=$(echo "$required" | cut -d. -f2)

    inst_minor="${inst_minor:-0}"
    req_minor="${req_minor:-0}"

    if [ "$inst_major" -gt "$req_major" ]; then
        return 0
    elif [ "$inst_major" -eq "$req_major" ] && [ "$inst_minor" -ge "$req_minor" ]; then
        return 0
    fi
    return 1
}

# ---------------------------------------------------------------------------
#  resolve_ngc_image [driver_version]
#  Selects the newest compatible NGC trtllm container tag for the given
#  NVIDIA driver version.  If driver_version is omitted, auto-detects.
#
#  Echoes the full image URI (e.g. nvcr.io/nvidia/tritonserver:25.05-trtllm-python-py3)
#  Returns 1 if no compatible image found.
# ---------------------------------------------------------------------------
resolve_ngc_image() {
    local driver_ver="${1:-}"

    if [ -z "$driver_ver" ]; then
        driver_ver=$(detect_driver_version) \
            || { log_error "Cannot detect NVIDIA driver version"; return 1; }
    fi

    local tag min_drv trtllm_ver cuda_ver
    for entry in "${_NGC_TRTLLM_MATRIX[@]}"; do
        read -r tag min_drv trtllm_ver cuda_ver <<< "$entry"
        if _driver_ge "$driver_ver" "$min_drv"; then
            echo "${_NGC_TRITON_BASE}:${tag}${_NGC_TRITON_DEPLOY_SUFFIX}"
            return 0
        fi
    done

    log_error "Driver $driver_ver too old — oldest supported container (24.07) needs >= 550.54"
    log_error "Update your NVIDIA driver: https://www.nvidia.com/drivers"
    return 1
}

# ---------------------------------------------------------------------------
#  resolve_ngc_image_info [driver_version]
#  Like resolve_ngc_image but also prints the tag, TRT-LLM version, and
#  CUDA version to stderr for informational logging.
#  Echoes the full image URI to stdout.
# ---------------------------------------------------------------------------
resolve_ngc_image_info() {
    local driver_ver="${1:-}"

    if [ -z "$driver_ver" ]; then
        driver_ver=$(detect_driver_version) \
            || { log_error "Cannot detect NVIDIA driver version"; return 1; }
    fi

    local tag min_drv trtllm_ver cuda_ver
    for entry in "${_NGC_TRTLLM_MATRIX[@]}"; do
        read -r tag min_drv trtllm_ver cuda_ver <<< "$entry"
        if _driver_ge "$driver_ver" "$min_drv"; then
            log_info "Driver $driver_ver >= $min_drv → NGC tag $tag (TRT-LLM $trtllm_ver, CUDA $cuda_ver)"
            echo "${_NGC_TRITON_BASE}:${tag}${_NGC_TRITON_DEPLOY_SUFFIX}"
            return 0
        fi
    done

    log_error "Driver $driver_ver too old — oldest supported container (24.07) needs >= 550.54"
    log_error "Update your NVIDIA driver: https://www.nvidia.com/drivers"
    return 1
}

# ---------------------------------------------------------------------------
#  ensure_ngc_image <image>
#  Pulls the NGC image if not already present locally.
# ---------------------------------------------------------------------------
ensure_ngc_image() {
    local image="$1"

    if docker image inspect "$image" &>/dev/null; then
        log_info "Image already present: $image"
        return 0
    fi

    log_step "Pulling NGC image: $image"
    log_info "This may take a while (15-30 GB) ..."

    if retry 2 10 docker pull "$image"; then
        log_info "Image pulled: $image"
    else
        log_error "Failed to pull image: $image"
        log_error "Check network connectivity and NGC credentials"
        return 1
    fi
}

# ---------------------------------------------------------------------------
#  check_docker_gpu_ready
#  Verifies Docker + NVIDIA Container Toolkit + GPU access.
#  Returns 1 on hard failure, 0 on success (with warnings for soft issues).
# ---------------------------------------------------------------------------
check_docker_gpu_ready() {
    if ! command -v docker &>/dev/null; then
        log_error "Docker not found"
        log_error "Install: https://docs.docker.com/engine/install/"
        return 1
    fi

    if ! docker info &>/dev/null 2>&1; then
        log_error "Docker daemon not running or insufficient permissions"
        log_error "Try: sudo systemctl start docker && sudo usermod -aG docker \$USER"
        return 1
    fi

    local gpu_ok=false
    if docker info 2>/dev/null | grep -qi 'nvidia'; then
        gpu_ok=true
    elif command -v nvidia-container-cli &>/dev/null; then
        gpu_ok=true
    elif [ -f /etc/nvidia-container-runtime/config.toml ]; then
        gpu_ok=true
    fi

    if ! $gpu_ok; then
        log_error "NVIDIA Container Toolkit not detected"
        log_error "Install: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
        return 1
    fi

    log_info "Docker + NVIDIA GPU runtime: OK"
    return 0
}

# ---------------------------------------------------------------------------
#  print_container_recommendation
#  Detects the driver, resolves the best NGC image, and prints a
#  human-readable recommendation.  Non-fatal: only warns on failure.
#  Intended to be called during Phase A prerequisites check.
# ---------------------------------------------------------------------------
print_container_recommendation() {
    local driver_ver
    if ! driver_ver=$(detect_driver_version); then
        log_warn "Cannot detect NVIDIA driver — skipping container recommendation"
        return 0
    fi

    log_info "NVIDIA driver: $driver_ver"

    local image
    if image=$(resolve_ngc_image_info "$driver_ver"); then
        log_info "Recommended TRT-LLM build container: $image"
        log_info "Phase B: bash scripts/bash/build_engines.sh"
    else
        log_warn "No compatible NGC TRT-LLM container for driver $driver_ver"
        log_warn "Upgrade your NVIDIA driver to at least 550.54 for container support"
    fi
}

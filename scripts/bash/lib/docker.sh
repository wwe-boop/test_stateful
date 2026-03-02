#!/bin/bash
# ===========================================================================
#  docker.sh — Docker / NGC container helpers
#
#  Functions: detect_driver_version, detect_gpu_compute_cap,
#             resolve_ngc_image, ensure_ngc_image,
#             check_docker_gpu_ready, print_container_recommendation,
#             resolve_ngc_tag, resolve_ngc_python_version,
#             resolve_ngc_entry, build_combined_triton_image,
#             ensure_combined_triton_image, _check_ngc_manifest
#  Depends:   lib/logging.sh, lib/utils.sh
#
#  NGC compatibility matrix loaded from scripts/bash/ngc_matrix.conf
#  (falls back to built-in defaults if the file is missing).
#  Update via: bash scripts/bash/autorun.sh update-matrix
#
#  Container strategy:
#    Phase B (engine build):  trtllm-python-py3  (unmodified NGC image)
#    Phase C (Triton deploy): combined image      (trtllm + onnxruntime backend)
#
#  The combined image is built locally via multi-stage Dockerfile, extracting
#  only /opt/tritonserver/backends/onnxruntime from the full py3 image.
#  This adds ~500 MB to the trtllm base (~17 GB), while py3 shares most
#  Docker layers so incremental download is ~3-5 GB.
# ===========================================================================

[[ -n "${_LIB_DOCKER_LOADED:-}" ]] && return 0
_LIB_DOCKER_LOADED=1

_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_LIB_DIR}/logging.sh"
source "${_LIB_DIR}/utils.sh"

# NGC image base and suffixes
_NGC_TRITON_BASE="nvcr.io/nvidia/tritonserver"
_NGC_TRTLLM_SUFFIX="-trtllm-python-py3"
_NGC_FULL_SUFFIX="-py3"

# Combined image tag prefix (locally built: trtllm + onnxruntime backend)
_COMBINED_IMAGE_NAME="qwen3-tts-triton-base"

# Qwen3 project minimum requirements (QK-Norm needs TRT-LLM >= 1.0.0)
_QWEN3_MIN_TRTLLM="1.0.0"
_QWEN3_MIN_DRIVER="575.57"

# ---------------------------------------------------------------------------
#  _load_ngc_matrix
#  Populates _NGC_TRTLLM_MATRIX from ngc_matrix.conf.
#  Falls back to a minimal built-in default if the file is missing.
#  Each entry: "tag  min_driver  trtllm_version  cuda_version  python_version  size_gb"
# ---------------------------------------------------------------------------
_NGC_TRTLLM_MATRIX=()

_load_ngc_matrix() {
    [[ ${#_NGC_TRTLLM_MATRIX[@]} -gt 0 ]] && return 0

    local conf_path="${_LIB_DIR}/../ngc_matrix.conf"

    if [ -f "$conf_path" ]; then
        while IFS= read -r line; do
            local stripped="${line##"${line%%[![:space:]]*}"}"
            [[ -z "$stripped" || "$stripped" == \#* ]] && continue
            _NGC_TRTLLM_MATRIX+=("$stripped")
        done < "$conf_path"
    fi

    if [ ${#_NGC_TRTLLM_MATRIX[@]} -eq 0 ]; then
        log_warn "ngc_matrix.conf not found or empty — using built-in fallback"
        _NGC_TRTLLM_MATRIX=(
            "25.11  580.95  1.0.3  13.0  3.12  12.25"
            "25.05  570.124 0.19.0 12.8  3.12  17.0"
            "24.07  550.54  0.11.0 12.4  3.10  23.0"
        )
    fi
}

_load_ngc_matrix

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
#  detect_gpu_compute_cap
#  Returns the compute capability of the first GPU (e.g. "8.0", "8.9").
#  Returns 1 if nvidia-smi is unavailable or no GPU found.
# ---------------------------------------------------------------------------
detect_gpu_compute_cap() {
    if ! command -v nvidia-smi &>/dev/null; then return 1; fi
    nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null \
        | head -1 | tr -d ' '
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
#  _version_ge <installed> <required>
#  Semantic version comparison (major.minor.patch).  Returns 0 if installed
#  >= required.  Missing components default to 0.
# ---------------------------------------------------------------------------
_version_ge() {
    local inst="$1" req="$2"
    local i1 i2 i3 r1 r2 r3
    IFS='.' read -r i1 i2 i3 <<< "$inst"
    IFS='.' read -r r1 r2 r3 <<< "$req"
    i1="${i1:-0}"; i2="${i2:-0}"; i3="${i3:-0}"
    r1="${r1:-0}"; r2="${r2:-0}"; r3="${r3:-0}"

    if   [ "$i1" -gt "$r1" ]; then return 0
    elif [ "$i1" -lt "$r1" ]; then return 1
    elif [ "$i2" -gt "$r2" ]; then return 0
    elif [ "$i2" -lt "$r2" ]; then return 1
    elif [ "$i3" -ge "$r3" ]; then return 0
    fi
    return 1
}

# ---------------------------------------------------------------------------
#  _check_ngc_manifest <image>
#  Returns 0 if the Docker registry has a manifest for <image>, 1 otherwise.
#  Used to skip matrix entries for containers not yet published.
# ---------------------------------------------------------------------------
_check_ngc_manifest() {
    local image="$1"
    docker manifest inspect "$image" > /dev/null 2>&1
}

# ---------------------------------------------------------------------------
#  _resolve_best_entry [driver_version]
#  Core resolver: finds the newest NGC matrix entry that satisfies both
#  driver compatibility AND the project minimum TRT-LLM version.
#
#  When _NGC_VERIFY_MANIFEST=1, each candidate is checked against the
#  Docker registry via _check_ngc_manifest.  Entries whose manifest is
#  unavailable (not yet released) are skipped with a warning.
#
#  On success, echoes: "tag min_driver trtllm_version cuda_version python_version size_gb"
#  On failure, prints targeted error messages and returns 1.
# ---------------------------------------------------------------------------
_resolve_best_entry() {
    local driver_ver="${1:-${TARGET_DRIVER:-}}"

    if [ -z "$driver_ver" ]; then
        driver_ver=$(detect_driver_version) \
            || { log_error "Cannot detect NVIDIA driver version"; return 1; }
    elif [ -n "${TARGET_DRIVER:-}" ] && [ -z "${1:-}" ]; then
        log_info "Using target driver version: $driver_ver (override via TARGET_DRIVER)"
    fi

    local tag min_drv trtllm_ver cuda_ver py_ver size_gb
    local best_driver_match=""

    for entry in "${_NGC_TRTLLM_MATRIX[@]}"; do
        read -r tag min_drv trtllm_ver cuda_ver py_ver size_gb <<< "$entry"
        if _driver_ge "$driver_ver" "$min_drv"; then
            if [ -z "$best_driver_match" ]; then
                best_driver_match="$tag $min_drv $trtllm_ver $cuda_ver"
            fi
            if _version_ge "$trtllm_ver" "$_QWEN3_MIN_TRTLLM"; then
                if [ "${_NGC_VERIFY_MANIFEST:-0}" = "1" ]; then
                    local _img="${_NGC_TRITON_BASE}:${tag}${_NGC_TRTLLM_SUFFIX}"
                    if docker image inspect "$_img" &>/dev/null; then
                        : # image already local — skip remote manifest check
                    elif ! _check_ngc_manifest "$_img"; then
                        log_warn "NGC $tag not yet available on registry, trying next..."
                        continue
                    fi
                fi
                echo "$tag $min_drv $trtllm_ver $cuda_ver ${py_ver:-3.12} ${size_gb:--}"
                return 0
            fi
        fi
    done

    if [ -n "$best_driver_match" ]; then
        local _bt _bd _bv _bc
        read -r _bt _bd _bv _bc <<< "$best_driver_match"
        log_error "Driver $driver_ver supports NGC $_bt (TRT-LLM $_bv), but Qwen3 requires TRT-LLM >= $_QWEN3_MIN_TRTLLM"
        log_error "Qwen3 QK-Norm requires TRT-LLM >= $_QWEN3_MIN_TRTLLM (NGC 25.09+, driver >= $_QWEN3_MIN_DRIVER)"
        log_error "Upgrade your NVIDIA driver to >= $_QWEN3_MIN_DRIVER: https://www.nvidia.com/drivers"
    else
        log_error "Driver $driver_ver too old — no compatible NGC container found"
        log_error "Minimum driver for Qwen3: >= $_QWEN3_MIN_DRIVER"
        log_error "Update your NVIDIA driver: https://www.nvidia.com/drivers"
    fi
    return 1
}

# ---------------------------------------------------------------------------
#  resolve_ngc_image [driver_version]
#  Selects the newest compatible NGC trtllm container tag for the given
#  NVIDIA driver version.  If driver_version is omitted, auto-detects.
#  Rejects containers with TRT-LLM below the Qwen3 minimum.
#
#  Echoes the full image URI (e.g. nvcr.io/nvidia/tritonserver:25.09-trtllm-python-py3)
#  Returns 1 if no compatible image found.
# ---------------------------------------------------------------------------
resolve_ngc_image() {
    local entry
    entry=$(_resolve_best_entry "$@") || return 1

    local tag _rest
    read -r tag _rest <<< "$entry"
    echo "${_NGC_TRITON_BASE}:${tag}${_NGC_TRTLLM_SUFFIX}"
}

# ---------------------------------------------------------------------------
#  resolve_ngc_image_info [driver_version]
#  Like resolve_ngc_image but also prints the tag, TRT-LLM version, and
#  CUDA version to stderr for informational logging.
#  Echoes the full image URI to stdout.
# ---------------------------------------------------------------------------
resolve_ngc_image_info() {
    local entry
    entry=$(_resolve_best_entry "$@") || return 1

    local tag min_drv trtllm_ver cuda_ver py_ver size_gb
    read -r tag min_drv trtllm_ver cuda_ver py_ver size_gb <<< "$entry"

    local driver_ver="${1:-}"
    if [ -z "$driver_ver" ]; then
        driver_ver=$(detect_driver_version 2>/dev/null) || true
    fi
    log_info "Driver ${driver_ver:-?} >= $min_drv → NGC tag $tag (TRT-LLM $trtllm_ver, CUDA $cuda_ver, Python ${py_ver:-?})"
    echo "${_NGC_TRITON_BASE}:${tag}${_NGC_TRTLLM_SUFFIX}"
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

    if ! docker info &>/dev/null; then
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
#  resolve_ngc_tag [driver_version]
#  Like resolve_ngc_image but echoes only the NGC tag (e.g. "25.09"),
#  not the full image URI.  Used to construct image names for both
#  trtllm and py3 variants.
# ---------------------------------------------------------------------------
resolve_ngc_tag() {
    local entry
    entry=$(_resolve_best_entry "$@") || return 1

    local tag _rest
    read -r tag _rest <<< "$entry"
    echo "$tag"
}

# ---------------------------------------------------------------------------
#  resolve_ngc_entry [driver_version]
#  Returns the full matrix entry (all 6 fields) for the best compatible
#  NGC container.  Useful for callers that need python_version or size.
#  Output: "tag  min_driver  trtllm_version  cuda_version  python_version  size_gb"
# ---------------------------------------------------------------------------
resolve_ngc_entry() {
    _resolve_best_entry "$@"
}

# ---------------------------------------------------------------------------
#  resolve_ngc_python_version [driver_version]
#  Returns the Python version used by the best compatible NGC container.
#  Falls back to "3.12" if the field is missing.
# ---------------------------------------------------------------------------
resolve_ngc_python_version() {
    local entry
    entry=$(_resolve_best_entry "$@") || return 1

    local tag min_drv trtllm_ver cuda_ver py_ver size_gb
    read -r tag min_drv trtllm_ver cuda_ver py_ver size_gb <<< "$entry"

    echo "${py_ver:-3.12}"
}

# ---------------------------------------------------------------------------
#  build_combined_triton_image [ngc_tag]
#
#  Builds a combined Docker image that merges:
#    - trtllm-python-py3  (TRT-LLM + Python backends)
#    - py3                (source for ONNX Runtime backend)
#
#  The result is tagged as: qwen3-tts-triton-base:<ngc_tag>
#
#  Both source images are pulled if not present.  Since they share the
#  same CUDA/Ubuntu base layers, incremental download for py3 is ~3-5 GB.
#  The final image adds ~500 MB over the trtllm base.
# ---------------------------------------------------------------------------
build_combined_triton_image() {
    local ngc_tag="${1:-}"

    if [ -z "$ngc_tag" ]; then
        ngc_tag=$(resolve_ngc_tag) \
            || { log_error "Cannot determine NGC tag"; return 1; }
    fi

    local trtllm_image="${_NGC_TRITON_BASE}:${ngc_tag}${_NGC_TRTLLM_SUFFIX}"
    local full_image="${_NGC_TRITON_BASE}:${ngc_tag}${_NGC_FULL_SUFFIX}"
    local combined_tag="${_COMBINED_IMAGE_NAME}:${ngc_tag}"

    # Already built?
    if docker image inspect "$combined_tag" &>/dev/null; then
        log_info "Combined image already exists: $combined_tag"
        echo "$combined_tag"
        return 0
    fi

    log_step "Building combined Triton image (TRT-LLM + ONNX Runtime)"
    log_info "  TRT-LLM base: $trtllm_image"
    log_info "  ORT source:   $full_image"
    log_info "  Output tag:   $combined_tag"

    # Pull source images
    ensure_ngc_image "$trtllm_image" || return 1

    log_info "Pulling py3 image for ONNX Runtime backend (shared layers = fast) ..."
    ensure_ngc_image "$full_image" || return 1

    # Build via inline Dockerfile (empty context dir to avoid sending workspace)
    log_info "Extracting ONNX Runtime backend into combined image ..."

    local build_ctx
    build_ctx=$(mktemp -d)

    if ! docker build -t "$combined_tag" -f - "$build_ctx" <<DOCKERFILE
FROM ${full_image} AS ort_source
FROM ${trtllm_image}
COPY --from=ort_source /opt/tritonserver/backends/onnxruntime /opt/tritonserver/backends/onnxruntime
DOCKERFILE
    then
        rm -rf "$build_ctx"
        log_error "Failed to build combined image"
        return 1
    fi
    rm -rf "$build_ctx"

    local size
    size=$(docker image inspect "$combined_tag" --format '{{.Size}}' 2>/dev/null)
    if [ -n "$size" ]; then
        local size_gb
        size_gb=$(awk -v s="$size" 'BEGIN {printf "%.1f", s/1073741824}')
        log_info "Combined image ready: $combined_tag (${size_gb} GB)"
    else
        log_info "Combined image ready: $combined_tag"
    fi

    echo "$combined_tag"
}

# ---------------------------------------------------------------------------
#  ensure_combined_triton_image [ngc_tag]
#  Returns the combined image tag, building it if needed.
#  Echoes the full image tag to stdout.
# ---------------------------------------------------------------------------
ensure_combined_triton_image() {
    local ngc_tag="${1:-}"

    if [ -z "$ngc_tag" ]; then
        ngc_tag=$(resolve_ngc_tag) \
            || { log_error "Cannot determine NGC tag"; return 1; }
    fi

    local combined_tag="${_COMBINED_IMAGE_NAME}:${ngc_tag}"

    if docker image inspect "$combined_tag" &>/dev/null; then
        log_info "Combined image already exists: $combined_tag"
        echo "$combined_tag"
        return 0
    fi

    build_combined_triton_image "$ngc_tag"
}

# ---------------------------------------------------------------------------
#  print_container_recommendation
#  Detects the driver, resolves the best NGC image, and prints a
#  human-readable recommendation.  Non-fatal: only warns on failure.
#  Intended to be called during Phase A prerequisites check.
# ---------------------------------------------------------------------------
print_container_recommendation() {
    local driver_ver
    if [ -n "${TARGET_DRIVER:-}" ]; then
        driver_ver="$TARGET_DRIVER"
        log_info "Target driver (override): $driver_ver"
    elif ! driver_ver=$(detect_driver_version); then
        log_warn "Cannot detect NVIDIA driver — skipping container recommendation"
        return 0
    else
        log_info "NVIDIA driver: $driver_ver"
    fi

    local image
    if image=$(resolve_ngc_image_info "$driver_ver"); then
        log_info "Recommended container (Phase B engine build): $image"
        log_info "Triton deploy image (Phase C) will add ONNX Runtime backend automatically."
        log_info "Phase B: bash scripts/bash/build_engines.sh"
    else
        log_warn "No compatible NGC container for driver $driver_ver (Qwen3 needs TRT-LLM >= $_QWEN3_MIN_TRTLLM)"
        log_warn "Upgrade your NVIDIA driver to >= $_QWEN3_MIN_DRIVER: https://www.nvidia.com/drivers"
    fi
}

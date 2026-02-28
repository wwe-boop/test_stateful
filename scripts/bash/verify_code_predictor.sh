#!/bin/bash
# ===========================================================================
#  verify_code_predictor.sh — Verify Code Predictor ONNX → TRT compilation
#
#  Runs the §5.5 verification checklist inside an NGC TensorRT container:
#    1. ONNX → TRT compilation (unrolled + single-stage)
#    2. Accuracy comparison (TRT vs ONNX Runtime)
#    3. Performance benchmark (B=1, B=8)
#
#  Prerequisites:
#    - Code Predictor ONNX files at workspace/exported/<variant>/
#      (produced by export_models.sh or export_05_code_predictor.py)
#    - Docker with NVIDIA Container Toolkit
#
#  Usage:
#    bash scripts/bash/verify_code_predictor.sh                      # all variants, both modes
#    bash scripts/bash/verify_code_predictor.sh --variant design-1.7b
#    bash scripts/bash/verify_code_predictor.sh --mode unrolled      # unrolled only
#    bash scripts/bash/verify_code_predictor.sh --fp32               # FP32 precision
#    bash scripts/bash/verify_code_predictor.sh --dry-run
#
#  Output: workspace/exported/<variant>/code_predictor_*.plan
#          workspace/exported/<variant>/trt_verification_report.json
# ===========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
source "${SCRIPT_DIR}/tools.sh"

EXPORTED_DIR="${REPO_ROOT}/workspace/exported"
SCRIPTS_DIR="${REPO_ROOT}/scripts/python"
VARIANT=""
MODE="both"
PRECISION_FLAG="--bf16"
BATCH_SIZES="1,8"
DRY_RUN=false
USER_IMAGE="${TRTLLM_IMAGE:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --variant)      VARIANT="$2"; shift 2 ;;
        --mode)         MODE="$2"; shift 2 ;;
        --image)        USER_IMAGE="$2"; shift 2 ;;
        --fp32)         PRECISION_FLAG="--fp32"; shift ;;
        --batch-sizes)  BATCH_SIZES="$2"; shift 2 ;;
        --dry-run)      DRY_RUN=true; shift ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --variant <name>       Verify a specific model variant"
            echo "  --mode <mode>          unrolled | single_stage | both (default: both)"
            echo "  --image <uri>          Override NGC container image"
            echo "  --fp32                 Build FP32 engines instead of BF16"
            echo "  --batch-sizes <list>   Comma-separated batch sizes (default: 1,8)"
            echo "  --dry-run              Show docker command without executing"
            echo "  -h, --help             Show this help"
            exit 0
            ;;
        *) log_error "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Preflight ──
log_step "Code Predictor TRT Verification (§5.5)"

check_docker_gpu_ready || exit 1

# ── Resolve container image ──
if [ -n "$USER_IMAGE" ]; then
    TRTLLM_IMAGE="$USER_IMAGE"
    log_info "Using user-specified image: $TRTLLM_IMAGE"
else
    log_info "Auto-detecting best NGC container ..."
    _NGC_VERIFY_MANIFEST=1
    TRTLLM_IMAGE=$(resolve_ngc_image_info) \
        || { log_error "Cannot determine compatible container. Use --image to specify manually."; exit 1; }
fi

ensure_ngc_image "$TRTLLM_IMAGE" || exit 1

# ── Discover variants ──
discover_cp_variants() {
    local found=()
    for d in "$EXPORTED_DIR"/*/; do
        local vname
        vname=$(basename "$d")
        if [ -f "$d/code_predictor_unrolled.onnx" ] || \
           [ -f "$d/code_predictor_single_stage.onnx" ]; then
            found+=("$vname")
        fi
    done
    echo "${found[@]}"
}

if [ -n "$VARIANT" ]; then
    VARIANTS=("$VARIANT")
else
    read -ra VARIANTS <<< "$(discover_cp_variants)"
    if [ ${#VARIANTS[@]} -eq 0 ]; then
        log_error "No Code Predictor ONNX files found in $EXPORTED_DIR/"
        log_error "Run export_models.sh first (Phase A)."
        exit 1
    fi
    log_info "Discovered variants: ${VARIANTS[*]}"
fi

# ── Run verification for each variant ──
verify_variant() {
    local variant="$1"
    local variant_dir="$EXPORTED_DIR/$variant"

    log_step "Verifying: $variant (mode=$MODE)"

    local docker_cmd=(
        docker run --rm --gpus all
        -v "$variant_dir:/mnt/model"
        -v "$SCRIPTS_DIR:/mnt/scripts:ro"
        "$TRTLLM_IMAGE"
        bash -c "pip install -q --index-url https://mirrors.bfsu.edu.cn/pypi/web/simple onnxruntime 2>&1 | tail -3; python3 /mnt/scripts/verify_code_predictor_trt.py --model-dir /mnt/model --mode $MODE $PRECISION_FLAG --batch-sizes $BATCH_SIZES"
    )

    if $DRY_RUN; then
        log_info "[DRY RUN] Would execute:"
        log_info "  ${docker_cmd[*]}"
        return 0
    fi

    if "${docker_cmd[@]}"; then
        log_info "Verification PASSED: $variant"
        return 0
    else
        log_error "Verification FAILED: $variant"
        return 1
    fi
}

FAILED=0
SUCCEEDED=0

for variant in "${VARIANTS[@]}"; do
    if verify_variant "$variant"; then
        SUCCEEDED=$((SUCCEEDED + 1))
    else
        FAILED=$((FAILED + 1))
    fi
done

# ── Summary ──
echo ""
log_step "Verification Summary"
log_info "  Image:     $TRTLLM_IMAGE"
log_info "  Mode:      $MODE"
log_info "  Succeeded: $SUCCEEDED"
if [ "$FAILED" -gt 0 ]; then
    log_error "  Failed: $FAILED"
    log_info ""
    log_info "If unrolled build failed, try single-stage fallback (§5.6):"
    log_info "  bash $0 --mode single_stage"
fi

if [ "$SUCCEEDED" -gt 0 ] && ! $DRY_RUN; then
    log_info ""
    log_info "Reports: $EXPORTED_DIR/*/trt_verification_report.json"
    log_info "Engines: $EXPORTED_DIR/*/code_predictor_*.plan"
fi

exit "$FAILED"

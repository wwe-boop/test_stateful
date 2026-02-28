#!/bin/bash
# ===========================================================================
#  verify_talker_trtllm.sh — Verify Talker Backbone TRT-LLM engine (Phase 2, Item 9)
#
#  Two-step flow:
#    1. Host: Generate PyTorch reference (verify_talker_trtllm_ref.py) → .npz
#    2. Container: Run TRT-LLM engine vs reference (verify_talker_trtllm.py) → report
#
#  Prerequisites:
#    - TRT-LLM engine at workspace/exported/<variant>/trtllm_engine/
#    - For step 1: conda env qwen3-tts (or venv) with qwen_tts; workspace/exported/<variant>/weights/
#    - Docker with NVIDIA Container Toolkit
#
#  Usage:
#    bash scripts/bash/verify_talker_trtllm.sh                 # all variants with engine
#    bash scripts/bash/verify_talker_trtllm.sh --variant design-1.7b
#    bash scripts/bash/verify_talker_trtllm.sh --n-decode-steps 5
#    bash scripts/bash/verify_talker_trtllm.sh --skip-ref       # skip host ref gen, only container verify
#    bash scripts/bash/verify_talker_trtllm.sh --dry-run
#
#  Output: workspace/exported/<variant>/talker_trtllm_ref.npz (step 1)
#          workspace/exported/<variant>/trtllm_verification_report.json (step 2)
# ===========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
source "${SCRIPT_DIR}/tools.sh"

EXPORTED_DIR="${REPO_ROOT}/workspace/exported"
SCRIPTS_PY="${REPO_ROOT}/scripts/python"
VARIANT=""
N_DECODE_STEPS=5
SKIP_REF=false
DRY_RUN=false
USER_IMAGE="${TRTLLM_IMAGE:-}"
GPU_DEVICES="all"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --variant)          VARIANT="$2"; shift 2 ;;
        --n-decode-steps)   N_DECODE_STEPS="$2"; shift 2 ;;
        --skip-ref)         SKIP_REF=true; shift ;;
        --image)            USER_IMAGE="$2"; shift 2 ;;
        --gpu)              GPU_DEVICES="device=$2"; shift 2 ;;
        --dry-run)          DRY_RUN=true; shift ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --variant <name>       Verify a specific model variant"
            echo "  --n-decode-steps <N>   Decode steps for ref + engine (default: 5)"
            echo "  --skip-ref             Skip PyTorch reference generation; only run container verify"
            echo "  --image <uri>          Override NGC container image"
            echo "  --gpu <id>             Use only GPU id (e.g. 1) to avoid OOM when another GPU is busy"
            echo "  --dry-run              Show commands without executing"
            echo "  -h, --help             Show this help"
            exit 0
            ;;
        *) log_error "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Discover variants (must have trtllm_engine) ──
discover_talker_variants() {
    local found=()
    for d in "$EXPORTED_DIR"/*/; do
        local vname
        vname=$(basename "$d")
        if [ -f "$d/trtllm_engine/config.json" ] || [ -f "$d/trtllm_engine/rank0.engine" ]; then
            found+=("$vname")
        fi
    done
    echo "${found[@]}"
}

if [ -n "$VARIANT" ]; then
    if [ ! -f "$EXPORTED_DIR/$VARIANT/trtllm_engine/config.json" ] && [ ! -f "$EXPORTED_DIR/$VARIANT/trtllm_engine/rank0.engine" ]; then
        log_error "TRT-LLM engine not found: $EXPORTED_DIR/$VARIANT/trtllm_engine/"
        log_error "Run build_engines.sh first (Phase B)."
        exit 1
    fi
    VARIANTS=("$VARIANT")
else
    read -ra VARIANTS <<< "$(discover_talker_variants)"
    if [ ${#VARIANTS[@]} -eq 0 ]; then
        log_error "No TRT-LLM engines found in $EXPORTED_DIR/*/trtllm_engine/"
        log_error "Run build_engines.sh first (Phase B)."
        exit 1
    fi
    log_info "Discovered variants: ${VARIANTS[*]}"
fi

# ── Step 1: Generate PyTorch reference on host ──
run_ref_gen() {
    local variant="$1"
    if $SKIP_REF; then
        if [ ! -f "$EXPORTED_DIR/$variant/talker_trtllm_ref.npz" ]; then
            log_error "Reference file missing: $EXPORTED_DIR/$variant/talker_trtllm_ref.npz (run without --skip-ref first)"
            return 1
        fi
        log_info "Using existing reference: $EXPORTED_DIR/$variant/talker_trtllm_ref.npz"
        return 0
    fi
    if [ ! -f "$EXPORTED_DIR/$variant/weights/config.json" ]; then
        log_error "Weights config missing: $EXPORTED_DIR/$variant/weights/config.json (run export_models.sh / export_06 first)"
        return 1
    fi
    log_step "Generating PyTorch reference: $variant"
    if $DRY_RUN; then
        log_info "[DRY RUN] Would run: python3 scripts/python/verify_talker_trtllm_ref.py --variant $variant --n-decode-steps $N_DECODE_STEPS"
        return 0
    fi
    ( cd "${REPO_ROOT}" && python3 "${SCRIPTS_PY}/verify_talker_trtllm_ref.py" --variant "$variant" --n-decode-steps "$N_DECODE_STEPS" ) || return 1
    log_info "Reference saved: $EXPORTED_DIR/$variant/talker_trtllm_ref.npz"
    return 0
}

# ── Step 2: Run TRT-LLM verification in container ──
run_container_verify() {
    local variant="$1"
    local variant_dir="$EXPORTED_DIR/$variant"
    if [ ! -f "$variant_dir/talker_trtllm_ref.npz" ]; then
        log_error "Reference file missing: $variant_dir/talker_trtllm_ref.npz"
        return 1
    fi
    log_step "Verifying TRT-LLM engine: $variant"
    local docker_cmd=(
        docker run --rm --gpus "$GPU_DEVICES"
        -v "$variant_dir:/mnt/model"
        -v "$SCRIPTS_PY:/mnt/scripts:ro"
        "$TRTLLM_IMAGE"
        python3 /mnt/scripts/verify_talker_trtllm.py
            --engine-dir /mnt/model/trtllm_engine
            --ref-file /mnt/model/talker_trtllm_ref.npz
            --n-decode-steps "$N_DECODE_STEPS"
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

# ── Resolve container image (for step 2) ──
log_step "Talker Backbone TRT-LLM Verification (Phase 2, Item 9)"

if ! $SKIP_REF; then
    log_info "Step 1: PyTorch reference (host); Step 2: TRT-LLM verify (container)"
else
    log_info "Step 2 only: TRT-LLM verify (container)"
fi

if [ -n "$USER_IMAGE" ]; then
    TRTLLM_IMAGE="$USER_IMAGE"
    log_info "Using user-specified image: $TRTLLM_IMAGE"
else
    log_info "Auto-detecting best NGC container ..."
    _NGC_VERIFY_MANIFEST=1
    TRTLLM_IMAGE=$(resolve_ngc_image_info) \
        || { log_error "Cannot determine compatible container. Use --image to specify manually."; exit 1; }
fi

if ! $SKIP_REF || ! $DRY_RUN; then
    check_docker_gpu_ready || exit 1
    ensure_ngc_image "$TRTLLM_IMAGE" || exit 1
fi

FAILED=0
SUCCEEDED=0

for variant in "${VARIANTS[@]}"; do
    if run_ref_gen "$variant" && run_container_verify "$variant"; then
        SUCCEEDED=$((SUCCEEDED + 1))
    else
        FAILED=$((FAILED + 1))
    fi
done

# ── Summary ──
echo ""
log_step "Verification Summary"
log_info "  Image:     $TRTLLM_IMAGE"
log_info "  Succeeded:  $SUCCEEDED"
if [ "$FAILED" -gt 0 ]; then
    log_error "  Failed:    $FAILED"
fi
if [ "$SUCCEEDED" -gt 0 ] && ! $DRY_RUN; then
    log_info ""
    log_info "Reports: $EXPORTED_DIR/*/trtllm_verification_report.json"
fi

exit "$FAILED"

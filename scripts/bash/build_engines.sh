#!/bin/bash
# ===========================================================================
#  build_engines.sh — Phase B: Compile TRT-LLM engines inside a container
#
#  Reads TRT-LLM checkpoints produced by export_models.sh (Phase A) and
#  compiles them into TensorRT engines using trtllm-build inside an NGC
#  TRT-LLM container.
#
#  The container image is auto-selected based on the host NVIDIA driver
#  version (see lib/docker.sh for the compatibility matrix).  Use --image
#  to override.
#
#  Prerequisites:
#    - NVIDIA GPU with driver >= 550.54
#    - Docker with NVIDIA Container Toolkit (docker run --gpus all)
#    - Checkpoints at workspace/exported/<variant>/trtllm_checkpoint/
#      (produced by: bash scripts/bash/autorun.sh setup  or  setup_env.sh / export_models.sh)
#
#  Usage:
#    bash scripts/bash/build_engines.sh                          # auto everything
#    bash scripts/bash/build_engines.sh --variant base-1.7b      # single variant
#    bash scripts/bash/build_engines.sh --max-batch-size 4       # custom batch size
#    bash scripts/bash/build_engines.sh --image <custom-image>   # override container
#    bash scripts/bash/build_engines.sh --target-driver 575.57  # build for production driver
#    bash scripts/bash/build_engines.sh --dry-run                # show docker command
#    bash scripts/bash/build_engines.sh --pull-only              # pull image, don't build
#
#  Environment variables:
#    TRTLLM_IMAGE     Docker image override (default: auto-detect from driver)
#    MAX_BATCH_SIZE   Max batch     (default: 8)
#    MAX_INPUT_LEN    Prefill len   (default: 512)
#    MAX_SEQ_LEN      Total seq len (default: 4096)
#    ENGINE_DTYPE     bfloat16|float16 (default: bfloat16)
#
#  Output: workspace/exported/<variant>/trtllm_engine/
# ===========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
source "${SCRIPT_DIR}/tools.sh"

# ── Engine build defaults (matching architecture.md §6.2) ──
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-8}"
MAX_INPUT_LEN="${MAX_INPUT_LEN:-512}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
ENGINE_DTYPE="${ENGINE_DTYPE:-bfloat16}"

EXPORTED_DIR="${REPO_ROOT}/workspace/exported"
VARIANT=""
DRY_RUN=false
PULL_ONLY=false
USER_IMAGE="${TRTLLM_IMAGE:-}"
TARGET_DRIVER="${TARGET_DRIVER:-}"

# ── Argument parsing ──
while [[ $# -gt 0 ]]; do
    case "$1" in
        --variant)        VARIANT="$2"; shift 2 ;;
        --image)          USER_IMAGE="$2"; shift 2 ;;
        --max-batch-size) MAX_BATCH_SIZE="$2"; shift 2 ;;
        --max-input-len)  MAX_INPUT_LEN="$2"; shift 2 ;;
        --max-seq-len)    MAX_SEQ_LEN="$2"; shift 2 ;;
        --dtype)          ENGINE_DTYPE="$2"; shift 2 ;;
        --target-driver)  TARGET_DRIVER="$2"; export TARGET_DRIVER; shift 2 ;;
        --dry-run)        DRY_RUN=true; shift ;;
        --pull-only)      PULL_ONLY=true; shift ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --variant <name>       Build for a specific model variant"
            echo "  --image <uri>          Override NGC container image (default: auto-detect)"
            echo "  --target-driver <ver>  Target NVIDIA driver for NGC container selection"
            echo "                         (e.g. 575.57 for production machines)"
            echo "  --max-batch-size N     Max batch size (default: 8)"
            echo "  --max-input-len N      Max input length for prefill (default: 512)"
            echo "  --max-seq-len N        Max sequence length incl. KV cache (default: 4096)"
            echo "  --dtype bf16|fp16      Engine precision (default: bfloat16)"
            echo "  --dry-run              Show docker command without executing"
            echo "  --pull-only            Pull the container image and exit"
            echo "  -h, --help             Show this help"
            exit 0
            ;;
        *) log_error "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Preflight: Docker + GPU ──
log_step "Phase B: TRT-LLM Engine Build"

check_docker_gpu_ready || exit 1

# ── Resolve container image ──
if [ -n "$USER_IMAGE" ]; then
    TRTLLM_IMAGE="$USER_IMAGE"
    log_info "Using user-specified image: $TRTLLM_IMAGE"
else
    log_info "Auto-detecting best NGC container for this GPU driver ..."
    _NGC_VERIFY_MANIFEST=1
    TRTLLM_IMAGE=$(resolve_ngc_image_info) \
        || { log_error "Cannot determine compatible container. Use --image to specify manually."; exit 1; }
fi

# ── Pull image ──
ensure_ngc_image "$TRTLLM_IMAGE" || exit 1

if $PULL_ONLY; then
    log_info "Image ready. Use 'bash scripts/bash/build_engines.sh' to build engines."
    exit 0
fi

# ── Check checkpoints ──
if [ ! -d "$EXPORTED_DIR" ]; then
    log_error "No exported models found at: $EXPORTED_DIR"
    log_error "Run 'autorun.sh setup' or export_models.sh first (Phase A)."
    exit 1
fi

discover_checkpoints() {
    local found=()
    for ckpt_dir in "$EXPORTED_DIR"/*/trtllm_checkpoint; do
        if [ -f "$ckpt_dir/config.json" ]; then
            local vname
            vname=$(basename "$(dirname "$ckpt_dir")")
            found+=("$vname")
        fi
    done
    echo "${found[@]}"
}

if [[ -n "$VARIANT" && "$VARIANT" != all* ]]; then
    CKPT_DIR="$EXPORTED_DIR/$VARIANT/trtllm_checkpoint"
    if [ ! -f "$CKPT_DIR/config.json" ]; then
        log_error "TRT-LLM checkpoint not found: $CKPT_DIR"
        log_error "Run: python scripts/export/export_04_talker_backbone.py --variant $VARIANT"
        exit 1
    fi
    VARIANTS=("$VARIANT")
else
    read -ra VARIANTS <<< "$(discover_checkpoints)"
    if [ ${#VARIANTS[@]} -eq 0 ]; then
        log_error "No TRT-LLM checkpoints found in $EXPORTED_DIR/*/trtllm_checkpoint/"
        log_error "Run 'autorun.sh setup' or export_models.sh first."
        exit 1
    fi
    log_info "Discovered checkpoints: ${VARIANTS[*]}"
fi

# ── Build engine for each variant ──
build_engine() {
    local variant="$1"
    local ckpt_dir="$EXPORTED_DIR/$variant/trtllm_checkpoint"
    local engine_dir="$EXPORTED_DIR/$variant/trtllm_engine"

    log_step "Building TRT-LLM engine: $variant"
    log_info "  Checkpoint:  $ckpt_dir"
    log_info "  Engine out:  $engine_dir"
    log_info "  Image:       $TRTLLM_IMAGE"
    log_info "  Config:      batch=$MAX_BATCH_SIZE, input=$MAX_INPUT_LEN, seq=$MAX_SEQ_LEN, dtype=$ENGINE_DTYPE"

    mkdir -p "$engine_dir"

    # max_prompt_embedding_table_size: Talker receives inputs_embeds from
    # the Orchestrator, so we need prompt_embedding_table to pass them in.
    # Size = max_input_len (prefill embeds occupy up to this many positions).
    local docker_cmd=(
        docker run --rm --gpus all
        -v "$EXPORTED_DIR/$variant:/mnt/model"
        "$TRTLLM_IMAGE"
        trtllm-build
            --checkpoint_dir /mnt/model/trtllm_checkpoint
            --output_dir /mnt/model/trtllm_engine
            --gemm_plugin "$ENGINE_DTYPE"
            --gpt_attention_plugin "$ENGINE_DTYPE"
            --max_batch_size "$MAX_BATCH_SIZE"
            --max_input_len "$MAX_INPUT_LEN"
            --max_seq_len "$MAX_SEQ_LEN"
            --max_prompt_embedding_table_size "$MAX_INPUT_LEN"
            --gather_context_logits
            --gather_all_token_logits
            --paged_kv_cache disable
    )

    if $DRY_RUN; then
        log_info "[DRY RUN] Would execute:"
        log_info "  ${docker_cmd[*]}"
        return 0
    fi

    log_info "Running trtllm-build in container ..."
    if "${docker_cmd[@]}"; then
        log_info "Engine built successfully: $engine_dir"
        if ls "$engine_dir"/*.engine 1>/dev/null 2>&1; then
            local engine_size
            engine_size=$(du -sh "$engine_dir" | cut -f1)
            log_info "  Engine size: $engine_size"
        fi
    else
        log_error "trtllm-build failed for variant: $variant"
        return 1
    fi
}

# ── Main loop ──
FAILED=0
SUCCEEDED=0

for variant in "${VARIANTS[@]}"; do
    if build_engine "$variant"; then
        SUCCEEDED=$((SUCCEEDED + 1))
    else
        FAILED=$((FAILED + 1))
    fi
done

# ── Summary ──
echo ""
log_step "Engine Build Summary"
log_info "  Image:     $TRTLLM_IMAGE"
log_info "  Succeeded: $SUCCEEDED"
if [ "$FAILED" -gt 0 ]; then
    log_error "  Failed: $FAILED"
fi

if [ "$SUCCEEDED" -gt 0 ] && ! $DRY_RUN; then
    echo ""
    log_info "Engines ready at: $EXPORTED_DIR/*/trtllm_engine/"
    log_info "Next: deploy with Triton Inference Server"
fi

exit "$FAILED"

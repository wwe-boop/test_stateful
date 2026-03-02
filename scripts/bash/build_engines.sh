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
#  Supports remote builds for cross-GPU development (e.g. develop on RTX
#  4090, build engines on A800 via SSH).  Use --remote <user@host>.
#
#  Prerequisites:
#    - NVIDIA GPU with driver >= 550.54
#    - Docker with NVIDIA Container Toolkit (docker run --gpus all)
#    - Checkpoints at workspace/exported/<variant>/trtllm_checkpoint/
#      (produced by: bash scripts/bash/autorun.sh setup  or  setup_env.sh / export_models.sh)
#    - For remote builds: SSH key auth + Docker + NVIDIA GPU on remote host
#
#  Usage:
#    bash scripts/bash/build_engines.sh                          # auto everything
#    bash scripts/bash/build_engines.sh --variant base-1.7b      # single variant
#    bash scripts/bash/build_engines.sh --max-batch-size 4       # custom batch size
#    bash scripts/bash/build_engines.sh --image <custom-image>   # override container
#    bash scripts/bash/build_engines.sh --target-driver 575.57  # build for production driver
#    bash scripts/bash/build_engines.sh --remote user@a800       # build on remote GPU
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
#
#  Pure TRT (--pure-trt): Build talker_context.engine and talker_decode_fused.engine
#  from ONNX via trtexec (no TRT-LLM). Requires talker_context.onnx and
#  talker_decode_fused.onnx from export_04a/04b. Uses same NGC image (has trtexec).
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
PURE_TRT=false
USER_IMAGE="${TRTLLM_IMAGE:-}"
TARGET_DRIVER="${TARGET_DRIVER:-}"
REMOTE_HOST=""
REMOTE_DIR="/tmp/qwen3-tts-engine-build"

# ── Shared helpers ──

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

# Returns the trtllm-build arguments (shared between local and remote builds)
_trtllm_build_args() {
    echo "--checkpoint_dir /mnt/model/trtllm_checkpoint"
    echo "--output_dir /mnt/model/trtllm_engine"
    echo "--gemm_plugin ${ENGINE_DTYPE}"
    echo "--gpt_attention_plugin ${ENGINE_DTYPE}"
    echo "--max_batch_size ${MAX_BATCH_SIZE}"
    echo "--max_input_len ${MAX_INPUT_LEN}"
    echo "--max_seq_len ${MAX_SEQ_LEN}"
    echo "--max_prompt_embedding_table_size ${MAX_INPUT_LEN}"
    echo "--gather_context_logits"
    echo "--gather_all_token_logits"
    echo "--paged_kv_cache disable"
}

# Talker dimensions for Pure TRT (trtexec). Read from config if present, else fallback.
# Output: H num_kv_heads head_dim num_layers
_get_talker_dims() {
    local variant="$1"
    local cfg="$EXPORTED_DIR/$variant/trtllm_checkpoint/config.json"
    if [ -f "$cfg" ]; then
        python3 -c "
import json
c = json.load(open('$cfg'))
print(c.get('hidden_size', 1024), c.get('num_key_value_heads', 2), c.get('head_size', 64), c.get('num_hidden_layers', 28))
" 2>/dev/null || echo "1024 2 64 28"
        return
    fi
    case "$variant" in
        base-0.6b|custom-0.6b)   echo "1024 2 64 28" ;;
        base-1.7b|custom-1.7b|design-1.7b) echo "2048 8 128 28" ;;
        *) echo "1024 2 64 28" ;;
    esac
}

# Build Pure TRT engines (context + fused decode) from ONNX via trtexec
build_talker_trt() {
    local variant="$1"
    local variant_dir="$EXPORTED_DIR/$variant"
    local ctx_onnx="$variant_dir/talker_context.onnx"
    local dec_onnx="$variant_dir/talker_decode_fused.onnx"

    if [ ! -f "$ctx_onnx" ] || [ ! -f "$dec_onnx" ]; then
        log_error "Pure TRT: missing ONNX files for $variant (need talker_context.onnx and talker_decode_fused.onnx)"
        log_error "  Run: python scripts/export/export_04a_talker_context.py --variant $variant"
        log_error "       python scripts/export/export_04b_talker_decode_fused.py --variant $variant"
        return 1
    fi

    local dims
    dims=($(_get_talker_dims "$variant"))
    local H="${dims[0]:-1024}" KV_HEADS="${dims[1]:-2}" HEAD_DIM="${dims[2]:-64}" NUM_LAYERS="${dims[3]:-28}"

    log_step "Building Pure TRT engines: $variant (H=$H, kv_heads=$KV_HEADS, head_dim=$HEAD_DIM, layers=$NUM_LAYERS)"
    log_info "  Context ONNX:  $ctx_onnx"
    log_info "  Decode ONNX:   $dec_onnx"
    log_info "  Image:         $TRTLLM_IMAGE"

    if $DRY_RUN; then
        log_info "[DRY RUN] Would run trtexec for context and decode engines"
        return 0
    fi

    # Context engine
    log_info "Building context engine (trtexec) ..."
    local ctx_cmd=(
        docker run --rm --gpus all
        -v "$variant_dir:/mnt/model"
        "$TRTLLM_IMAGE"
        trtexec --onnx=/mnt/model/talker_context.onnx
        --saveEngine=/mnt/model/talker_context.engine
        --bf16
        "--minShapes=input_embeds:1x1x${H},position_ids:3x1x1"
        "--optShapes=input_embeds:1x128x${H},position_ids:3x1x128"
        "--maxShapes=input_embeds:${MAX_BATCH_SIZE}x${MAX_INPUT_LEN}x${H},position_ids:3x${MAX_BATCH_SIZE}x${MAX_INPUT_LEN}"
        --memPoolSize=workspace:4096
    )
    if ! "${ctx_cmd[@]}"; then
        log_error "trtexec context engine failed for $variant"
        return 1
    fi

    # Decode engine: build min/opt/max shapes for input_embeds, position_ids, and all past_kv_{i}_k/v
    local dec_min="input_embeds:1x1x${H},position_ids:3x1x1"
    local dec_opt="input_embeds:4x1x${H},position_ids:3x4x1"
    local dec_max="input_embeds:${MAX_BATCH_SIZE}x1x${H},position_ids:3x${MAX_BATCH_SIZE}x1"
    local i=0
    while [ "$i" -lt "$NUM_LAYERS" ]; do
        dec_min="$dec_min,past_kv_${i}_k:1x${KV_HEADS}x1x${HEAD_DIM},past_kv_${i}_v:1x${KV_HEADS}x1x${HEAD_DIM}"
        dec_opt="$dec_opt,past_kv_${i}_k:4x${KV_HEADS}x512x${HEAD_DIM},past_kv_${i}_v:4x${KV_HEADS}x512x${HEAD_DIM}"
        dec_max="$dec_max,past_kv_${i}_k:${MAX_BATCH_SIZE}x${KV_HEADS}x${MAX_SEQ_LEN}x${HEAD_DIM},past_kv_${i}_v:${MAX_BATCH_SIZE}x${KV_HEADS}x${MAX_SEQ_LEN}x${HEAD_DIM}"
        i=$((i + 1))
    done

    log_info "Building decode fused engine (trtexec) ..."
    local dec_cmd=(
        docker run --rm --gpus all
        -v "$variant_dir:/mnt/model"
        "$TRTLLM_IMAGE"
        trtexec --onnx=/mnt/model/talker_decode_fused.onnx
        --saveEngine=/mnt/model/talker_decode_fused.engine
        --bf16
        --memPoolSize=workspace:8192
    )
    dec_cmd+=(--minShapes="$dec_min" --optShapes="$dec_opt" --maxShapes="$dec_max")
    if ! "${dec_cmd[@]}"; then
        log_error "trtexec decode engine failed for $variant"
        return 1
    fi

    log_info "Pure TRT engines built: $variant_dir/talker_context.engine, $variant_dir/talker_decode_fused.engine"
    return 0
}

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
        --remote)         REMOTE_HOST="$2"; shift 2 ;;
        --remote-dir)     REMOTE_DIR="$2"; shift 2 ;;
        --dry-run)        DRY_RUN=true; shift ;;
        --pull-only)      PULL_ONLY=true; shift ;;
        --pure-trt)       PURE_TRT=true; shift ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --variant <name>       Build for a specific model variant"
            echo "  --image <uri>          Override NGC container image (default: auto-detect)"
            echo "  --target-driver <ver>  Target NVIDIA driver for NGC container selection"
            echo "                         (e.g. 575.57 for production machines)"
            echo "  --remote <user@host>   Build engines on remote GPU via SSH"
            echo "                         (cross-GPU: e.g. develop on 4090, deploy on A800)"
            echo "  --remote-dir <path>    Remote workspace dir (default: /tmp/qwen3-tts-engine-build)"
            echo "  --max-batch-size N     Max batch size (default: 8)"
            echo "  --max-input-len N      Max input length for prefill (default: 512)"
            echo "  --max-seq-len N        Max sequence length incl. KV cache (default: 4096)"
            echo "  --dtype bf16|fp16      Engine precision (default: bfloat16)"
            echo "  --dry-run              Show docker command without executing"
            echo "  --pull-only            Pull the container image and exit"
            echo "  --pure-trt             Build Pure TRT engines (talker_context, talker_decode_fused) from ONNX via trtexec"
            echo "  -h, --help             Show this help"
            exit 0
            ;;
        *) log_error "Unknown argument: $1"; exit 1 ;;
    esac
done

# ===========================================================================
#  Remote build mode — build engines on a remote GPU via SSH
#
#  Workflow: rsync checkpoints → SSH docker trtllm-build → rsync engines back
#  Use case: develop on RTX 4090 (sm_89), deploy on A800/A100 (sm_80)
# ===========================================================================

if [ -n "$REMOTE_HOST" ]; then

    remote_build_engine() {
        local variant="$1"
        local ckpt_dir="$EXPORTED_DIR/$variant/trtllm_checkpoint"
        local engine_dir="$EXPORTED_DIR/$variant/trtllm_engine"
        local remote_work="$REMOTE_DIR/$variant"

        log_step "Building TRT-LLM engine (remote): $variant"
        log_info "  Checkpoint:  $ckpt_dir"
        log_info "  Remote:      $REMOTE_HOST:$remote_work"
        log_info "  Image:       $TRTLLM_IMAGE"
        log_info "  Target GPU:  $REMOTE_GPU (sm_${REMOTE_CC//./})"
        log_info "  Config:      batch=$MAX_BATCH_SIZE, input=$MAX_INPUT_LEN, seq=$MAX_SEQ_LEN, dtype=$ENGINE_DTYPE"

        if $DRY_RUN; then
            log_info "[DRY RUN] Would: rsync checkpoint → build on $REMOTE_HOST → rsync engine back"
            return 0
        fi

        # Sync checkpoint to remote
        ssh "$REMOTE_HOST" "mkdir -p '${remote_work}'" \
            || { log_error "Failed to create remote dir: $remote_work"; return 1; }

        log_info "Syncing checkpoint to $REMOTE_HOST ..."
        rsync -az --info=progress2 \
            "$ckpt_dir/" "$REMOTE_HOST:${remote_work}/trtllm_checkpoint/" \
            || { log_error "rsync failed: checkpoint → remote"; return 1; }

        # Build engine on remote GPU
        log_info "Running trtllm-build on remote GPU ..."
        local remote_cmd="docker run --rm --gpus all"
        remote_cmd+=" -v ${remote_work}:/mnt/model"
        remote_cmd+=" ${TRTLLM_IMAGE}"
        remote_cmd+=" trtllm-build"
        remote_cmd+=" $(_trtllm_build_args | tr '\n' ' ')"

        if ! ssh "$REMOTE_HOST" "$remote_cmd"; then
            log_error "Remote trtllm-build failed for variant: $variant"
            return 1
        fi

        # Sync engine back
        mkdir -p "$engine_dir"
        log_info "Syncing engine from $REMOTE_HOST ..."
        rsync -az --info=progress2 \
            "$REMOTE_HOST:${remote_work}/trtllm_engine/" "$engine_dir/" \
            || { log_error "rsync failed: remote engine → local"; return 1; }

        log_info "Engine built successfully: $engine_dir (target: sm_${REMOTE_CC//./})"
    }

    # ── Remote preflight ──
    log_step "Phase B: Remote TRT-LLM Engine Build → $REMOTE_HOST"

    require_cmd rsync "rsync required for remote build (apt install rsync)"

    log_info "Verifying SSH connectivity to $REMOTE_HOST ..."
    if ! ssh -o ConnectTimeout=10 -o BatchMode=yes "$REMOTE_HOST" true 2>/dev/null; then
        log_error "Cannot connect to $REMOTE_HOST via SSH"
        log_error "Set up SSH key auth: ssh-copy-id $REMOTE_HOST"
        exit 1
    fi

    log_info "Checking remote Docker + GPU ..."
    if ! ssh "$REMOTE_HOST" "command -v docker &>/dev/null && nvidia-smi &>/dev/null"; then
        log_error "Remote host requires Docker + NVIDIA GPU driver"
        exit 1
    fi

    # Detect remote GPU info (single SSH call)
    REMOTE_GPU_INFO=$(ssh "$REMOTE_HOST" \
        "nvidia-smi --query-gpu=name,driver_version,compute_cap \
         --format=csv,noheader,nounits 2>/dev/null | head -1")
    REMOTE_GPU=$(echo "$REMOTE_GPU_INFO" | cut -d',' -f1 | xargs)
    REMOTE_DRIVER=$(echo "$REMOTE_GPU_INFO" | cut -d',' -f2 | xargs)
    REMOTE_CC=$(echo "$REMOTE_GPU_INFO" | cut -d',' -f3 | xargs)
    log_info "Remote GPU: $REMOTE_GPU (sm_${REMOTE_CC//./}, driver $REMOTE_DRIVER)"

    # Show local vs remote GPU info
    if command -v nvidia-smi &>/dev/null; then
        LOCAL_GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | xargs)
        LOCAL_CC=$(detect_gpu_compute_cap 2>/dev/null || true)
        if [ -n "$LOCAL_CC" ] && [ "$LOCAL_CC" != "$REMOTE_CC" ]; then
            log_info "Local GPU:  $LOCAL_GPU (sm_${LOCAL_CC//./}) — checkpoint export only"
            log_info "Engine will target remote sm_${REMOTE_CC//./} ($REMOTE_GPU)"
        fi
    fi

    # ── Resolve NGC image for remote driver ──
    if [ -n "$USER_IMAGE" ]; then
        TRTLLM_IMAGE="$USER_IMAGE"
        log_info "Using user-specified image: $TRTLLM_IMAGE"
    else
        log_info "Selecting NGC container for remote driver $REMOTE_DRIVER ..."
        TRTLLM_IMAGE=$(resolve_ngc_image_info "$REMOTE_DRIVER") \
            || { log_error "No compatible container for remote driver $REMOTE_DRIVER. Use --image."; exit 1; }
    fi

    # ── Ensure image on remote ──
    if ! ssh "$REMOTE_HOST" "docker image inspect '${TRTLLM_IMAGE}' &>/dev/null"; then
        log_info "Pulling NGC image on remote host (this may take a while) ..."
        ssh "$REMOTE_HOST" "docker pull '${TRTLLM_IMAGE}'" \
            || { log_error "Failed to pull $TRTLLM_IMAGE on remote"; exit 1; }
    else
        log_info "Image already present on remote: $TRTLLM_IMAGE"
    fi

    if $PULL_ONLY; then
        log_info "Image ready on remote. Re-run without --pull-only to build engines."
        exit 0
    fi

    # ── Check checkpoints (local) ──
    if [ ! -d "$EXPORTED_DIR" ]; then
        log_error "No exported models at: $EXPORTED_DIR"
        log_error "Run Phase A first: autorun.sh setup"
        exit 1
    fi

    if [[ -n "$VARIANT" && "$VARIANT" != all* ]]; then
        if [ ! -f "$EXPORTED_DIR/$VARIANT/trtllm_checkpoint/config.json" ]; then
            log_error "Checkpoint not found: $EXPORTED_DIR/$VARIANT/trtllm_checkpoint"
            log_error "Run: python scripts/export/export_04_talker_backbone.py --variant $VARIANT"
            exit 1
        fi
        VARIANTS=("$VARIANT")
    else
        read -ra VARIANTS <<< "$(discover_checkpoints)"
        if [ ${#VARIANTS[@]} -eq 0 ]; then
            log_error "No TRT-LLM checkpoints in $EXPORTED_DIR/*/trtllm_checkpoint/"
            log_error "Run Phase A first: autorun.sh setup"
            exit 1
        fi
        log_info "Discovered checkpoints: ${VARIANTS[*]}"
    fi

    # ── Remote build loop ──
    FAILED=0
    SUCCEEDED=0

    for variant in "${VARIANTS[@]}"; do
        if remote_build_engine "$variant"; then
            SUCCEEDED=$((SUCCEEDED + 1))
        else
            FAILED=$((FAILED + 1))
        fi
    done

    # Clean up remote workspace
    if [ "$SUCCEEDED" -gt 0 ] && ! $DRY_RUN; then
        log_info "Cleaning up remote workspace ..."
        ssh "$REMOTE_HOST" "rm -rf '${REMOTE_DIR}'" 2>/dev/null || true
    fi

    # ── Summary ──
    echo ""
    log_step "Remote Engine Build Summary"
    log_info "  Remote:    $REMOTE_HOST ($REMOTE_GPU, sm_${REMOTE_CC//./})"
    log_info "  Image:     $TRTLLM_IMAGE"
    log_info "  Succeeded: $SUCCEEDED"
    [ "$FAILED" -gt 0 ] && log_error "  Failed: $FAILED"

    if [ "$SUCCEEDED" -gt 0 ] && ! $DRY_RUN; then
        echo ""
        log_info "Engines at: $EXPORTED_DIR/*/trtllm_engine/"
        log_info "Target arch: sm_${REMOTE_CC//./} ($REMOTE_GPU)"
        log_info "Deploy on sm_${REMOTE_CC//./} GPU: bash scripts/bash/build_triton.sh run"
    fi

    exit "$FAILED"
fi

# ===========================================================================
#  Pure TRT build mode (--pure-trt): trtexec context + decode engines from ONNX
# ===========================================================================
discover_pure_trt_variants() {
    local found=()
    for vdir in "$EXPORTED_DIR"/*/; do
        [ -d "$vdir" ] || continue
        if [ -f "${vdir}talker_context.onnx" ] && [ -f "${vdir}talker_decode_fused.onnx" ]; then
            found+=("$(basename "$vdir")")
        fi
    done
    echo "${found[@]}"
}

if [ "$PURE_TRT" = true ]; then
    log_step "Phase B: Pure TRT Engine Build (trtexec)"

    check_docker_gpu_ready || exit 1
    if [ -n "$USER_IMAGE" ]; then
        TRTLLM_IMAGE="$USER_IMAGE"
    else
        _NGC_VERIFY_MANIFEST=1
        TRTLLM_IMAGE=$(resolve_ngc_image_info) \
            || { log_error "Cannot determine NGC container. Use --image."; exit 1; }
    fi
    ensure_ngc_image "$TRTLLM_IMAGE" || exit 1

    if [[ -n "$VARIANT" && "$VARIANT" != all* ]]; then
        PURE_VARIANTS=("$VARIANT")
    else
        read -ra PURE_VARIANTS <<< "$(discover_pure_trt_variants)"
        if [ ${#PURE_VARIANTS[@]} -eq 0 ]; then
            log_error "No Pure TRT ONNX found in $EXPORTED_DIR (need talker_context.onnx + talker_decode_fused.onnx per variant)"
            log_error "Run export_04a_talker_context.py and export_04b_talker_decode_fused.py first."
            exit 1
        fi
        log_info "Discovered Pure TRT variants: ${PURE_VARIANTS[*]}"
    fi

    FAILED=0
    SUCCEEDED=0
    for variant in "${PURE_VARIANTS[@]}"; do
        if build_talker_trt "$variant"; then
            SUCCEEDED=$((SUCCEEDED + 1))
        else
            FAILED=$((FAILED + 1))
        fi
    done
    echo ""
    log_step "Pure TRT Build Summary"
    log_info "  Succeeded: $SUCCEEDED"
    [ "$FAILED" -gt 0 ] && log_error "  Failed: $FAILED"
    exit "$FAILED"
fi

# ===========================================================================
#  Local build mode (TRT-LLM)
# ===========================================================================

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

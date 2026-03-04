#!/bin/bash
# ===========================================================================
#  build_engines.sh — Phase B: Compile TensorRT engines from ONNX (trtexec)
#
#  Reads ONNX files produced by Phase A (export_models.sh / export_all.py) and
#  compiles them into TensorRT engines using trtexec inside an NGC container
#  (tritonserver:xx.yy-py3, trtexec at /usr/src/tensorrt/bin/trtexec).
#
#  Builds: talker_unified.engine (single engine for prefill+decode, per variant),
#  plus text_embedder.engine, codec_embedder.engine, speaker_encoder.engine,
#  speech_tokenizer_encoder.engine, code2wav_decoder.engine
#  (shared or per-variant as per Phase A layout).
#
#  Prerequisites:
#    - NVIDIA GPU with driver >= 550.54
#    - Docker with NVIDIA Container Toolkit (docker run --gpus all)
#    - ONNX at workspace/exported/<variant>/ and workspace/exported/tokenizer/
#
#  Usage:
#    bash scripts/bash/build_engines.sh                    # all variants
#    bash scripts/bash/build_engines.sh --variant base-1.7b
#    bash scripts/bash/build_engines.sh --max-batch-size 4
#    bash scripts/bash/build_engines.sh --dry-run
#    bash scripts/bash/build_engines.sh --pull-only
#
#  Environment variables:
#    NGC_IMAGE        Docker image override (default: auto-detect from driver)
#    MAX_BATCH_SIZE   Max batch (default: 16)
#    MAX_INPUT_LEN    Prefill len (default: 512)
#    MAX_SEQ_LEN      Total seq len (default: 1024)
#    ENGINE_DTYPE     bfloat16|float16 (default: bfloat16)
#
#  Output: workspace/exported/<variant>/*.engine, workspace/exported/tokenizer/*.engine
# ===========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
source "${SCRIPT_DIR}/tools.sh"

# trtexec path inside NGC tritonserver image
TRTEXEC="/usr/src/tensorrt/bin/trtexec"

# ── Engine build defaults (TRT memory optimization: BF16 I/O, seq=1024, batch=16) ──
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-16}"
MAX_INPUT_LEN="${MAX_INPUT_LEN:-512}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"
ENGINE_DTYPE="${ENGINE_DTYPE:-bfloat16}"

EXPORTED_DIR="${REPO_ROOT}/workspace/exported"
TOKENIZER_DIR="${EXPORTED_DIR}/tokenizer"
VARIANT=""
DRY_RUN=false
PULL_ONLY=false
USER_IMAGE="${NGC_IMAGE:-}"
TARGET_DRIVER="${TARGET_DRIVER:-}"

# Talker dimensions for trtexec. Read from model config.json, else fallback.
# Output: H num_kv_heads head_dim num_layers
_get_talker_dims() {
    local variant="$1"

    # Variant name → model directory name
    local -A _VARIANT_DIR=(
        [base-0.6b]=Qwen3-TTS-12Hz-0.6B-Base
        [custom-0.6b]=Qwen3-TTS-12Hz-0.6B-CustomVoice
        [base-1.7b]=Qwen3-TTS-12Hz-1.7B-Base
        [custom-1.7b]=Qwen3-TTS-12Hz-1.7B-CustomVoice
        [design-1.7b]=Qwen3-TTS-12Hz-1.7B-VoiceDesign
    )
    local model_dir="${REPO_ROOT}/workspace/models/${_VARIANT_DIR[$variant]:-}"
    local cfg="$model_dir/config.json"

    if [ -f "$cfg" ]; then
        python3 -c "
import json, sys
c = json.load(open('$cfg'))
tc = c.get('talker_config', c)
h = tc.get('hidden_size', 1024)
nkv = tc.get('num_key_value_heads', 8)
head_dim = tc.get('head_dim', h // tc.get('num_attention_heads', 16))
nlayers = tc.get('num_hidden_layers', 28)
print(h, nkv, head_dim, nlayers)
" 2>/dev/null && return
    fi

    # Fallback: both 0.6b and 1.7b share head_dim=128, kv_heads=8, layers=28
    case "$variant" in
        base-0.6b|custom-0.6b)   echo "1024 8 128 28" ;;
        base-1.7b|custom-1.7b|design-1.7b) echo "2048 8 128 28" ;;
        *) echo "1024 8 128 28" ;;
    esac
}

# Build single Talker unified TRT engine (prefill + decode) from ONNX via trtexec.
# S_past min=1: prefill passes dummy past_kv; decode uses real KV cache.
build_talker_unified_trt() {
    local variant="$1"
    local variant_dir="$EXPORTED_DIR/$variant"
    local unified_onnx="$variant_dir/talker_unified.onnx"

    if [ ! -f "$unified_onnx" ]; then
        log_error "Unified TRT: missing ONNX for $variant (need talker_unified.onnx)"
        log_error "  Run: python scripts/export/export_04_talker_unified.py --variant $variant"
        return 1
    fi

    local dims
    dims=($(_get_talker_dims "$variant"))
    local H="${dims[0]:-1024}" KV_HEADS="${dims[1]:-2}" HEAD_DIM="${dims[2]:-64}" NUM_LAYERS="${dims[3]:-28}"

    log_step "Building Talker unified TRT engine: $variant (H=$H, kv_heads=$KV_HEADS, head_dim=$HEAD_DIM, layers=$NUM_LAYERS)"
    log_info "  Unified ONNX: $unified_onnx"
    log_info "  Image:       $NGC_IMAGE"

    if $DRY_RUN; then
        log_info "[DRY RUN] Would run trtexec for talker_unified.engine"
        return 0
    fi

    # Single engine: min/opt/max for input_embeds, position_ids, and all past_kv_{i}_k/v
    # All tensors sharing the "batch" dim must have the same batch value per profile.
    # min: batch=1, S=1, S_past=1 (dummy); opt: batch=1, S=1, S_past=128 (decode hot path); max: batch=B, S=512, S_past=4096
    local OPT_BATCH=1 OPT_S_PAST=128
    local unif_min="input_embeds:1x1x${H},position_ids:3x1x1"
    local unif_opt="input_embeds:${OPT_BATCH}x1x${H},position_ids:3x${OPT_BATCH}x1"
    local unif_max="input_embeds:${MAX_BATCH_SIZE}x${MAX_INPUT_LEN}x${H},position_ids:3x${MAX_BATCH_SIZE}x${MAX_INPUT_LEN}"
    local i=0
    while [ "$i" -lt "$NUM_LAYERS" ]; do
        unif_min="$unif_min,past_kv_${i}_k:1x${KV_HEADS}x1x${HEAD_DIM},past_kv_${i}_v:1x${KV_HEADS}x1x${HEAD_DIM}"
        unif_opt="$unif_opt,past_kv_${i}_k:${OPT_BATCH}x${KV_HEADS}x${OPT_S_PAST}x${HEAD_DIM},past_kv_${i}_v:${OPT_BATCH}x${KV_HEADS}x${OPT_S_PAST}x${HEAD_DIM}"
        unif_max="$unif_max,past_kv_${i}_k:${MAX_BATCH_SIZE}x${KV_HEADS}x${MAX_SEQ_LEN}x${HEAD_DIM},past_kv_${i}_v:${MAX_BATCH_SIZE}x${KV_HEADS}x${MAX_SEQ_LEN}x${HEAD_DIM}"
        i=$((i + 1))
    done

    # BF16 I/O formats: input_embeds=bf16, position_ids=int64, all KV=bf16 (58 inputs, 60 outputs)
    local io_in="bf16:chw,int64:chw"
    local io_out="bf16:chw,int64:chw,bf16:chw,bf16:chw"
    i=0
    while [ "$i" -lt "$NUM_LAYERS" ]; do
        io_in="$io_in,bf16:chw,bf16:chw"
        io_out="$io_out,bf16:chw,bf16:chw"
        i=$((i + 1))
    done

    log_info "Building talker_unified.engine (trtexec, BF16 I/O) ..."
    local unif_cmd=(
        docker run --rm --gpus all
        -v "$variant_dir:/mnt/model"
        "$NGC_IMAGE"
        $TRTEXEC --onnx=/mnt/model/talker_unified.onnx
        --saveEngine=/mnt/model/talker_unified.engine
        --bf16
        --memPoolSize=workspace:8192
        --minShapes="$unif_min"
        --optShapes="$unif_opt"
        --maxShapes="$unif_max"
        --inputIOFormats="$io_in"
        --outputIOFormats="$io_out"
    )
    if ! "${unif_cmd[@]}"; then
        log_error "trtexec talker_unified engine failed for $variant"
        return 1
    fi

    log_info "Talker unified TRT engine built: $variant_dir/talker_unified.engine"
    return 0
}

# Build peripheral TRT engines (text_embedder, codec_embedder, speaker_encoder, speech_tokenizer_encoder, code2wav_decoder)
build_peripheral_engines() {
    if $DRY_RUN; then
        log_info "[DRY RUN] Would run trtexec for text_embedder, codec_embedder, speaker_encoder, speech_tokenizer_encoder, code2wav_decoder"
        return 0
    fi
    local image="$1"
    local failed=0

    # Text embedder: per-variant (Embedding + ResizeMLP, required for lightweight deploy)
    for vdir in "$EXPORTED_DIR"/*/; do
        [ -d "$vdir" ] || continue
        local onnx="${vdir}text_embedder.onnx"
        [ -f "$onnx" ] || continue
        local variant_name
        variant_name=$(basename "$vdir")
        log_info "Building text_embedder.engine for $variant_name ..."
        local te_cmd=(
            docker run --rm --gpus all -v "$vdir:/mnt/model" "$image"
            $TRTEXEC --onnx=/mnt/model/text_embedder.onnx
            --saveEngine=/mnt/model/text_embedder.engine
            --bf16
            --minShapes=token_ids:1x1
            --optShapes=token_ids:1x64
            --maxShapes=token_ids:${MAX_BATCH_SIZE}x${MAX_INPUT_LEN}
            --memPoolSize=workspace:4096
        )
        if [ -f "${onnx}.data" ]; then
            log_info "  (has external data file)"
        fi
        if ! "${te_cmd[@]}"; then
            log_error "text_embedder trtexec failed for $variant_name"
            failed=$((failed + 1))
        fi
    done

    # Codec embedder: per-variant (Embedding lookup, required for lightweight deploy)
    for vdir in "$EXPORTED_DIR"/*/; do
        [ -d "$vdir" ] || continue
        local onnx="${vdir}codec_embedder.onnx"
        [ -f "$onnx" ] || continue
        local variant_name
        variant_name=$(basename "$vdir")
        log_info "Building codec_embedder.engine for $variant_name ..."
        if ! docker run --rm --gpus all -v "$vdir:/mnt/model" "$image" \
            $TRTEXEC --onnx=/mnt/model/codec_embedder.onnx \
            --saveEngine=/mnt/model/codec_embedder.engine \
            --bf16 \
            --minShapes=codec_ids:1x1 \
            --optShapes=codec_ids:1x8 \
            --maxShapes=codec_ids:${MAX_BATCH_SIZE}x64 \
            --memPoolSize=workspace:1024; then
            log_error "codec_embedder trtexec failed for $variant_name"
            failed=$((failed + 1))
        fi
    done

    # Speaker encoder: per-variant, only if ONNX exists (base variants)
    for vdir in "$EXPORTED_DIR"/*/; do
        [ -d "$vdir" ] || continue
        local onnx="${vdir}speaker_encoder.onnx"
        [ -f "$onnx" ] || continue
        local variant_name
        variant_name=$(basename "$vdir")
        log_info "Building speaker_encoder.engine for $variant_name ..."
        if ! docker run --rm --gpus all -v "$vdir:/mnt/model" "$image" \
            $TRTEXEC --onnx=/mnt/model/speaker_encoder.onnx \
            --saveEngine=/mnt/model/speaker_encoder.engine \
            --bf16 \
            --minShapes=mel:1x1x128 \
            --optShapes=mel:1x300x128 \
            --maxShapes=mel:${MAX_BATCH_SIZE}x1000x128 \
            --memPoolSize=workspace:1024; then
            log_error "speaker_encoder trtexec failed for $variant_name"
            failed=$((failed + 1))
        fi
    done

    # Speech tokenizer encoder (shared, in tokenizer dir)
    # Min waveform = 960 samples (1 codec frame at stride 8×6×5×4=960, 24kHz → 40ms).
    # Smaller values produce 0-length intermediates that TRT cannot handle.
    if [ -f "$TOKENIZER_DIR/speech_tokenizer_encoder.onnx" ]; then
        log_info "Building speech_tokenizer_encoder.engine ..."
        if ! docker run --rm --gpus all -v "$TOKENIZER_DIR:/mnt/model" "$image" \
            $TRTEXEC --onnx=/mnt/model/speech_tokenizer_encoder.onnx \
            --saveEngine=/mnt/model/speech_tokenizer_encoder.engine \
            --bf16 \
            --minShapes=waveform:1x1x960 \
            --optShapes=waveform:1x1x48000 \
            --maxShapes=waveform:1x1x480000 \
            --memPoolSize=workspace:4096; then
            log_error "speech_tokenizer_encoder trtexec failed"
            failed=$((failed + 1))
        fi
    else
        log_info "speech_tokenizer_encoder.onnx not found, skipping"
    fi

    # Code2Wav decoder (shared)
    if [ -f "$TOKENIZER_DIR/code2wav_decoder.onnx" ]; then
        log_info "Building code2wav_decoder.engine ..."
        if ! docker run --rm --gpus all -v "$TOKENIZER_DIR:/mnt/model" "$image" \
            $TRTEXEC --onnx=/mnt/model/code2wav_decoder.onnx \
            --saveEngine=/mnt/model/code2wav_decoder.engine \
            --bf16 \
            --minShapes=codes:1x16x1 \
            --optShapes=codes:1x16x50 \
            --maxShapes=codes:${MAX_BATCH_SIZE}x16x500 \
            --memPoolSize=workspace:4096; then
            log_error "code2wav_decoder trtexec failed"
            failed=$((failed + 1))
        fi
    else
        log_error "code2wav_decoder.onnx not found (required)"
        failed=$((failed + 1))
    fi

    [ "$failed" -eq 0 ]
}

# Discover variants that have talker_unified ONNX (for TRT build)
discover_trt_variants() {
    local found=()
    for vdir in "$EXPORTED_DIR"/*/; do
        [ -d "$vdir" ] || continue
        if [ -f "${vdir}talker_unified.onnx" ]; then
            found+=("$(basename "$vdir")")
        fi
    done
    echo "${found[@]}"
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
        --dry-run)        DRY_RUN=true; shift ;;
        --pull-only)      PULL_ONLY=true; shift ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --variant <name>       Build for a specific model variant"
            echo "  --image <uri>          Override NGC container image (default: auto-detect)"
            echo "  --target-driver <ver>  Target NVIDIA driver for NGC container selection"
            echo "  --max-batch-size N     Max batch size (default: 8)"
            echo "  --max-input-len N      Max input length for prefill (default: 512)"
            echo "  --max-seq-len N        Max sequence length incl. KV cache (default: 4096)"
            echo "  --dtype bf16|fp16      Engine precision (default: bfloat16)"
            echo "  --dry-run              Show docker commands without executing"
            echo "  --pull-only            Pull the container image and exit"
            echo "  -h, --help             Show this help"
            exit 0
            ;;
        *) log_error "Unknown argument: $1"; exit 1 ;;
    esac
done

# ===========================================================================
#  Phase B: TRT engine build (trtexec from ONNX)
# ===========================================================================

log_step "Phase B: TensorRT Engine Build (trtexec)"

check_docker_gpu_ready || exit 1

if [ -n "$USER_IMAGE" ]; then
    NGC_IMAGE="$USER_IMAGE"
    log_info "Using user-specified image: $NGC_IMAGE"
else
    _NGC_VERIFY_MANIFEST=1
    NGC_IMAGE=$(resolve_ngc_image_info) \
        || { log_error "Cannot determine NGC container. Use --image."; exit 1; }
fi
ensure_ngc_image "$NGC_IMAGE" || exit 1

if $PULL_ONLY; then
    log_info "Image ready. Re-run without --pull-only to build engines."
    exit 0
fi

if [ ! -d "$EXPORTED_DIR" ]; then
    log_error "No exported models at: $EXPORTED_DIR"
    log_error "Run Phase A first: autorun.sh setup or export_all.py"
    exit 1
fi

if [[ -n "$VARIANT" && "$VARIANT" != all* ]]; then
    if [ ! -f "$EXPORTED_DIR/$VARIANT/talker_unified.onnx" ]; then
        log_error "Talker unified ONNX not found for $VARIANT. Run export_04_talker_unified.py first."
        exit 1
    fi
    VARIANTS=("$VARIANT")
else
    read -ra VARIANTS <<< "$(discover_trt_variants)"
    if [ ${#VARIANTS[@]} -eq 0 ]; then
        log_error "No talker_unified.onnx found in $EXPORTED_DIR (run export_04_talker_unified.py per variant)"
        exit 1
    fi
    log_info "Discovered variants: ${VARIANTS[*]}"
fi

FAILED=0
SUCCEEDED=0

for variant in "${VARIANTS[@]}"; do
    if build_talker_unified_trt "$variant"; then
        SUCCEEDED=$((SUCCEEDED + 1))
    else
        FAILED=$((FAILED + 1))
    fi
done

# Peripheral engines (once, shared or per-variant)
if ! build_peripheral_engines "$NGC_IMAGE"; then
    FAILED=$((FAILED + 1))
fi

echo ""
log_step "Engine Build Summary"
log_info "  Image:     $NGC_IMAGE"
log_info "  Succeeded: $SUCCEEDED (talker variants)"
[ "$FAILED" -gt 0 ] && log_error "  Failed: $FAILED"

if [ "$SUCCEEDED" -gt 0 ] && ! $DRY_RUN; then
    echo ""
    log_info "Engines: $EXPORTED_DIR/<variant>/*.engine, $TOKENIZER_DIR/*.engine"
    log_info "Next: bash scripts/bash/build_triton.sh assemble --engine-mode trt && build_triton.sh run"
fi

exit "$FAILED"

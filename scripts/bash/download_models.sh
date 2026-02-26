#!/bin/bash
# ===========================================================================
#  download_models.sh — Standalone Qwen3-TTS model downloader
#
#  Can be used independently of autorun.sh to download or update models.
#
#  Usage:
#    bash scripts/bash/download_models.sh [target_dir] [variant] [source]
#
#  Examples:
#    bash scripts/bash/download_models.sh                      # defaults
#    bash scripts/bash/download_models.sh ./models base-1.7b   # explicit
#    bash scripts/bash/download_models.sh ./models all hf      # all via HF
#    bash scripts/bash/download_models.sh ./models base-1.7b modelscope
#    MODEL_SOURCE=modelscope bash scripts/bash/download_models.sh
#
#  Environment variables:
#    MODEL_VARIANT    default: base-1.7b
#    MODEL_SOURCE     default: auto  (auto | hf | modelscope)
#    HF_MIRROR        HuggingFace mirror URL (for China users)
#    HF_ENDPOINT      HuggingFace API endpoint override
# ===========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
source "${SCRIPT_DIR}/tools.sh"

# Handle flags before positional args
if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    echo "Usage: $(basename "$0") [target_dir] [variant] [source]"
    echo ""
    list_model_variants
    echo ""
    echo "Sources: auto (default), hf (HuggingFace), modelscope"
    echo ""
    echo "Environment variables:"
    echo "  MODEL_VARIANT    Model variant (default: base-1.7b)"
    echo "  MODEL_SOURCE     Download source (default: auto)"
    echo "  HF_MIRROR        HuggingFace mirror URL"
    echo "  HF_ENDPOINT      HuggingFace API endpoint"
    exit 0
fi

if [ "${1:-}" = "--list" ] || [ "${1:-}" = "-l" ]; then
    list_model_variants
    exit 0
fi

TARGET_DIR="${1:-${REPO_ROOT}/workspace/models}"
VARIANT="${MODEL_VARIANT:-${2:-base-1.7b}}"
SOURCE="${MODEL_SOURCE:-${3:-auto}}"

# ---------------------------------------------------------------------------

echo ""
echo -e "${_CLR_BLUE}Qwen3-TTS Model Downloader${_CLR_RESET}"
echo ""
echo "  Target:    $TARGET_DIR"
echo "  Variant:   $VARIANT"
echo "  Source:     $SOURCE"
echo ""

ensure_git_lfs || exit 1

download_qwen3_tts_models "$TARGET_DIR" "$VARIANT" "$SOURCE"

echo ""
echo -e "${_CLR_GREEN}Done.${_CLR_RESET} Models are at: $TARGET_DIR"
echo ""

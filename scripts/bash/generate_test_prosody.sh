#!/bin/bash
# ===========================================================================
#  generate_test_prosody.sh — Generate test-prosody-mini via LLM
#
#  Usage:
#    bash scripts/bash/generate_test_prosody.sh
#    bash scripts/bash/generate_test_prosody.sh --count 10 --provider dashscope
#    bash scripts/bash/generate_test_prosody.sh --provider ark
# ===========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null)" \
    || REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/setup_data_synth_env.sh"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if command -v conda >/dev/null 2>&1; then
    # Prefer project env when available.
    eval "$(conda shell.bash hook 2>/dev/null)" || true
    if conda env list 2>/dev/null | awk '{print $1}' | grep -qx "qwen3-tts"; then
        conda activate qwen3-tts >/dev/null 2>&1 || true
    fi
fi

exec "${PYTHON_BIN}" "${REPO_ROOT}/eval/data_synth/generate_test_prosody.py" "$@"

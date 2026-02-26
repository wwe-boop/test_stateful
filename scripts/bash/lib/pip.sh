#!/bin/bash
# ===========================================================================
#  pip.sh — pip install helpers
#
#  Functions: pip_install, pip_install_requirements,
#             install_torch_cuda, install_qwen3_tts, validate_python_env
#  Depends:   lib/logging.sh, lib/prerequisites.sh
# ===========================================================================

[[ -n "${_LIB_PIP_LOADED:-}" ]] && return 0
_LIB_PIP_LOADED=1

_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_LIB_DIR}/logging.sh"
source "${_LIB_DIR}/prerequisites.sh"

# ---------------------------------------------------------------------------
#  pip_install <package> [package...]
# ---------------------------------------------------------------------------
pip_install() {
    if [ $# -eq 0 ]; then
        log_error "Usage: pip_install <package> [package...]"
        return 1
    fi
    log_step "pip install $*"
    python3 -m pip install --upgrade "$@" \
        || { log_error "pip install failed"; return 1; }
}

# ---------------------------------------------------------------------------
#  pip_install_requirements [requirements_file]
#  Defaults to requirements.txt in the current directory.
# ---------------------------------------------------------------------------
pip_install_requirements() {
    local req_file="${1:-requirements.txt}"
    if [ ! -f "$req_file" ]; then
        log_error "Requirements file not found: $req_file"
        return 1
    fi
    log_step "Installing from $req_file..."
    python3 -m pip install --upgrade -r "$req_file" \
        || { log_error "pip install from $req_file failed"; return 1; }
}

# ---------------------------------------------------------------------------
#  install_torch_cuda [cuda_version]
#  Installs PyTorch + torchaudio with the correct CUDA build.
#  Auto-detects the CUDA version when not provided.
#  Skips if PyTorch with CUDA is already importable.
# ---------------------------------------------------------------------------
install_torch_cuda() {
    local cuda_ver="${1:-}"

    if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        local existing
        existing=$(python3 -c "import torch; print(torch.__version__)")
        log_info "PyTorch $existing (CUDA) already installed, skipping"
        return 0
    fi

    if [ -z "$cuda_ver" ]; then
        cuda_ver=$(detect_cuda_version) || true
    fi

    if [ -z "$cuda_ver" ]; then
        log_warn "No CUDA detected — installing CPU-only PyTorch"
        pip_install torch torchaudio
        return $?
    fi

    local tag
    tag=$(cuda_to_torch_tag "$cuda_ver")
    log_step "Installing PyTorch (CUDA $cuda_ver → $tag)..."

    python3 -m pip install --upgrade torch torchaudio \
        --index-url "https://download.pytorch.org/whl/${tag}" \
        || { log_error "PyTorch install failed for $tag"; return 1; }

    if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        local tv cr
        tv=$(python3 -c "import torch; print(torch.__version__)")
        cr=$(python3 -c "import torch; print(torch.version.cuda)")
        log_info "PyTorch $tv installed (CUDA runtime: $cr)"
    else
        log_warn "PyTorch installed but torch.cuda.is_available() == False"
        log_warn "Check that your NVIDIA driver is compatible with CUDA $cuda_ver"
    fi
}

# ---------------------------------------------------------------------------
#  install_qwen3_tts <source_dir>
#  Editable-installs the Qwen3-TTS package from the submodule.
# ---------------------------------------------------------------------------
install_qwen3_tts() {
    local src_dir="$1"

    if [ -z "$src_dir" ] || [ ! -f "$src_dir/pyproject.toml" ]; then
        log_error "Qwen3-TTS source not found at: ${src_dir:-<not set>}"
        return 1
    fi

    if python3 -c "import qwen_tts" 2>/dev/null; then
        local ver
        ver=$(python3 -c \
            "import importlib.metadata; print(importlib.metadata.version('qwen-tts'))" \
            2>/dev/null || echo "unknown")
        log_info "qwen-tts ($ver) already installed, skipping"
        return 0
    fi

    log_step "Installing Qwen3-TTS from $src_dir..."
    python3 -m pip install -e "$src_dir" \
        || { log_error "Qwen3-TTS install failed"; return 1; }
    log_info "Qwen3-TTS installed"
}

# ---------------------------------------------------------------------------
#  validate_python_env
#  Quick smoke-test: imports the core packages and reports versions.
# ---------------------------------------------------------------------------
validate_python_env() {
    log_step "Validating Python environment..."

    python3 - <<'PYEOF'
import sys, importlib

required = {
    "torch":        "torch",
    "torchaudio":   "torchaudio",
    "transformers": "transformers",
    "qwen_tts":     "qwen-tts",
}

ok, fail = 0, 0
for mod, label in required.items():
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, "__version__", "?")
        print(f"  {label:20s} {ver}")
        ok += 1
    except ImportError:
        print(f"  {label:20s} ** MISSING **")
        fail += 1

# CUDA check
try:
    import torch
    if torch.cuda.is_available():
        print(f"  {'CUDA':20s} {torch.version.cuda}  "
              f"(GPU: {torch.cuda.get_device_name(0)})")
    else:
        print(f"  {'CUDA':20s} not available")
except Exception:
    pass

sys.exit(1 if fail else 0)
PYEOF

    if [ $? -ne 0 ]; then
        log_error "Some packages are missing — see above"
        return 1
    fi
    log_info "Python environment OK"
}

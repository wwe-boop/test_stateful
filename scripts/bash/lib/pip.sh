#!/bin/bash
# ===========================================================================
#  pip.sh — pip install helpers
#
#  Functions: pip_install, pip_install_requirements, install_torch_cuda,
#             install_flash_attn, install_qwen3_tts,
#             install_safetensors, install_onnx_export_deps,
#             validate_python_env
#  Depends:   lib/logging.sh, lib/prerequisites.sh, lib/network.sh
# ===========================================================================

[[ -n "${_LIB_PIP_LOADED:-}" ]] && return 0
_LIB_PIP_LOADED=1

_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_LIB_DIR}/logging.sh"
source "${_LIB_DIR}/prerequisites.sh"
source "${_LIB_DIR}/network.sh"

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

    # --index-url overrides pip.conf; re-inject user's configured mirror
    # as --extra-index-url so dependencies (nvidia-cudnn, sympy, etc.)
    # can be fetched from the faster mirror
    local pip_args=(--upgrade torch torchaudio
        --index-url "https://download.pytorch.org/whl/${tag}")

    local _cfg_index
    _cfg_index=$(python3 -m pip config get global.index-url 2>/dev/null) || true
    if [ -n "$_cfg_index" ]; then
        pip_args+=(--extra-index-url "$_cfg_index")
        log_info "附加依赖源: $_cfg_index"
    fi

    python3 -m pip install "${pip_args[@]}" \
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
#  install_flash_attn
#  Installs flash-attn with the correct pre-built wheel for the current
#  PyTorch / CUDA / Python combination.  Downloads from GitHub releases
#  via the project's mirror infrastructure (github_url).
#  Falls back to source build if no pre-built wheel matches.
# ---------------------------------------------------------------------------
install_flash_attn() {
    if python3 -c "import flash_attn" 2>/dev/null; then
        local ver
        ver=$(python3 -c "import flash_attn; print(flash_attn.__version__)")
        log_info "flash-attn ($ver) already installed, skipping"
        return 0
    fi

    if ! python3 -c "import torch" 2>/dev/null; then
        log_error "PyTorch not installed — install_flash_attn requires PyTorch first"
        return 1
    fi

    local fa_ver torch_ver cuda_major py_ver abi
    fa_ver=$(python3 -c "
import json, urllib.request
data = json.loads(urllib.request.urlopen(
    'https://pypi.org/pypi/flash-attn/json', timeout=10).read())
print(data['info']['version'])
" 2>/dev/null) || true

    if [ -z "$fa_ver" ]; then
        log_warn "无法从 PyPI 获取 flash-attn 最新版本，使用默认 2.8.3"
        fa_ver="2.8.3"
    fi

    read -r torch_ver cuda_major py_ver abi < <(python3 -c "
import torch, sys
tv = '.'.join(torch.__version__.split('+')[0].split('.')[:2])
cu = torch.version.cuda.split('.')[0] if torch.version.cuda else ''
pv = f'{sys.version_info.major}{sys.version_info.minor}'
abi = 'TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE'
print(tv, cu, pv, abi)
")

    if [ -z "$cuda_major" ]; then
        log_warn "CUDA not available, skipping flash-attn"
        return 0
    fi

    local wheel="flash_attn-${fa_ver}+cu${cuda_major}torch${torch_ver}cxx11abi${abi}-cp${py_ver}-cp${py_ver}-linux_x86_64.whl"
    local gh_release="https://github.com/Dao-AILab/flash-attention/releases/download/v${fa_ver}/${wheel}"
    local download_url
    download_url=$(github_url "$gh_release")

    log_step "Installing flash-attn ${fa_ver} (torch${torch_ver} cu${cuda_major} cp${py_ver} abi=${abi})..."
    log_info "Wheel URL: $download_url"

    local tmp_wheel="/tmp/${wheel}"
    if curl -fSL -o "$tmp_wheel" --connect-timeout 15 --max-time 600 \
            --retry 2 --retry-delay 5 "$download_url" 2>/dev/null \
       && [ -s "$tmp_wheel" ] \
       && python3 -c "import zipfile; zipfile.ZipFile('$tmp_wheel')" 2>/dev/null; then
        log_info "预编译 wheel 下载成功，安装中..."
        python3 -m pip install "$tmp_wheel" \
            && { rm -f "$tmp_wheel"; log_info "flash-attn ${fa_ver} installed"; return 0; }
        log_warn "预编译 wheel 安装失败，尝试源码编译..."
        rm -f "$tmp_wheel"
    else
        rm -f "$tmp_wheel"
        log_warn "预编译 wheel 下载失败 ($wheel)，尝试源码编译..."
    fi

    pip_install flash-attn --no-build-isolation \
        || { log_warn "flash-attn source build failed (non-fatal)"; return 1; }
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
#  install_safetensors
#  Installs safetensors — needed for writing TRT-LLM checkpoint files.
#  Idempotent, lightweight, no CUDA dependency.
# ---------------------------------------------------------------------------
install_safetensors() {
    if python3 -c "import safetensors" 2>/dev/null; then
        log_info "safetensors already installed, skipping"
        return 0
    fi

    log_info "Installing safetensors ..."
    python3 -m pip install safetensors -q \
        || log_warn "safetensors install failed (non-fatal, checkpoint will fall back to .bin)"
}

# ---------------------------------------------------------------------------
#  install_onnx_export_deps
#  Installs packages needed for ONNX export:
#    onnx, onnxruntime, onnxscript (torch.onnx 内部依赖), onnxsim (简化)
#  Idempotent — skips if already installed.
# ---------------------------------------------------------------------------
install_onnx_export_deps() {
    local missing=()

    python3 -c "import onnx" 2>/dev/null \
        || missing+=(onnx)

    python3 -c "import onnxruntime" 2>/dev/null \
        || missing+=(onnxruntime)

    python3 -c "import onnxscript" 2>/dev/null \
        || missing+=(onnxscript)

    python3 -c "import onnxsim" 2>/dev/null \
        || missing+=(onnxsim)

    if [ ${#missing[@]} -eq 0 ]; then
        log_info "ONNX export dependencies already installed"
        return 0
    fi

    log_step "Installing ONNX export dependencies: ${missing[*]}"
    python3 -m pip install --upgrade "${missing[@]}" \
        || { log_error "Failed to install ONNX dependencies"; return 1; }
    log_info "ONNX export dependencies installed"
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

optional = {
    "safetensors":  "safetensors",
    "onnx":         "onnx",
    "onnxruntime":  "onnxruntime",
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

for mod, label in optional.items():
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, "__version__", "?")
        print(f"  {label:20s} {ver}")
    except ImportError:
        print(f"  {label:20s} (not installed)")

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

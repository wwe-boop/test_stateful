#!/bin/bash
# ===========================================================================
#  pip.sh — pip install helpers
#
#  Functions: pip_install, pip_install_requirements
#  Depends:   lib/logging.sh
# ===========================================================================

[[ -n "${_LIB_PIP_LOADED:-}" ]] && return 0
_LIB_PIP_LOADED=1

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/logging.sh"

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

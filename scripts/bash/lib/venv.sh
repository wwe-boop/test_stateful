#!/bin/bash
# ===========================================================================
#  venv.sh — Python virtual-environment management
#
#  Functions: ensure_venv_cmd, create_venv, ensure_venv
#  Depends:   lib/logging.sh, lib/utils.sh
# ===========================================================================

[[ -n "${_LIB_VENV_LOADED:-}" ]] && return 0
_LIB_VENV_LOADED=1

_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_LIB_DIR}/logging.sh"
source "${_LIB_DIR}/utils.sh"

VENV_CMD=""

# ---------------------------------------------------------------------------
#  ensure_venv_cmd
#  Detects (or installs) a venv tool: mamba > conda > python3-venv.
#  Sets the global VENV_CMD variable.
# ---------------------------------------------------------------------------
ensure_venv_cmd() {
    if [ -n "$VENV_CMD" ]; then
        return 0
    fi

    if command -v mamba &>/dev/null; then
        VENV_CMD="mamba"
        log_info "Found venv tool: mamba"
        return 0
    fi

    if command -v conda &>/dev/null; then
        VENV_CMD="conda"
        log_info "Found venv tool: conda"
        return 0
    fi

    if command -v python3 &>/dev/null && python3 -m venv -h &>/dev/null; then
        VENV_CMD="venv"
        log_info "Found venv tool: python3 venv"
        return 0
    fi

    log_warn "No venv tool found, installing miniforge..."
    local INSTALLER="Miniforge3-$(uname)-$(uname -m).sh"
    retry 3 10 curl -fSL --connect-timeout 10 --max-time 600 \
        -O "https://github.com/conda-forge/miniforge/releases/latest/download/${INSTALLER}" \
        || { log_error "Failed to download miniforge"; return 1; }
    bash "${INSTALLER}" -b -p "$HOME/miniforge" \
        || { log_error "Failed to install miniforge"; return 1; }
    rm -f "${INSTALLER}"
    eval "$("$HOME/miniforge/bin/conda" shell.bash hook)"
    VENV_CMD="mamba"
    log_info "Installed miniforge, using: mamba"
}

_venv_exists() {
    local env_name="$1"
    case "$VENV_CMD" in
        mamba|conda)
            $VENV_CMD env list 2>/dev/null | awk '{print $1}' | grep -qx "$env_name"
            ;;
        venv)
            [ -d "$env_name" ] && [ -f "$env_name/bin/activate" ]
            ;;
        *)
            return 1
            ;;
    esac
}

_activate_venv() {
    local env_name="$1"
    case "$VENV_CMD" in
        mamba|conda)
            if ! declare -f conda &>/dev/null; then
                local _conda_exe
                _conda_exe=$(command -v conda 2>/dev/null || echo "${CONDA_EXE:-}")
                if [ -z "$_conda_exe" ]; then
                    log_error "conda executable not found"
                    return 1
                fi
                eval "$("$_conda_exe" shell.bash hook 2>/dev/null)" || true
            fi
            conda activate "$env_name" 2>/dev/null && return 0

            local env_path
            env_path=$($VENV_CMD env list 2>/dev/null \
                | awk -v name="$env_name" '$1==name {print $NF}')
            if [ -n "$env_path" ] && [ -f "$env_path/bin/activate" ]; then
                source "$env_path/bin/activate"
                return 0
            fi
            log_error "Could not activate conda env '$env_name'"
            return 1
            ;;
        venv)
            source "$env_name/bin/activate"
            ;;
    esac
}

# ---------------------------------------------------------------------------
#  create_venv <env_name> [python_version]
#  Creates a new environment (idempotent — skips if it exists).
# ---------------------------------------------------------------------------
create_venv() {
    local ENV_NAME="$1"
    local PYTHON_VERSION="${2:-3.10}"

    if [ -z "$ENV_NAME" ]; then
        log_error "Usage: create_venv <env_name> [python_version]"
        return 1
    fi

    ensure_venv_cmd || return 1

    if _venv_exists "$ENV_NAME"; then
        log_warn "Environment '$ENV_NAME' already exists, skipping creation"
        return 0
    fi

    log_step "Creating environment '$ENV_NAME' (python=$PYTHON_VERSION)..."

    case "$VENV_CMD" in
        mamba)
            mamba create -n "$ENV_NAME" -c conda-forge python="$PYTHON_VERSION" -y \
                || { log_error "mamba create failed"; return 1; }
            ;;
        conda)
            conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y \
                || { log_error "conda create failed"; return 1; }
            ;;
        venv)
            local SYS_PY_VER
            SYS_PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
            if [ "$SYS_PY_VER" != "$PYTHON_VERSION" ]; then
                log_warn "System python is $SYS_PY_VER, requested $PYTHON_VERSION"
                log_warn "venv cannot switch python version, will use $SYS_PY_VER"
            fi
            python3 -m venv "$ENV_NAME" \
                || { log_error "python3 -m venv failed"; return 1; }
            ;;
    esac

    log_info "Environment '$ENV_NAME' created successfully"
}

# ---------------------------------------------------------------------------
#  ensure_venv <env_name> [python_version]
#  Creates (if needed) AND activates the environment.
# ---------------------------------------------------------------------------
ensure_venv() {
    local ENV_NAME="$1"
    local PYTHON_VERSION="${2:-3.10}"

    if [ -z "$ENV_NAME" ]; then
        log_error "Usage: ensure_venv <env_name> [python_version]"
        return 1
    fi

    ensure_venv_cmd || return 1

    if _venv_exists "$ENV_NAME"; then
        log_info "Environment '$ENV_NAME' already exists"
    else
        create_venv "$ENV_NAME" "$PYTHON_VERSION" || return 1
    fi

    _activate_venv "$ENV_NAME" || { log_error "Failed to activate '$ENV_NAME'"; return 1; }
    log_info "Activated: $ENV_NAME ($(python3 --version 2>&1))"
}

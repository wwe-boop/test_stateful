#!/bin/bash
# ===========================================================================
#  engine.sh — Standalone TTS Engine lifecycle helpers
#
#  Functions: resolve_variant_model_dir, engine_start, engine_stop,
#             engine_status, engine_health_check
#  Depends:   lib/logging.sh, lib/utils.sh
#
#  Manages the standalone TTS engine server (python -m engine.server)
#  as a background process with PID file tracking.
# ===========================================================================

[[ -n "${_LIB_ENGINE_LOADED:-}" ]] && return 0
_LIB_ENGINE_LOADED=1

_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_LIB_DIR}/logging.sh"
source "${_LIB_DIR}/utils.sh"

ENGINE_GRPC_PORT="${ENGINE_GRPC_PORT:-50051}"
ENGINE_WEBSOCKET_PORT="${ENGINE_WEBSOCKET_PORT:-50052}"
ENGINE_HEALTH_PORT="${ENGINE_HEALTH_PORT:-8080}"
ENGINE_IMAGE="${ENGINE_IMAGE:-qwen3-engine:26.02}"
ENGINE_CONTAINER_NAME="${ENGINE_CONTAINER_NAME:-qwen3-engine}"
# Standalone engine: interpreter selection (Phase A conda env is not auto-activated for deploy).
#   ENGINE_PYTHON        If set, must be an executable python with torch + deps.
#   QWEN3_TTS_ENV_NAME   Conda env name to look up (default: qwen3-tts).

# ---------------------------------------------------------------------------
#  resolve_engine_python_bin <repo_root>
#  Prints a python executable path for engine.server. Prefers ENGINE_PYTHON,
#  then <repo>/.venv/bin/python, then conda env QWEN3_TTS_ENV_NAME, else python3.
# ---------------------------------------------------------------------------
resolve_engine_python_bin() {
    local repo_root="$1"
    local env_name="${QWEN3_TTS_ENV_NAME:-qwen3-tts}"

    if [ -n "${ENGINE_PYTHON:-}" ] && [ -x "$ENGINE_PYTHON" ]; then
        printf '%s\n' "$ENGINE_PYTHON"
        return 0
    fi
    if [ -x "$repo_root/.venv/bin/python" ]; then
        printf '%s\n' "$repo_root/.venv/bin/python"
        return 0
    fi
    local list_cmd=""
    command -v conda &>/dev/null && list_cmd=conda
    [ -z "$list_cmd" ] && command -v mamba &>/dev/null && list_cmd=mamba
    if [ -n "$list_cmd" ]; then
        local prefix
        prefix=$($list_cmd env list 2>/dev/null | awk -v n="$env_name" '$1 == n { print $NF; exit }')
        if [ -n "$prefix" ] && [ -x "$prefix/bin/python" ]; then
            printf '%s\n' "$prefix/bin/python"
            return 0
        fi
    fi
    if command -v python3 &>/dev/null; then
        command -v python3
        return 0
    fi
    if command -v python &>/dev/null; then
        command -v python
        return 0
    fi
    return 1
}

# ---------------------------------------------------------------------------
#  resolve_variant_model_dir <variant>
#  Maps a variant name to the HuggingFace model directory name.
#  Echoes the directory name (not full path) on success, returns 1 on failure.
# ---------------------------------------------------------------------------
resolve_variant_model_dir() {
    local variant="$1"
    case "$variant" in
        design-1.7b) echo "Qwen3-TTS-12Hz-1.7B-VoiceDesign" ;;
        custom-1.7b) echo "Qwen3-TTS-12Hz-1.7B-CustomVoice" ;;
        base-1.7b)   echo "Qwen3-TTS-12Hz-1.7B-Base" ;;
        custom-0.6b) echo "Qwen3-TTS-12Hz-0.6B-CustomVoice" ;;
        base-0.6b)   echo "Qwen3-TTS-12Hz-0.6B-Base" ;;
        *)
            log_error "Unknown variant: $variant"
            return 1
            ;;
    esac
}

# ---------------------------------------------------------------------------
#  resolve_engine_paths <repo_root> <variant>
#  Auto-discovers tokenizer, weights, and engine directories.
#  Sets global variables: _ENGINE_TOKENIZER_DIR, _ENGINE_WEIGHTS_DIR,
#  _ENGINE_DIR (TRT plan directory).
#  Returns 1 if critical paths are missing.
# ---------------------------------------------------------------------------
resolve_engine_paths() {
    local repo_root="$1"
    local variant="$2"
    local exported_dir="$repo_root/workspace/exported"
    local models_dir="$repo_root/workspace/models"

    # Tokenizer dir: workspace/models/<model_dir>
    local model_dir_name
    model_dir_name=$(resolve_variant_model_dir "$variant") || return 1
    _ENGINE_TOKENIZER_DIR="$models_dir/$model_dir_name"
    if [ ! -d "$_ENGINE_TOKENIZER_DIR" ]; then
        log_error "Tokenizer directory not found: $_ENGINE_TOKENIZER_DIR"
        log_error "Run Phase A first: autorun.sh setup"
        return 1
    fi

    # Weights dir: workspace/exported/<variant>/weights
    _ENGINE_WEIGHTS_DIR="$exported_dir/$variant/weights"
    if [ ! -d "$_ENGINE_WEIGHTS_DIR" ]; then
        log_error "Weights directory not found: $_ENGINE_WEIGHTS_DIR"
        log_error "Run Phase A first: autorun.sh setup"
        return 1
    fi

    # TRT fused engine:
    #   - Phase B (build_engines.sh): workspace/exported/<variant>/talker_code2wav_fused.engine
    #   - Triton assemble copy:       .../engines/talker_code2wav_fused/model.plan
    local fused_subdir="$exported_dir/$variant/engines/talker_code2wav_fused"
    local fused_flat="$exported_dir/$variant/talker_code2wav_fused.engine"
    if [ -d "$fused_subdir" ] && { [ -f "$fused_subdir/model.plan" ] || [ -f "$fused_subdir/talker_code2wav_fused.engine" ]; }; then
        _ENGINE_DIR="$fused_subdir"
    elif [ -f "$fused_flat" ]; then
        _ENGINE_DIR="$exported_dir/$variant"
    else
        _ENGINE_DIR=""
        log_warn "TRT engine directory not found, engine will run in stub/ONNX mode"
    fi

    return 0
}

# ---------------------------------------------------------------------------
#  engine_pid_file <repo_root>
#  Returns the PID file path for the standalone engine.
# ---------------------------------------------------------------------------
engine_pid_file() {
    echo "$1/workspace/.engine.pid"
}

# ---------------------------------------------------------------------------
#  engine_log_file <repo_root>
#  Returns the log file path for the standalone engine.
# ---------------------------------------------------------------------------
engine_log_file() {
    echo "$1/workspace/engine.log"
}

# ---------------------------------------------------------------------------
#  engine_start <repo_root> <variant> [options...]
#
#  Starts the standalone TTS engine server as a background process.
#  Options:
#    --port <N>            gRPC port (default: $ENGINE_GRPC_PORT)
#    --ws-port <N>         WebSocket port (default: $ENGINE_WEBSOCKET_PORT)
#    --device <N>          GPU device (default: 0)
#    --max-batch <N>       Max batch size (default: 48)
#    --max-sessions <N>    Max concurrent sessions (default: 128)
#    --foreground          Run in foreground (don't daemonize)
# ---------------------------------------------------------------------------
engine_start() {
    local repo_root="$1"
    local variant="$2"
    shift 2

    local port="$ENGINE_GRPC_PORT"
    local ws_port="$ENGINE_WEBSOCKET_PORT"
    local device=0
    local max_batch=48
    local max_sessions=128
    local foreground=false

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --port)          port="$2"; shift 2 ;;
            --ws-port)       ws_port="$2"; shift 2 ;;
            --device)        device="$2"; shift 2 ;;
            --max-batch)     max_batch="$2"; shift 2 ;;
            --max-sessions)  max_sessions="$2"; shift 2 ;;
            --foreground)    foreground=true; shift ;;
            *)               shift ;;
        esac
    done

    local pid_file
    pid_file=$(engine_pid_file "$repo_root")
    local log_file
    log_file=$(engine_log_file "$repo_root")

    # Check if already running
    if [ -f "$pid_file" ]; then
        local old_pid
        old_pid=$(cat "$pid_file" 2>/dev/null)
        if [ -n "$old_pid" ] && kill -0 "$old_pid" 2>/dev/null; then
            log_warn "Engine already running (PID $old_pid)"
            log_info "Stop first: deploy.sh stop"
            return 1
        fi
        rm -f "$pid_file"
    fi

    resolve_engine_paths "$repo_root" "$variant" || return 1

    local pybin
    pybin=$(resolve_engine_python_bin "$repo_root") || {
        log_error "No python interpreter found for standalone engine"
        return 1
    }
    if ! "$pybin" -c "import torch" 2>/dev/null; then
        log_error "Selected Python cannot import torch: $pybin"
        log_error "Activate Phase A environment: conda activate ${QWEN3_TTS_ENV_NAME:-qwen3-tts}"
        log_error "Or set ENGINE_PYTHON to a Python that has PyTorch (and project deps) installed."
        return 1
    fi

    log_step "Starting Standalone TTS Engine"
    log_info "  Python:       $pybin"
    log_info "  Variant:      $variant"
    log_info "  Tokenizer:    $_ENGINE_TOKENIZER_DIR"
    log_info "  Weights:      $_ENGINE_WEIGHTS_DIR"
    log_info "  TRT Engines:  ${_ENGINE_DIR:-stub mode}"
    log_info "  GPU Device:   $device"
    log_info "  Max Batch:    $max_batch"
    log_info "  Max Sessions: $max_sessions"
    log_info "  gRPC Port:    $port"
    log_info "  WS Port:      $ws_port"

    local cmd=(
        "$pybin" -m engine.server
        --tokenizer-dir "$_ENGINE_TOKENIZER_DIR"
        --weights-dir "$_ENGINE_WEIGHTS_DIR"
        --device "$device"
        --max-batch "$max_batch"
        --max-sessions "$max_sessions"
        --port "$port"
        --ws-port "$ws_port"
    )

    if [ -n "$_ENGINE_DIR" ]; then
        cmd+=(--engine-dir "$_ENGINE_DIR")
    fi

    if $foreground; then
        log_info "Running in foreground (Ctrl+C to stop)..."
        cd "$repo_root" && exec "${cmd[@]}"
    fi

    mkdir -p "$(dirname "$log_file")"
    cd "$repo_root" && nohup "${cmd[@]}" > "$log_file" 2>&1 &
    local engine_pid=$!

    echo "$engine_pid" > "$pid_file"
    log_info "Engine started (PID $engine_pid)"
    log_info "  Log file: $log_file"
    log_info "  PID file: $pid_file"

    return 0
}

# ---------------------------------------------------------------------------
#  engine_stop <repo_root>
#  Stops the standalone TTS engine gracefully (SIGTERM, then SIGKILL).
# ---------------------------------------------------------------------------
engine_stop() {
    local repo_root="$1"
    local pid_file
    pid_file=$(engine_pid_file "$repo_root")

    if [ ! -f "$pid_file" ]; then
        log_info "No engine PID file found (not running?)"
        return 0
    fi

    local pid
    pid=$(cat "$pid_file" 2>/dev/null)
    if [ -z "$pid" ]; then
        rm -f "$pid_file"
        log_info "Empty PID file, cleaned up"
        return 0
    fi

    if ! kill -0 "$pid" 2>/dev/null; then
        rm -f "$pid_file"
        log_info "Engine process (PID $pid) not running, cleaned up PID file"
        return 0
    fi

    log_info "Stopping engine (PID $pid)..."
    kill -TERM "$pid" 2>/dev/null

    local elapsed=0
    while [ "$elapsed" -lt 10 ]; do
        if ! kill -0 "$pid" 2>/dev/null; then
            rm -f "$pid_file"
            log_info "Engine stopped gracefully"
            return 0
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done

    log_warn "Engine did not stop within 10s, sending SIGKILL..."
    kill -9 "$pid" 2>/dev/null
    sleep 1
    rm -f "$pid_file"
    log_info "Engine killed"
    return 0
}

# ---------------------------------------------------------------------------
#  engine_status <repo_root>
#  Returns status: "none" | "running" | "healthy"
# ---------------------------------------------------------------------------
engine_status() {
    local repo_root="$1"
    local pid_file
    pid_file=$(engine_pid_file "$repo_root")

    if [ ! -f "$pid_file" ]; then
        echo "none"
        return 0
    fi

    local pid
    pid=$(cat "$pid_file" 2>/dev/null)
    if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
        rm -f "$pid_file"
        echo "none"
        return 0
    fi

    echo "running"
    return 0
}

# ---------------------------------------------------------------------------
#  engine_health_check <port> [timeout_sec]
#  Waits for the engine gRPC port to become reachable.
# ---------------------------------------------------------------------------
engine_health_check() {
    local port="${1:-$ENGINE_GRPC_PORT}"
    local timeout="${2:-60}"
    local elapsed=0
    local interval=2

    while [ "$elapsed" -lt "$timeout" ]; do
        if command -v grpc_health_probe &>/dev/null; then
            if grpc_health_probe -addr "localhost:${port}" -connect-timeout 1s &>/dev/null; then
                log_info "Engine healthy on port $port"
                return 0
            fi
        else
            # Fallback: check if port is open
            if (echo >/dev/tcp/localhost/"$port") 2>/dev/null; then
                log_info "Engine port $port is open (gRPC health probe not installed)"
                return 0
            fi
        fi
        sleep "$interval"
        elapsed=$((elapsed + interval))
    done

    log_error "Engine health check timed out after ${timeout}s"
    return 1
}

# ---------------------------------------------------------------------------
#  engine_docker_image_has_app <image_tag>
#  Returns 0 if the image was built from Dockerfile.engine (bundled engine/ under /app).
#  Returns 1 if the tag points at a wrong image (e.g. base TensorRT retagged as qwen3-engine).
# ---------------------------------------------------------------------------
engine_docker_image_has_app() {
    local image="$1"
    docker run --rm --entrypoint "" "$image" \
        python3 -c "import engine.server" &>/dev/null
}

# ---------------------------------------------------------------------------
#  engine_build_image <repo_root> [image_tag]
#  Builds the Docker image for the standalone engine.
# ---------------------------------------------------------------------------
engine_build_image() {
    local repo_root="$1"
    local image_tag="${2:-$ENGINE_IMAGE}"
    local dockerfile="$repo_root/Dockerfile.engine"

    if [ ! -f "$dockerfile" ]; then
        log_error "Dockerfile not found: $dockerfile"
        return 1
    fi

    log_step "Building engine Docker image: $image_tag"
    # BuildKit: enables RUN --mount cache for pip (faster rebuilds; see Dockerfile.engine).
    DOCKER_BUILDKIT=1 docker build -t "$image_tag" -f "$dockerfile" "$repo_root" \
        || { log_error "Docker build failed"; return 1; }
    log_info "Image built: $image_tag"
}

# ---------------------------------------------------------------------------
#  engine_start_docker <repo_root> <variant> [options...]
#
#  Starts the standalone TTS engine server inside a Docker container.
#  Options: same as engine_start, plus --image <tag>
# ---------------------------------------------------------------------------
engine_start_docker() {
    local repo_root="$1"
    local variant="$2"
    shift 2

    local port="$ENGINE_GRPC_PORT"
    local ws_port="$ENGINE_WEBSOCKET_PORT"
    local health_port="$ENGINE_HEALTH_PORT"
    local device=0
    local max_batch=48
    local max_sessions=128
    local max_seq_len=""
    local image="$ENGINE_IMAGE"
    local container_name="$ENGINE_CONTAINER_NAME"

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --port)          port="$2"; shift 2 ;;
            --ws-port)       ws_port="$2"; shift 2 ;;
            --device)        device="$2"; shift 2 ;;
            --max-batch)     max_batch="$2"; shift 2 ;;
            --max-sessions)  max_sessions="$2"; shift 2 ;;
            --max-seq-len)   max_seq_len="$2"; shift 2 ;;
            --image)         image="$2"; shift 2 ;;
            --name)          container_name="$2"; shift 2 ;;
            *)               shift ;;
        esac
    done

    if docker inspect "$container_name" &>/dev/null; then
        log_warn "Container '$container_name' already exists"
        if docker inspect -f '{{.State.Running}}' "$container_name" 2>/dev/null | grep -q true; then
            log_warn "Container is running. Stop first: deploy.sh stop"
            return 1
        fi
        docker rm "$container_name" >/dev/null 2>&1
    fi

    resolve_engine_paths "$repo_root" "$variant" || return 1

    log_step "Starting Engine (Docker: $image)"
    log_info "  Container:    $container_name"
    log_info "  Variant:      $variant"
    log_info "  Tokenizer:    $_ENGINE_TOKENIZER_DIR"
    log_info "  Weights:      $_ENGINE_WEIGHTS_DIR"
    log_info "  TRT Engines:  ${_ENGINE_DIR:-stub mode}"
    log_info "  GPU Device:   $device"
    log_info "  Max Batch:    $max_batch"
    log_info "  Max Sessions: $max_sessions"
    log_info "  Max Seq Len:  ${max_seq_len:-auto}"
    log_info "  gRPC Port:    $port"
    log_info "  WS Port:      $ws_port"

    # Image bundles engine/ + engine.yaml under /app; mount only workspace/ for data.
    local workspace_host="$repo_root/workspace"
    local tk_mount="/data/models/$(basename "$_ENGINE_TOKENIZER_DIR")"
    local wt_mount="/data/exported/$variant/weights"
    local eng_mount=""
    if [[ -n "${_ENGINE_DIR:-}" ]]; then
        eng_mount="/data${_ENGINE_DIR#"$workspace_host"}"
    fi

    local -a run_cmd=(
        python3 -m engine.server
        --config /app/engine.yaml
        --tokenizer-dir "$tk_mount"
        --weights-dir "$wt_mount"
    )
    if [[ -n "$eng_mount" ]]; then
        run_cmd+=( --engine-dir "$eng_mount" )
    fi
    run_cmd+=(
        --device "$device"
        --max-batch "$max_batch"
        --max-sessions "$max_sessions"
        --port "$port"
        --ws-port "$ws_port"
    )

    local -a env_args=(
        -e "ENGINE_SCHEDULER_MAX_BATCH_SIZE=$max_batch"
        -e "ENGINE_SERVER_WEBSOCKET_PORT=$ws_port"
        -e "ENGINE_SERVER_HEALTH_PORT=$health_port"
    )
    if [[ -n "$max_seq_len" ]]; then
        env_args+=( -e "ENGINE_SCHEDULER_MAX_SEQ_LEN=$max_seq_len" )
    fi

    docker run --gpus all -d \
        --name "$container_name" \
        -w /app \
        -e "PYTHONPATH=/app" \
        -v "$workspace_host:/data:ro" \
        -p "${port}:${port}" \
        -p "${ws_port}:${ws_port}" \
        -p "${health_port}:${health_port}" \
        --shm-size=4g \
        "${env_args[@]}" \
        "$image" \
        "${run_cmd[@]}" \
        || { log_error "Docker run failed"; return 1; }

    log_info "Container started: $container_name"
    log_info "  Logs: docker logs -f $container_name"
}

# ---------------------------------------------------------------------------
#  engine_stop_docker [container_name]
#  Stops the engine Docker container.
# ---------------------------------------------------------------------------
engine_stop_docker() {
    local container_name="${1:-$ENGINE_CONTAINER_NAME}"
    if ! docker inspect "$container_name" &>/dev/null; then
        log_info "No container '$container_name' found"
        return 0
    fi
    log_info "Stopping container '$container_name'..."
    docker stop "$container_name" >/dev/null 2>&1
    docker rm "$container_name" >/dev/null 2>&1
    log_info "Container stopped and removed"
}

# ---------------------------------------------------------------------------
#  engine_show_status <repo_root>
#  Prints detailed status info (for human consumption).
# ---------------------------------------------------------------------------
engine_show_status() {
    local repo_root="$1"
    local pid_file
    pid_file=$(engine_pid_file "$repo_root")
    local log_file
    log_file=$(engine_log_file "$repo_root")
    local status
    status=$(engine_status "$repo_root")

    case "$status" in
        none)
            log_info "Standalone engine: NOT RUNNING"
            ;;
        running|healthy)
            local pid
            pid=$(cat "$pid_file" 2>/dev/null)
            log_info "Standalone engine: RUNNING (PID $pid)"
            if [ -f "$log_file" ]; then
                log_info "  Log file: $log_file"
                log_info "  Last 5 lines:"
                tail -5 "$log_file" 2>/dev/null | while IFS= read -r line; do
                    echo "    $line"
                done
            fi
            ;;
    esac
}

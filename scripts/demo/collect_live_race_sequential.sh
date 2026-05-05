#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"

VARIANT="${MODEL_VARIANT:-custom-1.7b}"
TEXT="${QWEN_DEMO_TEXT:-你好，这是千问3 TTS token级流式语音演示。}"
SPEAKER="${QWEN_DEMO_DEFAULT_SPEAKER:-Serena}"
LANGUAGE="${QWEN_DEMO_DEFAULT_LANGUAGE:-auto}"
CACHE_MODE="${QWEN_DEMO_CACHE_MODE:-hit}"
OUTPUT="${QWEN_DEMO_RACE_TRACE:-${REPO_ROOT}/workspace/demo_traces/race_default.json}"
TRITON_GRPC_PORT="${TRITON_GRPC_PORT:-8001}"
TRITON_MAX_BATCH_SLOTS="${TRITON_MAX_BATCH_SLOTS:-128}"
TRITON_MAX_SESSIONS="${TRITON_MAX_SESSIONS:-128}"
ENGINE_WS_PORT="${ENGINE_WEBSOCKET_PORT:-50052}"
ENGINE_MAX_BATCH="${ENGINE_MAX_BATCH_SIZE:-128}"
ENGINE_MAX_SESSIONS="${ENGINE_MAX_SESSIONS:-16}"
TIMEOUT_SEC="${QWEN_DEMO_RACE_TIMEOUT_SEC:-120}"
ENGINE_RETRIES="${QWEN_DEMO_ENGINE_CAPTURE_RETRIES:-4}"
OFFICIAL_WARMUP_ROUNDS="${QWEN_DEMO_OFFICIAL_WARMUP_ROUNDS:-1}"
OFFICIAL_WARMUP_TEXT="${QWEN_DEMO_OFFICIAL_WARMUP_TEXT:-你好。}"
RUN_OFFICIAL=true
RUN_ENGINE=true
RUN_TRITON=true
LEAVE_TRITON_RUNNING=true
STRICT=false

usage() {
  cat <<EOF
Usage: scripts/demo/collect_live_race_sequential.sh [options]

Sequentially captures real Performance PK traces/audio without keeping all large backends in GPU memory at once.
It writes ${OUTPUT} and leaves Triton running by default so the WebUI can replay the captured result and run live concurrency.

Options:
  --variant <name>             Model variant (default: ${VARIANT})
  --text <text>                Text used for all four backends
  --speaker <name>             Speaker (default: ${SPEAKER})
  --language <name>            Language (default: ${LANGUAGE})
  --cache-mode <hit|miss|auto> Cache mode (default: ${CACHE_MODE})
  --output <path>              Race trace JSON output (default: workspace/demo_traces/race_default.json)
  --triton-slots <N>           Triton MAX_BATCH_SLOTS for the final Triton run (default: ${TRITON_MAX_BATCH_SLOTS})
  --engine-sessions <N>        Standalone engine max sessions during single-request capture (default: ${ENGINE_MAX_SESSIONS})
  --timeout-sec <N>            Per-backend measurement timeout (default: ${TIMEOUT_SEC})
  --engine-retries <N>         Bare-engine WebSocket capture retries (default: ${ENGINE_RETRIES})
  --official-warmup-rounds <N> Unmeasured warmup passes per official PyTorch mode (default: ${OFFICIAL_WARMUP_ROUNDS})
  --official-warmup-text <txt> Text used for official PyTorch warmup
  --skip-official              Do not run official PyTorch baselines
  --skip-engine                Do not run standalone bare engine
  --skip-triton                Do not run Triton
  --leave-triton-stopped       Stop Triton after capture instead of leaving it ready for WebUI
  --strict                     Fail immediately when a requested backend cannot be measured
  -h, --help                   Show this help

Notes:
  This script intentionally stops compose-managed engine/Triton between phases.
  That is the point: official PyTorch, bare engine and Triton get isolated VRAM instead of racing each other for cache.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --variant)
      VARIANT="$2"
      shift 2
      ;;
    --text)
      TEXT="$2"
      shift 2
      ;;
    --speaker)
      SPEAKER="$2"
      shift 2
      ;;
    --language)
      LANGUAGE="$2"
      shift 2
      ;;
    --cache-mode)
      CACHE_MODE="$2"
      shift 2
      ;;
    --output)
      OUTPUT="$2"
      shift 2
      ;;
    --triton-slots)
      TRITON_MAX_BATCH_SLOTS="$2"
      shift 2
      ;;
    --engine-sessions)
      ENGINE_MAX_SESSIONS="$2"
      shift 2
      ;;
    --timeout-sec)
      TIMEOUT_SEC="$2"
      shift 2
      ;;
    --engine-retries)
      ENGINE_RETRIES="$2"
      shift 2
      ;;
    --official-warmup-rounds)
      OFFICIAL_WARMUP_ROUNDS="$2"
      shift 2
      ;;
    --official-warmup-text)
      OFFICIAL_WARMUP_TEXT="$2"
      shift 2
      ;;
    --skip-official)
      RUN_OFFICIAL=false
      shift
      ;;
    --skip-engine)
      RUN_ENGINE=false
      shift
      ;;
    --skip-triton)
      RUN_TRITON=false
      shift
      ;;
    --leave-triton-stopped)
      LEAVE_TRITON_RUNNING=false
      shift
      ;;
    --strict)
      STRICT=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

cd "${REPO_ROOT}"

log() {
  printf '\n[%s] %s\n' "$(date +%H:%M:%S)" "$*"
}

choose_python() {
  if [[ -n "${QWEN_DEMO_PYTHON:-}" ]]; then
    printf '%s\n' "${QWEN_DEMO_PYTHON}"
    return
  fi
  if [[ -x "${HOME}/miniforge3/envs/qwen3-tts/bin/python" ]]; then
    printf '%s\n' "${HOME}/miniforge3/envs/qwen3-tts/bin/python"
    return
  fi
  if [[ -x "${HOME}/miniconda3/envs/qwen3-tts/bin/python" ]]; then
    printf '%s\n' "${HOME}/miniconda3/envs/qwen3-tts/bin/python"
    return
  fi
  printf '%s\n' "python3"
}

stop_gateway() {
  local gateway="$1"
  log "Stopping compose ${gateway} to release GPU memory"
  bash scripts/bash/compose.sh down --gateway "${gateway}" >/dev/null 2>&1 || true
}

start_engine() {
  log "Starting standalone bare engine (${VARIANT}, max_sessions=${ENGINE_MAX_SESSIONS})"
  bash scripts/bash/compose.sh up \
    --gateway engine \
    --variant "${VARIANT}" \
    --max-batch "${ENGINE_MAX_BATCH}" \
    --max-sessions "${ENGINE_MAX_SESSIONS}"
}

wait_engine_websocket() {
  local url="http://localhost:${ENGINE_WS_PORT}/v1/capabilities"
  local timeout="${QWEN_DEMO_ENGINE_WS_READY_TIMEOUT_SEC:-90}"
  local elapsed=0
  log "Waiting for bare-engine WebSocket capabilities: ${url}"
  while [[ "${elapsed}" -lt "${timeout}" ]]; do
    if curl -fsS "${url}" >/dev/null 2>&1; then
      log "Bare-engine WebSocket ready"
      return 0
    fi
    sleep 1
    elapsed=$((elapsed + 1))
  done
  echo "Bare-engine WebSocket did not become ready after ${timeout}s: ${url}" >&2
  return 1
}

start_triton() {
  log "Starting Triton (${VARIANT}, MAX_BATCH_SLOTS=${TRITON_MAX_BATCH_SLOTS}, MAX_SESSIONS=${TRITON_MAX_SESSIONS})"
  TRITON_MAX_BATCH_SLOTS="${TRITON_MAX_BATCH_SLOTS}" \
  TRITON_MAX_SESSIONS="${TRITON_MAX_SESSIONS}" \
    bash scripts/bash/compose.sh up --gateway triton --variant "${VARIANT}" --prepare
}

run_collector() {
  local label="$1"
  shift
  local args=(
    --text "${TEXT}"
    --speaker "${SPEAKER}"
    --language "${LANGUAGE}"
    --cache-mode "${CACHE_MODE}"
    --timeout-sec "${TIMEOUT_SEC}"
    --official-warmup-rounds "${OFFICIAL_WARMUP_ROUNDS}"
    --official-warmup-text "${OFFICIAL_WARMUP_TEXT}"
    --output "${OUTPUT}"
  )
  if ${STRICT}; then
    args+=(--fail-fast)
  fi
  log "Collecting ${label}"
  "${PYTHON_BIN}" scripts/demo/collect_demo_traces.py "${args[@]}" "$@"
}

dump_gateway_logs() {
  local gateway="$1"
  local container=""
  case "${gateway}" in
    engine) container="${ENGINE_CONTAINER_NAME:-qwen3-engine}" ;;
    triton) container="${TRITON_CONTAINER_NAME:-qwen3-tts-triton}" ;;
    *) return 0 ;;
  esac
  if ! docker ps -a --format '{{.Names}}' | grep -Fxq "${container}"; then
    return 0
  fi
  mkdir -p workspace/logs
  local log_path="workspace/logs/${gateway}-capture-$(date +%Y%m%d-%H%M%S).log"
  docker logs --tail 400 "${container}" > "${log_path}" 2>&1 || true
  log "Saved ${gateway} container logs: ${log_path}"
}

PYTHON_BIN="$(choose_python)"
mkdir -p "$(dirname "${OUTPUT}")"
rm -f "${OUTPUT}"

log "Sequential live race capture will write: ${OUTPUT}"
log "Backends: official=$(${RUN_OFFICIAL} && echo yes || echo no), engine=$(${RUN_ENGINE} && echo yes || echo no), triton=$(${RUN_TRITON} && echo yes || echo no)"

if ! ${RUN_OFFICIAL} && ! ${RUN_ENGINE} && ! ${RUN_TRITON}; then
  run_collector "fixture baseline"
fi

if ${RUN_OFFICIAL}; then
  stop_gateway engine
  stop_gateway triton
  QWEN_DEMO_ENABLE_OFFICIAL_LIVE=1 run_collector "official PyTorch baselines" --live-official
fi

if ${RUN_ENGINE}; then
  stop_gateway triton
  start_engine
  wait_engine_websocket || true
  run_collector "bare engine" --live-engine --engine-ws "ws://localhost:${ENGINE_WS_PORT}/v1/ws" --engine-retries "${ENGINE_RETRIES}"
  dump_gateway_logs engine
  stop_gateway engine
fi

if ${RUN_TRITON}; then
  stop_gateway engine
  start_triton
  run_collector "Triton" --live-triton --triton-grpc "localhost:${TRITON_GRPC_PORT}"
  if ! ${LEAVE_TRITON_RUNNING}; then
    stop_gateway triton
  fi
elif ! ${LEAVE_TRITON_RUNNING}; then
  stop_gateway triton
fi

log "Sequential live race capture complete"
printf 'Trace: %s\n' "${OUTPUT}"
if ${RUN_TRITON} && ${LEAVE_TRITON_RUNNING}; then
  printf 'Triton is left running on localhost:%s for WebUI live replay/concurrency.\n' "${TRITON_GRPC_PORT}"
fi

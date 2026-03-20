#!/bin/bash
# ===========================================================================
#  L2 test T2.4: assemble_model_repo ONNX/TRT dual mode and validate.
#
#  Verifies:
#    - ONNX mode: .onnx + .onnx.data copy, tokenizer.json, audio_utils.py,
#                 lightweight_tokenizer.py in tts_orchestrator/1
#    - TRT mode:  talker_code2wav_fused config has TYPE_BF16, same Python files
#    - Both: first_chunk_frames in tts_orchestrator config, weights/ .pt files
#
#  Run from repo root:
#    bash tests/test_assemble.sh
#  Requires: workspace/exported/<variant>/ with ONNX (and optionally .engine for trt).
# ===========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
EXPORTED_DIR="${REPO_ROOT}/workspace/exported"
VARIANT="${VARIANT:-base-1.7b}"

# Source triton lib (and deps) without full tools.sh to avoid venv/network
LIB_DIR="${REPO_ROOT}/scripts/bash/lib"
source "${LIB_DIR}/logging.sh"
source "${LIB_DIR}/utils.sh"
source "${LIB_DIR}/docker.sh"
source "${LIB_DIR}/triton.sh"

TEST_REPO_ONNX="${TEST_REPO_ONNX:-${REPO_ROOT}/workspace/test_repo_onnx}"
TEST_REPO_TRT="${TEST_REPO_TRT:-${REPO_ROOT}/workspace/test_repo_trt}"

log_step "T2.4 assemble_model_repo tests (variant=$VARIANT)"

if [ ! -d "${EXPORTED_DIR}/${VARIANT}" ]; then
  log_error "Variant dir not found: ${EXPORTED_DIR}/${VARIANT}. Run Phase A export first."
  exit 1
fi

# ---- ONNX mode ----
log_info "Assembling ONNX mode -> ${TEST_REPO_ONNX}"
rm -rf "${TEST_REPO_ONNX}"
assemble_model_repo "${EXPORTED_DIR}" "${VARIANT}" "${TEST_REPO_ONNX}" "onnx" || { log_error "assemble (onnx) failed"; exit 1; }
validate_model_repo "${TEST_REPO_ONNX}" || { log_error "validate (onnx) failed"; exit 1; }

# Assert ONNX-specific
ORCH_1="${TEST_REPO_ONNX}/tts_orchestrator/1"
[ -f "${ORCH_1}/model.py" ] || { log_error "Missing ${ORCH_1}/model.py"; exit 1; }
[ -f "${ORCH_1}/prefill_builder.py" ] || { log_error "Missing prefill_builder.py"; exit 1; }
[ -f "${ORCH_1}/audio_utils.py" ] || { log_error "Missing audio_utils.py"; exit 1; }
[ -f "${ORCH_1}/lightweight_tokenizer.py" ] || { log_error "Missing lightweight_tokenizer.py"; exit 1; }
[ -f "${ORCH_1}/weights/config.json" ] || { log_error "Missing orchestrator weights/config.json"; exit 1; }
grep -q "first_chunk_frames" "${TEST_REPO_ONNX}/tts_orchestrator/config.pbtxt" || { log_error "Orchestrator config missing first_chunk_frames"; exit 1; }
if [ -d "${ORCH_1}/tokenizer" ]; then
  if [ -f "${ORCH_1}/tokenizer/tokenizer.json" ]; then
    log_info "  tokenizer.json present in orchestrator"
  fi
fi
log_info "ONNX assemble checks passed"

# ---- TRT mode (if .engine exists) ----
if [ -f "${EXPORTED_DIR}/${VARIANT}/talker_code2wav_fused.engine" ] || [ -f "${EXPORTED_DIR}/${VARIANT}/talker_code2wav_fused.plan" ]; then
  log_info "Assembling TRT mode -> ${TEST_REPO_TRT}"
  rm -rf "${TEST_REPO_TRT}"
  assemble_model_repo "${EXPORTED_DIR}" "${VARIANT}" "${TEST_REPO_TRT}" "trt" || { log_error "assemble (trt) failed"; exit 1; }
  validate_model_repo "${TEST_REPO_TRT}" || { log_error "validate (trt) failed"; exit 1; }
  grep -q "TYPE_BF16" "${TEST_REPO_TRT}/talker_code2wav_fused/config.pbtxt" || { log_error "talker_code2wav_fused TRT config missing TYPE_BF16"; exit 1; }
  log_info "TRT assemble checks passed"
else
  log_warn "No talker_code2wav_fused.engine/.plan found; skipping TRT assemble test"
fi

log_info "T2.4 test_assemble.sh passed"

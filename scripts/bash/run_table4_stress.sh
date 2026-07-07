#!/usr/bin/env bash
# Run Table 4 concurrency stress sweep (Poisson arrival, fixed trace replay).
#
# Usage:
#   bash scripts/bash/run_table4_stress.sh
#   bash scripts/bash/run_table4_stress.sh --smoke          # C=1,8 only
#   TRITON=5090-host:8001 VARIANT=stateful_triton bash scripts/bash/run_table4_stress.sh
#
# Prerequisites:
#   - conda activate qwen3-tts (or venv with tritonclient)
#   - Triton tts_orchestrator running at $TRITON

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

TRITON="${TRITON:-localhost:8001}"
TRACE="${TRACE:-${REPO_ROOT}/eval/fixtures/steadystream_stress_v1.json}"
VARIANT="${VARIANT:-stateful_triton}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/workspace/table4_runs}"
SEEDS="${SEEDS:-42,123,456}"
WARMUP="${WARMUP:-2}"
TIMEOUT="${TIMEOUT:-180}"
SMOKE=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) SMOKE=true; shift ;;
    --triton) TRITON="$2"; shift 2 ;;
    --variant) VARIANT="$2"; shift 2 ;;
    --out-root) OUT_ROOT="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ "${SMOKE}" == "true" ]]; then
  CONCURRENCY_LEVELS="1 8"
  SEEDS="42"
else
  CONCURRENCY_LEVELS="1 8 16 32 64 128"
fi

IFS=',' read -r -a SEED_ARR <<< "${SEEDS}"

echo "=== Table 4 stress sweep ==="
echo "Triton:      ${TRITON}"
echo "Trace:       ${TRACE}"
echo "Variant:     ${VARIANT}"
echo "Out root:    ${OUT_ROOT}"
echo "Concurrency: ${CONCURRENCY_LEVELS}"
echo "Seeds:       ${SEEDS}"

mkdir -p "${OUT_ROOT}"

for seed in "${SEED_ARR[@]}"; do
  for c in ${CONCURRENCY_LEVELS}; do
    run_dir="${OUT_ROOT}/${VARIANT}_c${c}_seed${seed}"
    echo ""
    echo "--- ${VARIANT}  concurrency=${c}  seed=${seed} ---"
    python3 "${REPO_ROOT}/scripts/python/steadystream_stress_client.py" \
      --triton "${TRITON}" \
      --trace "${TRACE}" \
      --concurrency "${c}" \
      --seed "${seed}" \
      --variant "${VARIANT}" \
      --out-dir "${run_dir}" \
      --warmup "${WARMUP}" \
      --timeout "${TIMEOUT}" \
      || echo "WARN: run failed for c=${c} seed=${seed} (continuing)"
  done
done

echo ""
echo "=== Generating Table 4 markdown ==="
python3 "${REPO_ROOT}/eval/make_table4.py" \
  --results-dir "${OUT_ROOT}" \
  --variant "${VARIANT}" \
  --output "${OUT_ROOT}/table4_${VARIANT}.md"

echo "Done. Results: ${OUT_ROOT}"

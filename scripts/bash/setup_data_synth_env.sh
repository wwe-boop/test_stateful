#!/bin/bash
# ===========================================================================
#  setup_data_synth_env.sh — Load LLM credentials for eval data synthesis
#
#  Usage:
#    source scripts/bash/setup_data_synth_env.sh
#    bash scripts/bash/generate_test_prosody.sh --count 5 --dry-run
#
#  Credentials live in repo-root `.env.data_synth` (gitignored).
#  Template: eval/data_synth/env.example
# ===========================================================================

setup_data_synth_env() {
    local script_dir repo_root env_file

    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    repo_root="$(git -C "${script_dir}" rev-parse --show-toplevel 2>/dev/null || true)"
    if [ -z "$repo_root" ]; then
        repo_root="$(cd "${script_dir}/../.." && pwd)"
    fi

    env_file="${DATA_SYNTH_ENV_FILE:-${repo_root}/.env.data_synth}"
    if [ ! -f "$env_file" ]; then
        echo "[data-synth] Missing ${env_file}" >&2
        echo "[data-synth] Copy template: cp eval/data_synth/env.example .env.data_synth" >&2
        return 1
    fi

    set -a
    # shellcheck disable=SC1090
    source "$env_file"
    set +a

    export DATA_SYNTH_PROVIDER="${DATA_SYNTH_PROVIDER:-dashscope}"
    export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
    export DATA_SYNTH_MODEL="${DATA_SYNTH_MODEL:-qwen-plus}"
    export ARK_BASE_URL="${ARK_BASE_URL:-https://ark.cn-beijing.volces.com/api/v3}"

    case "${DATA_SYNTH_PROVIDER}" in
        dashscope|openai|qwen)
            if [ -z "${OPENAI_API_KEY:-}" ]; then
                echo "[data-synth] OPENAI_API_KEY is required for provider=${DATA_SYNTH_PROVIDER}" >&2
                return 1
            fi
            ;;
        ark|volcengine|doubao)
            if [ -z "${ARK_API_KEY:-}" ] || [ -z "${ARK_MODEL:-}" ]; then
                echo "[data-synth] ARK_API_KEY and ARK_MODEL are required for provider=ark" >&2
                return 1
            fi
            ;;
        *)
            echo "[data-synth] Unsupported DATA_SYNTH_PROVIDER=${DATA_SYNTH_PROVIDER}" >&2
            return 1
            ;;
    esac

    echo "[data-synth] Loaded ${env_file} (provider=${DATA_SYNTH_PROVIDER})"
}

if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    setup_data_synth_env || return 1
else
    setup_data_synth_env || exit 1
    echo "Environment ready. Example:"
    echo "  python eval/data_synth/generate_test_prosody.py --count 5"
fi

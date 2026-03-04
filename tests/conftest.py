"""
Pytest conftest: add model_repository/tts_orchestrator/1 to path for imports.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ORCH_1 = REPO_ROOT / "model_repository" / "tts_orchestrator" / "1"
if ORCH_1.is_dir() and str(ORCH_1) not in sys.path:
    sys.path.insert(0, str(ORCH_1))

WORKSPACE_EXPORTED = REPO_ROOT / "workspace" / "exported"
TOKENIZER_DIR = WORKSPACE_EXPORTED / "tokenizer" / "Qwen3-TTS-Tokenizer-12Hz"
WEIGHTS_DIR_BASE = WORKSPACE_EXPORTED / "base-1.7b" / "weights"

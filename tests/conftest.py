"""
Pytest conftest: shared path setup for all test subdirectories.

Path constants are in tests/paths.py (importable from test modules).
This conftest adds orchestrator source to sys.path for legacy unit tests.
"""
import sys
from tests.paths import REPO_ROOT, VARIANT

ORCH_1 = REPO_ROOT / "model_repository" / "tts_orchestrator" / "1"
if ORCH_1.is_dir() and str(ORCH_1) not in sys.path:
    sys.path.insert(0, str(ORCH_1))

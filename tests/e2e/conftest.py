"""Auto-apply 'e2e' marker to all tests in this directory."""
import pytest

# Standalone CLI scripts that are NOT pytest tests
collect_ignore = [
    "test_triton_tts.py",
    "test_concurrent_tts.py",
    "test_greedy_baseline.py",
    "full_chain_audio_listen.py",
    "compare_official_vs_triton_audio.py",
    "generate_audio_compare.py",
    "verify_fused_triton_backend.py",
]


def pytest_collection_modifyitems(items):
    for item in items:
        if "/e2e/" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)

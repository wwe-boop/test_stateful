"""Auto-apply 'integration' marker to all tests in this directory."""
import pytest

# Standalone CLI scripts that are NOT pytest tests
collect_ignore = [
    "compare_official_vs_fused_onnx.py",
    "verify_code2wav_streaming.py",
    "verify_trt_talker.py",
    "verify_e2e_trt.py",
    "verify_e2e_trt_ref.py",
    "verify_precision_ort.py",
    "verify_code_predictor_trt.py",
    "verify_speech_tokenizer_encoder.py",
    "verify_e2e.py",
    "verify_prototype_parity.py",
    "verify_multi_variant.py",
    "pad_tolerance_experiment.py",
]


def pytest_collection_modifyitems(items):
    for item in items:
        if "/integration/" in str(item.fspath):
            item.add_marker(pytest.mark.integration)

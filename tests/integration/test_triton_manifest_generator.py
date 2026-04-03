"""Regression tests for triton_manifest + generate_triton_configs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_PY = REPO_ROOT / "scripts" / "python"
sys.path.insert(0, str(SCRIPTS_PY))

from generate_triton_configs import (  # noqa: E402
    generate_configs,
    render_code2wav_streaming,
    render_orchestrator,
    render_speech_tokenizer_encoder_trt,
    render_talker_unified_trt,
    triton_io_float_pbtxt_from_manifest,
)
from triton_manifest_io import build_manifest_for_export, load_manifest  # noqa: E402


FIXTURE_MANIFEST = REPO_ROOT / "tests" / "data" / "triton_manifest_custom_1_7b.json"
FIXTURE_LAYOUT = REPO_ROOT / "workspace" / "exported" / "custom-1.7b" / "code2wav_state_layout.json"


def test_render_talker_unified_28_layers_bf16():
    talker = {
        "hidden_size": 2048,
        "num_kv_heads": 8,
        "head_dim": 128,
        "num_layers": 28,
        "vocab_size": 3072,
    }
    text = render_talker_unified_trt(talker, "bf16")
    assert 'name: "talker_unified"' in text
    assert "TYPE_BF16" in text
    assert text.count('name: "past_kv_') == 28 * 2
    assert 'name: "past_kv_27_v"' in text
    assert 'name: "present_kv_27_k"' in text


def test_render_code2wav_trt_bf16_has_8_kv_layers():
    text = render_code2wav_streaming("trt", "bf16")
    assert 'name: "code2wav"' in text
    assert 'name: "past_kv_7_v"' in text
    assert 'name: "conv_state_16"' in text
    assert 'name: "new_transconv_overlap_3"' in text


def test_render_orchestrator_custom_variant():
    orch = {
        "tts_model_type": "custom_voice",
        "supported_task_types": "custom_voice",
        "max_decode_steps": "4096",
        "audio_chunk_frames": "25",
        "first_chunk_frames": "4",
        "weights_dir": "/models/tts_orchestrator/1/weights",
        "tokenizer_dir": "/models/tts_orchestrator/1/tokenizer",
    }
    text = render_orchestrator("custom-1.7b", orch)
    assert 'string_value: "custom-1.7b"' in text
    assert 'string_value: "custom_voice"' in text


def test_render_speech_tokenizer_encoder_trt():
    text = render_speech_tokenizer_encoder_trt()
    assert "TYPE_INT64" in text and "audio_codes" in text


def test_load_manifest_merges_orchestrator_defaults(tmp_path):
    mpath = tmp_path / "triton_manifest.json"
    mpath.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "variant": "custom-1.7b",
                "code2wav_fused": {
                    "num_code2wav_hidden_layers": 8,
                    "c2w_state_input_names": [],
                    "c2w_state_output_names": [],
                    "initial_state_shapes": [],
                },
            }
        ),
        encoding="utf-8",
    )
    m = load_manifest(mpath, output_repo=None)
    assert m["orchestrator"]["tts_model_type"] == "custom_voice"


def test_build_manifest_for_export_roundtrip():
    if not FIXTURE_LAYOUT.is_file():
        pytest.skip("workspace exported layout not present")
    lay = json.loads(FIXTURE_LAYOUT.read_text(encoding="utf-8"))
    wc = {
        "talker_hidden_size": 2048,
        "talker_num_kv_heads": 8,
        "talker_head_dim": 128,
        "talker_num_layers": 28,
        "talker_vocab_size": 3072,
        "talker_num_heads": 16,
    }
    m = build_manifest_for_export("custom-1.7b", wc, lay)
    assert m["schema_version"] == 1
    assert m.get("triton_io_float_dtype") == "bf16"
    assert m["code2wav_fused"]["c2w_state_input_names"][0] == "c2w_past_kv_0_k"


def test_triton_io_float_dtype_from_manifest_defaults_fp32():
    assert triton_io_float_pbtxt_from_manifest({}) == "TYPE_FP32"  # missing key defaults fp32
    assert triton_io_float_pbtxt_from_manifest({"triton_io_float_dtype": "bf16"}) == "TYPE_BF16"


def test_generate_configs_minimal_repo(tmp_path):
    if not FIXTURE_MANIFEST.is_file():
        pytest.fail("missing fixture manifest")
    manifest = json.loads(FIXTURE_MANIFEST.read_text(encoding="utf-8"))
    # Minimal fake repo: only fused + orchestrator
    for name in ("talker_code2wav_fused", "tts_orchestrator"):
        (tmp_path / name / "1").mkdir(parents=True)
    (tmp_path / "talker_code2wav_fused" / "1" / "model.plan").write_text("stub")
    (tmp_path / "tts_orchestrator" / "1" / ".keep").write_text("")
    generate_configs(manifest, tmp_path, "trt", engine_dtype="bf16")
    fused_cfg = (tmp_path / "talker_code2wav_fused" / "config.pbtxt").read_text()
    assert "tensorrt" in fused_cfg
    assert 'name: "input_embeds"' in fused_cfg
    assert 'name: "cache_position"' in fused_cfg
    assert 'name: "wav"' in fused_cfg
    assert "c2w_past_kv_0_k" in fused_cfg
    assert 'name: "past_kv_0_k"' in fused_cfg
    assert "TYPE_BF16" in fused_cfg
    orch_cfg = (tmp_path / "tts_orchestrator" / "config.pbtxt").read_text()
    assert "tts_orchestrator" in orch_cfg
    assert "custom-1.7b" in orch_cfg


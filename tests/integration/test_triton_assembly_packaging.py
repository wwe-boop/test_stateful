"""Regression tests for assembled model package contents."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_MANIFEST = REPO_ROOT / "tests" / "data" / "triton_manifest_custom_1_7b.json"


def _write_custom_export(exported_dir: Path) -> None:
    variant_dir = exported_dir / "custom-1.7b"
    variant_dir.mkdir(parents=True)
    (exported_dir / "artifact_manifest.json").write_text(
        json.dumps(
            {
                "artifact_schema_version": 1,
                "ngc_tag": "25.03",
                "ngc_image": "nvcr.io/nvidia/tritonserver:25.03-py3",
                "tensorrt_version": "10.9.0",
                "gpu_sm": "sm_89",
                "engine_dtype": "bf16",
                "engines": {},
            }
        ),
        encoding="utf-8",
    )
    manifest = json.loads(FIXTURE_MANIFEST.read_text(encoding="utf-8"))
    manifest["package"] = {
        "schema_version": 1,
        "layout": "triton_model_version",
        "model_package_dir": "/models/tts_orchestrator/1",
        "runtime_dir": "runtime",
        "weights_dir": "weights",
        "tokenizer_dir": "tokenizer",
        "manifest": "runtime/triton_manifest.json",
        "runtime_artifacts": {
            "trt": "runtime/model.plan",
            "onnx": "runtime/model.onnx",
        },
        "optional_assets": {
            "speaker_encoder": "runtime/speaker_encoder.onnx",
            "speech_tokenizer_encoder": "runtime/speech_tokenizer_encoder.onnx",
            "speech_tokenizer_codec_fused": "runtime/speech_tokenizer_codec_fused.onnx",
        },
    }
    (variant_dir / "triton_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    (variant_dir / "talker_code2wav_fused.engine").write_bytes(b"fake-plan")
    weights_dir = variant_dir / "weights"
    weights_dir.mkdir()
    (weights_dir / "config.json").write_text(
        json.dumps(
            {
                "talker_hidden_size": 2048,
                "talker_num_heads": 16,
                "talker_num_kv_heads": 8,
                "talker_head_dim": 128,
                "talker_num_layers": 28,
                "talker_vocab_size": 3072,
            }
        ),
        encoding="utf-8",
    )

    tokenizer_dir = exported_dir / "tokenizer"
    tokenizer_dir.mkdir()
    (tokenizer_dir / "speech_tokenizer_encoder.onnx").write_bytes(b"fake-onnx")
    (tokenizer_dir / "code2wav_decoder.engine").write_bytes(b"fake-engine")

    model_dir = exported_dir.parent / "models" / "Qwen3-TTS-12Hz-1.7B-CustomVoice"
    model_dir.mkdir(parents=True)
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")


def _assemble(exported_dir: Path, repo_dir: Path, variant: str, engine_mode: str) -> None:
    cmd = f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/bash/lib/triton.sh"
assemble_model_repo "$1" "$2" "$3" "$4" "1"
"""
    subprocess.run(
        ["bash", "-c", cmd, "bash", str(exported_dir), variant, str(repo_dir), engine_mode],
        cwd=REPO_ROOT,
        check=True,
    )


def test_custom_trt_package_excludes_verification_and_icl_assets(tmp_path):
    exported_dir = tmp_path / "exported"
    repo_dir = tmp_path / "model_repository"
    _write_custom_export(exported_dir)

    _assemble(exported_dir, repo_dir, "custom-1.7b", "trt")

    package_dir = repo_dir / "tts_orchestrator" / "1"
    runtime_dir = package_dir / "runtime"
    assert not (repo_dir / "triton_manifest.json").exists()
    assert not (repo_dir / "artifact_manifest.json").exists()
    assert (runtime_dir / "model.plan").is_file()
    assert (package_dir / "artifact_manifest.json").is_file()
    assert (runtime_dir / "artifact_manifest.json").is_file()
    assert not (runtime_dir / "speech_tokenizer_encoder.onnx").exists()
    assert not (package_dir / "tokenizer" / "code2wav_decoder.engine").exists()
    assert not (repo_dir / "speech_tokenizer_encoder").exists()

    manifest = json.loads((runtime_dir / "triton_manifest.json").read_text(encoding="utf-8"))
    optional_assets = manifest["package"]["optional_assets"]
    assert optional_assets == {}

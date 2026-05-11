# English comments only.
"""Load Triton deployment manifest (triton_manifest.json)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def load_weights_config(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def weights_to_talker_section(w: Dict[str, Any]) -> Dict[str, int]:
    """Map export weights/config.json keys to manifest talker section."""
    h = int(w.get("talker_hidden_size", 2048))
    n_heads = int(w.get("talker_num_heads", 16))
    kv = int(w.get("talker_num_kv_heads", 8))
    hd = w.get("talker_head_dim")
    if hd is None:
        hd = h // n_heads
    return {
        "hidden_size": h,
        "num_kv_heads": kv,
        "head_dim": int(hd),
        "num_layers": int(w.get("talker_num_layers", 28)),
        "vocab_size": int(w.get("talker_vocab_size", 3072)),
    }


def variant_orchestrator_defaults(
    variant: str,
    model_package_dir: str = "/models/tts_orchestrator/1",
) -> Dict[str, str]:
    if variant.startswith("base-"):
        tts = "base"
        tasks = "voice_clone_icl,voice_clone_xvec"
    elif variant.startswith("custom-"):
        tts = "custom_voice"
        tasks = "custom_voice"
    elif variant.startswith("design-"):
        tts = "voice_design"
        tasks = "voice_design"
    else:
        tts = "unknown"
        tasks = "unknown"
    return {
        "tts_model_type": tts,
        "supported_task_types": tasks,
        "max_decode_steps": "4096",
        "audio_chunk_frames": "25",
        "first_chunk_frames": "4",
        "model_package_dir": model_package_dir,
    }


def package_defaults() -> Dict[str, Any]:
    return {
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


def load_manifest(
    manifest_path: Path,
    output_repo: Optional[Path] = None,
    model_package_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load JSON manifest.
    Optionally fill talker section from tts_orchestrator/<version>/weights/config.json in output_repo.
    """
    with open(manifest_path, encoding="utf-8") as f:
        manifest: Dict[str, Any] = json.load(f)

    variant = manifest.get("variant", "unknown")
    package_model_package_dir = model_package_dir or "/models/tts_orchestrator/1"
    package = manifest.get("package")
    if model_package_dir is None and isinstance(package, dict):
        raw_package_dir = package.get("model_package_dir")
        if raw_package_dir:
            package_model_package_dir = str(raw_package_dir)
    package_version = Path(package_model_package_dir).name or "1"

    if not manifest.get("talker") and output_repo is not None:
        wc = output_repo / "tts_orchestrator" / package_version / "weights" / "config.json"
        if wc.is_file():
            manifest["talker"] = weights_to_talker_section(load_weights_config(wc))
            logger.info("Filled manifest.talker from orchestrator weights/config.json")

    orch = manifest.get("orchestrator")
    defaults = variant_orchestrator_defaults(variant, package_model_package_dir)
    if orch is None:
        manifest["orchestrator"] = dict(defaults)
    else:
        merged = dict(defaults)
        merged.update(orch)
        manifest["orchestrator"] = merged
    if model_package_dir is not None:
        manifest["orchestrator"]["model_package_dir"] = package_model_package_dir

    package = manifest.get("package")
    if not isinstance(package, dict):
        manifest["package"] = package_defaults()
    else:
        merged_package = package_defaults()
        for key, value in package.items():
            if isinstance(value, dict) and isinstance(merged_package.get(key), dict):
                nested = dict(merged_package[key])
                nested.update(value)
                merged_package[key] = nested
            else:
                merged_package[key] = value
        manifest["package"] = merged_package
    if model_package_dir is not None:
        manifest["package"]["model_package_dir"] = package_model_package_dir

    return manifest


def build_manifest_for_export(
    variant: str,
    weights_config: Dict[str, Any],
    code2wav_layout: Dict[str, Any],
    engine_mode: str = "trt",
    engine_dtype: str = "bf16",
    triton_io_float_dtype: str = "bf16",
) -> Dict[str, Any]:
    """Build a full manifest dict after export (e.g. export_09).

    engine_dtype: TensorRT builder precision (e.g. trtexec --bf16); Phase B reads this for prec flags.
    triton_io_float_dtype: Float tensor I/O for trtexec --inputIOFormats/--outputIOFormats and Triton config.pbtxt.
        ONNX graph remains FP32 (ONNX_EXPORT_DTYPE); TRT may insert reformats at boundaries.

    The ``architecture`` section provides a complete, self-contained model
    description for the standalone engine.  Priority: manifest > engine.yaml > defaults.
    """
    talker = weights_to_talker_section(weights_config)
    orch = variant_orchestrator_defaults(variant)

    architecture: Dict[str, Any] = {
        "num_layers": talker["num_layers"],
        "hidden_size": talker["hidden_size"],
        "kv_heads": talker["num_kv_heads"],
        "head_dim": talker["head_dim"],
        "codec_vocab_size": talker["vocab_size"],
        "logits_topk": code2wav_layout.get("logits_topk", 50),
        "cp_num_stages": code2wav_layout.get("cp_num_stages", 15),
        "n_c2w_layers": code2wav_layout["num_code2wav_hidden_layers"],
        "c2w_kv_heads": code2wav_layout.get("c2w_kv_heads", 16),
        "c2w_head_dim": code2wav_layout.get("c2w_head_dim", 64),
        "c2w_sliding_window": code2wav_layout.get("c2w_sliding_window", 72),
        "n_c2w_conv_states": len([
            n for n in code2wav_layout.get("c2w_state_input_names", [])
            if "conv_state" in n
        ]),
        "n_c2w_transconv_states": len([
            n for n in code2wav_layout.get("c2w_state_input_names", [])
            if "transconv" in n
        ]),
        "dtype": engine_dtype,
    }

    return {
        "schema_version": 2,
        "variant": variant,
        "engine_mode": engine_mode,
        "engine_dtype": engine_dtype,
        "triton_io_float_dtype": triton_io_float_dtype,
        "package": {
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
        },
        "engine_profile": {
            "profile_schema_version": 1,
            "engine_mode": engine_mode,
            "engine_dtype": engine_dtype,
            "triton_io_float_dtype": triton_io_float_dtype,
            "builder": "trtexec",
        },
        "architecture": architecture,
        "talker": talker,
        "code2wav_fused": {
            "num_code2wav_hidden_layers": code2wav_layout["num_code2wav_hidden_layers"],
            "c2w_state_input_names": code2wav_layout["c2w_state_input_names"],
            "c2w_state_output_names": code2wav_layout["c2w_state_output_names"],
            "initial_state_shapes": code2wav_layout["initial_state_shapes"],
            "packed_kv": bool(code2wav_layout.get("packed_kv", False)),
            "c2w_kv_heads": int(code2wav_layout.get("c2w_kv_heads", 16)),
            "c2w_head_dim": int(code2wav_layout.get("c2w_head_dim", 64)),
            "c2w_sliding_window": int(code2wav_layout.get("c2w_sliding_window", 72)),
            "logits_topk": int(code2wav_layout.get("logits_topk", 50)),
            "cp_num_stages": int(code2wav_layout.get("cp_num_stages", 15)),
        },
        "orchestrator": orch,
    }

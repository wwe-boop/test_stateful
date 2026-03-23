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


def variant_orchestrator_defaults(variant: str) -> Dict[str, str]:
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
        "weights_dir": "/models/tts_orchestrator/1/weights",
        "tokenizer_dir": "/models/tts_orchestrator/1/tokenizer",
    }


def load_manifest(
    manifest_path: Path,
    output_repo: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Load JSON manifest.
    Optionally fill talker section from tts_orchestrator/1/weights/config.json in output_repo.
    """
    with open(manifest_path, encoding="utf-8") as f:
        manifest: Dict[str, Any] = json.load(f)

    variant = manifest.get("variant", "unknown")
    if not manifest.get("talker") and output_repo is not None:
        wc = output_repo / "tts_orchestrator" / "1" / "weights" / "config.json"
        if wc.is_file():
            manifest["talker"] = weights_to_talker_section(load_weights_config(wc))
            logger.info("Filled manifest.talker from orchestrator weights/config.json")

    orch = manifest.get("orchestrator")
    defaults = variant_orchestrator_defaults(variant)
    if orch is None:
        manifest["orchestrator"] = dict(defaults)
    else:
        merged = dict(defaults)
        merged.update(orch)
        manifest["orchestrator"] = merged

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
    """
    talker = weights_to_talker_section(weights_config)
    orch = variant_orchestrator_defaults(variant)
    return {
        "schema_version": 1,
        "variant": variant,
        "engine_mode": engine_mode,
        "engine_dtype": engine_dtype,
        "triton_io_float_dtype": triton_io_float_dtype,
        "talker": talker,
        "code2wav_fused": {
            "num_code2wav_hidden_layers": code2wav_layout["num_code2wav_hidden_layers"],
            "c2w_state_input_names": code2wav_layout["c2w_state_input_names"],
            "c2w_state_output_names": code2wav_layout["c2w_state_output_names"],
            "initial_state_shapes": code2wav_layout["initial_state_shapes"],
        },
        "orchestrator": orch,
    }

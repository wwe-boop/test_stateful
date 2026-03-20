#!/usr/bin/env python3
# English comments only.
"""
Generate Triton config.pbtxt files from triton_manifest.json.

Replaces heredoc logic in scripts/bash/lib/triton.sh for assemble_model_repo.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts" / "python"))

from triton_manifest_io import load_manifest

logger = logging.getLogger(__name__)

# Code2wav streaming TRT conv / transconv specs (must match former triton.sh heredoc).
_CONV_SPECS: List[Tuple[str, int, int, int]] = [
    ("conv_state_0", -1, 512, 2),
    ("conv_state_1", -1, 1024, 6),
    ("conv_state_2", -1, 1024, 6),
    ("conv_state_3", -1, 1024, 6),
    ("conv_state_4", -1, 768, 6),
    ("conv_state_5", -1, 768, 18),
    ("conv_state_6", -1, 768, 54),
    ("conv_state_7", -1, 384, 6),
    ("conv_state_8", -1, 384, 18),
    ("conv_state_9", -1, 384, 54),
    ("conv_state_10", -1, 192, 6),
    ("conv_state_11", -1, 192, 18),
    ("conv_state_12", -1, 192, 54),
    ("conv_state_13", -1, 96, 6),
    ("conv_state_14", -1, 96, 18),
    ("conv_state_15", -1, 96, 54),
    ("conv_state_16", -1, 96, 6),
]

_TRANSCONV_SPECS: List[Tuple[str, int, int, int]] = [
    ("transconv_overlap_0", -1, 768, 8),
    ("transconv_overlap_1", -1, 384, 5),
    ("transconv_overlap_2", -1, 192, 4),
    ("transconv_overlap_3", -1, 96, 3),
]


def normalize_engine_dtype(short: str) -> str:
    s = (short or "bf16").lower().strip()
    if s in ("bfloat16",):
        return "bf16"
    if s in ("float16",):
        return "fp16"
    if s in ("float32", "float"):
        return "fp32"
    return s


def to_triton_dtype(short: str) -> str:
    s = normalize_engine_dtype(short)
    mapping = {
        "float": "TYPE_FP32",
        "float32": "TYPE_FP32",
        "fp32": "TYPE_FP32",
        "float16": "TYPE_FP16",
        "fp16": "TYPE_FP16",
        "bfloat16": "TYPE_BF16",
        "bf16": "TYPE_BF16",
        "int32": "TYPE_INT32",
        "int64": "TYPE_INT64",
        "string": "TYPE_STRING",
        "bool": "TYPE_BOOL",
    }
    return mapping.get(s, "TYPE_FP32")


def render_minimal_onnx(model_name: str) -> str:
    return f'''name: "{model_name}"
backend: "onnxruntime"
max_batch_size: 0

instance_group [
  {{
    count: 1
    kind: KIND_GPU
    gpus: [ 0 ]
  }}
]
'''


def render_minimal_trt(model_name: str) -> str:
    return f'''name: "{model_name}"
backend: "tensorrt"
max_batch_size: 0

instance_group [
  {{
    count: 1
    kind: KIND_GPU
    gpus: [ 0 ]
  }}
]
'''


def render_speaker_encoder_trt(engine_dtype: str) -> str:
    ft = to_triton_dtype(engine_dtype)
    return f'''name: "speaker_encoder"
backend: "tensorrt"
max_batch_size: 0

input [
  {{ name: "mel"  data_type: {ft}  dims: [ -1, -1, 128 ] }}
]
output [
  {{ name: "speaker_embedding"  data_type: {ft}  dims: [ -1, 1024 ] }}
]

instance_group [
  {{ count: 1  kind: KIND_GPU  gpus: [ 0 ] }}
]
'''


def render_speech_tokenizer_encoder_trt() -> str:
    return '''name: "speech_tokenizer_encoder"
backend: "tensorrt"
max_batch_size: 0

input [
  { name: "waveform"  data_type: TYPE_FP32  dims: [ -1, 1, -1 ] }
]
output [
  { name: "audio_codes"  data_type: TYPE_INT64  dims: [ -1, 16, -1 ] }
]

instance_group [
  { count: 1  kind: KIND_GPU  gpus: [ 0 ] }
]
'''


def render_talker_unified_trt(talker: Dict[str, Any], engine_dtype: str) -> str:
    ft = to_triton_dtype(engine_dtype)
    H = int(talker.get("hidden_size", 2048))
    kv = int(talker.get("num_kv_heads", 8))
    hd = int(talker.get("head_dim", 128))
    nl = int(talker.get("num_layers", 28))
    v = int(talker.get("vocab_size", 3072))

    parts: List[str] = [
        'name: "talker_unified"',
        'backend: "tensorrt"',
        "max_batch_size: 0",
        "",
        "input [",
        f"  {{ name: \"input_embeds\"  data_type: {ft}  dims: [ -1, -1, {H} ] }}",
        "]",
        "input [",
        '  { name: "position_ids"  data_type: TYPE_INT64  dims: [ -1, 3, -1 ] }',
        "]",
    ]
    for i in range(nl):
        parts.append("input [")
        parts.append(
            f'  {{ name: "past_kv_{i}_k"  data_type: {ft}  dims: [ -1, {kv}, -1, {hd} ] }}'
        )
        parts.append("]")
        parts.append("input [")
        parts.append(
            f'  {{ name: "past_kv_{i}_v"  data_type: {ft}  dims: [ -1, {kv}, -1, {hd} ] }}'
        )
        parts.append("]")
    parts.extend(
        [
            "output [",
            f'  {{ name: "codec_sum"  data_type: {ft}  dims: [ -1, 1, {H} ] }}',
            "]",
            "output [",
            '  { name: "full_codec"  data_type: TYPE_INT64  dims: [ -1, 16 ] }',
            "]",
            "output [",
            f'  {{ name: "hidden"  data_type: {ft}  dims: [ -1, -1, {H} ] }}',
            "]",
            "output [",
            f'  {{ name: "logits"  data_type: {ft}  dims: [ -1, -1, {v} ] }}',
            "]",
        ]
    )
    for i in range(nl):
        parts.append("output [")
        parts.append(
            f'  {{ name: "present_kv_{i}_k"  data_type: {ft}  dims: [ -1, {kv}, -1, {hd} ] }}'
        )
        parts.append("]")
        parts.append("output [")
        parts.append(
            f'  {{ name: "present_kv_{i}_v"  data_type: {ft}  dims: [ -1, {kv}, -1, {hd} ] }}'
        )
        parts.append("]")
    parts.extend(
        [
            "instance_group [",
            "  { count: 1  kind: KIND_GPU  gpus: [ 0 ] }",
            "]",
            "",
        ]
    )
    return "\n".join(parts)


def _dims_pbtxt(tup: Tuple[int, ...]) -> str:
    return ", ".join(str(x) for x in tup)


def render_code2wav_streaming(engine_mode: str, engine_dtype: str) -> str:
    if engine_mode == "onnx":
        return '''name: "code2wav"
backend: "onnxruntime"
max_batch_size: 0

input [
  { name: "codes"  data_type: TYPE_INT64  dims: [ -1, 16, 4 ] }
]
input [
  { name: "cache_position"  data_type: TYPE_INT64  dims: [ -1, 4 ] }
]

instance_group [
  { count: 1  kind: KIND_GPU  gpus: [ 0 ] }
]
'''
    ft = to_triton_dtype(engine_dtype)
    parts: List[str] = [
        'name: "code2wav"',
        'backend: "tensorrt"',
        "max_batch_size: 0",
        "",
        "input [",
        '  { name: "codes"  data_type: TYPE_INT64  dims: [ -1, 16, 4 ] }',
        "]",
        "input [",
        '  { name: "cache_position"  data_type: TYPE_INT64  dims: [ -1, 4 ] }',
        "]",
    ]
    for i in range(8):
        parts.append("input [")
        parts.append(
            f'  {{ name: "past_kv_{i}_k"  data_type: {ft}  dims: [ -1, 16, -1, 64 ] }}'
        )
        parts.append("]")
        parts.append("input [")
        parts.append(
            f'  {{ name: "past_kv_{i}_v"  data_type: {ft}  dims: [ -1, 16, -1, 64 ] }}'
        )
        parts.append("]")
    for name, a, b, c in _CONV_SPECS:
        parts.append("input [")
        parts.append(f'  {{ name: "{name}"  data_type: {ft}  dims: [ {_dims_pbtxt((a, b, c))} ] }}')
        parts.append("]")
    for name, a, b, c in _TRANSCONV_SPECS:
        parts.append("input [")
        parts.append(f'  {{ name: "{name}"  data_type: {ft}  dims: [ {_dims_pbtxt((a, b, c))} ] }}')
        parts.append("]")
    parts.append("output [")
    parts.append(f'  {{ name: "wav"  data_type: {ft}  dims: [ -1, 1, 7680 ] }}')
    parts.append("]")
    for i in range(8):
        parts.append("output [")
        parts.append(
            f'  {{ name: "present_kv_{i}_k"  data_type: {ft}  dims: [ -1, 16, -1, 64 ] }}'
        )
        parts.append("]")
        parts.append("output [")
        parts.append(
            f'  {{ name: "present_kv_{i}_v"  data_type: {ft}  dims: [ -1, 16, -1, 64 ] }}'
        )
        parts.append("]")
    for name, a, b, c in _CONV_SPECS:
        parts.append("output [")
        parts.append(
            f'  {{ name: "new_{name}"  data_type: {ft}  dims: [ {_dims_pbtxt((a, b, c))} ] }}'
        )
        parts.append("]")
    for name, a, b, c in _TRANSCONV_SPECS:
        parts.append("output [")
        parts.append(
            f'  {{ name: "new_{name}"  data_type: {ft}  dims: [ {_dims_pbtxt((a, b, c))} ] }}'
        )
        parts.append("]")
    parts.extend(
        [
            "instance_group [",
            "  { count: 1  kind: KIND_GPU  gpus: [ 0 ] }",
            "]",
            "",
        ]
    )
    return "\n".join(parts)


def render_orchestrator(variant: str, orch: Dict[str, str]) -> str:
    tts = orch.get("tts_model_type", "unknown")
    tasks = orch.get("supported_task_types", "unknown")
    max_decode = orch.get("max_decode_steps", "4096")
    achunk = orch.get("audio_chunk_frames", "25")
    fchunk = orch.get("first_chunk_frames", "4")
    wdir = orch.get("weights_dir", "/models/tts_orchestrator/1/weights")
    tdir = orch.get("tokenizer_dir", "/models/tts_orchestrator/1/tokenizer")
    return f'''name: "tts_orchestrator"
backend: "python"
max_batch_size: 0

model_transaction_policy {{
  decoupled: true
}}

input [
  {{
    name: "request"
    data_type: TYPE_STRING
    dims: [ 1 ]
  }}
]

output [
  {{
    name: "audio_chunk"
    data_type: TYPE_FP32
    dims: [ -1 ]
  }},
  {{
    name: "is_final"
    data_type: TYPE_BOOL
    dims: [ 1 ]
  }}
]

instance_group [
  {{
    count: 1
    kind: KIND_GPU
    gpus: [ 0 ]
  }}
]

parameters: {{
  key: "model_variant"
  value: {{ string_value: "{variant}" }}
}}
parameters: {{
  key: "tts_model_type"
  value: {{ string_value: "{tts}" }}
}}
parameters: {{
  key: "supported_task_types"
  value: {{ string_value: "{tasks}" }}
}}
parameters: {{
  key: "weights_dir"
  value: {{ string_value: "{wdir}" }}
}}
parameters: {{
  key: "tokenizer_dir"
  value: {{ string_value: "{tdir}" }}
}}
parameters: {{
  key: "max_decode_steps"
  value: {{ string_value: "{max_decode}" }}
}}
parameters: {{
  key: "audio_chunk_frames"
  value: {{ string_value: "{achunk}" }}
}}
parameters: {{
  key: "first_chunk_frames"
  value: {{ string_value: "{fchunk}" }}
}}
'''


def model_has_engine(model_dir: Path) -> Tuple[bool, bool]:
    """Returns (has_plan, has_onnx)."""
    v1 = model_dir / "1"
    return (v1 / "model.plan").is_file(), (v1 / "model.onnx").is_file()


def generate_configs(
    manifest: Dict[str, Any],
    output_repo: Path,
    engine_mode: str,
    engine_dtype: Optional[str] = None,
) -> None:
    emode = engine_mode.lower().strip()
    ed = normalize_engine_dtype(
        engine_dtype or manifest.get("engine_dtype") or "bf16"
    )
    variant = str(manifest.get("variant", "unknown"))
    talker = manifest.get("talker") or {}
    orch = manifest.get("orchestrator") or {}

    models = [
        "speaker_encoder",
        "speech_tokenizer_codec_fused",
        "talker_code2wav_fused",
        "speech_tokenizer_encoder",
        "talker_unified",
        "code2wav",
    ]

    for name in models:
        mdir = output_repo / name
        if not mdir.is_dir():
            continue
        has_plan, has_onnx = model_has_engine(mdir)
        if not has_plan and not has_onnx:
            continue

        text: str
        if name == "speaker_encoder":
            if emode == "trt" and has_plan:
                text = render_speaker_encoder_trt(ed)
            else:
                text = render_minimal_onnx(name)
        elif name == "speech_tokenizer_encoder":
            if emode == "trt" and has_plan:
                text = render_speech_tokenizer_encoder_trt()
            else:
                text = render_minimal_onnx(name)
        elif name == "talker_unified":
            if emode == "trt" and has_plan:
                text = render_talker_unified_trt(talker, ed)
            else:
                text = render_minimal_onnx(name)
        elif name == "code2wav":
            text = render_code2wav_streaming(emode, ed)
        else:
            # speech_tokenizer_codec_fused, talker_code2wav_fused
            if emode == "trt" and has_plan:
                text = render_minimal_trt(name)
            else:
                text = render_minimal_onnx(name)

        cfg_path = mdir / "config.pbtxt"
        cfg_path.write_text(text, encoding="utf-8")
        logger.info("Wrote %s", cfg_path)

    orch_dir = output_repo / "tts_orchestrator"
    if orch_dir.is_dir():
        otxt = render_orchestrator(variant, orch)
        (orch_dir / "config.pbtxt").write_text(otxt, encoding="utf-8")
        logger.info("Wrote %s", orch_dir / "config.pbtxt")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True, help="Path to triton_manifest.json")
    p.add_argument(
        "--output-repo",
        type=Path,
        required=True,
        help="Assembled model_repository root",
    )
    p.add_argument(
        "--engine-mode",
        choices=["onnx", "trt"],
        required=True,
        help="Must match assembled engine files",
    )
    p.add_argument(
        "--engine-dtype",
        default=None,
        help="Override manifest engine_dtype (bf16|fp16|fp32)",
    )
    args = p.parse_args()

    manifest = load_manifest(args.manifest, output_repo=args.output_repo)
    mmode = str(manifest.get("engine_mode", args.engine_mode)).lower()
    if mmode != args.engine_mode:
        logger.warning(
            "CLI engine_mode=%s differs from manifest engine_mode=%s (using CLI)",
            args.engine_mode,
            mmode,
        )

    generate_configs(
        manifest,
        args.output_repo,
        args.engine_mode,
        engine_dtype=args.engine_dtype,
    )


if __name__ == "__main__":
    main()

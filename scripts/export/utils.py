"""
Shared utilities for ONNX export scripts.
"""

import os
import sys
import json
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import torch
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
THIRD_PARTY = REPO_ROOT / "third_party" / "Qwen3-TTS"
DEFAULT_MODELS_DIR = REPO_ROOT / "workspace" / "models"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "workspace" / "exported"

sys.path.insert(0, str(THIRD_PARTY))

logger = logging.getLogger("onnx_export")

MODEL_VARIANTS = {
    "base-1.7b": "Qwen3-TTS-12Hz-1.7B-Base",
    "custom-1.7b": "Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "design-1.7b": "Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "base-0.6b": "Qwen3-TTS-12Hz-0.6B-Base",
    "custom-0.6b": "Qwen3-TTS-12Hz-0.6B-CustomVoice",
}

TOKENIZER_DIR_NAME = "Qwen3-TTS-Tokenizer-12Hz"

DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "fp32": torch.float32,
    "float32": torch.float32,
}

DTYPE_NAMES = {torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32"}

DEFAULT_DTYPE = torch.bfloat16

ONNX_EXPORT_DTYPE = torch.float32


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


MIN_FREE_VRAM_GB = 2.0


def auto_detect_device() -> str:
    """Pick the GPU with the most free VRAM; fall back to CPU if none qualifies.

    Scans all visible CUDA devices, selects the one with the most free memory.
    Falls back to CPU if the best GPU has less than MIN_FREE_VRAM_GB free.
    Returns 'cuda:N' (specific GPU) or 'cpu'.
    """
    if not torch.cuda.is_available():
        logger.info("No GPU detected → using cpu")
        return "cpu"

    n_gpus = torch.cuda.device_count()
    best_idx, best_free = -1, 0.0

    for i in range(n_gpus):
        free, total = torch.cuda.mem_get_info(i)
        free_gb = free / (1024 ** 3)
        total_gb = total / (1024 ** 3)
        name = torch.cuda.get_device_name(i)
        logger.info(
            f"  GPU {i}: {name}  free={free_gb:.1f} GiB / {total_gb:.1f} GiB"
        )
        if free_gb > best_free:
            best_free = free_gb
            best_idx = i

    if best_idx < 0 or best_free < MIN_FREE_VRAM_GB:
        logger.warning(
            f"No GPU with >= {MIN_FREE_VRAM_GB:.0f} GiB free VRAM "
            f"(best: GPU {best_idx} with {best_free:.1f} GiB free) → falling back to cpu"
        )
        return "cpu"

    device = f"cuda:{best_idx}"
    logger.info(
        f"Selected GPU {best_idx} ({best_free:.1f} GiB free) → using {device}"
    )
    return device


def resolve_device(user_device: Optional[str]) -> str:
    """Resolve a user-supplied --device arg to a concrete device string.

    - None or ""  → auto_detect_device() (scan all GPUs, pick best)
    - "cuda"      → auto_detect_device() (bare 'cuda' would default to GPU 0)
    - "cuda:N"    → use as-is (user picked a specific GPU)
    - "cpu"       → use as-is
    """
    if not user_device or user_device == "cuda":
        return auto_detect_device()
    return user_device


def resolve_dtype(dtype_str: str) -> torch.dtype:
    """Map a CLI dtype string (e.g. 'bf16') to torch.dtype."""
    dt = DTYPE_MAP.get(dtype_str.lower())
    if dt is None:
        raise ValueError(
            f"Unknown dtype '{dtype_str}'. "
            f"Available: bf16, fp16, fp32"
        )
    return dt


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert a PyTorch tensor to float32 numpy, handling BF16/FP16."""
    t = tensor.detach().cpu()
    if t.dtype in (torch.bfloat16, torch.float16):
        t = t.float()
    return t.numpy()


def add_common_args(parser) -> None:
    """Add --device, --dtype, --models-dir, --output-dir to an ArgumentParser."""
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device: cpu | cuda | cuda:0 | cuda:1 ... (default: auto-select GPU with most free VRAM)",
    )
    parser.add_argument(
        "--dtype", type=str, default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Target inference precision for embedding weights and TRT engine build "
             "(default: bf16). ONNX export always uses fp32 internally.",
    )
    parser.add_argument("--models-dir", type=str, default=None,
                        help="Models directory (default: workspace/models)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: workspace/exported)")


def resolve_model_path(variant: str, models_dir: Optional[str] = None) -> Path:
    base = Path(models_dir) if models_dir else DEFAULT_MODELS_DIR
    dir_name = MODEL_VARIANTS.get(variant)
    if dir_name is None:
        raise ValueError(
            f"Unknown variant '{variant}'. Available: {list(MODEL_VARIANTS.keys())}"
        )
    path = base / dir_name
    if not path.exists():
        raise FileNotFoundError(f"Model directory not found: {path}")
    return path


def resolve_tokenizer_path(models_dir: Optional[str] = None) -> Path:
    base = Path(models_dir) if models_dir else DEFAULT_MODELS_DIR
    path = base / TOKENIZER_DIR_NAME
    if not path.exists():
        raise FileNotFoundError(f"Tokenizer directory not found: {path}")
    return path


def ensure_output_dir(output_dir: Optional[str], variant: str) -> Path:
    base = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    out = base / variant
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_tts_model(
    model_path: Path,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
):
    """Load Qwen3TTSForConditionalGeneration without speech tokenizer."""
    from qwen_tts.core.models import Qwen3TTSConfig, Qwen3TTSForConditionalGeneration
    from transformers import AutoConfig, AutoModel

    AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
    AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration)

    model = AutoModel.from_pretrained(
        str(model_path),
        dtype=dtype,
        device_map=device,
        local_files_only=True,
        trust_remote_code=True,
    )
    model.eval()
    return model


def load_speech_tokenizer(tokenizer_path: Path, device: str = "cpu", dtype: torch.dtype = torch.float32):
    """Load the speech tokenizer (encoder + decoder) from the tokenizer checkpoint."""
    from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Model
    from qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Config
    from transformers import AutoConfig, AutoModel

    AutoConfig.register("qwen3_tts_tokenizer_12hz", Qwen3TTSTokenizerV2Config)
    AutoModel.register(Qwen3TTSTokenizerV2Config, Qwen3TTSTokenizerV2Model)

    model = AutoModel.from_pretrained(
        str(tokenizer_path),
        dtype=dtype,
        device_map=device,
        local_files_only=True,
        trust_remote_code=True,
    )
    model.eval()
    return model


def verify_onnx(onnx_path: str, test_inputs: Dict[str, np.ndarray],
                 torch_outputs: Dict[str, np.ndarray],
                 atol: float = 1e-4, rtol: float = 1e-3) -> bool:
    """Verify ONNX model output against PyTorch reference."""
    import onnxruntime as ort

    session = ort.InferenceSession(
        onnx_path,
        providers=["CPUExecutionProvider"],
    )

    ort_outputs = session.run(None, test_inputs)

    output_names = [o.name for o in session.get_outputs()]
    all_close = True
    for i, name in enumerate(output_names):
        ref = torch_outputs.get(name)
        if ref is None:
            ref = list(torch_outputs.values())[i]
        if not np.allclose(ort_outputs[i], ref, atol=atol, rtol=rtol):
            max_diff = np.max(np.abs(ort_outputs[i] - ref))
            logger.warning(f"  Output '{name}': max_diff={max_diff:.6f} (atol={atol})")
            all_close = False
        else:
            logger.info(f"  Output '{name}': OK (max_diff={np.max(np.abs(ort_outputs[i] - ref)):.6f})")

    return all_close


_WEIGHT_EXTENSIONS = (".safetensors", ".bin")


def has_model_weights(model_dir: Path) -> bool:
    """Check if a model directory contains actual weight files (safetensors/bin)."""
    for ext in _WEIGHT_EXTENSIONS:
        if list(model_dir.glob(f"*{ext}")):
            return True
    return False


def _onnx_save_safe(onnx_model, onnx_path: str) -> None:
    """Save ONNX model, auto-switching to external data when >2 GiB."""
    import onnx

    PROTO_LIMIT = 2 * 1024 * 1024 * 1024  # 2 GiB
    model_size = onnx_model.ByteSize()

    if model_size < PROTO_LIMIT:
        onnx.save(onnx_model, onnx_path)
    else:
        logger.info(f"  Model size {model_size / (1024**3):.2f} GiB > 2 GiB, saving with external data")
        ext_data_path = os.path.basename(onnx_path) + ".data"
        onnx.save_model(
            onnx_model, onnx_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=ext_data_path,
            size_threshold=1024,
        )


def _onnx_check_safe(onnx_path: str) -> None:
    """Run onnx.checker, handling >2 GiB models by checking from path."""
    import onnx

    try:
        onnx_model = onnx.load(onnx_path, load_external_data=False)
        onnx.checker.check_model(onnx_model)
    except ValueError:
        onnx.checker.check_model(onnx_path)


def simplify_onnx(onnx_path: str) -> bool:
    """Run onnxsim to simplify the ONNX model in-place.

    Returns True on success, False if simplification failed (non-fatal).
    """
    try:
        import onnx
        import onnxsim

        logger.info("Simplifying with onnxsim ...")
        onnx_model = onnx.load(onnx_path)
        simplified, ok = onnxsim.simplify(onnx_model)
        if ok:
            _onnx_save_safe(simplified, onnx_path)
            logger.info(
                f"  Simplified: {len(onnx_model.graph.node)} → "
                f"{len(simplified.graph.node)} nodes"
            )
            return True
        else:
            logger.warning("  onnxsim returned check=False, keeping original")
            return False
    except Exception as e:
        logger.warning(f"  onnxsim failed (non-fatal): {e}")
        return False


def _traceable_sdpa_mask(
    batch_size,
    cache_position,
    kv_length,
    kv_offset=0,
    mask_function=None,
    attention_mask=None,
    **kwargs,
):
    """JIT-traceable causal mask builder that replaces the vmap-based version.

    Constructs a standard lower-triangular causal mask using only basic tensor
    ops so that torch.jit.trace / torch.onnx.export can record the graph.
    """
    q_length = cache_position.shape[0]
    q_pos = cache_position.unsqueeze(1)
    kv_pos = torch.arange(kv_length, device=cache_position.device).unsqueeze(0) + kv_offset
    causal_mask = kv_pos <= q_pos
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, q_length, kv_length)

    if attention_mask is not None and attention_mask.ndim == 2:
        pad = attention_mask[:, -kv_length:].bool()
        causal_mask = causal_mask & pad[:, None, None, :]

    return causal_mask


@contextmanager
def _patch_vmap_mask():
    """Temporarily replace the vmap-based sdpa_mask with a JIT-traceable version.

    transformers >=4.50 uses torch.vmap inside create_causal_mask which crashes
    under torch.jit.trace (RuntimeError: unordered_map::at). This context
    manager swaps it out for the duration of ONNX export.
    """
    try:
        from transformers import masking_utils
        from transformers.masking_utils import AttentionMaskInterface
    except ImportError:
        yield
        return

    orig = AttentionMaskInterface._global_mapping.get("sdpa")
    if orig is None:
        yield
        return

    AttentionMaskInterface._global_mapping["sdpa"] = _traceable_sdpa_mask
    logger.info("  Patched sdpa_mask → traceable version (no vmap)")
    try:
        yield
    finally:
        AttentionMaskInterface._global_mapping["sdpa"] = orig
        logger.info("  Restored original sdpa_mask")


def export_onnx(
    model: torch.nn.Module,
    dummy_inputs: tuple,
    input_names: list,
    output_names: list,
    dynamic_axes: dict,
    onnx_path: str,
    opset_version: int = 18,
    simplify: bool = True,
) -> None:
    """Export a PyTorch module to ONNX, then validate and optionally simplify.

    IMPORTANT: Model and dummy_inputs MUST be in FP32. ONNX export does not
    support BF16/FP16 reliably (ComplexDouble RoPE, BF16 Conv, etc.).
    Precision reduction is handled later by TRT engine build or explicit weight casting.

    Uses the legacy TorchScript exporter (dynamo=False) for maximum compatibility
    with models that contain data-dependent shapes (e.g. Mimi's dynamic padding).
    """
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    logger.info(f"Exporting to {onnx_path} (fp32 graph) ...")

    with _patch_vmap_mask():
        torch.onnx.export(
            model,
            dummy_inputs,
            onnx_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True,
            dynamo=False,
        )

    import onnx

    file_size = os.path.getsize(onnx_path)

    if file_size < 2 * 1024 * 1024 * 1024:
        onnx_model = onnx.load(onnx_path)
        try:
            onnx.checker.check_model(onnx_model)
        except ValueError:
            onnx.checker.check_model(onnx_path)
        logger.info(f"ONNX model exported: {onnx_path}")
        logger.info(f"  Initializers: {len(onnx_model.graph.initializer)}")
        logger.info(f"  Nodes (before simplify): {len(onnx_model.graph.node)}")
        del onnx_model
    else:
        _onnx_check_safe(onnx_path)
        logger.info(f"ONNX model exported (large): {onnx_path}")

    if simplify:
        simplify_onnx(onnx_path)

    file_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    logger.info(f"  File size: {file_size_mb:.1f} MB")

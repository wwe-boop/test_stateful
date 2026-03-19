"""
Shared utilities and reusable nn.Modules for ONNX export scripts.
"""

import os
import sys
import json
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
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


def _patch_decoder_rotary_from_weights(decoder) -> None:
    """When decoder_config was None at load, RoPE inv_freq is loaded from checkpoint and can have
    a different head_dim than the built attention layers. Rebuild rotary_emb.inv_freq so
    cos/sin have shape [..., head_dim] matching the attention layer's query_states."""
    if not hasattr(decoder, "pre_transformer") or not hasattr(decoder.pre_transformer, "layers"):
        return
    layers = decoder.pre_transformer.layers
    if not layers:
        return
    attn = getattr(layers[0], "self_attn", None)
    if attn is None:
        return
    cfg = getattr(decoder, "config", None)
    if cfg is None:
        return
    head_dim = getattr(attn, "head_dim", None)
    if head_dim is None:
        head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    setattr(cfg, "head_dim", head_dim)
    rotary = getattr(decoder.pre_transformer, "rotary_emb", None)
    if rotary is None or not hasattr(rotary, "inv_freq"):
        return
    # inv_freq length must be head_dim/2 (forward does cat(freqs, freqs) -> cos/sin last dim = head_dim)
    current_len = rotary.inv_freq.shape[0]
    expected_len = head_dim // 2
    # Always rebuild so cos/sin match the attention layer's query (handles checkpoint vs build mismatch)
    rope_theta = float(getattr(cfg, "rope_theta", 10000.0))
    device = rotary.inv_freq.device
    inv_freq = 1.0 / (
        rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
    )
    rotary.register_buffer("inv_freq", inv_freq, persistent=False)
    if hasattr(rotary, "attention_scaling"):
        scaling = getattr(rotary, "attention_scaling", None)
        if scaling is not None and isinstance(scaling, torch.Tensor):
            pass  # keep existing
        else:
            rotary.attention_scaling = 1.0
    logger.info(
        "Patched decoder rotary_emb.inv_freq: head_dim=%s (inv_freq len %s -> %s)",
        head_dim, current_len, expected_len,
    )


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
    # When config had decoder_config=None, default config can mismatch checkpoint; fix RoPE head_dim
    if hasattr(model, "decoder"):
        _patch_decoder_rotary_from_weights(model.decoder)
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
        ort_out = np.asarray(ort_outputs[i])
        ref_has_nan = np.isnan(ref).any()
        ort_has_nan = np.isnan(ort_out).any()
        if ref_has_nan or ort_has_nan:
            logger.warning(
                f"  Output '{name}': ref_has_nan={ref_has_nan}, ort_has_nan={ort_has_nan}"
            )
            all_close = False
            continue
        if not np.allclose(ort_out, ref, atol=atol, rtol=rtol):
            diff = np.abs(ort_out.astype(np.float64) - ref.astype(np.float64))
            max_diff = np.max(diff)
            logger.warning(f"  Output '{name}': max_diff={max_diff:.6f} (atol={atol})")
            all_close = False
        else:
            diff = np.abs(ort_out.astype(np.float64) - ref.astype(np.float64))
            logger.info(f"  Output '{name}': OK (max_diff={np.max(diff):.6f})")

    return all_close


_WEIGHT_EXTENSIONS = (".safetensors", ".bin")


def has_model_weights(model_dir: Path) -> bool:
    """Check if a model directory contains actual weight files (safetensors/bin)."""
    for ext in _WEIGHT_EXTENSIONS:
        if list(model_dir.glob(f"*{ext}")):
            return True
    return False


def _remove_stale_data_file(onnx_path: str) -> None:
    """Remove existing .data file to prevent onnx.save_model from appending to it."""
    data_file = onnx_path + ".data"
    if os.path.isfile(data_file):
        os.remove(data_file)


def _onnx_save_safe(onnx_model, onnx_path: str) -> None:
    """Save ONNX model, auto-switching to external data when >2 GiB.

    Computes real weight size from in-memory raw_data (not ByteSize, which
    under-counts when stale data_location=EXTERNAL flags are present).
    """
    import onnx

    PROTO_LIMIT = 2 * 1024 * 1024 * 1024  # 2 GiB

    total_weight_bytes = 0
    for init in onnx_model.graph.initializer:
        if init.raw_data:
            total_weight_bytes += len(init.raw_data)
        elif init.float_data:
            total_weight_bytes += len(init.float_data) * 4
        elif init.int32_data:
            total_weight_bytes += len(init.int32_data) * 4
        elif init.int64_data:
            total_weight_bytes += len(init.int64_data) * 8

    if total_weight_bytes < PROTO_LIMIT:
        for init in onnx_model.graph.initializer:
            if init.data_location == 1:
                init.data_location = 0
        onnx.save(onnx_model, onnx_path)
    else:
        logger.info(
            f"  Weights {total_weight_bytes / (1024**3):.2f} GiB >= 2 GiB, "
            f"saving with external data"
        )
        ext_data_path = os.path.basename(onnx_path) + ".data"
        _remove_stale_data_file(onnx_path)
        onnx.save_model(
            onnx_model, onnx_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=ext_data_path,
            size_threshold=1024,
        )


def _consolidate_onnx_external_data(onnx_path: str) -> None:
    """Consolidate per-tensor external data files into a single .data file.

    torch.onnx.export saves large models (>2 GiB) with one file per tensor.
    TensorRT prefers a single .data file.  This loads the model with all
    external data in memory, re-saves with all_tensors_to_one_file=True,
    then removes the per-tensor files.
    """
    import onnx

    model = onnx.load(onnx_path, load_external_data=False)

    ext_locations: set[str] = set()
    for init in model.graph.initializer:
        if init.data_location == 1:
            for entry in init.external_data:
                if entry.key == "location":
                    ext_locations.add(entry.value)
                    break

    if not ext_locations:
        del model
        return

    data_file = os.path.basename(onnx_path) + ".data"
    if len(ext_locations) == 1 and data_file in ext_locations:
        del model
        return

    logger.info(
        f"  Consolidating {len(ext_locations)} per-tensor files → {data_file}"
    )
    del model

    model = onnx.load(onnx_path, load_external_data=True)
    _remove_stale_data_file(onnx_path)
    onnx.save_model(
        model,
        onnx_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_file,
        size_threshold=1024,
    )
    del model

    base_dir = os.path.dirname(onnx_path) or "."
    cleaned = 0
    for loc in ext_locations:
        if loc == data_file:
            continue
        filepath = os.path.join(base_dir, loc)
        if os.path.isfile(filepath):
            os.remove(filepath)
            cleaned += 1
    logger.info(f"  Cleaned up {cleaned} per-tensor files")


def _onnx_check_safe(onnx_path: str) -> None:
    """Run onnx.checker, handling >2 GiB models with external data.

    When external .data file exists, run check from model dir so relative
    paths in the proto resolve correctly. Fallback: make ValidationError
    non-fatal (TRT/ORT may still load the model).
    """
    import onnx

    onnx_path = os.path.abspath(onnx_path)
    ext_data = onnx_path + ".data"
    if os.path.exists(ext_data):
        model_dir = os.path.dirname(onnx_path)
        model_name = os.path.basename(onnx_path)
        try:
            if model_dir:
                orig_cwd = os.getcwd()
                os.chdir(model_dir)
                try:
                    onnx.checker.check_model(model_name)
                finally:
                    os.chdir(orig_cwd)
            else:
                onnx.checker.check_model(onnx_path)
        except Exception as e:
            if "doesn't exist or is not accessible" in str(e) or "ValidationError" in type(e).__name__:
                logger.warning(
                    "ONNX checker failed (external data path); model may still work: %s",
                    e,
                )
            else:
                raise
        return
    try:
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
    except ValueError:
        onnx.checker.check_model(onnx_path)


def simplify_onnx(onnx_path: str) -> bool:
    """Run onnxsim to simplify the ONNX model in-place.

    Returns True on success, False if simplification failed (non-fatal).
    Uses _onnx_save_safe which handles >2 GiB models with external data.

    Skips models with external data (>2 GiB weights) because onnxsim 0.x
    serializes the full model via protobuf (hard 2 GiB limit).  TensorRT
    and ONNX Runtime apply their own graph optimizations at load time.
    """
    ext_data = onnx_path + ".data"
    if os.path.exists(ext_data):
        ext_mb = os.path.getsize(ext_data) / (1024 * 1024)
        logger.info(
            f"Skipping onnxsim: model has {ext_mb:.0f} MB external data "
            f"(exceeds protobuf 2 GiB serialize limit). "
            f"TRT/ORT will optimize the graph at engine build time."
        )
        return False

    try:
        import onnx
        import onnxsim

        logger.info("Simplifying with onnxsim ...")
        onnx_model = onnx.load(onnx_path)
        n_before = len(onnx_model.graph.node)

        simplified, ok = onnxsim.simplify(onnx_model)
        del onnx_model
        if not ok:
            logger.warning("  onnxsim returned check=False, keeping original")
            return False

        n_after = len(simplified.graph.node)
        logger.info(f"  Simplified: {n_before} → {n_after} nodes")

        _onnx_save_safe(simplified, onnx_path)
        return True
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
    do_constant_folding: bool = True,
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

    # Break shared storage only when needed (e.g. fused QKV views of one buffer).
    # Cloning every parameter can double external data size (~6GB -> ~12GB) when
    # the exporter then writes each tensor; we only clone tensors that share
    # storage with another so TRT gets correct per-tensor shapes.
    def _storage_id(t):
        return t.untyped_storage().data_ptr() if t.numel() > 0 else None

    seen_storage = {}
    for p in model.parameters():
        sid = _storage_id(p.data)
        if sid is not None:
            seen_storage[sid] = seen_storage.get(sid, 0) + 1
    for b in model.buffers():
        sid = _storage_id(b.data)
        if sid is not None:
            seen_storage[sid] = seen_storage.get(sid, 0) + 1

    for p in model.parameters():
        sid = _storage_id(p.data)
        if p.data.is_floating_point():
            d = p.data.detach().float().contiguous()
            if seen_storage.get(sid, 1) > 1:
                d = d.clone()
            p.data = d
        else:
            if seen_storage.get(sid, 1) > 1:
                p.data = p.data.detach().contiguous().clone()
            elif not p.data.is_contiguous():
                p.data = p.data.detach().contiguous()
    for b in model.buffers():
        sid = _storage_id(b.data)
        if b.data.is_floating_point():
            d = b.data.detach().float().contiguous()
            if seen_storage.get(sid, 1) > 1:
                d = d.clone()
            b.data = d
        else:
            if seen_storage.get(sid, 1) > 1:
                b.data = b.data.detach().contiguous().clone()
            elif not b.data.is_contiguous():
                b.data = b.data.detach().contiguous()

    _remove_stale_data_file(onnx_path)

    with _patch_vmap_mask():
        torch.onnx.export(
            model,
            dummy_inputs,
            onnx_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=do_constant_folding,
            dynamo=False,
        )

    import onnx

    # torch.onnx.export saves >2 GiB models with one file per tensor.
    # Consolidate into a single .data file for TensorRT compatibility.
    _consolidate_onnx_external_data(onnx_path)

    # Use safe check when model has external data: onnx.load(load_external_data=True)
    # loads full weights into memory; check_model() calls SerializeToString() which
    # hits protobuf's 2 GiB limit. load_external_data=False keeps only graph in memory.
    ext_data_path = onnx_path + ".data"
    if os.path.exists(ext_data_path):
        _onnx_check_safe(onnx_path)
        onnx_model = onnx.load(onnx_path, load_external_data=False)
        logger.info(f"ONNX model exported: {onnx_path}")
        logger.info(f"  Initializers: {len(onnx_model.graph.initializer)}")
        logger.info(f"  Nodes (before simplify): {len(onnx_model.graph.node)}")
        del onnx_model
    else:
        onnx_model = onnx.load(onnx_path)
        try:
            onnx.checker.check_model(onnx_model)
        except ValueError:
            onnx.checker.check_model(onnx_path)
        except Exception as e:
            from google.protobuf.message import EncodeError
            if isinstance(e, EncodeError):
                _onnx_check_safe(onnx_path)
            else:
                raise
        logger.info(f"ONNX model exported: {onnx_path}")
        logger.info(f"  Initializers: {len(onnx_model.graph.initializer)}")
        logger.info(f"  Nodes (before simplify): {len(onnx_model.graph.node)}")
        del onnx_model

    if simplify:
        simplify_onnx(onnx_path)

    file_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    ext_data = onnx_path + ".data"
    if os.path.exists(ext_data):
        ext_mb = os.path.getsize(ext_data) / (1024 * 1024)
        logger.info(f"  File size: {file_size_mb:.1f} MB + {ext_mb:.1f} MB external data")
    else:
        logger.info(f"  File size: {file_size_mb:.1f} MB")


# ---------------------------------------------------------------------------
#  Reusable nn.Modules for fused ONNX export (used by 04a, 04b, 05)
# ---------------------------------------------------------------------------

class CodePredictorUnrolled(nn.Module):
    """All Code Predictor stages fully unrolled into a single forward pass (no KV Cache).

    Used by export_04 (fused context), export_05 (fused decode), and
    export_code_predictor (standalone CP export). Shared here so the dependency is explicit.

    Each of the (num_code_groups-1) stages:
      1. Appends the new codec embedding to the sequence
      2. Projects through small_to_mtp_projection
      3. Full prefill through 5-layer Transformer
      4. Takes last hidden → lm_head[stage] → argmax → next token

    See architecture.md §5.4 for design rationale.
    """

    def __init__(self, code_predictor, talker_codec_embedding):
        super().__init__()
        self.transformer_layers = code_predictor.model.layers
        self.norm = code_predictor.model.norm
        self.rotary_emb = code_predictor.model.rotary_emb
        self.projection = code_predictor.small_to_mtp_projection

        self.codec_embeddings = code_predictor.model.codec_embedding
        self.talker_codec_embedding = talker_codec_embedding
        self.lm_heads = code_predictor.lm_head

        self.num_stages = len(self.lm_heads)
        self.hidden_size = code_predictor.config.hidden_size

    def _transformer_forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        device = x.device

        position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
        position_embeddings = self.rotary_emb(x, position_ids)

        causal_mask = torch.triu(
            torch.full((S, S), float('-inf'), device=device, dtype=x.dtype),
            diagonal=1,
        )
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        hidden = x
        for layer in self.transformer_layers:
            layer_out = layer(
                hidden,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=None,
                output_attentions=False,
                use_cache=False,
                cache_position=torch.arange(S, device=device),
                position_embeddings=position_embeddings,
            )
            hidden = layer_out[0]

        return self.norm(hidden)

    def forward(self, past_hidden: torch.Tensor, codec_token_0: torch.Tensor) -> torch.Tensor:
        """
        Args:
            past_hidden:   [B, 1, talker_hidden_size] - last hidden from Talker
            codec_token_0: [B] - first codec token (sampled from Talker logits)
        Returns:
            codec_tokens:  [B, num_stages] - predicted codec tokens for codebooks 1..num_code_groups-1
        """
        embed_0 = self.talker_codec_embedding(codec_token_0).unsqueeze(1)
        sequence = self.projection(torch.cat([past_hidden, embed_0], dim=1))

        output_tokens = []
        for stage in range(self.num_stages):
            hidden = self._transformer_forward(sequence)
            logits = self.lm_heads[stage](hidden[:, -1:, :])
            token = logits.argmax(dim=-1).squeeze(-1)
            output_tokens.append(token)

            if stage < self.num_stages - 1:
                next_embed = self.projection(
                    self.codec_embeddings[stage](token).unsqueeze(1)
                )
                sequence = torch.cat([sequence, next_embed], dim=1)

        return torch.stack(output_tokens, dim=1)

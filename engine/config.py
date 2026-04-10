"""Engine configuration — split into engine params (YAML) and model params (manifest).

Loading priority for model architecture:
    model_manifest.json (from engine_dir) > engine.yaml model overrides > defaults

Loading priority for engine params:
    CLI args > ENGINE_* env vars > engine.yaml > defaults

Usage:
    cfg = load_config("engine.yaml", cli_overrides={...})
    model_arch = load_model_manifest(engine_dir, cfg)
    model_config = to_model_config(model_arch, cfg)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)

try:
    import yaml
except ImportError:
    yaml = None


# ---------------------------------------------------------------------------
# Model architecture (from manifest, not engine.yaml)
# ---------------------------------------------------------------------------

@dataclass
class ModelArchConfig:
    """Model architecture — loaded from model_manifest / triton_manifest.json.

    These values are determined at export time and travel with the model
    artifacts.  The standalone engine reads them from the manifest file
    in engine_dir instead of requiring manual configuration.
    """
    variant: str = ""
    num_layers: int = 28
    hidden_size: int = 2048
    kv_heads: int = 8
    head_dim: int = 128
    codec_vocab_size: int = 3072
    logits_topk: int = 50
    n_c2w_layers: int = 8
    c2w_kv_heads: int = 16
    c2w_head_dim: int = 64
    c2w_sliding_window: int = 72
    n_c2w_conv_states: int = 17
    n_c2w_transconv_states: int = 4
    dtype: str = "bf16"
    tts_model_type: str = "unknown"
    supported_task_types: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Engine config dataclasses (engine.yaml)
# ---------------------------------------------------------------------------

@dataclass
class PathsConfig:
    """File paths — resolved relative to repo root or absolute."""
    tokenizer_dir: str = ""
    weights_dir: str = ""
    engine_dir: str = ""


@dataclass
class ServerConfig:
    """gRPC server settings."""
    port: int = 50051
    health_port: int = 8080
    max_sessions: int = 128
    request_timeout_sec: float = 120.0
    warmup_rounds: int = 3


@dataclass
class SchedulerConfig:
    """Decode batch scheduling parameters."""
    max_batch_size: int = 48
    max_seq_len: int = 512
    # MLFQ
    mlfq_q1_threshold: int = 50
    mlfq_q2_threshold: int = 200
    mlfq_aging_interval: int = 100
    mlfq_starvation_limit: int = 50
    # Eviction
    max_idle_sec: float = 10.0
    max_queue_size: int = 256
    # Session timeout
    session_timeout_sec: float = 300.0
    # Pad phase: minimum pad steps before silence detection activates
    min_pad_steps: int = 4


@dataclass
class PrefixCacheConfig:
    """Cross-request prefix KV cache."""
    enabled: bool = True
    max_entries: int = 16
    max_prefix_len: int = 512


@dataclass
class SpliterConfig:
    """Text segmentation / Spliter parameters."""
    ema_ratio_initial: float = 5.5
    ema_alpha: float = 0.1
    ema_overflow_alpha: float = 0.5
    ema_min_ratio: float = 2.0
    ema_max_ratio: float = 10.0
    max_concurrent_segments: int = 2
    prefill_len: int = 12
    safety_margin: int = 8
    # Punct-tier split mins as fractions of dynamic cap (see spliter/driver.compute_thresholds)
    l1_split_cap_ratio: float = 0.70
    l2_split_cap_ratio: float = 0.80
    l3_split_cap_ratio: float = 0.90


@dataclass
class SamplingConfig:
    """Decode sampling parameters (defaults, overridable per-request)."""
    do_sample: bool = True
    temperature: float = 0.9
    repetition_penalty: float = 1.05
    top_k: int = 50
    random_seed: int = 0


@dataclass
class PrefillConfig:
    """Prefill / CustomVoice speaker defaults (see engine.yaml)."""
    default_speaker: str = "vivian"
    fallback_speaker: str = "vivian"


@dataclass
class EngineConfig:
    """Top-level engine configuration (no model architecture)."""
    paths: PathsConfig = field(default_factory=PathsConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    prefix_cache: PrefixCacheConfig = field(default_factory=PrefixCacheConfig)
    spliter: SpliterConfig = field(default_factory=SpliterConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    prefill: PrefillConfig = field(default_factory=PrefillConfig)


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _deep_update(base: dict, override: dict) -> dict:
    """Recursively merge override into base dict."""
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _apply_env_overrides(raw: dict) -> dict:
    """Apply ENGINE_* environment variable overrides.

    E.g. ENGINE_SCHEDULER_MAX_BATCH_SIZE=32 → scheduler.max_batch_size = 32
    """
    prefix = "ENGINE_"
    for key, val in os.environ.items():
        if not key.startswith(prefix):
            continue
        parts = key[len(prefix):].lower().split("_", 1)
        if len(parts) < 2:
            continue
        section, field_name = parts[0], parts[1]
        if section not in raw:
            raw[section] = {}
        raw[section][field_name] = _coerce_value(val)
    return raw


def _coerce_value(val: str):
    """Best-effort type coercion for env vars."""
    if val.lower() in ("true", "yes"):
        return True
    if val.lower() in ("false", "no"):
        return False
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        pass
    return val


def _dict_to_config(raw: dict) -> EngineConfig:
    """Build EngineConfig from a raw dict, ignoring unknown keys."""
    cfg = EngineConfig()
    for section_name, section_cls in [
        ("paths", PathsConfig),
        ("server", ServerConfig),
        ("scheduler", SchedulerConfig),
        ("prefix_cache", PrefixCacheConfig),
        ("spliter", SpliterConfig),
        ("sampling", SamplingConfig),
        ("prefill", PrefillConfig),
    ]:
        section_data = raw.get(section_name, {})
        if not isinstance(section_data, dict):
            continue
        section_obj = getattr(cfg, section_name)
        for k, v in section_data.items():
            if hasattr(section_obj, k):
                expected_type = type(getattr(section_obj, k))
                try:
                    setattr(section_obj, k, expected_type(v))
                except (ValueError, TypeError):
                    logger.warning("Cannot set %s.%s = %r", section_name, k, v)
    return cfg


# ---------------------------------------------------------------------------
# Public: load engine config
# ---------------------------------------------------------------------------

_AUTO_CONFIG_NAMES = ("engine.yaml", "engine.yml")


def load_config(
    config_path: Optional[str] = None,
    cli_overrides: Optional[dict] = None,
) -> EngineConfig:
    """Load engine config from YAML file + env vars + CLI overrides.

    Priority: CLI > env vars > YAML file > defaults.
    This only loads engine-level params.  Model architecture is loaded
    separately via ``load_model_manifest()``.

    When *config_path* is not given, auto-discovers engine.yaml / engine.yml
    in the current working directory.
    """
    raw: dict = {}

    if not config_path:
        for name in _AUTO_CONFIG_NAMES:
            candidate = Path(name)
            if candidate.exists():
                config_path = str(candidate)
                break

    if config_path:
        p = Path(config_path)
        if p.exists():
            if yaml is None:
                raise ImportError(
                    "PyYAML required for YAML config. Install: pip install pyyaml"
                )
            with open(p, encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
            logger.info("Loaded config from %s", p)
        else:
            logger.warning("Config file not found: %s, using defaults", p)

    _apply_env_overrides(raw)

    if cli_overrides:
        _deep_update(raw, cli_overrides)

    return _dict_to_config(raw)


# ---------------------------------------------------------------------------
# Public: load model manifest (architecture)
# ---------------------------------------------------------------------------

MANIFEST_FILENAMES = ("triton_manifest.json", "model_manifest.json")


def load_model_manifest(
    engine_dir: str,
    engine_config: Optional[EngineConfig] = None,
    tokenizer_dir: str = "",
) -> ModelArchConfig:
    """Load model architecture from manifest file in engine_dir.

    Search order for manifest:
      1. engine_dir/triton_manifest.json  (generated by export_09)
      2. engine_dir/model_manifest.json

    If no manifest is found, falls back to auto-detection from
    tokenizer config.json, then to hardcoded defaults.

    Priority: manifest > engine.yaml "model" overrides > auto-detect > defaults.
    """
    arch = ModelArchConfig()
    manifest_data: dict = {}

    # --- Try loading manifest from engine_dir ---
    if engine_dir:
        engine_path = Path(engine_dir)
        for fname in MANIFEST_FILENAMES:
            manifest_file = engine_path / fname
            if manifest_file.is_file():
                try:
                    with open(manifest_file, encoding="utf-8") as f:
                        manifest_data = json.load(f)
                    logger.info("Loaded model manifest from %s", manifest_file)
                except (json.JSONDecodeError, OSError) as e:
                    logger.warning("Failed to read %s: %s", manifest_file, e)
                break

    # --- Extract architecture section ---
    arch_section = manifest_data.get("architecture", {})

    if not arch_section and manifest_data.get("talker"):
        talker = manifest_data["talker"]
        c2w = manifest_data.get("code2wav_fused", {})
        arch_section = {
            "num_layers": talker.get("num_layers"),
            "hidden_size": talker.get("hidden_size"),
            "kv_heads": talker.get("num_kv_heads"),
            "head_dim": talker.get("head_dim"),
            "codec_vocab_size": talker.get("vocab_size"),
            "n_c2w_layers": c2w.get("num_code2wav_hidden_layers"),
            "dtype": manifest_data.get("engine_dtype", "bf16"),
        }
        arch_section = {k: v for k, v in arch_section.items() if v is not None}
        logger.info("Built architecture from legacy manifest talker/code2wav sections")

    # --- Fallback: auto-detect from tokenizer config.json ---
    if not arch_section and tokenizer_dir:
        arch_section = _detect_from_tokenizer(tokenizer_dir)

    # --- Apply architecture to ModelArchConfig ---
    if manifest_data.get("variant"):
        arch.variant = manifest_data["variant"]
    orchestrator = manifest_data.get("orchestrator", {})
    if isinstance(orchestrator, dict):
        tts_model_type = orchestrator.get("tts_model_type")
        if tts_model_type:
            arch.tts_model_type = str(tts_model_type)
        supported = orchestrator.get("supported_task_types")
        if isinstance(supported, str) and supported.strip():
            arch.supported_task_types = tuple(
                s.strip() for s in supported.split(",") if s.strip()
            )
        elif isinstance(supported, list):
            arch.supported_task_types = tuple(
                str(s).strip() for s in supported if str(s).strip()
            )

    applied = []
    for k, v in arch_section.items():
        if hasattr(arch, k):
            expected_type = type(getattr(arch, k))
            try:
                setattr(arch, k, expected_type(v))
                applied.append(f"{k}={v}")
            except (ValueError, TypeError):
                pass

    if applied:
        logger.info("Model architecture: %s", ", ".join(applied))

    return arch


def _detect_from_tokenizer(tokenizer_dir: str) -> dict:
    """Fallback: extract model dimensions from HuggingFace config.json."""
    config_path = Path(tokenizer_dir) / "config.json"
    if not config_path.exists():
        return {}

    try:
        with open(config_path, encoding="utf-8") as f:
            hf_cfg = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to read %s: %s", config_path, e)
        return {}

    talker = hf_cfg.get("talker_config", {})
    if not talker:
        return {}

    detected = {
        "num_layers": talker.get("num_hidden_layers"),
        "hidden_size": talker.get("hidden_size"),
        "kv_heads": talker.get("num_key_value_heads"),
        "head_dim": talker.get("head_dim"),
        "codec_vocab_size": talker.get("vocab_size"),
    }
    detected = {k: v for k, v in detected.items() if v is not None}

    if detected:
        logger.info("Auto-detected from %s: %s", config_path.name,
                     ", ".join(f"{k}={v}" for k, v in detected.items()))
    return detected


# ---------------------------------------------------------------------------
# Public: convert to backend ModelConfig
# ---------------------------------------------------------------------------

def torch_dtype(dtype_str: str) -> torch.dtype:
    mapping = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    return mapping.get(dtype_str, torch.bfloat16)


def to_model_config(arch: ModelArchConfig, cfg: EngineConfig):
    """Convert ModelArchConfig + EngineConfig → backend.kv_cache_pool.ModelConfig."""
    from .backend.kv_cache_pool import ModelConfig
    return ModelConfig(
        num_layers=arch.num_layers,
        kv_heads=arch.kv_heads,
        head_dim=arch.head_dim,
        max_seq_len=cfg.scheduler.max_seq_len,
        hidden_size=arch.hidden_size,
        dtype=torch_dtype(arch.dtype),
        codec_vocab_size=arch.codec_vocab_size,
        logits_topk=arch.logits_topk,
        n_c2w_layers=arch.n_c2w_layers,
        c2w_kv_heads=arch.c2w_kv_heads,
        c2w_head_dim=arch.c2w_head_dim,
        c2w_sliding_window=arch.c2w_sliding_window,
        n_c2w_conv_states=arch.n_c2w_conv_states,
        n_c2w_transconv_states=arch.n_c2w_transconv_states,
    )

"""Engine configuration: YAML-based, replacing triton_manifest.json.

Loads from engine.yaml with env-var overrides.  All parameters have
sensible defaults so the engine can start with an empty config.

Example minimal engine.yaml:
    model:
      variant: custom-1.7b
      weights_dir: /path/to/weights
      engine_dir: /path/to/engines
      tokenizer_dir: /path/to/tokenizer
"""

from __future__ import annotations

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
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ModelArchConfig:
    """Model architecture — auto-detected from weights/config.json if omitted."""
    variant: str = "custom-1.7b"
    # Talker
    num_layers: int = 28
    hidden_size: int = 1536
    kv_heads: int = 8
    head_dim: int = 64
    codec_vocab_size: int = 2176
    # Code2Wav
    n_c2w_layers: int = 8
    c2w_kv_heads: int = 16
    c2w_head_dim: int = 64
    c2w_sliding_window: int = 72
    n_c2w_conv_states: int = 17
    n_c2w_transconv_states: int = 4
    # Precision
    dtype: str = "bf16"


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
    max_seq_len: int = 2048
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
    max_concurrent_segments: int = 2
    prefill_len: int = 12


@dataclass
class SamplingConfig:
    """Decode sampling parameters (defaults, overridable per-request)."""
    temperature: float = 1.0
    repetition_penalty: float = 1.05
    top_k: int = 50


@dataclass
class EngineConfig:
    """Top-level engine configuration."""
    model: ModelArchConfig = field(default_factory=ModelArchConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    prefix_cache: PrefixCacheConfig = field(default_factory=PrefixCacheConfig)
    spliter: SpliterConfig = field(default_factory=SpliterConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)

    def torch_dtype(self) -> torch.dtype:
        mapping = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        return mapping.get(self.model.dtype, torch.bfloat16)


# ---------------------------------------------------------------------------
# Loading
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
    """Build EngineConfig from a flat dict, ignoring unknown keys."""
    cfg = EngineConfig()
    for section_name, section_cls in [
        ("model", ModelArchConfig),
        ("paths", PathsConfig),
        ("server", ServerConfig),
        ("scheduler", SchedulerConfig),
        ("prefix_cache", PrefixCacheConfig),
        ("spliter", SpliterConfig),
        ("sampling", SamplingConfig),
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


def load_config(
    config_path: Optional[str] = None,
    cli_overrides: Optional[dict] = None,
) -> EngineConfig:
    """Load engine config from YAML file + env vars + CLI overrides.

    Priority: CLI > env vars > YAML file > defaults.
    """
    raw: dict = {}

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


def to_model_config(cfg: EngineConfig):
    """Convert EngineConfig.model → backend.kv_cache_pool.ModelConfig."""
    from .backend.kv_cache_pool import ModelConfig
    m = cfg.model
    return ModelConfig(
        num_layers=m.num_layers,
        kv_heads=m.kv_heads,
        head_dim=m.head_dim,
        max_seq_len=cfg.scheduler.max_seq_len,
        hidden_size=m.hidden_size,
        dtype=cfg.torch_dtype(),
        codec_vocab_size=m.codec_vocab_size,
        n_c2w_layers=m.n_c2w_layers,
        c2w_kv_heads=m.c2w_kv_heads,
        c2w_head_dim=m.c2w_head_dim,
        c2w_sliding_window=m.c2w_sliding_window,
        n_c2w_conv_states=m.n_c2w_conv_states,
        n_c2w_transconv_states=m.n_c2w_transconv_states,
    )

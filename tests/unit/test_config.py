"""Tests for engine.config — YAML config loading."""

import os
import tempfile

import pytest

from engine.config import (
    EngineConfig,
    load_config,
    _coerce_value,
    _apply_env_overrides,
    _dict_to_config,
    to_model_config,
)


class TestCoercion:
    def test_bool_true(self):
        assert _coerce_value("true") is True
        assert _coerce_value("yes") is True

    def test_bool_false(self):
        assert _coerce_value("false") is False
        assert _coerce_value("no") is False

    def test_int(self):
        assert _coerce_value("42") == 42

    def test_float(self):
        assert _coerce_value("3.14") == pytest.approx(3.14)

    def test_string(self):
        assert _coerce_value("hello") == "hello"


class TestEnvOverrides:
    def test_applies_engine_prefix(self):
        raw: dict = {"scheduler": {"max_batch_size": 48}}
        env = {"ENGINE_SCHEDULER_MAX_BATCH_SIZE": "32"}
        with _patch_env(env):
            _apply_env_overrides(raw)
        assert raw["scheduler"]["max_batch_size"] == 32

    def test_ignores_non_engine(self):
        raw: dict = {}
        env = {"OTHER_VAR": "123"}
        with _patch_env(env):
            _apply_env_overrides(raw)
        assert "other" not in raw


class TestLoadConfig:
    def test_defaults(self):
        cfg = load_config()
        assert cfg.scheduler.max_batch_size == 48
        assert cfg.model.num_layers == 28
        assert cfg.prefix_cache.enabled is True

    def test_yaml_file(self):
        try:
            import yaml
        except ImportError:
            pytest.skip("PyYAML not installed")

        content = "scheduler:\n  max_batch_size: 16\nmodel:\n  variant: base-0.6b\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(content)
            f.flush()
            cfg = load_config(f.name)
        os.unlink(f.name)
        assert cfg.scheduler.max_batch_size == 16
        assert cfg.model.variant == "base-0.6b"

    def test_cli_overrides(self):
        cfg = load_config(cli_overrides={"scheduler": {"max_batch_size": 8}})
        assert cfg.scheduler.max_batch_size == 8

    def test_missing_file_uses_defaults(self):
        cfg = load_config("/nonexistent/path.yaml")
        assert cfg.scheduler.max_batch_size == 48


class TestToModelConfig:
    def test_conversion(self):
        cfg = EngineConfig()
        mc = to_model_config(cfg)
        assert mc.num_layers == cfg.model.num_layers
        assert mc.kv_heads == cfg.model.kv_heads
        assert mc.c2w_sliding_window == cfg.model.c2w_sliding_window


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

class _patch_env:
    def __init__(self, env_vars: dict):
        self._vars = env_vars
        self._orig = {}

    def __enter__(self):
        for k, v in self._vars.items():
            self._orig[k] = os.environ.get(k)
            os.environ[k] = v

    def __exit__(self, *args):
        for k in self._vars:
            if self._orig[k] is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = self._orig[k]

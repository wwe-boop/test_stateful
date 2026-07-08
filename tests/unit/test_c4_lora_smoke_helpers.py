import importlib.util
import sys
from pathlib import Path

import torch


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "python"
    / "run_c4_lora_smoke_train.py"
)

SPEC = importlib.util.spec_from_file_location("c4_lora", SCRIPT_PATH)
c4_lora = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = c4_lora
SPEC.loader.exec_module(c4_lora)


class TinyAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = torch.nn.Module()
        self.self_attn.q_proj = torch.nn.Linear(4, 6, bias=False)
        self.self_attn.v_proj = torch.nn.Linear(4, 4, bias=False)
        self.self_attn.o_proj = torch.nn.Linear(4, 4, bias=False)


def test_lora_linear_preserves_shape_and_starts_as_noop():
    base = torch.nn.Linear(4, 3, bias=False)
    layer = c4_lora.LoRALinear(base, rank=2, alpha=4)
    x = torch.randn(5, 4)

    assert layer(x).shape == (5, 3)
    assert torch.allclose(layer(x), base(x))
    assert layer.lora_A.requires_grad is True
    assert layer.lora_B.requires_grad is True
    assert base.weight.requires_grad is False


def test_inject_lora_matches_suffixes_only():
    model = TinyAttention()

    info = c4_lora.inject_lora(
        model,
        target_suffixes=("self_attn.q_proj", "self_attn.v_proj"),
        rank=2,
        alpha=4,
        dropout=0.0,
    )

    assert info["matched_module_count"] == 2
    assert isinstance(model.self_attn.q_proj, c4_lora.LoRALinear)
    assert isinstance(model.self_attn.v_proj, c4_lora.LoRALinear)
    assert isinstance(model.self_attn.o_proj, torch.nn.Linear)
    assert model.self_attn.o_proj.weight.requires_grad is False


def test_lora_state_dict_only_contains_adapter_weights():
    model = TinyAttention()
    c4_lora.inject_lora(
        model,
        target_suffixes=("self_attn.q_proj",),
        rank=2,
        alpha=4,
        dropout=0.0,
    )

    keys = set(c4_lora.lora_state_dict(model))

    assert "self_attn.q_proj.lora_A" in keys
    assert "self_attn.q_proj.lora_B" in keys
    assert all("base" not in key for key in keys)

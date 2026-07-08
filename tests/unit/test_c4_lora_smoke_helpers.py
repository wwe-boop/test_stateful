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


def test_load_lora_adapter_restores_adapter_weights(tmp_path):
    model = TinyAttention()
    c4_lora.inject_lora(
        model,
        target_suffixes=("self_attn.q_proj",),
        rank=2,
        alpha=4,
        dropout=0.0,
    )
    with torch.no_grad():
        model.self_attn.q_proj.lora_A.fill_(0.25)
        model.self_attn.q_proj.lora_B.fill_(0.5)
    saved_state = c4_lora.lora_state_dict(model)
    adapter_path = tmp_path / "adapter.pt"
    torch.save(
        {
            "metadata": {"source": "unit-test"},
            "state_dict": saved_state,
        },
        adapter_path,
    )

    with torch.no_grad():
        model.self_attn.q_proj.lora_A.zero_()
        model.self_attn.q_proj.lora_B.zero_()

    info = c4_lora.load_lora_adapter(model, adapter_path)

    assert info["loaded_tensor_count"] == 2
    assert info["loaded_param_count"] == 20
    assert info["metadata"] == {"source": "unit-test"}
    assert torch.allclose(model.self_attn.q_proj.lora_A.cpu(), saved_state["self_attn.q_proj.lora_A"])
    assert torch.allclose(model.self_attn.q_proj.lora_B.cpu(), saved_state["self_attn.q_proj.lora_B"])


def test_build_sample_schedule_uses_full_epochs_when_steps_is_zero():
    schedule = c4_lora.build_sample_schedule(
        num_items=3,
        steps=0,
        epochs=2,
        shuffle=False,
        seed=123,
    )

    assert [item["row_index"] for item in schedule] == [0, 1, 2, 0, 1, 2]
    assert [item["epoch"] for item in schedule] == [0, 0, 0, 1, 1, 1]
    assert [item["cycle"] for item in schedule] == [0, 0, 0, 0, 0, 0]


def test_build_sample_schedule_repeats_epoch_plan_to_requested_steps():
    schedule = c4_lora.build_sample_schedule(
        num_items=3,
        steps=5,
        epochs=1,
        shuffle=False,
        seed=123,
    )

    assert [item["row_index"] for item in schedule] == [0, 1, 2, 0, 1]
    assert [item["cycle"] for item in schedule] == [0, 0, 0, 1, 1]


def test_build_sample_schedule_shuffle_is_seeded():
    first = c4_lora.build_sample_schedule(
        num_items=5,
        steps=0,
        epochs=2,
        shuffle=True,
        seed=123,
    )
    second = c4_lora.build_sample_schedule(
        num_items=5,
        steps=0,
        epochs=2,
        shuffle=True,
        seed=123,
    )

    assert first == second
    assert [item["row_index"] for item in first] != [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]

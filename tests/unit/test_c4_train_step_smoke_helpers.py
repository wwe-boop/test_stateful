import importlib.util
import sys
from pathlib import Path

import torch


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "python"
    / "run_c4_train_step_smoke.py"
)

SPEC = importlib.util.spec_from_file_location("c4_train_step", SCRIPT_PATH)
c4_train_step = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = c4_train_step
SPEC.loader.exec_module(c4_train_step)


class TinyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.talker = torch.nn.Module()
        self.talker.codec_head = torch.nn.Linear(2, 2, bias=False)
        self.talker.code_predictor = torch.nn.Module()
        self.talker.code_predictor.lm_head = torch.nn.Linear(2, 2, bias=False)
        self.talker.model = torch.nn.Linear(2, 2, bias=False)


def test_parse_prefixes_drops_empty_values():
    assert c4_train_step.parse_prefixes("a,b,, c ") == ("a", "b", "c")


def test_configure_trainable_only_enables_matching_prefixes():
    model = TinyModule()

    info = c4_train_step.configure_trainable(
        model,
        ("talker.codec_head", "talker.code_predictor.lm_head"),
    )

    assert info["trainable_params"] == 8
    assert model.talker.codec_head.weight.requires_grad is True
    assert model.talker.code_predictor.lm_head.weight.requires_grad is True
    assert model.talker.model.weight.requires_grad is False

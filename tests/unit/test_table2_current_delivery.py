import importlib.util
import sys
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "python"
    / "make_table2_current_delivery.py"
)

SPEC = importlib.util.spec_from_file_location("table2_delivery", SCRIPT_PATH)
table2_delivery = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = table2_delivery
SPEC.loader.exec_module(table2_delivery)


def minimal_row(label: str, cer: float) -> dict:
    return {
        "label": label,
        "f0_jump_mean_st": 1.0,
        "f0_jump_mean_st_seed_std": 0.1,
        "energy_jump_mean_db": 2.0,
        "energy_jump_mean_db_seed_std": 0.2,
        "pause_deviation_mean_ms": 3.0,
        "pause_deviation_mean_ms_seed_std": 0.3,
        "sim_delta_x100": 0.04,
        "sim_delta_x100_seed_std": 0.01,
        "fasl_mean_ms": 5.0,
        "fasl_mean_ms_seed_std": 0.5,
        "cer_mean": cer,
        "cer_seed_std": 0.002,
        "f0_coverage": 0.8,
        "energy_coverage": 0.9,
        "pause_coverage": 1.0,
    }


def test_render_delivery_keeps_full_steadystream_unmeasured():
    table2 = {
        "summary": {
            "stateless_once": minimal_row("无状态", 0.111),
        }
    }
    c4_validation = {
        "samples": 5,
        "segments": 42,
        "coded_segments": 42,
        "code_frames": 1831,
        "speech_seconds_from_codes": 146.48,
        "issue_count": 0,
    }
    c4_lora = {
        "model_dir": "workspace/hf_models/Qwen3-TTS-12Hz-0.6B-Base",
        "manifest_rows": 5,
        "steps": 2,
        "epochs": 1,
        "shuffle": True,
        "seed": 20260709,
        "trainable_params": 675840,
        "total_params": 915318848,
        "loaded_adapter": {
            "loaded_tensor_count": 132,
            "loaded_param_count": 675840,
        },
        "loss_history": [
            {"combined_loss": 14.618859},
            {"combined_loss": 14.419856},
        ],
    }

    rendered = table2_delivery.render_delivery(
        table2=table2,
        c4_validation=c4_validation,
        c4_lora=c4_lora,
    )

    assert "| 完整 SteadyStream | — | — | — | — | — | — | 未测" in rendered
    assert "API-synthetic C4 smoke validates plumbing only" in rendered
    assert "| codec frames | 1831 |" in rendered
    assert "| loaded adapter tensors | 132 |" in rendered
    assert "`14.618859 -> 14.419856`" in rendered


def test_render_delivery_reports_full_inference_row_when_present():
    table2 = {
        "summary": {
            "stateless_once": minimal_row("无状态", 0.111),
            "full_steadystream": minimal_row("完整 SteadyStream（C1+C2+C3, C4未训练）", 0.222),
        }
    }

    rendered = table2_delivery.render_delivery(
        table2=table2,
        c4_validation=None,
        c4_lora=None,
    )

    assert "Current measured inference rows: `2/6`" in rendered
    assert "Full-row guardrail" in rendered
    assert "| 完整 SteadyStream（C1+C2+C3, C4未训练） |" in rendered
    assert "22.20±0.20%" in rendered
    assert "未测：没有真实 C4 checkpoint" not in rendered

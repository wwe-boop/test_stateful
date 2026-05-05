from __future__ import annotations

import asyncio

import pytest

from demo_api.audio_assets import attach_audio_to_result, scheduled_start_ms
from demo_api.audio_store import AudioStore
from demo_api.jobs import ConcurrencyJobManager
from demo_api.schemas import RunMetrics, RunResult, percentile, summarize_ttft
from demo_api.schemas import normalize_backend_result
from demo_api.trace_store import TraceStore


def test_default_race_fixture_contains_backends(tmp_path):
    payload = TraceStore(workspace_dir=tmp_path).load_default_race()

    assert [result["backend"] for result in payload["results"]] == [
        "official_pytorch_offline",
        "official_pytorch_streaming",
        "bare_engine_streaming",
        "triton_trt_streaming",
    ]
    offline = payload["results"][0]
    assert "server_ttft_ms" not in offline["metrics"]
    assert offline["metrics"]["first_playable_ms"] == offline["metrics"]["official_approx_ttft_ms"]
    assert offline["metrics"]["official_approx_ttft_ms"] < offline["metrics"]["full_audio_ready_ms"]


def test_official_audio_schedule_uses_approx_ttft():
    metrics = {
        "official_approx_ttft_ms": 101,
        "first_playable_ms": 1320,
        "full_audio_ready_ms": 1320,
    }

    assert scheduled_start_ms(metrics, "official_pytorch_streaming") == pytest.approx(101)
    assert scheduled_start_ms(metrics, "official_pytorch_offline") == pytest.approx(101)


def test_official_normalization_uses_approx_ttft():
    result = normalize_backend_result(
        {
            "run_id": "official",
            "backend": "official_pytorch_streaming",
            "label": "Official PyTorch Streaming",
            "mode": "official",
            "source": "live_official_pytorch",
            "metrics": {
                "official_approx_ttft_ms": 105,
                "first_playable_ms": 3000,
                "full_audio_ready_ms": 3000,
            },
            "events": [{"type": "official_approx_ttft", "t_ms": 105, "meta": {}}],
            "audio": {"url": "/audio.wav", "scheduled_start_ms": 3000},
        }
    )

    assert result["metrics"]["first_playable_ms"] == pytest.approx(105)
    assert result["audio"]["scheduled_start_ms"] == pytest.approx(105)
    assert any("TTFT is approximated" in warning for warning in result["warnings"])


def test_percentile_interpolates():
    assert percentile([10, 20, 30, 40], 0.5) == pytest.approx(25.0)
    assert percentile([10, 20, 30, 40], 0.9) == pytest.approx(37.0)


def test_summarize_ttft_reports_distribution():
    summary = summarize_ttft([100, 120, 180, 220, 300])

    assert summary["count"] == 5
    assert summary["avg_ttft_ms"] == pytest.approx(184.0)
    assert summary["p50_ttft_ms"] == pytest.approx(180.0)
    assert summary["p90_ttft_ms"] == pytest.approx(268.0)
    assert summary["max_ttft_ms"] == pytest.approx(300.0)


def test_simulated_concurrency_does_not_attach_fake_audio():
    async def run_job():
        manager = ConcurrencyJobManager(enable_live=False)
        job = await manager.create_job({"concurrency": 4, "live": False})
        queue = await manager.subscribe(job)
        audio_seen = False
        while True:
            message = await asyncio.wait_for(queue.get(), timeout=2)
            if message.get("type") == "lane_update" and message.get("audio"):
                audio_seen = True
            if message.get("type") == "summary":
                return audio_seen, message

    audio_seen, summary = asyncio.run(run_job())
    assert audio_seen is False
    assert summary["source"] == "simulated"
    assert summary["count"] == 4


def test_missing_raw_audio_is_not_replaced_with_synthetic_wav(tmp_path):
    result = RunResult(
        run_id="fixture-row",
        backend="triton_trt_streaming",
        label="Triton TRT Streaming",
        mode="test",
        source="fixture",
        metrics=RunMetrics(first_playable_ms=24, audio_duration_ms=1600),
    )

    attach_audio_to_result(result, AudioStore(tmp_path))

    assert result.audio == {}
    assert result.warnings == [
        "No raw audio captured; playback disabled instead of attaching synthetic audio."
    ]
    assert list(tmp_path.iterdir()) == []

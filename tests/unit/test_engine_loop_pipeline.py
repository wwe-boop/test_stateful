"""Tests for engine loop pipeline correctness and session timeout."""

import asyncio
import queue
import time
import threading

import torch
import pytest

from engine.backend.kv_cache_pool import KVCachePool, ModelConfig, SlotKVState
from engine.backend.executor import StepOutput
from engine.backend.engine_loop import (
    EngineLoop,
    EngineSegment,
    EngineSessionGroup,
)
from engine.core.types import (
    EngineRequest,
    EngineResult,
    RequestType,
    ResultType,
    RequestPriority,
)


@pytest.fixture
def model_config():
    return ModelConfig(
        num_layers=2, kv_heads=2, head_dim=4, max_seq_len=16,
        n_c2w_layers=2, c2w_kv_heads=2, c2w_head_dim=4, c2w_sliding_window=8,
    )


class TestEngineSessionGroup:
    def test_created_at_is_set(self):
        req = EngineRequest(type=RequestType.NEW_SESSION, session_id="s1")
        group = EngineSessionGroup("s1", req)
        assert group.created_at > 0
        assert time.monotonic() - group.created_at < 1.0

    def test_active_slot_count(self):
        req = EngineRequest(type=RequestType.NEW_SESSION, session_id="s1")
        group = EngineSessionGroup("s1", req)
        seg = EngineSegment("s1", 0)
        seg.slot = SlotKVState(slot_id=0)
        seg.state = "active"
        group.segments[0] = seg
        assert group.active_slot_count == 1


class TestEngineLoopHealth:
    def test_health_stats_initial(self, model_config):
        inbox = queue.Queue()
        loop = asyncio.new_event_loop()

        class StubExecutor:
            kv_pool = KVCachePool(
                max_slots=4, config=model_config,
                device=torch.device("cpu"), preallocate=False,
            )

        engine_loop = EngineLoop(
            engine_inbox=inbox,
            async_loop=loop,
            executor=StubExecutor(),
            max_batch_size=4,
        )

        stats = engine_loop.health_stats()
        assert stats["running"] is False
        assert stats["active_sessions"] == 0
        assert stats["total_steps"] == 0
        assert stats["total_prefills"] == 0
        assert stats["total_evictions"] == 0
        assert stats["total_timeouts"] == 0
        assert stats["free_slots"] == 4
        loop.close()


class TestSessionTimeout:
    def test_timeout_detection(self, model_config):
        inbox = queue.Queue()
        loop = asyncio.new_event_loop()

        class StubExecutor:
            kv_pool = KVCachePool(
                max_slots=4, config=model_config,
                device=torch.device("cpu"), preallocate=False,
            )

        engine_loop = EngineLoop(
            engine_inbox=inbox,
            async_loop=loop,
            executor=StubExecutor(),
            max_batch_size=4,
            session_timeout_sec=0.05,
        )

        result_queue = asyncio.Queue()
        req = EngineRequest(
            type=RequestType.NEW_SESSION,
            session_id="s1",
            result_queue=result_queue,
        )
        engine_loop._handle_request(req)
        assert "s1" in engine_loop._groups

        engine_loop._groups["s1"].created_at = time.monotonic() - 1.0

        engine_loop._try_timeout_sessions()
        assert "s1" not in engine_loop._groups
        assert engine_loop._total_timeouts == 1
        loop.close()

    def test_no_timeout_within_limit(self, model_config):
        inbox = queue.Queue()
        loop = asyncio.new_event_loop()

        class StubExecutor:
            kv_pool = KVCachePool(
                max_slots=4, config=model_config,
                device=torch.device("cpu"), preallocate=False,
            )

        engine_loop = EngineLoop(
            engine_inbox=inbox,
            async_loop=loop,
            executor=StubExecutor(),
            max_batch_size=4,
            session_timeout_sec=300.0,
        )

        req = EngineRequest(
            type=RequestType.NEW_SESSION,
            session_id="s1",
        )
        engine_loop._handle_request(req)
        engine_loop._try_timeout_sessions()
        assert "s1" in engine_loop._groups
        assert engine_loop._total_timeouts == 0
        loop.close()


class TestProcessStepOutput:
    def test_process_updates_slot_state(self, model_config):
        inbox = queue.Queue()
        loop = asyncio.new_event_loop()

        pool = KVCachePool(
            max_slots=4, config=model_config,
            device=torch.device("cpu"), preallocate=False,
        )

        class StubExecutor:
            kv_pool = pool

        engine_loop = EngineLoop(
            engine_inbox=inbox,
            async_loop=loop,
            executor=StubExecutor(),
            max_batch_size=4,
        )

        slot = pool.allocate("s1")
        slot.past_len = 5
        slot.frame_idx = 3

        req = EngineRequest(type=RequestType.NEW_SESSION, session_id="s1")
        engine_loop._handle_request(req)

        seg = EngineSegment("s1", 0)
        seg.slot = slot
        seg.state = "active"
        engine_loop._groups["s1"].segments[0] = seg
        engine_loop._seg_by_slot[slot.slot_id] = seg

        output = StepOutput(
            slots=[slot],
            eos_flags=[False],
            audio_chunks=[b"\x00\x01"],
            batch_talker_kv=None,
            batch_c2w_kv=None,
            split_c2w_conv=[[]],
            split_c2w_transconv=[[]],
            codec_sum=torch.randn(1, 1, model_config.hidden_size),
            updated_tc=torch.zeros(1, model_config.codec_vocab_size, dtype=torch.int64),
        )

        engine_loop._process_step_output(output)
        assert slot.past_len == 6
        assert slot.frame_idx == 4
        assert slot.next_embed is not None
        loop.close()


class TestConfigNewFields:
    def test_server_config_has_warmup_and_health(self):
        from engine.config import ServerConfig
        sc = ServerConfig()
        assert sc.warmup_rounds == 3
        assert sc.health_port == 8080

    def test_scheduler_config_has_timeout(self):
        from engine.config import SchedulerConfig
        sc = SchedulerConfig()
        assert sc.session_timeout_sec == 300.0

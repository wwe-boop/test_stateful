"""Tests for engine loop pipeline correctness and session timeout."""

import asyncio
import queue
import time
import threading
from types import SimpleNamespace

import torch
import pytest

from engine.backend.kv_cache_pool import KVCachePool, ModelConfig, SlotKVState
from engine.backend.prefill import PrefillPlan, TaskType
from engine.backend.executor import Executor, StepOutput
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
            _device = torch.device("cpu")
            _config = model_config

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
            _device = torch.device("cpu")
            _config = model_config

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
            _device = torch.device("cpu")
            _config = model_config

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


class _ImmediateLoop:
    def call_soon_threadsafe(self, callback, *args):
        callback(*args)


class TestSessionCancel:
    def test_cancel_emits_session_done_and_removes_group(self, model_config):
        inbox = queue.Queue()
        loop = _ImmediateLoop()

        class StubExecutor:
            kv_pool = KVCachePool(
                max_slots=4, config=model_config,
                device=torch.device("cpu"), preallocate=False,
            )
            _device = torch.device("cpu")
            _config = model_config

        engine_loop = EngineLoop(
            engine_inbox=inbox,
            async_loop=loop,
            executor=StubExecutor(),
            max_batch_size=4,
        )

        result_queue = queue.Queue()
        new_req = EngineRequest(
            type=RequestType.NEW_SESSION,
            session_id="cancel-me",
            result_queue=result_queue,
        )
        engine_loop._handle_request(new_req)
        assert "cancel-me" in engine_loop._groups

        cancel_req = EngineRequest(
            type=RequestType.CANCEL_SESSION,
            session_id="cancel-me",
        )
        engine_loop._handle_request(cancel_req)

        result = result_queue.get_nowait()
        assert isinstance(result, EngineResult)
        assert result.type == ResultType.SESSION_DONE
        assert result.session_id == "cancel-me"
        assert result.metrics == {"cancelled": True}
        assert "cancel-me" not in engine_loop._groups

    def test_new_session_replacement_releases_existing_slot(self, model_config):
        inbox = queue.Queue()
        loop = _ImmediateLoop()

        class StubExecutor:
            kv_pool = KVCachePool(
                max_slots=4, config=model_config,
                device=torch.device("cpu"), preallocate=False,
            )
            _device = torch.device("cpu")
            _config = model_config

        engine_loop = EngineLoop(
            engine_inbox=inbox,
            async_loop=loop,
            executor=StubExecutor(),
            max_batch_size=4,
        )

        engine_loop._handle_request(
            EngineRequest(type=RequestType.NEW_SESSION, session_id="s1")
        )
        slot = StubExecutor.kv_pool.allocate("s1:0")
        seg = EngineSegment("s1", 0)
        seg.slot = slot
        seg.state = "active"
        engine_loop._groups["s1"].segments[0] = seg
        engine_loop._seg_by_slot[slot.slot_id] = seg

        engine_loop._handle_request(
            EngineRequest(type=RequestType.NEW_SESSION, session_id="s1")
        )

        assert slot.is_free is True
        assert slot.slot_id not in engine_loop._seg_by_slot
        assert engine_loop._groups["s1"].segments == {}
        assert StubExecutor.kv_pool.free_count == 4

    def test_duplicate_start_tokens_releases_replaced_segment_slot(self, model_config):
        inbox = queue.Queue()
        loop = _ImmediateLoop()

        class StubExecutor:
            kv_pool = KVCachePool(
                max_slots=4, config=model_config,
                device=torch.device("cpu"), preallocate=False,
            )
            _device = torch.device("cpu")
            _config = model_config

        engine_loop = EngineLoop(
            engine_inbox=inbox,
            async_loop=loop,
            executor=StubExecutor(),
            max_batch_size=4,
        )

        engine_loop._handle_request(
            EngineRequest(type=RequestType.NEW_SESSION, session_id="s1")
        )
        engine_loop._handle_request(
            EngineRequest(
                type=RequestType.START_TOKENS,
                session_id="s1",
                segment_idx=0,
                token_ids=[1],
            )
        )
        old_seg = engine_loop._groups["s1"].segments[0]
        slot = StubExecutor.kv_pool.allocate("s1:0")
        old_seg.slot = slot
        old_seg.state = "active"
        engine_loop._seg_by_slot[slot.slot_id] = old_seg

        engine_loop._handle_request(
            EngineRequest(
                type=RequestType.START_TOKENS,
                session_id="s1",
                segment_idx=0,
                token_ids=[2],
            )
        )

        new_seg = engine_loop._groups["s1"].segments[0]
        assert slot.is_free is True
        assert old_seg.slot is None
        assert new_seg is not old_seg
        assert new_seg.pending_token_ids == [2]
        assert new_seg.slot is None
        assert slot.slot_id not in engine_loop._seg_by_slot
        assert StubExecutor.kv_pool.free_count == 4

    def test_failed_prefill_cleanup_releases_slot_and_removes_session(self, model_config):
        inbox = queue.Queue()
        loop = _ImmediateLoop()

        class StubExecutor:
            kv_pool = KVCachePool(
                max_slots=4, config=model_config,
                device=torch.device("cpu"), preallocate=False,
            )
            _device = torch.device("cpu")
            _config = model_config

        engine_loop = EngineLoop(
            engine_inbox=inbox,
            async_loop=loop,
            executor=StubExecutor(),
            max_batch_size=4,
        )

        result_queue = queue.Queue()
        engine_loop._handle_request(
            EngineRequest(
                type=RequestType.NEW_SESSION,
                session_id="s1",
                result_queue=result_queue,
            )
        )
        seg = EngineSegment("s1", 0)
        seg.state = "pending_prefill"
        seg.slot = StubExecutor.kv_pool.allocate("s1:0")
        engine_loop._groups["s1"].segments[0] = seg
        engine_loop._seg_by_slot[seg.slot.slot_id] = seg

        engine_loop._cleanup_failed_prefills()

        result = result_queue.get_nowait()
        assert result.type == ResultType.ERROR
        assert result.session_id == "s1"
        assert "s1" not in engine_loop._groups
        assert StubExecutor.kv_pool.free_count == 4


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
            _device = torch.device("cpu")
            _config = model_config

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
        assert slot.next_embed is None
        assert slot.last_codec_sum is not None
        loop.close()


class TestConfigNewFields:
    def test_server_config_has_warmup_and_health(self):
        from engine.config import ServerConfig
        sc = ServerConfig()
        assert sc.warmup_rounds == 3
        assert sc.health_port == 8080

    def test_scheduler_config_has_timeout_and_pad_silence_thresholds(self):
        from engine.config import SchedulerConfig
        sc = SchedulerConfig()
        assert sc.session_timeout_sec == 300.0
        assert sc.pad_silence_peak_threshold == 5e-4
        assert sc.pad_silence_mean_abs_threshold == 2e-4


class TestPadSilenceDetection:
    def test_pad_silence_detection_uses_peak_and_mean_abs(self, model_config):
        inbox = queue.Queue()
        loop = asyncio.new_event_loop()

        class StubExecutor:
            kv_pool = KVCachePool(
                max_slots=4, config=model_config,
                device=torch.device("cpu"), preallocate=False,
            )
            _device = torch.device("cpu")
            _config = model_config

        engine_loop = EngineLoop(
            engine_inbox=inbox,
            async_loop=loop,
            executor=StubExecutor(),
            max_batch_size=4,
        )

        near_silence = torch.full((1920,), 1.5e-4, dtype=torch.float32)
        near_silence[0] = 4.5e-4
        assert engine_loop._is_pad_silence(near_silence.numpy().tobytes()) is True

        audible = torch.full((1920,), 1.5e-4, dtype=torch.float32)
        audible[0] = 8e-4
        assert engine_loop._is_pad_silence(audible.numpy().tobytes()) is False
        loop.close()


class _StubPrefillBuilder:
    def __init__(
        self,
        *,
        plan: PrefillPlan | None = None,
        suffix: tuple[torch.Tensor, list[torch.Tensor]] | None = None,
        cache_key: str = "cache-key",
        hidden_size: int = 2048,
    ):
        self._plan = plan
        self._suffix = suffix
        self._cache_key = cache_key
        self.w = SimpleNamespace(
            tts_pad_embed=torch.zeros(1, 1, hidden_size, dtype=torch.bfloat16),
        )

    def compute_cache_key(self, *args, **kwargs):
        return self._cache_key

    def build_plan_from_ids(self, **kwargs):
        if self._plan is None:
            raise AssertionError("build_plan_from_ids should not be called")
        return self._plan

    def build_suffix_from_ids(self, token_ids, include_eos=True):
        if self._suffix is None:
            raise AssertionError("build_suffix_from_ids should not be called")
        return self._suffix


class _StubExecutorForPrefill:
    def __init__(self, model_config):
        self._config = model_config
        self._device = torch.device("cpu")
        self.kv_pool = KVCachePool(
            max_slots=4,
            config=model_config,
            device=torch.device("cpu"),
            preallocate=False,
        )
        self.prefill_inputs: list[torch.Tensor] = []
        self.prefill_prefix_only_inputs: list[torch.Tensor] = []
        self.prefill_from_prefix_inputs: list[torch.Tensor] = []

    def make_zero_conv_states(self):
        return [torch.zeros(1, 1, 1)]

    def make_zero_transconv_states(self):
        return [torch.zeros(1, 1, 1)]

    def prefill(self, slot, embeds):
        self.prefill_inputs.append(embeds.clone())
        seq = int(embeds.shape[1])
        slot.talker_kv = torch.zeros(
            1,
            self._config.num_layers * 2,
            self._config.kv_heads,
            seq,
            self._config.head_dim,
        )
        slot.past_len = seq
        slot.position_offset = 0
        slot.frame_idx = 1
        slot.next_embed = torch.full(
            (1, 1, self._config.hidden_size),
            10.0,
        )
        slot.c2w_conv_states = self.make_zero_conv_states()
        slot.c2w_transconv_states = self.make_zero_transconv_states()
        slot.init_pingpong_buffers()
        slot.token_counts = torch.zeros(
            1,
            self._config.codec_vocab_size,
            dtype=torch.int64,
        )
        return b"", False

    def prefill_prefix_only(self, slot, embeds):
        self.prefill_prefix_only_inputs.append(embeds.clone())
        seq = int(embeds.shape[1])
        slot.talker_kv = torch.zeros(
            1,
            self._config.num_layers * 2,
            self._config.kv_heads,
            seq,
            self._config.head_dim,
        )
        slot.past_len = seq
        slot.position_offset = 0

    def prefill_from_prefix(self, slot, embeds):
        self.prefill_from_prefix_inputs.append(embeds.clone())
        seq = int(embeds.shape[1])
        slot.talker_kv = torch.zeros(
            1,
            self._config.num_layers * 2,
            self._config.kv_heads,
            slot.past_len + seq,
            self._config.head_dim,
        )
        slot.past_len += seq
        slot.frame_idx = 1
        slot.next_embed = torch.full(
            (1, 1, self._config.hidden_size),
            7.0,
        )
        slot.token_counts = torch.ones(
            1,
            self._config.codec_vocab_size,
            dtype=torch.int64,
        )
        return b"audio", False


def _make_steadystream_group(session_id: str, experimental: dict[str, str]):
    return EngineSessionGroup(
        session_id,
        EngineRequest(
            type=RequestType.NEW_SESSION,
            session_id=session_id,
            session_config=SimpleNamespace(experimental=experimental),
        ),
    )


def _make_talker_kv(model_config, seq: int) -> torch.Tensor:
    return torch.ones(
        1,
        model_config.num_layers * 2,
        model_config.kv_heads,
        seq,
        model_config.head_dim,
    )


class TestSteadyStreamCarry:
    def test_token_counts_reset_by_default(self, model_config):
        executor = _StubExecutorForPrefill(model_config)
        engine_loop = EngineLoop(
            engine_inbox=queue.Queue(),
            async_loop=_ImmediateLoop(),
            executor=executor,
            prefill_builder=_StubPrefillBuilder(hidden_size=model_config.hidden_size),
        )
        group = _make_steadystream_group(
            "s1",
            {"steadystream_variant": "kv_tail_only", "kv_tail_tokens": "16"},
        )
        seg = EngineSegment("s1", 0)
        seg.slot = SlotKVState(slot_id=0)
        seg.slot.talker_kv = _make_talker_kv(model_config, seq=4)
        seg.slot.past_len = 4
        seg.slot.token_counts = torch.ones(
            1,
            model_config.codec_vocab_size,
            dtype=torch.int64,
        )

        engine_loop._store_steadystream_carry(group, seg)

        assert "talker_kv" in group.steadystream_carry
        assert "token_counts" not in group.steadystream_carry

    def test_token_counts_can_be_inherited_explicitly(self, model_config):
        executor = _StubExecutorForPrefill(model_config)
        engine_loop = EngineLoop(
            engine_inbox=queue.Queue(),
            async_loop=_ImmediateLoop(),
            executor=executor,
            prefill_builder=_StubPrefillBuilder(hidden_size=model_config.hidden_size),
        )
        group = _make_steadystream_group(
            "s1",
            {
                "steadystream_variant": "kv_tail_only",
                "kv_tail_tokens": "16",
                "kv_inherit_token_counts": "true",
            },
        )
        seg = EngineSegment("s1", 0)
        seg.slot = SlotKVState(slot_id=0)
        seg.slot.talker_kv = _make_talker_kv(model_config, seq=4)
        seg.slot.past_len = 4
        seg.slot.token_counts = torch.ones(
            1,
            model_config.codec_vocab_size,
            dtype=torch.int64,
        )

        engine_loop._store_steadystream_carry(group, seg)

        assert "talker_kv" in group.steadystream_carry
        torch.testing.assert_close(
            group.steadystream_carry["token_counts"],
            seg.slot.token_counts,
        )

    def test_kv_tail_restore_preserves_logical_position_offset(self, model_config):
        executor = _StubExecutorForPrefill(model_config)
        engine_loop = EngineLoop(
            engine_inbox=queue.Queue(),
            async_loop=_ImmediateLoop(),
            executor=executor,
            prefill_builder=_StubPrefillBuilder(hidden_size=model_config.hidden_size),
        )
        group = _make_steadystream_group(
            "s1",
            {"steadystream_variant": "kv_tail_only", "kv_tail_tokens": "16"},
        )
        seg0 = EngineSegment("s1", 0)
        seg0.slot = SlotKVState(slot_id=0)
        seg0.slot.talker_kv = _make_talker_kv(model_config, seq=20)
        seg0.slot.past_len = 20
        seg0.slot.position_offset = 7

        engine_loop._store_steadystream_carry(group, seg0)

        assert group.steadystream_carry["talker_past_len"] == 16
        assert group.steadystream_carry["talker_logical_past_len"] == 27
        assert group.steadystream_carry["talker_position_offset"] == 11

        slot = SlotKVState(slot_id=1)
        engine_loop._restore_talker_kv_tail(
            slot,
            group.steadystream_carry["talker_kv"],
            group.steadystream_carry["talker_past_len"],
            group.steadystream_carry["talker_logical_past_len"],
        )

        assert slot.past_len == 16
        assert slot.position_offset == 11

    def test_kv_tail_snapshot_can_drop_terminal_token(self, model_config):
        executor = _StubExecutorForPrefill(model_config)
        engine_loop = EngineLoop(
            engine_inbox=queue.Queue(),
            async_loop=_ImmediateLoop(),
            executor=executor,
            prefill_builder=_StubPrefillBuilder(hidden_size=model_config.hidden_size),
        )
        group = _make_steadystream_group(
            "s1",
            {"steadystream_variant": "kv_tail_only", "kv_tail_tokens": "16"},
        )
        seg0 = EngineSegment("s1", 0)
        seg0.slot = SlotKVState(slot_id=0)
        kv = torch.arange(20, dtype=torch.float32).view(1, 1, 1, 20, 1)
        seg0.slot.talker_kv = kv.expand(
            1,
            model_config.num_layers * 2,
            model_config.kv_heads,
            20,
            model_config.head_dim,
        ).clone()
        seg0.slot.past_len = 20
        seg0.slot.position_offset = 7

        engine_loop._store_steadystream_carry(
            group,
            seg0,
            drop_talker_tail_tokens=1,
        )

        carry = group.steadystream_carry
        assert carry["talker_past_len"] == 16
        assert carry["talker_logical_past_len"] == 26
        assert carry["talker_position_offset"] == 10
        assert carry["talker_dropped_last_token"] is True
        assert carry["talker_dropped_tail_tokens"] == 1
        assert int(carry["talker_kv"][0, 0, 0, -1, 0].item()) == 18

    def test_terminal_drop_tokens_modes(self):
        slot = SlotKVState(slot_id=0)
        slot.frame_idx = 35
        slot.pad_start_frame = 30
        slot.pad_consecutive_silence = 2

        assert EngineLoop._steadystream_terminal_drop_tokens(slot) == 6
        assert (
            EngineLoop._steadystream_terminal_drop_tokens(slot, mode="pad_phase")
            == 6
        )
        assert (
            EngineLoop._steadystream_terminal_drop_tokens(slot, mode="eos_only")
            == 1
        )
        assert (
            EngineLoop._steadystream_terminal_drop_tokens(slot, mode="silence")
            == 3
        )
        assert (
            EngineLoop._steadystream_terminal_drop_tokens(
                slot,
                mode="silence",
                max_tokens=2,
            )
            == 2
        )

        slot.pad_start_frame = -1
        assert EngineLoop._steadystream_terminal_drop_tokens(slot) == 1

    def test_terminal_drop_mode_config(self):
        group = _make_steadystream_group(
            "s1",
            {
                "steadystream_variant": "kv_tail_only",
                "kv_terminal_drop_mode": "tail-silence",
                "kv_terminal_drop_max_tokens": "12",
            },
        )

        assert EngineLoop._steadystream_terminal_drop_mode(group) == "silence"
        assert EngineLoop._steadystream_terminal_drop_max_tokens(group) == 12

    def test_restore_reports_reset_or_inherited_token_counts(self, model_config):
        hidden = model_config.hidden_size
        req_embeds = torch.randn(1, 1, hidden)
        trailing = [torch.full((1, 1, hidden), 2.0)]
        executor = _StubExecutorForPrefill(model_config)
        engine_loop = EngineLoop(
            engine_inbox=queue.Queue(),
            async_loop=_ImmediateLoop(),
            executor=executor,
            prefill_builder=_StubPrefillBuilder(
                suffix=(req_embeds, trailing),
                hidden_size=hidden,
            ),
        )
        seg = EngineSegment("s1", 1)
        seg.pending_token_ids = [1, 2]
        seg.input_complete = True

        group = _make_steadystream_group(
            "s1",
            {"steadystream_variant": "kv_tail_only", "kv_prepend_prefix": "true"},
        )
        group.steadystream_carry = {
            "talker_kv": _make_talker_kv(model_config, seq=4),
            "talker_past_len": 4,
        }
        slot = SlotKVState(slot_id=0)
        metrics: dict[str, str] = {}

        assert engine_loop._try_prefill_from_steadystream_carry(
            group,
            seg,
            slot,
            TaskType.CUSTOM_VOICE,
            metrics,
        ) == (None, False)
        assert metrics["steadystream_token_counts"] == "reset"
        assert torch.count_nonzero(slot.token_counts) == 0
        assert metrics["steadystream_kv_position_offset"] == "0"
        assert metrics["steadystream_kv_dropped_last_token"] == "false"
        assert metrics["steadystream_kv_dropped_tail_tokens"] == "0"

        inherited_counts = torch.ones(
            1,
            model_config.codec_vocab_size,
            dtype=torch.int64,
        )
        group.steadystream_carry["token_counts"] = inherited_counts
        slot = SlotKVState(slot_id=0)
        metrics = {}

        assert engine_loop._try_prefill_from_steadystream_carry(
            group,
            seg,
            slot,
            TaskType.CUSTOM_VOICE,
            metrics,
        ) == (None, False)
        assert metrics["steadystream_token_counts"] == "inherited"
        torch.testing.assert_close(slot.token_counts, inherited_counts)

    def test_reprefill_history_recomputes_prefix_and_replay(self, model_config):
        hidden = model_config.hidden_size
        prefix = torch.full((1, 2, hidden), 1.0)
        replay = torch.full((1, 3, hidden), 2.0)
        request = torch.full((1, 1, hidden), 3.0)
        trailing = [torch.full((1, 1, hidden), 4.0)]
        plan = PrefillPlan(
            prefill_embeds=torch.cat([prefix, request], dim=1),
            trailing=trailing,
            cacheable_prefix_embeds=prefix,
            request_prefill_embeds=request,
        )
        executor = _StubExecutorForPrefill(model_config)
        engine_loop = EngineLoop(
            engine_inbox=queue.Queue(),
            async_loop=_ImmediateLoop(),
            executor=executor,
            prefill_builder=_StubPrefillBuilder(
                plan=plan,
                hidden_size=hidden,
            ),
        )
        seg = EngineSegment("s1", 1)
        seg.pending_token_ids = [1, 2]
        seg.input_complete = True
        group = _make_steadystream_group(
            "s1",
            {
                "steadystream_variant": "kv_tail_only",
                "kv_reprefill_history": "true",
                "kv_tail_tokens": "16",
            },
        )
        group.steadystream_carry = {
            "from_segment_idx": 0,
            "replay_embeds": replay,
            "replay_len": 3,
            "talker_kv": _make_talker_kv(model_config, seq=4),
            "talker_past_len": 4,
        }
        slot = SlotKVState(slot_id=0)
        metrics: dict[str, str] = {}

        assert engine_loop._try_prefill_from_steadystream_carry(
            group,
            seg,
            slot,
            TaskType.CUSTOM_VOICE,
            metrics,
        ) == (None, False)

        expected_prefill = torch.cat([prefix, replay], dim=1).to(model_config.dtype)
        assert len(executor.prefill_prefix_only_inputs) == 1
        torch.testing.assert_close(
            executor.prefill_prefix_only_inputs[0],
            expected_prefill,
        )
        assert executor.prefill_from_prefix_inputs == []
        assert metrics["steadystream_kv_tail"] == "reprefill_history"
        assert metrics["steadystream_kv_prefix"] == "recomputed"
        assert metrics["steadystream_replay_len"] == "3"
        assert metrics["steadystream_token_counts"] == "reset"
        assert slot.prefill_source == "steadystream_reprefill_kv_tail_only"
        torch.testing.assert_close(slot.next_embed, request.to(torch.float32))
        assert slot.steadystream_replay_embeds == []

    def test_restore_can_prepend_cached_prefix_before_kv_tail(self, model_config):
        hidden = model_config.hidden_size
        req_embeds = torch.randn(1, 1, hidden)
        trailing = [torch.full((1, 1, hidden), 2.0)]
        executor = _StubExecutorForPrefill(model_config)
        engine_loop = EngineLoop(
            engine_inbox=queue.Queue(),
            async_loop=_ImmediateLoop(),
            executor=executor,
            prefill_builder=_StubPrefillBuilder(
                suffix=(req_embeds, trailing),
                hidden_size=hidden,
            ),
        )
        seg = EngineSegment("s1", 1)
        seg.pending_token_ids = [1, 2]
        seg.input_complete = True
        group = _make_steadystream_group(
            "s1",
            {"steadystream_variant": "kv_tail_only", "kv_prepend_prefix": "true"},
        )
        group.steadystream_carry = {
            "talker_kv": _make_talker_kv(model_config, seq=4),
            "talker_past_len": 4,
        }
        cached_prefix = SimpleNamespace(
            prefix_len=2,
            talker_kv=torch.zeros(
                1,
                model_config.num_layers * 2,
                model_config.kv_heads,
                2,
                model_config.head_dim,
            ),
        )
        slot = SlotKVState(slot_id=0)
        metrics: dict[str, str] = {}

        assert engine_loop._try_prefill_from_steadystream_carry(
            group,
            seg,
            slot,
            TaskType.CUSTOM_VOICE,
            metrics,
            cached_prefix,
        ) == (None, False)

        assert slot.past_len == 6
        assert slot.position_offset == 0
        assert metrics["steadystream_kv_prefix"] == "cached"
        assert metrics["steadystream_kv_prefix_len"] == "2"
        assert metrics["steadystream_kv_tail_effective_len"] == "4"


class TestExecutorPositionIds:
    def test_position_ids_include_compacted_cache_offset(self, model_config):
        executor = Executor.__new__(Executor)
        executor._config = model_config
        executor._device = torch.device("cpu")
        executor._do_sample = False
        executor._temperature = 1.0
        executor._repetition_penalty = 1.0
        executor._c2w_conv_input_names = []
        executor._c2w_transconv_input_names = []

        slot = SlotKVState(slot_id=0)
        slot.past_len = 4
        slot.position_offset = 13
        slot.c2w_kv = None
        slot.c2w_conv_states = []
        slot.c2w_transconv_states = []
        slot.token_counts = torch.zeros(
            1,
            model_config.codec_vocab_size,
            dtype=torch.int64,
        )

        inputs = executor._build_fused_inputs(
            input_embeds=torch.zeros(1, 2, model_config.hidden_size),
            slots=[slot],
            batched_talker_kv=torch.zeros(
                1,
                model_config.num_layers * 2,
                model_config.kv_heads,
                slot.past_len,
                model_config.head_dim,
            ),
            past_seq_lens=torch.tensor([slot.past_len], dtype=torch.long),
            use_dummy_kv=False,
            sampling_mode="disabled",
        )

        assert inputs["position_ids"].shape == (1, 3, 2, 1)
        assert inputs["position_ids"][0, 0, :, 0].tolist() == [17, 18]


class TestPrefillBoundary:
    def test_full_prefill_can_prime_decode0_from_prefix_only_state(self, model_config):
        loop = _ImmediateLoop()
        executor = _StubExecutorForPrefill(model_config)
        hidden = model_config.hidden_size

        prefill = torch.randn(1, 3, hidden)
        trailing = [torch.full((1, 1, hidden), 2.0)]
        plan = PrefillPlan(
            prefill_embeds=prefill,
            trailing=trailing,
            prefix_cache_key="cache-key",
            cacheable_prefix_embeds=prefill[:, :2, :].clone(),
            request_prefill_embeds=prefill[:, 2:, :].clone(),
        )
        builder = _StubPrefillBuilder(plan=plan, hidden_size=hidden)

        engine_loop = EngineLoop(
            engine_inbox=queue.Queue(),
            async_loop=loop,
            executor=executor,
            prefill_builder=builder,
            max_batch_size=4,
        )

        result_queue = queue.Queue()
        engine_loop._handle_request(
            EngineRequest(
                type=RequestType.NEW_SESSION,
                session_id="s1",
                task_type="custom_voice",
                result_queue=result_queue,
            )
        )
        engine_loop._handle_request(
            EngineRequest(
                type=RequestType.START_TOKENS,
                session_id="s1",
                segment_idx=0,
                token_ids=[1, 2, 3],
            )
        )

        assert engine_loop._try_prefill_one() is True

        seg = engine_loop._groups["s1"].segments[0]
        slot = seg.slot
        assert slot is not None
        assert executor.prefill_inputs == []
        assert len(executor.prefill_prefix_only_inputs) == 1
        torch.testing.assert_close(
            executor.prefill_prefix_only_inputs[0],
            prefill[:, :2, :],
        )
        assert executor.prefill_from_prefix_inputs == []
        assert slot.prefill_source == "full_prefill_prefix_only"
        assert slot.past_len == 2
        assert slot.frame_idx == 0
        assert slot.text_idx == 0
        torch.testing.assert_close(
            slot.next_embed,
            prefill[:, 2:, :].to(torch.float32),
        )

    def test_prefix_cache_hit_restores_prefix_and_lets_decode0_consume_text(self, model_config):
        loop = _ImmediateLoop()
        executor = _StubExecutorForPrefill(model_config)
        hidden = model_config.hidden_size

        req_embeds = torch.randn(1, 1, hidden)
        trailing = [torch.full((1, 1, hidden), 2.0)]
        builder = _StubPrefillBuilder(
            suffix=(req_embeds, trailing),
            cache_key="cache-key",
            hidden_size=hidden,
        )

        engine_loop = EngineLoop(
            engine_inbox=queue.Queue(),
            async_loop=loop,
            executor=executor,
            prefill_builder=builder,
            max_batch_size=4,
        )
        cached_kv = torch.zeros(
            1,
            model_config.num_layers * 2,
            model_config.kv_heads,
            2,
            model_config.head_dim,
        )
        engine_loop._prefix_cache.put("cache-key", cached_kv, 2)

        result_queue = queue.Queue()
        engine_loop._handle_request(
            EngineRequest(
                type=RequestType.NEW_SESSION,
                session_id="s1",
                task_type="custom_voice",
                result_queue=result_queue,
            )
        )
        engine_loop._handle_request(
            EngineRequest(
                type=RequestType.START_TOKENS,
                session_id="s1",
                segment_idx=0,
                token_ids=[1, 2],
            )
        )

        assert engine_loop._try_prefill_one() is True

        seg = engine_loop._groups["s1"].segments[0]
        slot = seg.slot
        assert slot is not None
        assert executor.prefill_inputs == []
        assert executor.prefill_prefix_only_inputs == []
        assert executor.prefill_from_prefix_inputs == []
        assert slot.prefill_source == "prefix_cache_prefix_only"
        assert slot.past_len == 2
        assert slot.frame_idx == 0
        assert slot.text_idx == 0
        torch.testing.assert_close(
            slot.next_embed,
            req_embeds.to(torch.float32),
        )

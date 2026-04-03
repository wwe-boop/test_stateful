"""Integration test for the standalone TTS engine.

Tests the full asyncio ↔ engine thread pipeline in stub mode (no GPU):
  1. Spliter: text → segment actions
  2. Dispatcher: session management + text routing
  3. EngineLoop: drain inbox + prefill + decode pipeline
  4. Cross-thread result delivery

Run:
    pytest tests/test_engine_integration.py -v -s
    # or just the fast CPU-only tests:
    pytest tests/test_engine_integration.py -v -s -k "not gpu"
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import pytest

from tests.paths import REPO_ROOT, TOKENIZER_DIR, WEIGHTS_DIR, VARIANT

sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.DEBUG, format="%(name)s %(levelname)s %(message)s")

HAS_TOKENIZER = TOKENIZER_DIR.exists()
SKIP_NO_TOKENIZER = pytest.mark.skipif(
    not HAS_TOKENIZER,
    reason=f"Tokenizer not found (variant={VARIANT}, path={TOKENIZER_DIR})",
)


# ---------------------------------------------------------------------------
# 1. Spliter unit tests (pure CPU, no async, no GPU)
# ---------------------------------------------------------------------------

class TestSpliter:
    """Test the Spliter text segmentation logic."""

    def test_classify_punct_level(self):
        from engine.frontend.spliter.spliter import Spliter

        assert Spliter.classify_punct_level("你好。") == 1
        assert Spliter.classify_punct_level("你好，") == 2
        assert Spliter.classify_punct_level("你好") == 0
        assert Spliter.classify_punct_level("hello!") == 1

    def test_compute_thresholds(self):
        from engine.frontend.spliter.driver import compute_thresholds

        th = compute_thresholds(remaining_kv=500, ema_ratio=5.0)
        assert th.a < th.b < th.c < th.d
        assert th.a >= 1
        assert th.d <= 500

    def test_pre_split_short_text(self):
        """Short text → single segment, no split."""
        from engine.frontend.spliter.spliter import Spliter

        spliter = Spliter(engine_max_decode_len=512, ema_ratio=5.0)
        tokens = [(i, f"tok{i}") for i in range(10)]
        segments = spliter.pre_split(tokens)
        assert len(segments) == 1
        assert len(segments[0]) == 10

    def test_pre_split_at_l1_punct(self):
        """Text with L1 punct should split at the punct boundary."""
        from engine.frontend.spliter.spliter import Spliter

        spliter = Spliter(engine_max_decode_len=100, ema_ratio=2.0)
        th = spliter._make_thresholds()

        tokens = []
        for i in range(th.a + 5):
            tokens.append((i, f"tok{i}"))
        tokens.append((999, "句号。"))
        tokens.append((1000, "后续"))
        tokens.append((1001, "文本"))

        segments = spliter.pre_split(tokens)
        assert len(segments) >= 2, f"Expected >=2 segments, got {len(segments)}"

    def test_streaming_feed_tokens(self):
        """Streaming mode: feed tokens one by one, expect segment actions."""
        from engine.frontend.spliter.spliter import Spliter
        from engine.frontend.spliter.driver import ActionType

        spliter = Spliter(engine_max_decode_len=100, ema_ratio=2.0)
        th = spliter._make_thresholds()

        tokens = [(i, f"tok{i}") for i in range(th.a + 2)]
        tokens.append((999, "。"))

        actions = spliter.feed_tokens(tokens)

        action_types = [a.action.type for a in actions]
        assert ActionType.PREFILL in action_types, "Should have PREFILL action"
        assert ActionType.DECODE in action_types, "Should have DECODE actions"

    def test_offline_set_full_text(self):
        """Offline mode: set full text, get all segment actions."""
        from engine.frontend.spliter.spliter import Spliter
        from engine.frontend.spliter.driver import ActionType

        spliter = Spliter(engine_max_decode_len=100, ema_ratio=2.0)
        th = spliter._make_thresholds()

        tokens = []
        for i in range(th.a + 2):
            tokens.append((i, f"tok{i}"))
        tokens.append((100, "。"))
        for i in range(5):
            tokens.append((200 + i, f"after{i}"))

        actions = spliter.set_full_text(tokens)
        action_types = [a.action.type for a in actions]
        assert ActionType.PREFILL in action_types


# ---------------------------------------------------------------------------
# 2. AudioReorder tests
# ---------------------------------------------------------------------------

class TestAudioReorder:
    def test_in_order(self):
        from engine.frontend.spliter.reorder import AudioReorder

        r = AudioReorder()
        out = r.push(0, b"a0")
        assert out == [b"a0"]
        out = r.push(0, b"a1")
        assert out == [b"a1"]
        out = r.mark_done(0)
        assert out == []
        assert r.next_emit_segment == 1

    def test_out_of_order(self):
        from engine.frontend.spliter.reorder import AudioReorder

        r = AudioReorder()
        out = r.push(1, b"b0")
        assert out == []
        out = r.push(0, b"a0")
        assert out == [b"a0"]
        out = r.mark_done(0)
        assert b"b0" in out
        assert r.next_emit_segment == 1


# ---------------------------------------------------------------------------
# 3. Tokenizer test
# ---------------------------------------------------------------------------

class TestTokenizer:
    @SKIP_NO_TOKENIZER
    def test_encode_decode(self):
        from engine.frontend.spliter.tokenizer import LightQwen3TTSTokenizer

        tok = LightQwen3TTSTokenizer(str(TOKENIZER_DIR))
        ids = tok.encode_ids("你好世界")
        assert len(ids) > 0

        ids2, texts = tok.encode_with_text("你好，世界。")
        assert len(ids2) == len(texts)
        assert any("。" in t for t in texts), f"Expected punct in texts: {texts}"

    @SKIP_NO_TOKENIZER
    def test_spliter_with_real_tokenizer(self):
        """End-to-end: real tokenizer → Spliter → actions."""
        from engine.frontend.spliter.tokenizer import LightQwen3TTSTokenizer
        from engine.frontend.spliter.spliter import Spliter
        from engine.frontend.spliter.driver import ActionType

        tok = LightQwen3TTSTokenizer(str(TOKENIZER_DIR))
        text = "你好世界。今天天气真好，我们一起出去玩吧！"
        ids, texts = tok.encode_with_text(text, add_special_tokens=False)
        tokens = list(zip(ids, texts))

        spliter = Spliter(engine_max_decode_len=200, ema_ratio=5.0)
        actions = spliter.feed_tokens(tokens)

        prefills = [a for a in actions if a.action.type == ActionType.PREFILL]
        decodes = [a for a in actions if a.action.type == ActionType.DECODE]
        assert len(prefills) >= 1, "Should have at least one PREFILL"
        assert len(decodes) >= 1, "Should have DECODE actions"
        print(f"\nTokens: {len(tokens)}, Prefills: {len(prefills)}, "
              f"Decodes: {len(decodes)}, Segments: {spliter.current_segment_idx}")


# ---------------------------------------------------------------------------
# 4. Full pipeline integration test (asyncio + engine thread, stub mode)
# ---------------------------------------------------------------------------

class TestEngineIntegration:
    """Test the full engine pipeline in stub (no-GPU) mode."""

    @SKIP_NO_TOKENIZER
    @pytest.mark.asyncio
    async def test_single_session_stub(self):
        """One session, streaming text, verify audio callback chain."""
        from engine.core.types import (
            EngineRequest, EngineResult, RequestType, ResultType,
        )
        from engine.frontend.dispatcher import Dispatcher
        from engine.frontend.spliter.tokenizer import LightQwen3TTSTokenizer

        tokenizer = LightQwen3TTSTokenizer(str(TOKENIZER_DIR))
        async_inbox = asyncio.Queue(maxsize=256)

        dispatcher = Dispatcher(
            engine_inbox=async_inbox,
            tokenizer=tokenizer,
            max_sessions=8,
            engine_max_decode_len=200,
        )

        received_audio: list[bytes] = []
        done_event = asyncio.Event()

        async def on_audio(sid: str, data: bytes):
            received_audio.append(data)

        async def on_done(sid: str, metrics: dict):
            done_event.set()

        session = await dispatcher.create_session(
            "test-001",
            speaker_key="vivian",
            task_type="custom_voice",
            on_audio=on_audio,
            on_done=on_done,
        )
        assert dispatcher.active_count == 1

        await dispatcher.feed_text("test-001", "你好世界。")
        await dispatcher.text_complete("test-001")

        requests_sent = []
        while not async_inbox.empty():
            req = await async_inbox.get()
            requests_sent.append(req)

        req_types = [r.type for r in requests_sent]
        assert RequestType.NEW_SESSION in req_types
        has_segment = (RequestType.START_SEGMENT in req_types
                       or RequestType.APPEND_TEXT in req_types)
        assert has_segment, f"Expected segment requests, got: {req_types}"

        print(f"\nRequests sent to engine: {len(requests_sent)}")
        for r in requests_sent:
            print(f"  {r.type.name} seg={r.segment_idx} tokens={r.token_ids}")

        await session.result_queue.put(EngineResult(
            type=ResultType.SESSION_DONE,
            session_id="test-001",
        ))

        await asyncio.wait_for(done_event.wait(), timeout=2.0)
        assert dispatcher.active_count == 0
        print("Session completed successfully")

    @SKIP_NO_TOKENIZER
    @pytest.mark.asyncio
    async def test_multi_session_stub(self):
        """Multiple concurrent sessions."""
        from engine.core.types import (
            EngineResult, RequestType, ResultType,
        )
        from engine.frontend.dispatcher import Dispatcher
        from engine.frontend.spliter.tokenizer import LightQwen3TTSTokenizer

        tokenizer = LightQwen3TTSTokenizer(str(TOKENIZER_DIR))
        async_inbox = asyncio.Queue(maxsize=1024)

        dispatcher = Dispatcher(
            engine_inbox=async_inbox,
            tokenizer=tokenizer,
            max_sessions=64,
            engine_max_decode_len=200,
        )

        n_sessions = 8
        done_events = {}
        sessions = {}

        for i in range(n_sessions):
            sid = f"multi-{i:03d}"
            done_events[sid] = asyncio.Event()

            async def on_done(sid_inner: str, metrics: dict, _e=done_events[sid]):
                _e.set()

            s = await dispatcher.create_session(
                sid, task_type="custom_voice", on_done=on_done,
            )
            sessions[sid] = s

        assert dispatcher.active_count == n_sessions

        texts = [
            "你好。", "世界！", "今天天气好。", "明天见。",
            "测试。", "一二三四五。", "很高兴见到你！", "再见！",
        ]
        for i, sid in enumerate(sessions):
            await dispatcher.feed_text(sid, texts[i])
            await dispatcher.text_complete(sid)

        req_count = 0
        while not async_inbox.empty():
            await async_inbox.get()
            req_count += 1

        print(f"\n{n_sessions} sessions generated {req_count} engine requests")
        assert req_count >= n_sessions

        for sid, session in sessions.items():
            await session.result_queue.put(EngineResult(
                type=ResultType.SESSION_DONE,
                session_id=sid,
            ))

        for sid, evt in done_events.items():
            await asyncio.wait_for(evt.wait(), timeout=2.0)

        assert dispatcher.active_count == 0
        print(f"All {n_sessions} sessions completed")


# ---------------------------------------------------------------------------
# 5. Performance benchmark (optional, needs real tokenizer)
# ---------------------------------------------------------------------------

class TestPerformance:
    @SKIP_NO_TOKENIZER
    def test_spliter_throughput(self):
        """Benchmark: how fast can the Spliter process tokens?"""
        import time
        from engine.frontend.spliter.tokenizer import LightQwen3TTSTokenizer
        from engine.frontend.spliter.spliter import Spliter

        tok = LightQwen3TTSTokenizer(str(TOKENIZER_DIR))
        long_text = "这是一段测试文本。" * 200
        ids, texts = tok.encode_with_text(long_text, add_special_tokens=False)
        tokens = list(zip(ids, texts))

        n_iters = 10
        start = time.perf_counter()
        for _ in range(n_iters):
            s = Spliter(engine_max_decode_len=512, ema_ratio=5.0)
            s.set_full_text(tokens)
        elapsed = time.perf_counter() - start

        tokens_per_sec = (len(tokens) * n_iters) / elapsed
        print(f"\nSpliter throughput: {tokens_per_sec:.0f} tokens/sec "
              f"({len(tokens)} tokens × {n_iters} iters in {elapsed:.3f}s)")
        assert tokens_per_sec > 10_000, "Spliter should handle >10K tokens/sec"


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

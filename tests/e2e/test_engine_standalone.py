#!/usr/bin/env python3
"""
E2E test & benchmark for the standalone TTS engine (gRPC on port 50051).

Tests the engine with:
  1. Single-request smoke test
  2. Streaming text input (init -> text chunks -> text_complete)
  3. Multi-session concurrent requests (1, 2, 4 sessions)
  4. Long text rollover (medium / very long / streaming long)
  5. BadCase tests (empty text, whitespace, invalid task_type, etc.)
  6. Performance metrics: first-chunk latency, total latency, RTF, throughput

Usage (CLI — generates WAV files):
    python -m engine.server --config engine.yaml   # terminal 1
    python tests/e2e/test_engine_standalone.py      # terminal 2

    python tests/e2e/test_engine_standalone.py --host localhost --port 50051
    python tests/e2e/test_engine_standalone.py --concurrency 1,2,4,8
    python tests/e2e/test_engine_standalone.py --skip-concurrent --skip-badcase

Usage (pytest — auto-skip if engine not reachable):
    pytest tests/e2e/test_engine_standalone.py -v
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import struct
import sys
import time
import uuid
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from engine.gateway import tts_pb2, tts_pb2_grpc

GRPC_HOST = "localhost"
GRPC_PORT = 50051
SAMPLE_RATE = 24000
OUTPUT_DIR = REPO_ROOT / "workspace" / "audio_samples" / "engine"

# ---------------------------------------------------------------------------
# Result dataclass (mirrors test_concurrent_tts.py)
# ---------------------------------------------------------------------------

@dataclass
class TTSResult:
    session_id: str
    text: str
    first_chunk_ms: Optional[float] = None
    total_ms: float = 0.0
    num_chunks: int = 0
    total_samples: int = 0
    error: Optional[str] = None
    audio: Optional[np.ndarray] = None
    warnings: list = field(default_factory=list)

    @property
    def duration_sec(self) -> float:
        return self.total_samples / SAMPLE_RATE if self.total_samples > 0 else 0.0

    @property
    def rtf(self) -> float:
        if self.duration_sec <= 0 or self.total_ms <= 0:
            return 0.0
        return (self.total_ms / 1000) / self.duration_sec


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

TEST_TEXTS = [
    "你好，今天天气真好。",
    "欢迎来到人工智能语音合成的世界。",
    "技术创新推动着社会不断前进。",
    "我们正在测试多路并发的语音合成能力。",
    "这是第五路测试文本，用来验证系统的处理能力。",
    "深度学习让机器能够理解和生成自然的语音。",
    "云计算和边缘计算相结合，提供更好的用户体验。",
    "每一次迭代都让我们的系统变得更加完善。",
]

LONG_TEXT = (
    "人工智能正在深刻改变我们的世界。"
    "从语音识别到自然语言处理，从计算机视觉到机器人技术，"
    "AI的应用已经渗透到生活的方方面面。"
    "在医疗领域，AI可以辅助诊断疾病、发现新药物。"
    "在教育领域，AI可以提供个性化的学习方案。"
    "在交通领域，自动驾驶技术正在逐步成熟。"
    "未来，人工智能将继续推动社会进步，"
    "为人类创造更多的可能性。"
)

VERY_LONG_TEXT = (
    "在遥远的古代，人类就开始仰望星空，思考宇宙的奥秘。"
    "从古希腊的哲学家到中国的天文学家，人们不断探索着这个世界的本质。"
    "随着科学技术的发展，我们对宇宙的认知越来越深入。"
    "从伽利略的望远镜到哈勃太空望远镜，每一次技术的突破都让我们看到了更广阔的世界。"
    "如今，人工智能技术的发展为科学研究带来了前所未有的机遇。"
    "机器学习算法可以处理海量的天文数据，发现人眼无法察觉的规律。"
    "深度学习网络能够分析复杂的光谱数据，帮助我们理解恒星的演化过程。"
    "在医学领域，AI辅助诊断系统已经在某些疾病的识别上达到了专家水平。"
    "在交通出行方面，自动驾驶技术正在逐步走向成熟，有望彻底改变我们的出行方式。"
    "在教育领域，个性化学习系统能够根据每个学生的特点量身定制学习方案。"
    "在金融领域，智能风控系统可以实时监测交易异常，保护用户的资金安全。"
    "然而，技术的发展也带来了新的挑战和思考。"
    "如何确保人工智能的安全性和可控性？"
    "如何在自动化浪潮中保障就业和社会公平？"
    "这些问题需要我们全社会共同面对和解决。"
    "只有在技术进步与人文关怀之间找到平衡，"
    "才能让人工智能真正造福全人类，"
    "创造一个更加美好的未来。"
)


# ---------------------------------------------------------------------------
# WAV helpers
# ---------------------------------------------------------------------------

def _pcm_bytes_to_f32(pcm_data: bytes) -> np.ndarray:
    return np.frombuffer(pcm_data, dtype=np.float32)


def _save_wav(audio: np.ndarray, path: str):
    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * 32767).astype(np.int16)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_int16.tobytes())


# ---------------------------------------------------------------------------
# gRPC helpers
# ---------------------------------------------------------------------------

def _check_server(host: str, port: int) -> bool:
    import grpc
    channel = grpc.insecure_channel(f"{host}:{port}")
    try:
        grpc.channel_ready_future(channel).result(timeout=3)
        return True
    except grpc.FutureTimeoutError:
        return False
    finally:
        channel.close()


def _synthesize_oneshot(
    host: str, port: int,
    text: str,
    speaker: str = "Serena",
    task_type: str = "custom_voice",
    session_id: str = "",
    timeout: float = 120.0,
) -> TTSResult:
    """Send init → text → done in one shot, collect audio. Runs sync (for ThreadPool)."""
    sid = session_id or uuid.uuid4().hex[:12]
    result = TTSResult(session_id=sid, text=text)

    import grpc
    channel = grpc.insecure_channel(f"{host}:{port}")
    stub = tts_pb2_grpc.TTSServiceStub(channel)

    def request_gen():
        yield tts_pb2.SynthesizeRequest(
            init=tts_pb2.InitRequest(
                session_id=sid,
                speaker=speaker,
                task_type=task_type,
            )
        )
        yield tts_pb2.SynthesizeRequest(
            text=tts_pb2.TextChunk(text=text)
        )
        yield tts_pb2.SynthesizeRequest(
            done=tts_pb2.TextComplete()
        )

    chunks = []
    first_ts = None

    t0 = time.perf_counter()
    try:
        for resp in stub.SynthesizeStream(request_gen(), timeout=timeout):
            which = resp.WhichOneof("response")
            if which == "audio":
                if first_ts is None:
                    first_ts = time.perf_counter()
                chunks.append(_pcm_bytes_to_f32(resp.audio.pcm_data))
            elif which == "status":
                if resp.status.event == "error":
                    result.error = resp.status.message
                    break
                if resp.status.event == "done":
                    break
    except grpc.RpcError as e:
        result.error = f"gRPC {e.code().name}: {e.details()}"
    except Exception as e:
        result.error = str(e)
    finally:
        channel.close()

    elapsed = time.perf_counter() - t0
    result.total_ms = elapsed * 1000
    result.first_chunk_ms = (first_ts - t0) * 1000 if first_ts else None
    result.num_chunks = len(chunks)
    if chunks:
        result.audio = np.concatenate(chunks)
        result.total_samples = result.audio.size

    return result


async def _synthesize_streaming(
    host: str, port: int,
    init_speaker: str,
    init_task_type: str,
    text_chunks: list[str],
    chunk_delay_ms: float = 50.0,
    session_id: str = "",
    timeout: float = 120.0,
) -> TTSResult:
    """Send init → text_chunk* (with delays) → done, collect audio."""
    sid = session_id or uuid.uuid4().hex[:12]
    result = TTSResult(session_id=sid, text=" ".join(text_chunks))

    import grpc.aio as grpc_aio
    channel = grpc_aio.insecure_channel(f"{host}:{port}")
    stub = tts_pb2_grpc.TTSServiceStub(channel)

    chunks: List[np.ndarray] = []
    first_ts = None

    async def request_gen():
        yield tts_pb2.SynthesizeRequest(
            init=tts_pb2.InitRequest(
                session_id=sid,
                speaker=init_speaker,
                task_type=init_task_type,
            )
        )
        for chunk_text in text_chunks:
            if chunk_delay_ms > 0:
                await asyncio.sleep(chunk_delay_ms / 1000)
            yield tts_pb2.SynthesizeRequest(
                text=tts_pb2.TextChunk(text=chunk_text)
            )
        if chunk_delay_ms > 0:
            await asyncio.sleep(chunk_delay_ms / 1000)
        yield tts_pb2.SynthesizeRequest(
            done=tts_pb2.TextComplete()
        )

    t0 = time.perf_counter()
    try:
        response_stream = stub.SynthesizeStream(request_gen(), timeout=timeout)
        async for resp in response_stream:
            which = resp.WhichOneof("response")
            if which == "audio":
                if first_ts is None:
                    first_ts = time.perf_counter()
                chunks.append(_pcm_bytes_to_f32(resp.audio.pcm_data))
            elif which == "status":
                if resp.status.event == "error":
                    result.error = resp.status.message
                    break
                if resp.status.event == "done":
                    break
    except Exception as e:
        result.error = str(e)
    finally:
        await channel.close()

    elapsed = time.perf_counter() - t0
    result.total_ms = elapsed * 1000
    result.first_chunk_ms = (first_ts - t0) * 1000 if first_ts else None
    result.num_chunks = len(chunks)
    if chunks:
        result.audio = np.concatenate(chunks)
        result.total_samples = result.audio.size

    return result


# ---------------------------------------------------------------------------
# Printing helpers (same style as test_concurrent_tts.py)
# ---------------------------------------------------------------------------

def _print_result(result: TTSResult, label: str = ""):
    prefix = f"  [{label}]" if label else "  "
    if result.error:
        print(f"{prefix} ERROR: {result.error}")
        return
    if result.warnings:
        for w in result.warnings:
            print(f"{prefix} WARNING: {w}")
    fc = f"{result.first_chunk_ms:.0f}ms" if result.first_chunk_ms is not None else "N/A"
    print(
        f"{prefix} session={result.session_id}"
        f"  first_chunk={fc}"
        f"  total={result.total_ms:.0f}ms"
        f"  chunks={result.num_chunks}"
        f"  audio={result.duration_sec:.2f}s"
        f"  RTF={result.rtf:.2f}"
    )


def _print_summary(results: list[TTSResult], label: str):
    ok = [r for r in results if r.error is None and r.first_chunk_ms is not None]
    fail = [r for r in results if r.error is not None]
    print(f"\n  ── {label} Summary ──")
    print(f"  Total: {len(results)}  OK: {len(ok)}  Failed: {len(fail)}")
    if not ok:
        return
    first_chunks = [r.first_chunk_ms for r in ok]
    totals = [r.total_ms for r in ok]
    rtfs = [r.rtf for r in ok if r.rtf > 0]
    durations = [r.duration_sec for r in ok]
    print(f"  First-chunk latency:  min={min(first_chunks):.0f}ms  "
          f"median={statistics.median(first_chunks):.0f}ms  "
          f"max={max(first_chunks):.0f}ms  "
          f"mean={statistics.mean(first_chunks):.0f}ms")
    print(f"  Total latency:        min={min(totals):.0f}ms  "
          f"median={statistics.median(totals):.0f}ms  "
          f"max={max(totals):.0f}ms  "
          f"mean={statistics.mean(totals):.0f}ms")
    if rtfs:
        print(f"  RTF (wall/audio):     min={min(rtfs):.2f}  "
              f"median={statistics.median(rtfs):.2f}  "
              f"max={max(rtfs):.2f}")
    total_audio = sum(durations)
    wall_time = max(totals) / 1000
    effective_throughput = total_audio / wall_time if wall_time > 0 else 0
    print(f"  Aggregate throughput: {total_audio:.1f}s audio / {wall_time:.1f}s wall "
          f"= {effective_throughput:.2f}x realtime")


# ---------------------------------------------------------------------------
# Test 1: Single smoke
# ---------------------------------------------------------------------------

def test_single_smoke(host: str, port: int, output_dir: Path) -> TTSResult:
    print("\n" + "=" * 60)
    print("  Test 1: Single Request Smoke Test")
    print("=" * 60)

    result = _synthesize_oneshot(host, port,
        text="你好，这是单路测试。",
        speaker="Serena",
        session_id="smoke-single",
    )
    _print_result(result, "single")
    if result.audio is not None and result.audio.size > 0:
        out_path = str(output_dir / "test1_single_smoke.wav")
        _save_wav(result.audio, out_path)
        print(f"  Saved: {out_path}")

    return result


# ---------------------------------------------------------------------------
# Test 2: Streaming text input
# ---------------------------------------------------------------------------

def test_streaming_text(host: str, port: int, output_dir: Path) -> TTSResult:
    print("\n" + "=" * 60)
    print("  Test 2: Streaming Text Input")
    print("=" * 60)

    result = asyncio.run(_synthesize_streaming(
        host, port,
        init_speaker="Serena",
        init_task_type="custom_voice",
        text_chunks=[
            "你好，这是流式文本输入测试。",
            "我们正在验证",
            "文本追加功能",
            "是否工作正常。",
        ],
        chunk_delay_ms=200,
        session_id="stream-text",
    ))
    _print_result(result, "stream")
    if result.audio is not None and result.audio.size > 0:
        out_path = str(output_dir / "test2_streaming_text.wav")
        _save_wav(result.audio, out_path)
        print(f"  Saved: {out_path}")

    return result


# ---------------------------------------------------------------------------
# Test 3: Concurrent requests
# ---------------------------------------------------------------------------

def test_concurrent(host: str, port: int, concurrency: int, output_dir: Path) -> list[TTSResult]:
    print("\n" + "=" * 60)
    print(f"  Test 3: Concurrent Requests (concurrency={concurrency})")
    print("=" * 60)

    def _run_one(idx: int) -> TTSResult:
        text = TEST_TEXTS[idx % len(TEST_TEXTS)]
        return _synthesize_oneshot(
            host, port,
            text=text,
            speaker="Serena",
            session_id=f"concurrent-{idx}",
        )

    t0 = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(_run_one, i): i for i in range(concurrency)}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                results.append(result)
                _print_result(result, f"c{idx}")
            except Exception as e:
                print(f"  [c{idx}] EXCEPTION: {e}")
                results.append(TTSResult(session_id=f"concurrent-{idx}", text="", error=str(e)))

    wall_time = time.perf_counter() - t0
    print(f"\n  Wall time for {concurrency} concurrent: {wall_time:.2f}s")
    _print_summary(results, f"Concurrent x{concurrency}")

    for i, r in enumerate(results):
        if r.audio is not None and r.audio.size > 0:
            out_path = str(output_dir / f"test3_concurrent_{concurrency}x_{i}.wav")
            _save_wav(r.audio, out_path)

    return results


# ---------------------------------------------------------------------------
# Test 4: Long text rollover
# ---------------------------------------------------------------------------

def test_long_text(host: str, port: int, output_dir: Path) -> list[TTSResult]:
    print("\n" + "=" * 60)
    print("  Test 4: Long Text Rollover")
    print("=" * 60)

    results = []

    # 4a: medium long text
    print("\n  --- 4a: Medium long text ---")
    r1 = _synthesize_oneshot(host, port,
        text=LONG_TEXT, speaker="Serena", session_id="longtext-medium",
        timeout=180,
    )
    _print_result(r1, "medium-long")
    if r1.audio is not None and r1.audio.size > 0:
        out_path = str(output_dir / "test4a_long_text_medium.wav")
        _save_wav(r1.audio, out_path)
        print(f"  Saved: {out_path}")
    results.append(r1)

    # 4b: very long text — triggers multi-segment rollover + dynamic split
    print("\n  --- 4b: Very long text (multi-segment) ---")
    r2 = _synthesize_oneshot(host, port,
        text=VERY_LONG_TEXT, speaker="Serena", session_id="longtext-verylong",
        timeout=300,
    )
    _print_result(r2, "very-long")
    if r2.audio is not None and r2.audio.size > 0:
        out_path = str(output_dir / "test4b_long_text_verylong.wav")
        _save_wav(r2.audio, out_path)
        print(f"  Saved: {out_path}")
        print(f"  Audio duration: {r2.duration_sec:.2f}s "
              f"(~{len(VERY_LONG_TEXT)} chars)")
    results.append(r2)

    # 4c: streaming long text (init → append chunks → text_complete)
    print("\n  --- 4c: Streaming long text ---")
    sentences = [
        "人工智能正在深刻改变我们的世界。",
        "从语音识别到自然语言处理，从计算机视觉到机器人技术，AI的应用已经渗透到生活的方方面面。",
        "在医疗领域，AI可以辅助诊断疾病、发现新药物。在教育领域，AI可以提供个性化的学习方案。",
        "在交通领域，自动驾驶技术正在逐步成熟。未来，人工智能将继续推动社会进步，为人类创造更多的可能性。",
    ]
    r3 = asyncio.run(_synthesize_streaming(
        host, port,
        init_speaker="Serena",
        init_task_type="custom_voice",
        text_chunks=sentences,
        chunk_delay_ms=300,
        session_id="stream-long",
        timeout=180,
    ))
    _print_result(r3, "stream-long")
    if r3.audio is not None and r3.audio.size > 0:
        out_path = str(output_dir / "test4c_streaming_long.wav")
        _save_wav(r3.audio, out_path)
        print(f"  Saved: {out_path}")
    results.append(r3)

    # 4d: story from file (if exists)
    story_path = REPO_ROOT / "tests" / "data" / "story.txt"
    if story_path.is_file():
        print("\n  --- 4d: Story (long narrative) ---")
        story_text = story_path.read_text(encoding="utf-8").strip()
        print(f"  Story length: {len(story_text)} chars")
        r4 = _synthesize_oneshot(host, port,
            text=story_text, speaker="Serena", session_id="longtext-story",
            timeout=600,
        )
        _print_result(r4, "story")
        if r4.audio is not None and r4.audio.size > 0:
            out_path = str(output_dir / "test4d_story.wav")
            _save_wav(r4.audio, out_path)
            print(f"  Saved: {out_path}")
            print(f"  Audio duration: {r4.duration_sec:.2f}s (~{len(story_text)} chars)")
        results.append(r4)
    else:
        print(f"\n  --- 4d: Story SKIPPED (not found: {story_path}) ---")

    return results


# ---------------------------------------------------------------------------
# Test 5: BadCase tests
# ---------------------------------------------------------------------------

def test_badcases(host: str, port: int, output_dir: Path) -> list:
    print("\n" + "=" * 60)
    print("  Test 5: BadCase Tests")
    print("=" * 60)
    results = []

    # 5a: empty text
    print("\n  --- 5a: Empty text (should error) ---")
    r = _synthesize_oneshot(host, port, text="", speaker="Serena",
                            session_id="badcase-empty", timeout=15)
    expected_error = r.error is not None or r.total_samples == 0
    print(f"  Got error/empty: {expected_error} -> {'PASS' if expected_error else 'FAIL'}")
    if r.error:
        print(f"  Error msg: {r.error[:120]}")
    results.append(("5a_empty_text", expected_error, r))

    # 5b: whitespace-only text
    print("\n  --- 5b: Whitespace-only text (should error) ---")
    r = _synthesize_oneshot(host, port, text="   \n\t  ", speaker="Serena",
                            session_id="badcase-whitespace", timeout=15)
    expected_error = r.error is not None or r.total_samples == 0
    print(f"  Got error/empty: {expected_error} -> {'PASS' if expected_error else 'FAIL'}")
    if r.error:
        print(f"  Error msg: {r.error[:120]}")
    results.append(("5b_whitespace_text", expected_error, r))

    # 5c: invalid task_type
    print("\n  --- 5c: Invalid task_type (should error) ---")
    r = _synthesize_oneshot(host, port, text="测试", task_type="nonexistent_task",
                            session_id="badcase-invalid-task", timeout=15)
    expected_error = r.error is not None
    print(f"  Got error: {expected_error} -> {'PASS' if expected_error else 'FAIL'}")
    if r.error:
        print(f"  Error msg: {r.error[:120]}")
    results.append(("5c_invalid_task", expected_error, r))

    # 5d: streaming init → text_complete with no text
    print("\n  --- 5d: Streaming init → text_complete (no text) ---")
    r = asyncio.run(_synthesize_streaming(
        host, port,
        init_speaker="Serena", init_task_type="custom_voice",
        text_chunks=[],
        chunk_delay_ms=100, session_id="badcase-stream-notext", timeout=15,
    ))
    got_error_or_empty = r.error is not None or r.total_samples == 0
    print(f"  Error or empty: {got_error_or_empty} -> "
          f"{'PASS' if got_error_or_empty else 'FAIL (unexpected audio)'}")
    if r.error:
        print(f"  Error msg: {r.error[:120]}")
    results.append(("5d_stream_no_text", got_error_or_empty, r))

    # 5e: cancel mid-stream
    print("\n  --- 5e: Cancel mid-stream ---")
    r = _test_cancel(host, port)
    ok = r.error is None
    print(f"  Cancel clean: {ok} -> {'PASS' if ok else 'FAIL'}")
    _print_result(r, "cancel")
    results.append(("5e_cancel", ok, r))

    # 5f: single character text
    print("\n  --- 5f: Single character text ---")
    r = _synthesize_oneshot(host, port, text="好", speaker="Serena",
                            session_id="badcase-single-char", timeout=30)
    ok = r.error is None and r.total_samples > 0
    print(f"  Got audio: {ok} -> {'PASS' if ok else 'FAIL'}")
    _print_result(r, "single-char")
    if r.audio is not None and r.audio.size > 0:
        _save_wav(r.audio, str(output_dir / "test5f_single_char.wav"))
    results.append(("5f_single_char", ok, r))

    # 5g: pure punctuation
    print("\n  --- 5g: Pure punctuation text ---")
    r = _synthesize_oneshot(host, port, text="。。。！！！", speaker="Serena",
                            session_id="badcase-punctuation", timeout=30)
    _print_result(r, "punctuation")
    results.append(("5g_punctuation", True, r))

    # 5h: streaming with slow text input (simulates slow LLM)
    print("\n  --- 5h: Streaming with slow text input (1s delay) ---")
    r = asyncio.run(_synthesize_streaming(
        host, port,
        init_speaker="Serena", init_task_type="custom_voice",
        text_chunks=["你好，", "这是一个", "慢速输入的测试。"],
        chunk_delay_ms=1000, session_id="badcase-slow-stream", timeout=60,
    ))
    ok = r.error is None and r.total_samples > 0
    print(f"  Got audio: {ok} -> {'PASS' if ok else 'FAIL'}")
    _print_result(r, "slow-stream")
    if r.audio is not None and r.audio.size > 0:
        _save_wav(r.audio, str(output_dir / "test5h_slow_stream.wav"))
    results.append(("5h_slow_stream", ok, r))

    # Summary
    print("\n  ── BadCase Summary ──")
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    for name, ok, r in results:
        status = "PASS" if ok else "FAIL"
        err = f" error={r.error[:60]}" if r.error else ""
        samples = f" samples={r.total_samples}" if r.total_samples else ""
        print(f"  {status:4s}  {name:30s}{err}{samples}")
    print(f"\n  BadCase: {passed}/{total} passed")

    return results


def _test_cancel(host: str, port: int) -> TTSResult:
    """Send init + text, then cancel before done."""
    sid = uuid.uuid4().hex[:12]
    result = TTSResult(session_id=sid, text="(cancel test)")

    import grpc
    channel = grpc.insecure_channel(f"{host}:{port}")
    stub = tts_pb2_grpc.TTSServiceStub(channel)

    def request_gen():
        yield tts_pb2.SynthesizeRequest(
            init=tts_pb2.InitRequest(
                session_id=sid, speaker="Serena", task_type="custom_voice",
            )
        )
        yield tts_pb2.SynthesizeRequest(
            text=tts_pb2.TextChunk(text="这段文字将被取消。")
        )
        time.sleep(0.1)
        yield tts_pb2.SynthesizeRequest(
            cancel=tts_pb2.CancelRequest()
        )

    chunks = []
    t0 = time.perf_counter()
    try:
        for resp in stub.SynthesizeStream(request_gen(), timeout=15):
            which = resp.WhichOneof("response")
            if which == "audio":
                chunks.append(_pcm_bytes_to_f32(resp.audio.pcm_data))
            elif which == "status":
                break
    except grpc.RpcError:
        pass
    except Exception as e:
        result.error = str(e)
    finally:
        channel.close()

    result.total_ms = (time.perf_counter() - t0) * 1000
    result.num_chunks = len(chunks)
    if chunks:
        result.audio = np.concatenate(chunks)
        result.total_samples = result.audio.size

    return result


# ---------------------------------------------------------------------------
# pytest interface — auto-skip if engine not reachable
# ---------------------------------------------------------------------------

import pytest  # noqa: E402


@pytest.fixture(scope="module")
def engine_addr():
    if not _check_server(GRPC_HOST, GRPC_PORT):
        pytest.skip(
            f"Standalone engine not reachable at {GRPC_HOST}:{GRPC_PORT}. "
            f"Start with: python -m engine.server --config engine.yaml"
        )
    return GRPC_HOST, GRPC_PORT


class TestEngineSmokeAndStreaming:
    """Smoke + streaming tests against the standalone engine."""

    def test_single_smoke(self, engine_addr):
        host, port = engine_addr
        r = _synthesize_oneshot(host, port, text="你好，这是一个测试。", speaker="Serena")
        assert r.error is None, f"Synthesis failed: {r.error}"
        assert r.num_chunks >= 1
        assert r.total_samples >= SAMPLE_RATE * 0.1

    def test_english(self, engine_addr):
        host, port = engine_addr
        r = _synthesize_oneshot(host, port, text="Hello, how are you today?", speaker="Serena")
        assert r.error is None, f"Synthesis failed: {r.error}"
        assert r.total_samples >= SAMPLE_RATE * 0.1

    def test_streaming_text(self, engine_addr):
        host, port = engine_addr
        r = asyncio.run(_synthesize_streaming(
            host, port,
            init_speaker="Serena", init_task_type="custom_voice",
            text_chunks=["你好，", "这是流式测试。"],
            chunk_delay_ms=100,
        ))
        assert r.error is None, f"Streaming synthesis failed: {r.error}"
        assert r.total_samples >= SAMPLE_RATE * 0.1


class TestEngineLongText:
    """Long text rollover tests."""

    def test_medium_long(self, engine_addr):
        host, port = engine_addr
        r = _synthesize_oneshot(host, port, text=LONG_TEXT, speaker="Serena", timeout=180)
        assert r.error is None, f"Long text failed: {r.error}"
        assert r.duration_sec >= 1.0, f"Audio too short: {r.duration_sec:.2f}s"

    def test_very_long(self, engine_addr):
        host, port = engine_addr
        r = _synthesize_oneshot(host, port, text=VERY_LONG_TEXT, speaker="Serena", timeout=300)
        assert r.error is None, f"Very long text failed: {r.error}"
        assert r.duration_sec >= 3.0, f"Audio too short: {r.duration_sec:.2f}s"


class TestEngineBadCases:
    """Boundary conditions and error handling."""

    def test_empty_text(self, engine_addr):
        host, port = engine_addr
        r = _synthesize_oneshot(host, port, text="", speaker="Serena", timeout=15)
        assert r.error is not None or r.total_samples == 0

    def test_whitespace_text(self, engine_addr):
        host, port = engine_addr
        r = _synthesize_oneshot(host, port, text="   \n\t  ", speaker="Serena", timeout=15)
        assert r.error is not None or r.total_samples == 0

    def test_single_char(self, engine_addr):
        host, port = engine_addr
        r = _synthesize_oneshot(host, port, text="好", speaker="Serena", timeout=30)
        assert r.error is None
        assert r.total_samples > 0

    def test_cancel_stream(self, engine_addr):
        host, port = engine_addr
        r = _test_cancel(host, port)
        assert r.error is None


class TestEnginePerformance:
    """Performance baseline (relaxed thresholds for CI)."""

    def test_first_chunk_latency(self, engine_addr):
        host, port = engine_addr
        r = _synthesize_oneshot(host, port, text="今天天气真好。", speaker="Serena")
        assert r.error is None
        assert r.first_chunk_ms is not None
        print(f"\n  first_chunk={r.first_chunk_ms:.0f}ms  RTF={r.rtf:.2f}")
        assert r.first_chunk_ms < 30_000, f"First chunk too slow: {r.first_chunk_ms:.0f}ms"


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="E2E test & benchmark for standalone TTS engine (gRPC)"
    )
    parser.add_argument("--host", default=GRPC_HOST)
    parser.add_argument("--port", type=int, default=GRPC_PORT)
    parser.add_argument("--concurrency", default="1,2,4",
                        help="Concurrency levels, comma-separated (default: 1,2,4)")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR),
                        help="Output directory for WAV files")
    parser.add_argument("--skip-streaming", action="store_true")
    parser.add_argument("--skip-concurrent", action="store_true")
    parser.add_argument("--skip-long", action="store_true")
    parser.add_argument("--skip-badcase", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    concurrency_levels = [int(x.strip()) for x in args.concurrency.split(",")]

    print("=" * 60)
    print("  TTS Engine Standalone E2E Test & Benchmark")
    print("=" * 60)
    print(f"  Engine:      {args.host}:{args.port}")
    print(f"  Concurrency: {concurrency_levels}")
    print(f"  Output:      {output_dir}")

    if not _check_server(args.host, args.port):
        print(f"\nERROR: Engine not reachable at {args.host}:{args.port}")
        print("Start the engine first:")
        print("  python -m engine.server --config engine.yaml")
        sys.exit(1)
    print("  Server:      READY")

    all_results: dict[str, list[TTSResult]] = {}

    # Test 1: Single smoke
    r = test_single_smoke(args.host, args.port, output_dir)
    all_results["single"] = [r]

    # Test 2: Streaming text
    if not args.skip_streaming:
        r = test_streaming_text(args.host, args.port, output_dir)
        all_results["streaming"] = [r]

    # Test 3: Concurrent
    if not args.skip_concurrent:
        for level in concurrency_levels:
            results = test_concurrent(args.host, args.port, level, output_dir)
            all_results[f"concurrent_x{level}"] = results

    # Test 4: Long text
    if not args.skip_long:
        long_results = test_long_text(args.host, args.port, output_dir)
        all_results["long_text"] = long_results

    # Test 5: BadCases
    if not args.skip_badcase:
        test_badcases(args.host, args.port, output_dir)

    # Final summary
    print("\n" + "=" * 60)
    print("  FINAL SUMMARY")
    print("=" * 60)
    for name, results in all_results.items():
        ok = sum(1 for r in results if r.error is None)
        fail = sum(1 for r in results if r.error is not None)
        first_chunks = [r.first_chunk_ms for r in results if r.first_chunk_ms is not None]
        avg_first = statistics.mean(first_chunks) if first_chunks else 0
        print(f"  {name:20s}  OK={ok}  FAIL={fail}  avg_first_chunk={avg_first:.0f}ms")

    total_ok = sum(1 for results in all_results.values() for r in results if r.error is None)
    total_fail = sum(1 for results in all_results.values() for r in results if r.error is not None)
    print(f"\n  TOTAL: {total_ok} OK, {total_fail} FAILED")
    print(f"  Output: {output_dir.resolve()}")
    print(f"\n  Play audio: aplay {output_dir}/test1_single_smoke.wav")
    print(f"         or:  ffplay -autoexit {output_dir}/test1_single_smoke.wav")

    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo_api.audio_assets import attach_audio_to_result
from demo_api.audio_store import AudioStore
from demo_api.official_pytorch import is_official_streaming_info_warning
from demo_api.trace_store import TraceStore, normalize_race_payload, replace_backend_result
from demo_api.triton_client import TtsRequest, measure_once


async def main_async() -> None:
    parser = argparse.ArgumentParser(description="Collect WebUI demo race traces.")
    parser.add_argument("--text", default="你好，这是千问3 TTS token级流式语音演示。")
    parser.add_argument("--speaker", default="Serena")
    parser.add_argument("--language", default="auto")
    parser.add_argument("--cache-mode", default="hit", choices=["hit", "miss", "auto"])
    parser.add_argument("--triton-grpc", default="localhost:8001")
    parser.add_argument("--triton-model", default="tts_orchestrator")
    parser.add_argument("--live-triton", action="store_true", help="Replace fixture TRT trace with a live Triton measurement.")
    parser.add_argument("--engine-ws", default="ws://localhost:50052/v1/ws")
    parser.add_argument("--live-engine", action="store_true", help="Replace fixture bare-engine trace with a live standalone engine measurement.")
    parser.add_argument("--live-official", action="store_true", help="Replace fixture official PyTorch traces with live official measurements.")
    parser.add_argument("--timeout-sec", type=float, default=60.0)
    parser.add_argument("--engine-retries", type=int, default=4, help="Retry standalone engine capture after transient connection failures.")
    parser.add_argument("--official-warmup-text", default="你好。", help="Short unmeasured text used to warm the official PyTorch baseline.")
    parser.add_argument("--official-warmup-rounds", type=int, default=1, help="How many unmeasured warmup passes to run per official PyTorch mode.")
    parser.add_argument("--fail-fast", action="store_true", help="Exit non-zero when a requested live backend fails.")
    parser.add_argument("--output", default=str(REPO_ROOT / "workspace" / "demo_traces" / "race_default.json"))
    args = parser.parse_args()

    output_path = Path(args.output)
    store = TraceStore()
    audio_store = AudioStore()
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as handle:
            payload = normalize_race_payload(json.load(handle), source_path=output_path)
    else:
        payload = store.load_default_race()
    collection_warnings: list[str] = []
    payload["generated_at"] = datetime.now(timezone.utc).isoformat()
    payload["default_request"] = {
        "text": args.text,
        "speaker": args.speaker,
        "language": args.language,
        "cache_mode": args.cache_mode,
    }
    conditions = payload.setdefault("benchmark_conditions", {})
    conditions["race_mode"] = "sequential live capture, synchronized replay"
    conditions["live_capture_policy"] = "requested live backends are measured directly; failed backends remain fixture timing without synthetic audio"
    conditions["live_capture_generated_at"] = payload["generated_at"]
    conditions["official_pytorch_warmup"] = (
        f"{max(0, args.official_warmup_rounds)} unmeasured request(s) per mode before measured official requests"
        if args.live_official
        else "not run in this capture"
    )
    conditions["official_ttft_approximation"] = (
        "official PyTorch TTFT is approximated as first decode-stage code0 timestamp minus request start; captured full wav replay is scheduled at that approximate TTFT"
    )

    request = TtsRequest(
        text=args.text,
        speaker=args.speaker,
        language=args.language,
        cache_mode=args.cache_mode,
    )

    async def capture(name: str, backend: str, collect) -> None:
        nonlocal payload
        try:
            result = await collect()
            attach_audio_to_result(result, audio_store)
            payload = replace_backend_result(payload, backend, result.to_dict())
            collection_warnings.extend(
                warning
                for warning in result.warnings
                if not is_official_streaming_info_warning(warning)
            )
            print(
                f"captured {name}: source={result.source}, "
                f"audio={'yes' if result.audio.get('url') else 'no'}, "
                f"total_ms={result.metrics.total_ms}",
                file=sys.stderr,
            )
        except Exception as exc:
            message = f"{name} live measurement unavailable, keeping fixture trace/metrics without synthetic audio: {exc}"
            collection_warnings.append(message)
            print(message, file=sys.stderr)
            if args.fail_fast:
                raise

    if args.live_triton:
        await capture(
            "Triton",
            "triton_trt_streaming",
            lambda: measure_once(
                request,
                endpoint=args.triton_grpc,
                model_name=args.triton_model,
                timeout_sec=args.timeout_sec,
            ),
        )

    if args.live_engine:
        async def collect_engine():
            from demo_api import engine_client

            retries = max(1, args.engine_retries)
            last_exc: Exception | None = None
            for attempt in range(1, retries + 1):
                try:
                    return await engine_client.measure_once(
                        request,
                        url=args.engine_ws,
                        timeout_sec=args.timeout_sec,
                    )
                except Exception as exc:
                    last_exc = exc
                    if attempt >= retries:
                        break
                    sleep_sec = min(8.0, 1.5 * attempt)
                    print(
                        f"Bare engine attempt {attempt}/{retries} failed: {exc}; retrying in {sleep_sec:.1f}s",
                        file=sys.stderr,
                    )
                    await asyncio.sleep(sleep_sec)
            assert last_exc is not None
            raise last_exc

        await capture(
            "Bare engine",
            "bare_engine_streaming",
            collect_engine,
        )

    if args.live_official:
        os.environ["QWEN_DEMO_ENABLE_OFFICIAL_LIVE"] = "1"
        from demo_api.official_pytorch import OfficialPyTorchRunner

        official = OfficialPyTorchRunner()
        warmup_text = args.official_warmup_text.strip() or "你好。"
        warmup_rounds = max(0, int(args.official_warmup_rounds))
        if warmup_rounds > 0:
            for round_idx in range(warmup_rounds):
                for streaming_mode, mode_name in ((False, "official_pytorch_offline"), (True, "official_pytorch_streaming")):
                    try:
                        print(
                            f"official warmup {round_idx + 1}/{warmup_rounds} {mode_name}",
                            file=sys.stderr,
                        )
                        await asyncio.to_thread(
                            official.synthesize,
                            text=warmup_text,
                            speaker=args.speaker,
                            language=args.language,
                            streaming_mode=streaming_mode,
                        )
                    except Exception as exc:
                        message = f"Official warmup for {mode_name} failed: {exc}"
                        collection_warnings.append(message)
                        print(message, file=sys.stderr)
                        if args.fail_fast:
                            raise
        async def collect_official(streaming_mode: bool):
            return await asyncio.to_thread(
                official.synthesize,
                text=args.text,
                speaker=args.speaker,
                language=args.language,
                streaming_mode=streaming_mode,
            )

        for backend, streaming_mode, name in (
            ("official_pytorch_offline", False, "official_pytorch_offline"),
            ("official_pytorch_streaming", True, "official_pytorch_streaming"),
        ):
            await capture(
                name,
                backend,
                lambda streaming_mode=streaming_mode: collect_official(streaming_mode),
            )

    payload["warnings"] = collection_warnings

    out_path = output_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(out_path)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()

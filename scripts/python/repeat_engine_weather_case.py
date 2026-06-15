#!/usr/bin/env python3
"""Repeat one weather-case request against standalone engine WebSocket.

Runs the same text many times, records per-run duration/chunk stats, and saves
WAV files for anomalous or longest samples to help listen for hallucinations.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import uuid
import wave
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.python.raw_websocket import ws_close, ws_connect, ws_recv_frame, ws_send_json


DEFAULT_TEXT = (
    "北京现在天气晴，气温33℃，湿度19%，风是西风3级。"
    "白天温度较高，注意防晒和补水哦。"
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="ws://127.0.0.1:50052/v1/ws", help="Engine WebSocket URL")
    p.add_argument("--text", default=DEFAULT_TEXT, help="Text to synthesize")
    p.add_argument("--speaker", default="001", help="Speaker id")
    p.add_argument("--task-type", default="custom_voice", help="Task type")
    p.add_argument("--language", default="auto", help="Language")
    p.add_argument("--runs", type=int, default=100, help="Number of repeats")
    p.add_argument("--timeout", type=float, default=180.0, help="Per-request timeout in seconds")
    p.add_argument(
        "--mode",
        choices=["start", "oneshot"],
        default="start",
        help="WebSocket request mode: start/text/end or oneshot",
    )
    p.add_argument(
        "--out-dir",
        default="workspace/weather_repeat_100",
        help="Directory for JSON report and saved WAVs",
    )
    p.add_argument(
        "--anomaly-duration-sec",
        type=float,
        default=20.0,
        help="Treat audio longer than this as anomalous",
    )
    p.add_argument(
        "--anomaly-chunks",
        type=int,
        default=300,
        help="Treat chunk counts >= this as anomalous",
    )
    p.add_argument(
        "--save-topk",
        type=int,
        default=5,
        help="Also save WAVs for the top-K longest samples",
    )
    return p.parse_args()


def _save_wav(audio: np.ndarray, path: Path, sample_rate: int) -> None:
    audio = np.asarray(audio, dtype=np.float32)
    audio = np.clip(audio, -1.0, 1.0)
    audio_i16 = (audio * 32767.0).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_i16.tobytes())


def _build_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "task_type": args.task_type,
        "language": args.language,
        "speaker": args.speaker,
        "group_policy": "auto",
        "audio": {
            "encoding": "pcm_f32",
            "sample_rate": 24000,
            "channels": 1,
        },
    }


def _run_once(args: argparse.Namespace, index: int) -> dict[str, Any]:
    session_id = f"weather-repeat-{index:04d}-{uuid.uuid4().hex[:8]}"
    conn = None
    t0 = time.perf_counter()
    first_audio_sec = None
    audio_chunks: list[np.ndarray] = []
    events: list[dict[str, Any]] = []
    error = ""
    sample_rate = 24000

    try:
        conn = ws_connect(args.url, timeout=args.timeout)
        cfg = _build_config(args)
        if args.mode == "oneshot":
            payload = {
                "type": "oneshot",
                "session_id": session_id,
                "text": args.text,
                "config": cfg,
            }
            ws_send_json(conn, payload)
        else:
            ws_send_json(conn, {"type": "start", "session_id": session_id, "config": cfg})
            ws_send_json(conn, {"type": "text", "text": args.text, "seq_no": 0})
            ws_send_json(conn, {"type": "end"})

        while True:
            opcode, payload = ws_recv_frame(conn)
            if opcode == 0x2:
                if first_audio_sec is None:
                    first_audio_sec = time.perf_counter() - t0
                audio_chunks.append(np.frombuffer(payload, dtype=np.float32))
            elif opcode == 0x1:
                msg = json.loads(payload.decode("utf-8"))
                msg_type = str(msg.get("type") or "")
                if msg_type == "event":
                    evt = msg.get("event") or {}
                    evt_type = str(evt.get("type") or "")
                    events.append(
                        {
                            "type": evt_type,
                            "message": str(evt.get("message") or ""),
                            "meta": evt.get("meta") or {},
                        }
                    )
                    if evt_type == "start":
                        fmt = evt.get("audio") or evt.get("audio_format") or {}
                        sample_rate = int(fmt.get("sample_rate") or sample_rate)
                    if evt_type in {"done", "end", "error"}:
                        if evt_type == "error":
                            error = str(evt.get("message") or "error")
                        break
                else:
                    events.append({"type": msg_type, "message": str(msg)[:400], "meta": {}})
            elif opcode == 0x8:
                break
            elif opcode == 0x9:
                continue
            elif opcode == 0xA:
                continue
            else:
                error = f"unexpected opcode={opcode}"
                break
    except Exception as exc:  # noqa: BLE001
        error = repr(exc)
    finally:
        if conn is not None:
            try:
                ws_close(conn)
            except Exception:
                pass

    elapsed_sec = time.perf_counter() - t0
    audio = np.concatenate(audio_chunks) if audio_chunks else np.zeros((0,), dtype=np.float32)
    audio_sec = float(audio.size) / float(sample_rate) if audio.size else 0.0
    anomaly_reasons: list[str] = []
    if error:
        anomaly_reasons.append(f"error={error}")
    if audio_sec > args.anomaly_duration_sec:
        anomaly_reasons.append(f"audio_sec={audio_sec:.4f}>{args.anomaly_duration_sec:.4f}")
    if len(audio_chunks) >= args.anomaly_chunks:
        anomaly_reasons.append(f"audio_chunks={len(audio_chunks)}>={args.anomaly_chunks}")

    return {
        "index": index,
        "session_id": session_id,
        "ok": not error,
        "error": error,
        "mode": args.mode,
        "elapsed_sec": round(float(elapsed_sec), 4),
        "first_audio_sec": None if first_audio_sec is None else round(float(first_audio_sec), 4),
        "audio_sec": round(float(audio_sec), 4),
        "audio_chunks": len(audio_chunks),
        "audio_samples": int(audio.size),
        "sample_rate": int(sample_rate),
        "events_tail": events[-8:],
        "anomaly": bool(anomaly_reasons),
        "anomaly_reasons": anomaly_reasons,
        "_audio": audio,
    }


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    wav_dir = out_dir / "wav"
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    print(
        f"[repeat] url={args.url} mode={args.mode} runs={args.runs} speaker={args.speaker}",
        flush=True,
    )
    for idx in range(args.runs):
        rec = _run_once(args, idx)
        results.append(rec)
        print(
            f"[run] idx={idx:03d} sid={rec['session_id']} "
            f"ok={rec['ok']} dur={rec['audio_sec']:.4f}s chunks={rec['audio_chunks']} "
            f"elapsed={rec['elapsed_sec']:.4f}s anomaly={rec['anomaly']} "
            f"reasons={','.join(rec['anomaly_reasons']) or '-'}",
            flush=True,
        )

    anomalies = [r for r in results if r["anomaly"]]
    topk = sorted(results, key=lambda r: (r["audio_sec"], r["audio_chunks"]), reverse=True)[: max(args.save_topk, 0)]
    to_save = {r["session_id"]: r for r in anomalies}
    for r in topk:
        to_save.setdefault(r["session_id"], r)

    for rec in to_save.values():
        audio = rec.pop("_audio")
        if audio.size:
            tag = "anomaly" if rec["anomaly"] else "top"
            wav_path = wav_dir / f"{tag}_{rec['index']:03d}_{rec['session_id']}.wav"
            _save_wav(audio, wav_path, rec["sample_rate"])
            rec["saved_wav"] = str(wav_path)
        else:
            rec["saved_wav"] = ""

    for rec in results:
        if "_audio" in rec:
            rec.pop("_audio")
        if "saved_wav" not in rec:
            rec["saved_wav"] = ""

    ok_results = [r for r in results if r["ok"]]
    audio_secs = [r["audio_sec"] for r in ok_results]
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "url": args.url,
        "mode": args.mode,
        "text": args.text,
        "speaker": args.speaker,
        "task_type": args.task_type,
        "language": args.language,
        "runs": args.runs,
        "ok": len(ok_results),
        "fail": len(results) - len(ok_results),
        "anomaly_count": len(anomalies),
        "audio_sec_min": min(audio_secs) if audio_secs else None,
        "audio_sec_max": max(audio_secs) if audio_secs else None,
        "audio_sec_mean": statistics.mean(audio_secs) if audio_secs else None,
        "audio_sec_median": statistics.median(audio_secs) if audio_secs else None,
        "audio_chunks_min": min((r["audio_chunks"] for r in ok_results), default=None),
        "audio_chunks_max": max((r["audio_chunks"] for r in ok_results), default=None),
        "anomaly_sessions": [
            {
                "index": r["index"],
                "session_id": r["session_id"],
                "audio_sec": r["audio_sec"],
                "audio_chunks": r["audio_chunks"],
                "reasons": r["anomaly_reasons"],
                "saved_wav": r["saved_wav"],
            }
            for r in anomalies
        ],
        "saved_examples": [
            {
                "index": r["index"],
                "session_id": r["session_id"],
                "audio_sec": r["audio_sec"],
                "audio_chunks": r["audio_chunks"],
                "anomaly": r["anomaly"],
                "saved_wav": r["saved_wav"],
            }
            for r in sorted(to_save.values(), key=lambda x: (x["audio_sec"], x["audio_chunks"]), reverse=True)
        ],
    }

    report = {
        "summary": summary,
        "results": results,
    }
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[summary]")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[report] {report_path}")


if __name__ == "__main__":
    main()

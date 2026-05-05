from __future__ import annotations

import asyncio
import os
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .trace_store import TraceStore


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class RaceCaptureJob:
    job_id: str
    request: dict[str, Any]
    history: list[dict[str, Any]] = field(default_factory=list)
    subscribers: list[asyncio.Queue] = field(default_factory=list)
    done: bool = False
    returncode: int | None = None

    async def publish(self, message: dict[str, Any]) -> None:
        self.history.append(message)
        for subscriber in list(self.subscribers):
            await subscriber.put(message)


class RaceCaptureJobManager:
    def __init__(self, *, trace_store: TraceStore) -> None:
        self.trace_store = trace_store
        self._jobs: dict[str, RaceCaptureJob] = {}
        self._lock = asyncio.Lock()

    def get(self, job_id: str) -> RaceCaptureJob | None:
        return self._jobs.get(job_id)

    async def subscribe(self, job: RaceCaptureJob) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        for item in job.history:
            await queue.put(item)
        job.subscribers.append(queue)
        return queue

    def unsubscribe(self, job: RaceCaptureJob, queue: asyncio.Queue) -> None:
        if queue in job.subscribers:
            job.subscribers.remove(queue)

    async def create_job(self, request: dict[str, Any]) -> RaceCaptureJob:
        job = RaceCaptureJob(job_id=f"race-{uuid.uuid4().hex[:10]}", request=dict(request))
        self._jobs[job.job_id] = job
        asyncio.create_task(self._run_exclusive(job))
        return job

    async def _run_exclusive(self, job: RaceCaptureJob) -> None:
        if self._lock.locked():
            await job.publish(
                {
                    "type": "race_capture_error",
                    "job_id": job.job_id,
                    "message": "another race capture is already running; wait for it to finish",
                }
            )
            job.done = True
            return
        async with self._lock:
            await self._run(job)

    async def _run(self, job: RaceCaptureJob) -> None:
        cmd = self._build_command(job.request)
        await job.publish(
            {
                "type": "race_capture_started",
                "job_id": job.job_id,
                "command": " ".join(cmd),
            }
        )
        env = os.environ.copy()
        env["QWEN_DEMO_PYTHON"] = sys.executable
        env.setdefault("PYTHONUNBUFFERED", "1")
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(REPO_ROOT),
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        except Exception as exc:
            await job.publish(
                {
                    "type": "race_capture_error",
                    "job_id": job.job_id,
                    "message": f"failed to start race capture: {exc}",
                }
            )
            job.done = True
            return

        assert process.stdout is not None
        async for raw_line in process.stdout:
            line = raw_line.decode("utf-8", errors="replace").rstrip()
            if not line:
                continue
            await job.publish({"type": "race_capture_log", "job_id": job.job_id, "message": line})

        job.returncode = await process.wait()
        if job.returncode != 0:
            await job.publish(
                {
                    "type": "race_capture_error",
                    "job_id": job.job_id,
                    "message": f"race capture exited with code {job.returncode}",
                    "returncode": job.returncode,
                }
            )
            job.done = True
            return

        try:
            race = self.trace_store.load_default_race()
        except Exception as exc:
            await job.publish(
                {
                    "type": "race_capture_error",
                    "job_id": job.job_id,
                    "message": f"race capture completed but trace reload failed: {exc}",
                    "returncode": job.returncode,
                }
            )
            job.done = True
            return

        await job.publish(
            {
                "type": "race_capture_done",
                "job_id": job.job_id,
                "returncode": job.returncode,
                "race": race,
            }
        )
        job.done = True

    def _build_command(self, request: dict[str, Any]) -> list[str]:
        script = REPO_ROOT / "scripts" / "demo" / "collect_live_race_sequential.sh"
        cmd = [
            "bash",
            str(script),
            "--variant",
            str(request.get("variant") or os.environ.get("MODEL_VARIANT") or "custom-1.7b"),
            "--text",
            str(request.get("text") or "你好，这是千问3 TTS token级流式语音演示。"),
            "--speaker",
            str(request.get("speaker") or os.environ.get("QWEN_DEMO_DEFAULT_SPEAKER") or "Serena"),
            "--language",
            str(request.get("language") or os.environ.get("QWEN_DEMO_DEFAULT_LANGUAGE") or "auto"),
            "--cache-mode",
            str(request.get("cache_mode") or "hit"),
            "--timeout-sec",
            str(float(request.get("timeout_sec") or os.environ.get("QWEN_DEMO_RACE_TIMEOUT_SEC") or 120.0)),
            "--engine-retries",
            str(int(request.get("engine_retries") or os.environ.get("QWEN_DEMO_ENGINE_CAPTURE_RETRIES") or 4)),
            "--official-warmup-rounds",
            str(int(request.get("official_warmup_rounds") or os.environ.get("QWEN_DEMO_OFFICIAL_WARMUP_ROUNDS") or 1)),
            "--official-warmup-text",
            str(request.get("official_warmup_text") or os.environ.get("QWEN_DEMO_OFFICIAL_WARMUP_TEXT") or "你好。"),
            "--triton-slots",
            str(int(request.get("triton_slots") or os.environ.get("QWEN_DEMO_TRITON_MAX_BATCH_SLOTS") or os.environ.get("TRITON_MAX_BATCH_SLOTS") or 128)),
        ]
        if request.get("skip_official"):
            cmd.append("--skip-official")
        if request.get("skip_engine"):
            cmd.append("--skip-engine")
        if request.get("skip_triton"):
            cmd.append("--skip-triton")
        if request.get("strict"):
            cmd.append("--strict")
        return cmd

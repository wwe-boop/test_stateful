from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .schemas import BACKENDS, normalize_backend_result


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
WORKSPACE_TRACE_DIR = Path(
    os.environ.get("QWEN_DEMO_TRACE_DIR", str(REPO_ROOT / "workspace" / "demo_traces"))
)


class TraceStore:
    def __init__(
        self,
        *,
        workspace_dir: Path = WORKSPACE_TRACE_DIR,
        fixture_dir: Path = FIXTURE_DIR,
    ) -> None:
        self.workspace_dir = workspace_dir
        self.fixture_dir = fixture_dir

    def default_race_path(self) -> Path:
        workspace_path = self.workspace_dir / "race_default.json"
        if workspace_path.exists():
            return workspace_path
        return self.fixture_dir / "race_default.json"

    def load_default_race(self) -> dict[str, Any]:
        path = self.default_race_path()
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return normalize_race_payload(payload, source_path=path)

    def save_race_result(self, payload: dict[str, Any], filename: str = "race_default.json") -> Path:
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        path = self.workspace_dir / filename
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        return path


def normalize_race_payload(payload: dict[str, Any], *, source_path: Path | None = None) -> dict[str, Any]:
    raw_results = payload.get("results", []) or []
    by_backend = {
        result.get("backend"): normalize_backend_result(result)
        for result in raw_results
        if isinstance(result, dict)
    }
    missing = [backend for backend in BACKENDS if backend not in by_backend]
    if missing:
        where = f" in {source_path}" if source_path else ""
        raise ValueError(f"race trace is missing backend(s){where}: {', '.join(missing)}")

    normalized = dict(payload)
    normalized["results"] = [by_backend[backend] for backend in BACKENDS]
    normalized.setdefault("benchmark_conditions", {})
    normalized.setdefault("default_request", {})
    normalized.setdefault("generated_at", "")
    if source_path is not None:
        normalized["source_path"] = str(source_path)
    return normalized


def replace_backend_result(payload: dict[str, Any], backend: str, result: dict[str, Any]) -> dict[str, Any]:
    updated = dict(payload)
    results = []
    replaced = False
    for current in payload.get("results", []) or []:
        if current.get("backend") == backend:
            results.append(normalize_backend_result(result))
            replaced = True
        else:
            results.append(current)
    if not replaced:
        results.append(normalize_backend_result(result))
    updated["results"] = results
    return normalize_race_payload(updated)


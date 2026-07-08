"""Thin wrapper over OpenAI-compatible chat APIs (DashScope / Volcengine Ark)."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

from openai import OpenAI


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    api_key: str
    base_url: str
    model: str


def _require(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def load_llm_config(provider: str | None = None) -> LLMConfig:
    selected = (provider or os.environ.get("DATA_SYNTH_PROVIDER", "dashscope")).strip().lower()

    if selected in {"dashscope", "openai", "qwen"}:
        return LLMConfig(
            provider="dashscope",
            api_key=_require("OPENAI_API_KEY"),
            base_url=os.environ.get(
                "OPENAI_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ).strip(),
            model=os.environ.get("DATA_SYNTH_MODEL", "qwen-plus").strip(),
        )

    if selected in {"ark", "volcengine", "doubao"}:
        return LLMConfig(
            provider="ark",
            api_key=_require("ARK_API_KEY"),
            base_url=os.environ.get(
                "ARK_BASE_URL",
                "https://ark.cn-beijing.volces.com/api/v3",
            ).strip(),
            model=_require("ARK_MODEL"),
        )

    raise ValueError(
        f"Unsupported DATA_SYNTH_PROVIDER={selected!r}; use dashscope or ark"
    )


def _extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        payload = json.loads(cleaned)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        raise ValueError(f"LLM response does not contain JSON object: {text[:200]!r}")
    payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("LLM JSON payload must be an object")
    return payload


class LLMClient:
    def __init__(self, config: LLMConfig):
        self.config = config
        self._client = OpenAI(api_key=config.api_key, base_url=config.base_url)

    def generate_json(self, *, system_prompt: str, user_prompt: str, temperature: float = 0.9) -> dict[str, Any]:
        response = self._client.chat.completions.create(
            model=self.config.model,
            temperature=temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content or ""
        return _extract_json_object(content)

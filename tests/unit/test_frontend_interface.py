from __future__ import annotations

import asyncio

import pytest

from engine.core.types import (
    EngineResult,
    GroupPolicy,
    InputMode,
    RequestType,
    ResultType,
    SessionConfig,
)
from engine.frontend.interface import FrontendInterface


class _CharTokenizer:
    def encode_ids(self, text, add_special_tokens=False):
        return [ord(ch) for ch in text]

    def encode_with_text(self, text, add_special_tokens=False):
        ids = self.encode_ids(text, add_special_tokens=add_special_tokens)
        return ids, list(text)


@pytest.mark.asyncio
async def test_token_mode_preserves_whitespace_only_chunks():
    inbox = asyncio.Queue(maxsize=16)
    interface = FrontendInterface(
        engine_inbox=inbox,
        tokenizer=_CharTokenizer(),
        max_sessions=2,
        engine_max_decode_len=64,
    )

    session = await interface.create_session(
        "space-token",
        config=SessionConfig(
            task_type="custom_voice",
            speaker="Serena",
            input_mode=InputMode.TOKEN,
            group_policy=GroupPolicy.NONE,
        ),
    )

    await interface.push_text_input("space-token", " ")

    requests = []
    while not inbox.empty():
        requests.append(await inbox.get())

    assert [request.type for request in requests] == [
        RequestType.NEW_SESSION,
        RequestType.START_TOKENS,
    ]
    assert requests[-1].token_ids == [ord(" ")]

    await session.result_queue.put(
        EngineResult(type=ResultType.SESSION_DONE, session_id="space-token")
    )
    await asyncio.sleep(0)

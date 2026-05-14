from __future__ import annotations

import socket
import time

from tests.tools import serving_endpoints


class _FakeSocket:
    def __init__(self) -> None:
        self.timeouts: list[float] = []

    def settimeout(self, value: float) -> None:
        self.timeouts.append(value)


class _FakeConnection:
    def __init__(self) -> None:
        self.sock = _FakeSocket()


def test_websocket_oneshot_waits_through_short_socket_idle(monkeypatch):
    frames = [
        socket.timeout(),
        (
            0x1,
            b'{"type":"event","event":{"type":"done","meta":{}}}',
        ),
    ]

    def fake_recv_frame(_conn):
        item = frames.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    monkeypatch.setattr(serving_endpoints, "ws_recv_frame", fake_recv_frame)

    transport = serving_endpoints.EngineWebSocketTransport("ws://example.test/v1/ws")
    result = serving_endpoints.SynthesisResult("engine-websocket", "sid", "text")

    terminal = transport._read_until_timeout(
        _FakeConnection(),
        result,
        [],
        [],
        {"first_ts": None, "encoding": "pcm_f32"},
        deadline=time.perf_counter() + 1.0,
        stop_on_idle=False,
    )

    assert terminal is True
    assert result.error is None
    assert result.events == ["done"]


def test_websocket_idle_drain_still_returns_on_socket_idle(monkeypatch):
    def fake_recv_frame(_conn):
        raise socket.timeout()

    monkeypatch.setattr(serving_endpoints, "ws_recv_frame", fake_recv_frame)

    transport = serving_endpoints.EngineWebSocketTransport("ws://example.test/v1/ws")
    result = serving_endpoints.SynthesisResult("engine-websocket", "sid", "text")

    terminal = transport._read_until_timeout(
        _FakeConnection(),
        result,
        [],
        [],
        {"first_ts": None, "encoding": "pcm_f32"},
        deadline=time.perf_counter() + 1.0,
        stop_on_idle=True,
    )

    assert terminal is False
    assert result.error is None
    assert result.events == []

from __future__ import annotations

import torch
import pytest

from engine.backend.executor import TRTEngine


class _FakeContext:
    def __init__(
        self,
        *,
        output_shapes_before: dict[str, tuple[int, ...]],
        output_shapes_after: dict[str, tuple[int, ...]] | None = None,
        infer_shapes_result: list[str] | None = None,
    ):
        self.output_shapes_before = dict(output_shapes_before)
        self.output_shapes_after = dict(output_shapes_after or output_shapes_before)
        self.infer_shapes_result = list(infer_shapes_result or [])
        self.input_shapes: dict[str, tuple[int, ...]] = {}
        self.tensor_addresses: dict[str, int] = {}
        self.set_input_shape_calls: list[str] = []
        self.infer_shapes_calls = 0
        self.execute_calls = 0
        self._after_infer_shapes = False

    def set_input_shape(self, name: str, shape: tuple[int, ...]) -> None:
        self.set_input_shape_calls.append(name)
        self.input_shapes[name] = tuple(shape)

    def set_tensor_address(self, name: str, address: int) -> None:
        self.tensor_addresses[name] = address

    def infer_shapes(self) -> list[str]:
        self.infer_shapes_calls += 1
        self._after_infer_shapes = True
        return list(self.infer_shapes_result)

    def get_tensor_shape(self, name: str) -> tuple[int, ...]:
        if self._after_infer_shapes:
            return self.output_shapes_after[name]
        return self.output_shapes_before[name]

    def execute_async_v3(self, _stream_handle: int) -> None:
        self.execute_calls += 1


class _FakeStream:
    cuda_stream = 123


def _make_engine(
    ctx: _FakeContext,
    *,
    input_names: set[str] | None = None,
    output_dtypes: dict[str, torch.dtype] | None = None,
) -> TRTEngine:
    engine = TRTEngine.__new__(TRTEngine)
    engine._plan_path = "fake.plan"
    engine._device = torch.device("cpu")
    engine._engine = object()
    engine._context = ctx
    engine._input_names = set(input_names or {"input_embeds"})
    engine._output_dtypes = dict(output_dtypes or {"wav": torch.float32})
    engine._prev_input_shapes = {}
    engine._output_buffers = {}
    return engine


def test_infer_skips_unknown_inputs_and_resolves_dynamic_outputs():
    ctx = _FakeContext(
        output_shapes_before={"wav": (-1, 1920)},
        output_shapes_after={"wav": (1, 1920)},
    )
    engine = _make_engine(ctx, input_names={"input_embeds"})

    outputs = engine.infer(
        inputs={
            "input_embeds": torch.randn(1, 4),
            "cp_gumbel_noise": torch.randn(1, 15, 50),
        },
        output_names=["wav"],
        stream=_FakeStream(),
    )

    assert ctx.set_input_shape_calls == ["input_embeds"]
    assert "cp_gumbel_noise" not in ctx.tensor_addresses
    assert ctx.infer_shapes_calls == 1
    assert ctx.execute_calls == 1
    assert tuple(outputs["wav"].shape) == (1, 1920)


def test_infer_raises_for_missing_required_inputs():
    ctx = _FakeContext(output_shapes_before={"wav": (1, 1920)})
    engine = _make_engine(ctx, input_names={"input_embeds", "token_counts"})

    with pytest.raises(RuntimeError, match="missing required inputs"):
        engine.infer(
            inputs={"input_embeds": torch.randn(1, 4)},
            output_names=["wav"],
            stream=_FakeStream(),
        )


def test_infer_raises_when_output_shape_remains_dynamic():
    ctx = _FakeContext(
        output_shapes_before={"wav": (-1, 1920)},
        output_shapes_after={"wav": (-1, 1920)},
    )
    engine = _make_engine(ctx, input_names={"input_embeds"})

    with pytest.raises(RuntimeError, match="unresolved output shape"):
        engine.infer(
            inputs={"input_embeds": torch.randn(1, 4)},
            output_names=["wav"],
            stream=_FakeStream(),
        )

"""Executor: owns TRT engines and CUDA streams, executes GPU compute.

Replaces Triton BLS with direct TRT plan execution via torch, eliminating:
  - pb_utils.InferenceRequest overhead (~0.5ms/step)
  - Triton scheduling latency (~0.3ms/step)
  - dlpack round-trip for every KV tensor

Key optimisations:
  1. Packed KV tensors: 56 talker KV + 16 C2W KV bindings merged into
     2 packed tensors, reducing TRT I/O binding from ~201 to ~61.
  2. Pre-cached output metadata: dtype mapping and output buffer allocation
     happen once at init, not per-step.
  3. launch_decode_step() is asynchronous: enqueues CUDA kernels on a
     dedicated stream and returns a GPUFuture, letting the caller process
     previous step results on CPU while GPU is busy.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
import numpy as np

from .batch_helper import (
    pad_packed_kv,
    padded_attention_bias,
    split_packed_kv,
    uniform_past_seq_lens,
    zeros_attention_bias,
)
from .kv_cache_pool import KVCachePool, ModelConfig, SlotKVState

logger = logging.getLogger(__name__)

FUSED_CHUNK_T = 1
_FUSED_DUMMY_PAST_LEN = 1


# ---------------------------------------------------------------------------
# TRT Engine wrapper
# ---------------------------------------------------------------------------

class TRTEngine:
    """Thin wrapper around a TensorRT plan loaded via torch.

    Optimisations over naive per-call binding:
      - Output dtypes are cached once at load time.
      - Output buffers are pre-allocated for common shapes and reused.
      - Shapes are only set when they actually change.
    """

    def __init__(self, plan_path: str, device: torch.device):
        self._plan_path = plan_path
        self._device = device
        self._engine = None
        self._context = None
        self._output_dtypes: Dict[str, torch.dtype] = {}
        self._prev_input_shapes: Dict[str, tuple] = {}
        self._output_buffers: Dict[str, torch.Tensor] = {}

    def load(self) -> None:
        """Load TRT engine from .plan file."""
        import tensorrt as trt

        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)

        plan_path = Path(self._plan_path)
        if not plan_path.exists():
            raise FileNotFoundError(f"TRT plan not found: {plan_path}")

        with open(plan_path, "rb") as f:
            engine_bytes = f.read()

        self._engine = runtime.deserialize_cuda_engine(engine_bytes)
        if self._engine is None:
            raise RuntimeError(
                f"Failed to deserialize TRT engine: {plan_path}. "
                "TensorRT plan files are not compatible across different TRT "
                "library versions. Phase B builds engines with trtexec inside the "
                "NGC Triton image (e.g. libnvinfer 10.15.x for tritonserver:26.02); "
                "install a matching Python tensorrt, e.g. "
                "`pip install 'tensorrt==10.15.1.29'`, or rebuild engines after "
                "changing TensorRT."
            )

        self._context = self._engine.create_execution_context()
        self._cache_output_dtypes()
        logger.info("Loaded TRT engine: %s (%d I/O tensors)",
                     plan_path.name, self._engine.num_io_tensors)

    def _cache_output_dtypes(self) -> None:
        """Cache output tensor dtypes once at load time."""
        import tensorrt as trt
        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            if self._engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                dtype_trt = self._engine.get_tensor_dtype(name)
                self._output_dtypes[name] = self._trt_to_torch_dtype(dtype_trt)

    def get_io_names(self) -> tuple[list[str], list[str]]:
        """Return (input_names, output_names)."""
        import tensorrt as trt

        inputs, outputs = [], []
        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            if self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                inputs.append(name)
            else:
                outputs.append(name)
        return inputs, outputs

    def infer(
        self,
        inputs: Dict[str, torch.Tensor],
        output_names: List[str],
        stream: torch.cuda.Stream,
        output_overrides: Optional[Dict[str, torch.Tensor]] = None,
        context=None,
    ) -> Dict[str, torch.Tensor]:
        """Execute inference on the given CUDA stream.

        Uses torch tensors directly — zero copy via data_ptr().
        Skips set_input_shape for tensors whose shape hasn't changed.
        Re-uses output buffers when shapes match previous call.

        Args:
            output_overrides: pre-allocated tensors to use for specific outputs
                instead of the internal buffer cache.  TRT writes directly into
                these tensors — the caller owns them and must ensure they are not
                aliased with any input tensor.
            context: optional TRT execution context to use instead of the
                default one.  When provided, shape caching and output buffer
                reuse are disabled to avoid cross-context state conflicts
                (e.g. separate prefill context overlapping with decode).
        """
        ctx = context if context is not None else self._context
        use_cache = context is None

        for name, tensor in inputs.items():
            tensor = tensor.contiguous()
            shape = tuple(tensor.shape)
            if use_cache:
                if self._prev_input_shapes.get(name) != shape:
                    ctx.set_input_shape(name, shape)
                    self._prev_input_shapes[name] = shape
            else:
                ctx.set_input_shape(name, shape)
            ctx.set_tensor_address(name, tensor.data_ptr())

        outputs = {}
        for name in output_names:
            shape = tuple(ctx.get_tensor_shape(name))
            dtype_torch = self._output_dtypes.get(name, torch.float32)

            override = output_overrides.get(name) if output_overrides else None
            if override is not None and override.shape == shape and override.dtype == dtype_torch:
                out_tensor = override
            elif use_cache:
                existing = self._output_buffers.get(name)
                if existing is not None and existing.shape == shape and existing.dtype == dtype_torch:
                    out_tensor = existing
                else:
                    out_tensor = torch.empty(
                        shape, dtype=dtype_torch, device=self._device,
                    )
                    self._output_buffers[name] = out_tensor
            else:
                out_tensor = torch.empty(
                    shape, dtype=dtype_torch, device=self._device,
                )

            ctx.set_tensor_address(name, out_tensor.data_ptr())
            outputs[name] = out_tensor

        ctx.execute_async_v3(stream.cuda_stream)
        return outputs

    @staticmethod
    def _trt_to_torch_dtype(trt_dtype) -> torch.dtype:
        import tensorrt as trt
        mapping = {
            trt.float32: torch.float32,
            trt.float16: torch.float16,
            trt.bfloat16: torch.bfloat16,
            trt.int32: torch.int32,
            trt.int64: torch.int64,
            trt.int8: torch.int8,
            trt.bool: torch.bool,
        }
        return mapping.get(trt_dtype, torch.float32)


# ---------------------------------------------------------------------------
# GPU Future — handle to in-flight computation
# ---------------------------------------------------------------------------

@dataclass
class GPUFuture:
    """Handle to async GPU work.  Call wait() to synchronize."""
    _compute_stream: Any = None
    _raw: Dict[str, torch.Tensor] = field(default_factory=dict)
    _slots: List[SlotKVState] = field(default_factory=list)
    _original_past_lens: List[int] = field(default_factory=list)
    _padded_past_len: int = 0
    _seq: int = 1
    _c2w_conv_output_names: List[str] = field(default_factory=list)
    _c2w_transconv_output_names: List[str] = field(default_factory=list)
    _codec_eos_id: int = 2150
    _used_pingpong: bool = False

    def wait(self) -> StepOutput:
        """Synchronize GPU and extract results.

        Returns batch-level KV tensors for pool scatter (no per-slot splitting).
        Conv/transconv states are still split per-slot (heterogeneous shapes).
        """
        if self._compute_stream is not None:
            self._compute_stream.synchronize()

        batch_size = len(self._slots)
        raw = self._raw

        wav = raw.get("wav")
        codec_sum = raw.get("codec_sum")
        full_codec = raw.get("full_codec")
        updated_tc = raw.get("updated_token_counts")

        split_c2w_conv = []
        split_c2w_transconv = []
        conv_tensors = [raw.get(n) for n in self._c2w_conv_output_names]
        transconv_tensors = [raw.get(n) for n in self._c2w_transconv_output_names]
        for row_idx in range(batch_size):
            split_c2w_conv.append([
                t[row_idx:row_idx + 1] if t is not None else None
                for t in conv_tensors
            ])
            split_c2w_transconv.append([
                t[row_idx:row_idx + 1] if t is not None else None
                for t in transconv_tensors
            ])

        codec_eos_id = self._codec_eos_id
        eos_flags = []
        audio_chunks = []
        if full_codec is not None:
            eos_list = (full_codec[:, 0] == codec_eos_id).cpu().tolist()
        else:
            eos_list = [False] * batch_size
        if wav is not None:
            wav_cpu = wav.cpu().float()
        else:
            wav_cpu = None

        for row_idx in range(batch_size):
            eos_flags.append(eos_list[row_idx])
            if wav_cpu is not None:
                chunk = wav_cpu[row_idx].reshape(-1).numpy()
                audio_chunks.append(chunk.tobytes())
            else:
                audio_chunks.append(None)

        return StepOutput(
            slots=self._slots,
            eos_flags=eos_flags,
            audio_chunks=audio_chunks,
            batch_talker_kv=raw.get("talker_present_kv"),
            batch_c2w_kv=raw.get("c2w_present_kv"),
            original_past_lens=self._original_past_lens,
            padded_past_len=self._padded_past_len,
            split_c2w_conv=split_c2w_conv,
            split_c2w_transconv=split_c2w_transconv,
            codec_sum=codec_sum,
            updated_tc=updated_tc,
            used_pingpong=self._used_pingpong,
        )


@dataclass
class StepOutput:
    """Results from one decode step.

    Talker/C2W KV are kept as batch-level tensors for direct pool scatter
    (no per-slot splitting).  Conv/transconv states are split per-slot
    because they have heterogeneous shapes.

    When ``used_pingpong`` is True, TRT wrote conv/transconv outputs
    directly into each slot's write buffers.  The engine loop only needs
    to call ``slot.flip_c2w_buffers()`` — no copy or clone required.
    """
    slots: List[SlotKVState]
    eos_flags: List[bool]
    audio_chunks: List[Optional[bytes]]
    batch_talker_kv: Optional[torch.Tensor] = None
    batch_c2w_kv: Optional[torch.Tensor] = None
    original_past_lens: List[int] = field(default_factory=list)
    padded_past_len: int = 0
    split_c2w_conv: List[List[Optional[torch.Tensor]]] = field(default_factory=list)
    split_c2w_transconv: List[List[Optional[torch.Tensor]]] = field(default_factory=list)
    codec_sum: Optional[torch.Tensor] = None
    updated_tc: Optional[torch.Tensor] = None
    used_pingpong: bool = False


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class Executor:
    """Manages TRT engines and CUDA streams for pipelined decode.

    CUDA stream layout:
        compute_stream: fused Talker decode + Code2Wav in single engine call
    """

    def __init__(
        self,
        *,
        engine_dir: str = "",
        weights_dir: str = "",
        device_id: int = 0,
        max_batch_size: int = 48,
        max_seq_len: int = 512,
        model_config: Optional[ModelConfig] = None,
        do_sample: bool = True,
        temperature: float = 0.9,
        repetition_penalty: float = 1.05,
    ):
        self._engine_dir = Path(engine_dir) if engine_dir else None
        self._weights_dir = Path(weights_dir) if weights_dir else None
        self._device = torch.device("cuda", device_id)
        self._max_batch = max_batch_size
        self._max_seq_len = max_seq_len
        self._config = model_config or ModelConfig()
        self._do_sample = do_sample
        self._temperature = temperature
        self._repetition_penalty = repetition_penalty

        self._compute_stream = torch.cuda.Stream(device=self._device)
        self._prefill_stream = torch.cuda.Stream(device=self._device)

        self._fused_engine: Optional[TRTEngine] = None
        self._prefill_context = None
        self._embedding_weights = None
        self._kv_pool: Optional[KVCachePool] = None
        self._codec_eos_id: int = 2150

        self._c2w_conv_input_names: list[str] = []
        self._c2w_conv_output_names: list[str] = []
        self._c2w_transconv_input_names: list[str] = []
        self._c2w_transconv_output_names: list[str] = []
        self._manifest: dict = {}

        logger.info(
            "Executor created (device=%s, max_batch=%d, max_seq=%d)",
            self._device, max_batch_size, max_seq_len,
        )

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load TRT engines, embedding weights, and initialize KV pool."""
        fused_plan: Optional[Path] = None
        if self._engine_dir:
            mp = self._engine_dir / "model.plan"
            te = self._engine_dir / "talker_code2wav_fused.engine"
            if mp.exists():
                fused_plan = mp
            elif te.exists():
                fused_plan = te
            manifest_path = self._engine_dir / "triton_manifest.json"
            if manifest_path.exists():
                with open(manifest_path) as f:
                    self._manifest = json.load(f)
        if self._engine_dir and fused_plan is not None:
            self._fused_engine = TRTEngine(
                str(fused_plan), self._device,
            )
            self._fused_engine.load()
            self._discover_c2w_io_names()

            try:
                self._prefill_context = self._fused_engine._engine.create_execution_context()
                logger.info("Created separate prefill execution context")
            except Exception:
                self._prefill_context = None
                logger.info("Prefill shares execution context with decode")
        else:
            logger.warning("No TRT plan found, running in stub mode")

        self._kv_pool = KVCachePool(
            max_slots=self._max_batch,
            config=self._config,
            device=self._device,
        )
        logger.info("Executor loaded (engine=%s)", "TRT" if self._fused_engine else "stub")

    def set_embedding_weights(self, weights) -> None:
        self._embedding_weights = weights
        if hasattr(weights, 'codec_eos_id'):
            self._codec_eos_id = int(weights.codec_eos_id)
            logger.info("codec_eos_id set to %d from weights", self._codec_eos_id)

    def _discover_c2w_io_names(self) -> None:
        """Detect c2w conv/transconv I/O names from the loaded TRT engine.

        Also caches the static shape (batch=1) for each state tensor so
        we can create zero-initialized dummies for the first prefill call.
        """
        if self._fused_engine is None:
            return
        input_names, output_names = self._fused_engine.get_io_names()
        layout = self._manifest.get("code2wav_fused", {}) if self._manifest else {}
        layout_inputs = layout.get("c2w_state_input_names") or []
        layout_outputs = layout.get("c2w_state_output_names") or []

        def _natural_key(name: str) -> tuple[str, int]:
            m = re.search(r"^(.*?)(\d+)$", name)
            if m:
                return (m.group(1), int(m.group(2)))
            return (name, -1)

        def _ordered(names: list[str], prefix: str, layout_names: list[str]) -> list[str]:
            from_layout = [n for n in layout_names if n.startswith(prefix) and n in names]
            if from_layout:
                return from_layout
            return sorted((n for n in names if n.startswith(prefix)), key=_natural_key)

        self._c2w_conv_input_names = _ordered(
            input_names, "c2w_conv_state_", layout_inputs,
        )
        self._c2w_transconv_input_names = _ordered(
            input_names, "c2w_transconv_overlap_", layout_inputs,
        )
        self._c2w_conv_output_names = _ordered(
            output_names, "c2w_new_conv_state_", layout_outputs,
        )
        self._c2w_transconv_output_names = _ordered(
            output_names, "c2w_new_transconv_overlap_", layout_outputs,
        )

        eng = self._fused_engine._engine
        self._c2w_conv_shapes: list[tuple[int, ...]] = []
        for name in self._c2w_conv_input_names:
            shape = eng.get_tensor_shape(name)
            self._c2w_conv_shapes.append(tuple(1 if d == -1 else d for d in shape))
        self._c2w_transconv_shapes: list[tuple[int, ...]] = []
        for name in self._c2w_transconv_input_names:
            shape = eng.get_tensor_shape(name)
            self._c2w_transconv_shapes.append(tuple(1 if d == -1 else d for d in shape))

        logger.info(
            "C2W states: %d conv inputs, %d transconv inputs, "
            "%d conv outputs, %d transconv outputs",
            len(self._c2w_conv_input_names),
            len(self._c2w_transconv_input_names),
            len(self._c2w_conv_output_names),
            len(self._c2w_transconv_output_names),
        )

    @property
    def kv_pool(self) -> KVCachePool:
        return self._kv_pool

    def make_zero_conv_states(self) -> list[torch.Tensor]:
        """Create zero-initialized C2W conv states for one slot."""
        return [
            torch.zeros(shape, device=self._device, dtype=self._config.dtype)
            for shape in self._c2w_conv_shapes
        ]

    def make_zero_transconv_states(self) -> list[torch.Tensor]:
        """Create zero-initialized C2W transconv states for one slot."""
        return [
            torch.zeros(shape, device=self._device, dtype=self._config.dtype)
            for shape in self._c2w_transconv_shapes
        ]

    # ------------------------------------------------------------------
    # Prefill
    # ------------------------------------------------------------------

    def prefill(
        self,
        slot: SlotKVState,
        prefill_embeds: torch.Tensor,
    ) -> tuple[Optional[bytes], bool]:
        """Execute prefill for a single session on the dedicated prefill stream.

        Uses a separate CUDA stream so prefill does not block an
        in-flight decode step.  When a separate TRT execution context is
        available, prefill and decode can overlap on different streams.

        Results are written directly to the pre-allocated KV pool via
        scatter (no per-slot tensor references).

        Args:
            slot: the GPU slot to store resulting KV cache
            prefill_embeds: [1, S, H] bfloat16

        Returns:
            (prefill_audio_bytes, prefill_eos): wav bytes from prefill step
            and whether the first codec token is EOS.
        """
        self._kv_pool.init_kv_tensors(slot)

        if self._fused_engine is None:
            slot.past_len = int(prefill_embeds.shape[1])
            slot.c2w_conv_states = []
            slot.c2w_transconv_states = []
            return None, False

        seq = int(prefill_embeds.shape[1])

        inputs = self._build_fused_inputs(
            input_embeds=prefill_embeds.to(self._config.dtype),
            slots=[slot],
            batched_talker_kv=None,
            past_seq_lens=None,
            use_dummy_kv=True,
        )

        out_names = self._build_output_names()

        stream = self._prefill_stream
        prefill_ctx = self._prefill_context
        with torch.cuda.stream(stream):
            raw = self._fused_engine.infer(
                inputs, out_names, stream, context=prefill_ctx,
            )

        stream.synchronize()

        talker_kv = raw.get("talker_present_kv")
        if talker_kv is not None:
            stripped = talker_kv[:, :, :, 1:, :].contiguous()
            if self._kv_pool._preallocate:
                self._kv_pool.scatter_prefill_kv(slot.slot_id, stripped, seq)
            else:
                slot.talker_kv = stripped
        slot.past_len = seq

        c2w_kv = raw.get("c2w_present_kv")
        if c2w_kv is not None:
            c2w_kv = c2w_kv[:, :, :, 1:, :].contiguous()
            if self._kv_pool._preallocate:
                self._kv_pool.scatter_prefill_c2w_kv(slot.slot_id, c2w_kv)
            slot.c2w_kv = c2w_kv
        slot.c2w_conv_states = [raw[n].clone() for n in self._c2w_conv_output_names]
        slot.c2w_transconv_states = [raw[n].clone() for n in self._c2w_transconv_output_names]
        slot.init_pingpong_buffers()

        slot.frame_idx = 1
        slot.token_counts = raw.get(
            "updated_token_counts",
            torch.zeros(1, self._config.codec_vocab_size,
                        device=self._device, dtype=torch.int64),
        )
        codec_sum = raw.get("codec_sum")
        if codec_sum is not None:
            slot.next_embed = codec_sum

        wav = raw.get("wav")
        full_codec = raw.get("full_codec")
        prefill_audio: Optional[bytes] = None
        prefill_eos = False
        if wav is not None:
            prefill_audio = wav.cpu().float().reshape(-1).numpy().tobytes()
        if full_codec is not None and int(full_codec[0, 0].item()) == self._codec_eos_id:
            prefill_eos = True
        return prefill_audio, prefill_eos

    # ------------------------------------------------------------------
    # Decode step (async / pipelined)
    # ------------------------------------------------------------------

    def launch_decode_step(self, slots: List[SlotKVState]) -> GPUFuture:
        """Launch one fused decode step for a batch.  Returns immediately.

        Uses the pre-allocated KV pool for zero-copy gather (no pad+cat).
        The CUDA kernels run on self._compute_stream.  Call future.wait()
        to synchronize and get results.
        """
        batch_size = len(slots)

        input_embeds = torch.cat(
            [s.next_embed.to(self._config.dtype) for s in slots], dim=0,
        )

        original_past_lens = [s.past_len for s in slots]

        if self._fused_engine is None:
            for s in slots:
                s.past_len += 1
                s.frame_idx += 1
            return GPUFuture(_slots=slots)

        slot_ids = [s.slot_id for s in slots]
        max_past_len = max(original_past_lens) if original_past_lens else 0
        if max_past_len == 0:
            max_past_len = 1

        if self._kv_pool is not None and self._kv_pool._preallocate:
            batched_talker_kv = self._kv_pool.gather_talker_kv(
                slot_ids, max_past_len,
            )
            past_seq_lens = torch.tensor(
                original_past_lens, device=self._device, dtype=torch.long,
            )
        else:
            session_kv = [s.talker_kv for s in slots]
            batched_talker_kv, past_seq_lens = pad_packed_kv(
                session_kv, device=self._device, dtype=self._config.dtype,
            )

        padded_past_len = int(batched_talker_kv.shape[3]) if batched_talker_kv is not None else 0

        inputs = self._build_fused_inputs(
            input_embeds=input_embeds,
            slots=slots,
            batched_talker_kv=batched_talker_kv,
            past_seq_lens=past_seq_lens,
            use_dummy_kv=False,
        )

        out_names = self._build_output_names()

        # Build ping-pong output overrides: TRT writes directly into
        # each slot's write buffers, avoiding post-step clone/copy.
        output_overrides = self._build_pingpong_overrides(slots)

        with torch.cuda.stream(self._compute_stream):
            raw = self._fused_engine.infer(
                inputs, out_names, self._compute_stream,
                output_overrides=output_overrides,
            )

        return GPUFuture(
            _compute_stream=self._compute_stream,
            _raw=raw,
            _slots=slots,
            _original_past_lens=original_past_lens,
            _padded_past_len=padded_past_len,
            _seq=1,
            _c2w_conv_output_names=self._c2w_conv_output_names,
            _c2w_transconv_output_names=self._c2w_transconv_output_names,
            _codec_eos_id=self._codec_eos_id,
            _used_pingpong=output_overrides is not None,
        )

    def _build_pingpong_overrides(
        self, slots: List[SlotKVState],
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Build output_overrides dict for ping-pong zero-copy.

        For batch=1: TRT writes directly into the slot's write buffer.
            After the step, flip_c2w_buffers() swaps read↔write — zero copy.
        For batch>1: returns None (no overrides).  TRT writes to its own
            internal buffer; the engine loop then uses copy_c2w_and_flip()
            to scatter results into each slot's pre-allocated write buffer
            and flip — avoids per-step allocation while supporting
            heterogeneous slot ordering.
        """
        if len(slots) != 1:
            return None

        slot = slots[0]
        if not slot.pingpong_ready:
            return None

        overrides: Dict[str, torch.Tensor] = {}
        for idx, name in enumerate(self._c2w_conv_output_names):
            overrides[name] = slot._c2w_conv_write[idx]
        for idx, name in enumerate(self._c2w_transconv_output_names):
            overrides[name] = slot._c2w_transconv_write[idx]
        return overrides

    # ------------------------------------------------------------------
    # Input / output name builders
    # ------------------------------------------------------------------

    def _build_fused_inputs(
        self,
        input_embeds: torch.Tensor,
        slots: List[SlotKVState],
        batched_talker_kv: Optional[torch.Tensor],
        past_seq_lens: Optional[torch.Tensor],
        use_dummy_kv: bool,
    ) -> Dict[str, torch.Tensor]:
        """Build the full input dict for the fused TRT engine.

        Uses packed KV format: single tensor per cache type.
        """
        cfg = self._config
        batch = int(input_embeds.shape[0])
        seq = int(input_embeds.shape[1])

        if use_dummy_kv:
            if past_seq_lens is None:
                past_seq_lens = uniform_past_seq_lens(batch, 0, self._device)
            attn_bias = padded_attention_bias(
                past_seq_lens, seq, _FUSED_DUMMY_PAST_LEN,
                self._device, cfg.dtype,
            )
        else:
            padded_past_len = int(batched_talker_kv.shape[3]) if batched_talker_kv is not None else 0
            attn_bias = padded_attention_bias(
                past_seq_lens, seq, padded_past_len,
                self._device, cfg.dtype,
            )

        position_ids = torch.stack([
            torch.arange(s.past_len, s.past_len + seq,
                         device=self._device, dtype=torch.int64)
            .unsqueeze(0).expand(3, seq)
            for s in slots
        ], dim=0).unsqueeze(-1)

        cache_position = torch.stack([
            torch.full((FUSED_CHUNK_T,), s.frame_idx, device=self._device, dtype=torch.float32)
            for s in slots
        ], dim=0)

        tc = torch.cat([
            s.token_counts if s.token_counts is not None
            else torch.zeros(1, cfg.codec_vocab_size,
                             device=self._device, dtype=torch.int64)
            for s in slots
        ], dim=0)

        if self._do_sample:
            gumbel = torch.rand(
                batch, cfg.logits_topk,
                device=self._device, dtype=torch.float32,
            ).clamp(1e-8, 1.0)
            gumbel = -torch.log(-torch.log(gumbel))
            temperature = torch.full(
                (batch, 1), self._temperature,
                device=self._device, dtype=torch.float32,
            )
        else:
            gumbel = torch.zeros(
                batch, cfg.logits_topk,
                device=self._device, dtype=torch.float32,
            )
            temperature = torch.ones(
                batch, 1, device=self._device, dtype=torch.float32,
            )
        penalty = torch.full(
            (batch, 1), self._repetition_penalty,
            device=self._device, dtype=torch.float32,
        )

        d: Dict[str, torch.Tensor] = {
            "input_embeds": input_embeds.contiguous(),
            "position_ids": position_ids.contiguous(),
            "attention_bias": attn_bias.contiguous(),
            "token_counts": tc.contiguous(),
            "gumbel_noise": gumbel.contiguous(),
            "temperature": temperature.contiguous(),
            "penalty": penalty.contiguous(),
            "cache_position": cache_position.contiguous(),
        }

        # --- Talker KV (packed) ---
        if use_dummy_kv:
            d["talker_past_kv"] = torch.zeros(
                batch, cfg.num_layers * 2, cfg.kv_heads,
                _FUSED_DUMMY_PAST_LEN, cfg.head_dim,
                device=self._device, dtype=cfg.dtype,
            )
        else:
            d["talker_past_kv"] = batched_talker_kv.contiguous()

        # --- C2W KV (packed) with sliding window safety clamp ---
        # Relationship: c2w_attention_bias key_dim = c2w_past_len + FUSED_CHUNK_T
        # TRT profile max for c2w_past_kv dim3 = c2w_sliding_window - FUSED_CHUNK_T
        # because attention key_total = past + chunk_T <= c2w_sliding_window.
        # Different slots may be at different decode steps, so c2w_kv lengths
        # can differ.  We pad to the max length and mask padding in attn bias.
        c2w_window = cfg.c2w_sliding_window
        c2w_max_past = c2w_window - FUSED_CHUNK_T

        per_slot_c2w_lens = []
        for s in slots:
            if s.c2w_kv is not None:
                per_slot_c2w_lens.append(
                    min(int(s.c2w_kv.shape[3]), c2w_max_past)
                )
            else:
                per_slot_c2w_lens.append(0)
        c2w_past_len = max(per_slot_c2w_lens) if per_slot_c2w_lens else 0
        if c2w_past_len < 1:
            c2w_past_len = 1

        c2w_key_total = c2w_past_len + FUSED_CHUNK_T

        c2w_attn = torch.zeros(
            batch, 1, FUSED_CHUNK_T, c2w_key_total,
            device=self._device, dtype=cfg.dtype,
        )
        for bi, sl in enumerate(per_slot_c2w_lens):
            pad_cols = c2w_past_len - sl
            if pad_cols > 0:
                c2w_attn[bi, :, :, :pad_cols] = float("-inf")
        d["c2w_attention_bias"] = c2w_attn.contiguous()

        c2w_d1 = cfg.n_c2w_layers * 2
        c2w_d2 = cfg.c2w_kv_heads
        c2w_head = cfg.c2w_head_dim
        if batch == 1:
            s = slots[0]
            if s.c2w_kv is not None:
                kv = s.c2w_kv
                if kv.shape[3] > c2w_max_past:
                    kv = kv[:, :, :, -c2w_max_past:, :]
            else:
                kv = torch.zeros(
                    1, c2w_d1, c2w_d2, c2w_past_len, c2w_head,
                    device=self._device, dtype=cfg.dtype,
                )
            d["c2w_past_kv"] = kv.contiguous()
        else:
            padded = []
            for s, sl in zip(slots, per_slot_c2w_lens):
                if s.c2w_kv is not None:
                    kv = s.c2w_kv
                    if kv.shape[3] > c2w_max_past:
                        kv = kv[:, :, :, -c2w_max_past:, :]
                else:
                    kv = torch.zeros(
                        1, c2w_d1, c2w_d2, 0, c2w_head,
                        device=self._device, dtype=cfg.dtype,
                    )
                pad_cols = c2w_past_len - sl
                if pad_cols > 0:
                    kv = torch.nn.functional.pad(kv, (0, 0, pad_cols, 0))
                padded.append(kv)
            d["c2w_past_kv"] = torch.cat(padded, dim=0).contiguous()

        # --- C2W conv/transconv states (individual, heterogeneous shapes) ---
        has_conv = bool(slots[0].c2w_conv_states)
        for idx, name in enumerate(self._c2w_conv_input_names):
            if has_conv:
                if batch == 1:
                    d[name] = slots[0].c2w_conv_states[idx].contiguous()
                else:
                    d[name] = torch.cat(
                        [s.c2w_conv_states[idx] for s in slots], dim=0,
                    ).contiguous()
            else:
                shape = list(self._c2w_conv_shapes[idx])
                shape[0] = batch
                d[name] = torch.zeros(shape, device=self._device, dtype=cfg.dtype)

        has_transconv = bool(slots[0].c2w_transconv_states)
        for idx, name in enumerate(self._c2w_transconv_input_names):
            if has_transconv:
                if batch == 1:
                    d[name] = slots[0].c2w_transconv_states[idx].contiguous()
                else:
                    d[name] = torch.cat(
                        [s.c2w_transconv_states[idx] for s in slots], dim=0,
                    ).contiguous()
            else:
                shape = list(self._c2w_transconv_shapes[idx])
                shape[0] = batch
                d[name] = torch.zeros(shape, device=self._device, dtype=cfg.dtype)

        return d

    def _build_output_names(self) -> List[str]:
        names = [
            "wav", "codec_sum", "full_codec", "hidden", "logits",
            "updated_token_counts",
            "talker_present_kv", "c2w_present_kv",
        ]
        names.extend(self._c2w_conv_output_names)
        names.extend(self._c2w_transconv_output_names)
        return names

    # ------------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------------

    def warmup(self, n_rounds: int = 3) -> None:
        """Run dummy inferences to warm up TRT engines and CUDA caches.

        Triggers JIT compilation of TRT tactics and populates GPU L2 cache.
        Runs on the compute stream with a temporary slot.
        """
        if self._fused_engine is None or self._kv_pool is None:
            logger.info("Warmup skipped (no TRT engine)")
            return

        logger.info("Warmup: running %d rounds ...", n_rounds)
        cfg = self._config
        dummy_slot = self._kv_pool.allocate("__warmup__")
        if dummy_slot is None:
            logger.warning("Warmup skipped (no free slot)")
            return

        try:
            dummy_embeds = torch.randn(
                1, 4, cfg.hidden_size,
                device=self._device, dtype=cfg.dtype,
            )
            _ = self.prefill(dummy_slot, dummy_embeds)

            dummy_slot.next_embed = torch.randn(
                1, 1, cfg.hidden_size, device=self._device, dtype=torch.float32,
            )

            for i in range(n_rounds):
                future = self.launch_decode_step([dummy_slot])
                output = future.wait()
                if self._kv_pool._preallocate and output.batch_talker_kv is not None:
                    self._kv_pool.scatter_talker_kv(
                        [dummy_slot.slot_id], output.batch_talker_kv,
                        [dummy_slot.past_len], output.padded_past_len, 1,
                    )
                if output.used_pingpong and dummy_slot.pingpong_ready:
                    dummy_slot.flip_c2w_buffers()
                dummy_slot.past_len += 1
                dummy_slot.frame_idx += 1
                if output.codec_sum is not None:
                    dummy_slot.next_embed = output.codec_sum[:1]

            torch.cuda.synchronize(self._device)
            logger.info("Warmup complete (%d rounds)", n_rounds)
        finally:
            self._kv_pool.release(dummy_slot.slot_id)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Release GPU resources."""
        self._fused_engine = None
        self._prefill_context = None
        self._kv_pool = None
        torch.cuda.empty_cache()
        logger.info("Executor shutdown")

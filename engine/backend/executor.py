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
            raise RuntimeError(f"Failed to deserialize TRT engine: {plan_path}")

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
    ) -> Dict[str, torch.Tensor]:
        """Execute inference on the given CUDA stream.

        Uses torch tensors directly — zero copy via data_ptr().
        Skips set_input_shape for tensors whose shape hasn't changed.
        Re-uses output buffers when shapes match previous call.
        """
        ctx = self._context

        for name, tensor in inputs.items():
            tensor = tensor.contiguous()
            shape = tuple(tensor.shape)
            if self._prev_input_shapes.get(name) != shape:
                ctx.set_input_shape(name, shape)
                self._prev_input_shapes[name] = shape
            ctx.set_tensor_address(name, tensor.data_ptr())

        outputs = {}
        for name in output_names:
            shape = tuple(ctx.get_tensor_shape(name))
            dtype_torch = self._output_dtypes.get(name, torch.float32)

            existing = self._output_buffers.get(name)
            if existing is not None and existing.shape == shape and existing.dtype == dtype_torch:
                out_tensor = existing
            else:
                out_tensor = torch.empty(
                    shape, dtype=dtype_torch, device=self._device,
                )
                self._output_buffers[name] = out_tensor

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

        codec_eos_id = 2148
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
        )


@dataclass
class StepOutput:
    """Results from one decode step.

    Talker/C2W KV are kept as batch-level tensors for direct pool scatter
    (no per-slot splitting).  Conv/transconv states are split per-slot
    because they have heterogeneous shapes.
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
        max_seq_len: int = 2048,
        model_config: Optional[ModelConfig] = None,
    ):
        self._engine_dir = Path(engine_dir) if engine_dir else None
        self._weights_dir = Path(weights_dir) if weights_dir else None
        self._device = torch.device("cuda", device_id)
        self._max_batch = max_batch_size
        self._max_seq_len = max_seq_len
        self._config = model_config or ModelConfig()

        self._compute_stream = torch.cuda.Stream(device=self._device)
        self._prefill_stream = torch.cuda.Stream(device=self._device)

        self._fused_engine: Optional[TRTEngine] = None
        self._prefill_context = None
        self._embedding_weights = None
        self._kv_pool: Optional[KVCachePool] = None

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
        if self._engine_dir and (self._engine_dir / "model.plan").exists():
            self._fused_engine = TRTEngine(
                str(self._engine_dir / "model.plan"), self._device,
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

        if self._engine_dir and (self._engine_dir / "triton_manifest.json").exists():
            with open(self._engine_dir / "triton_manifest.json") as f:
                self._manifest = json.load(f)

        self._kv_pool = KVCachePool(
            max_slots=self._max_batch,
            config=self._config,
            device=self._device,
        )
        logger.info("Executor loaded (engine=%s)", "TRT" if self._fused_engine else "stub")

    def set_embedding_weights(self, weights) -> None:
        self._embedding_weights = weights

    def _discover_c2w_io_names(self) -> None:
        """Detect c2w conv/transconv I/O names from the loaded TRT engine."""
        if self._fused_engine is None:
            return
        input_names, output_names = self._fused_engine.get_io_names()
        self._c2w_conv_input_names = sorted(
            n for n in input_names
            if n.startswith("c2w_conv_state_")
        )
        self._c2w_transconv_input_names = sorted(
            n for n in input_names
            if n.startswith("c2w_transconv_overlap_")
        )
        self._c2w_conv_output_names = sorted(
            n for n in output_names
            if n.startswith("c2w_new_conv_state_")
        )
        self._c2w_transconv_output_names = sorted(
            n for n in output_names
            if n.startswith("c2w_new_transconv_overlap_")
        )
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

    # ------------------------------------------------------------------
    # Prefill
    # ------------------------------------------------------------------

    def prefill(
        self,
        slot: SlotKVState,
        prefill_embeds: torch.Tensor,
    ) -> None:
        """Execute prefill for a single session on the dedicated prefill stream.

        Uses a separate CUDA stream so prefill does not block an
        in-flight decode step.  When a separate TRT execution context is
        available, prefill and decode can overlap on different streams.

        Results are written directly to the pre-allocated KV pool via
        scatter (no per-slot tensor references).

        Args:
            slot: the GPU slot to store resulting KV cache
            prefill_embeds: [1, S, H] bfloat16
        """
        self._kv_pool.init_kv_tensors(slot)

        if self._fused_engine is None:
            slot.past_len = int(prefill_embeds.shape[1])
            slot.c2w_conv_states = []
            slot.c2w_transconv_states = []
            return

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
        with torch.cuda.stream(stream):
            raw = self._fused_engine.infer(inputs, out_names, stream)

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
            if self._kv_pool._preallocate:
                self._kv_pool.scatter_prefill_c2w_kv(slot.slot_id, c2w_kv)
            slot.c2w_kv = c2w_kv
        slot.c2w_conv_states = [raw[n] for n in self._c2w_conv_output_names]
        slot.c2w_transconv_states = [raw[n] for n in self._c2w_transconv_output_names]

        slot.frame_idx = 0
        slot.token_counts = raw.get(
            "updated_token_counts",
            torch.zeros(1, self._config.codec_vocab_size,
                        device=self._device, dtype=torch.int64),
        )
        codec_sum = raw.get("codec_sum")
        if codec_sum is not None:
            slot.next_embed = codec_sum

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

        with torch.cuda.stream(self._compute_stream):
            raw = self._fused_engine.infer(inputs, out_names, self._compute_stream)

        return GPUFuture(
            _compute_stream=self._compute_stream,
            _raw=raw,
            _slots=slots,
            _original_past_lens=original_past_lens,
            _padded_past_len=padded_past_len,
            _seq=1,
            _c2w_conv_output_names=self._c2w_conv_output_names,
            _c2w_transconv_output_names=self._c2w_transconv_output_names,
        )

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
            torch.full((3, seq), s.past_len, device=self._device, dtype=torch.int64)
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

        gumbel = torch.rand(batch, cfg.codec_vocab_size,
                            device=self._device, dtype=torch.float32)
        gumbel = -(-gumbel.clamp(min=1e-8).log()).clamp(min=1e-8).log()
        temperature = torch.ones(batch, 1, device=self._device, dtype=torch.float32)
        penalty = torch.ones(batch, 1, device=self._device, dtype=torch.float32)

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
        c2w_past_len = 0
        if slots[0].c2w_kv is not None:
            c2w_past_len = int(slots[0].c2w_kv.shape[3])
        if c2w_past_len > cfg.c2w_sliding_window:
            logger.warning(
                "C2W KV past_len %d exceeds sliding window %d, clamping",
                c2w_past_len, cfg.c2w_sliding_window,
            )
            c2w_past_len = cfg.c2w_sliding_window
        c2w_key_total = min(c2w_past_len + FUSED_CHUNK_T, cfg.c2w_sliding_window)
        if c2w_key_total <= 0:
            c2w_key_total = 1

        d["c2w_attention_bias"] = torch.zeros(
            batch, 1, FUSED_CHUNK_T, c2w_key_total,
            device=self._device, dtype=cfg.dtype,
        ).contiguous()

        if slots[0].c2w_kv is not None:
            def _clamp_c2w(kv: torch.Tensor) -> torch.Tensor:
                if kv.shape[3] > cfg.c2w_sliding_window:
                    return kv[:, :, :, -cfg.c2w_sliding_window:, :].contiguous()
                return kv

            if batch == 1:
                d["c2w_past_kv"] = _clamp_c2w(slots[0].c2w_kv).contiguous()
            else:
                d["c2w_past_kv"] = torch.cat(
                    [_clamp_c2w(s.c2w_kv) for s in slots], dim=0,
                ).contiguous()
        else:
            d["c2w_past_kv"] = torch.zeros(
                batch, cfg.n_c2w_layers * 2, cfg.c2w_kv_heads,
                1, cfg.c2w_head_dim,
                device=self._device, dtype=cfg.dtype,
            )

        # --- C2W conv/transconv states (individual, heterogeneous shapes) ---
        if slots[0].c2w_conv_states:
            for idx, name in enumerate(self._c2w_conv_input_names):
                if batch == 1:
                    d[name] = slots[0].c2w_conv_states[idx].contiguous()
                else:
                    d[name] = torch.cat(
                        [s.c2w_conv_states[idx] for s in slots], dim=0,
                    ).contiguous()

        if slots[0].c2w_transconv_states:
            for idx, name in enumerate(self._c2w_transconv_input_names):
                if batch == 1:
                    d[name] = slots[0].c2w_transconv_states[idx].contiguous()
                else:
                    d[name] = torch.cat(
                        [s.c2w_transconv_states[idx] for s in slots], dim=0,
                    ).contiguous()

        return d

    def _build_output_names(self) -> List[str]:
        names = [
            "wav", "codec_sum", "full_codec", "logits",
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
            self.prefill(dummy_slot, dummy_embeds)

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

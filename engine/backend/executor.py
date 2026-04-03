"""Executor: owns TRT engines and CUDA streams, executes GPU compute.

Replaces Triton BLS with direct TRT plan execution via torch, eliminating:
  - pb_utils.InferenceRequest overhead (~0.5ms/step)
  - Triton scheduling latency (~0.3ms/step)
  - dlpack round-trip for every KV tensor

Key optimisation: launch_decode_step() is asynchronous.  It enqueues CUDA
kernels on a dedicated stream and returns a GPUFuture immediately, letting
the caller process *previous* step results on CPU while GPU is busy.
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
    pad_talker_past_kv,
    padded_attention_bias,
    split_batched_kv,
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

    Supports two loading modes:
      1. torch_tensorrt (preferred): loads .plan directly
      2. tensorrt + torch: manual engine loading with zero-copy I/O
    """

    def __init__(self, plan_path: str, device: torch.device):
        self._plan_path = plan_path
        self._device = device
        self._engine = None
        self._context = None

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
        logger.info("Loaded TRT engine: %s (%d I/O tensors)",
                     plan_path.name, self._engine.num_io_tensors)

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
        """
        import tensorrt as trt

        ctx = self._context

        for name, tensor in inputs.items():
            tensor = tensor.contiguous()
            ctx.set_input_shape(name, tuple(tensor.shape))
            ctx.set_tensor_address(name, tensor.data_ptr())

        outputs = {}
        for name in output_names:
            shape = ctx.get_tensor_shape(name)
            dtype_trt = self._engine.get_tensor_dtype(name)
            dtype_torch = self._trt_to_torch_dtype(dtype_trt)
            out_tensor = torch.empty(
                tuple(shape), dtype=dtype_torch, device=self._device,
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
    _c2w_stream: Any = None
    _raw: Dict[str, torch.Tensor] = field(default_factory=dict)
    _slots: List[SlotKVState] = field(default_factory=list)
    _original_past_lens: List[int] = field(default_factory=list)
    _padded_past_len: int = 0
    _seq: int = 1
    _num_layers: int = 28
    _num_c2w_states: int = 37
    _c2w_state_output_names: List[str] = field(default_factory=list)

    def wait(self) -> StepOutput:
        """Synchronize GPU and extract per-session results."""
        if self._compute_stream is not None:
            self._compute_stream.synchronize()
        if self._c2w_stream is not None:
            self._c2w_stream.synchronize()

        batch_size = len(self._slots)
        raw = self._raw

        wav = raw.get("wav")
        codec_sum = raw.get("codec_sum")
        full_codec = raw.get("full_codec")
        logits = raw.get("logits")
        updated_tc = raw.get("updated_token_counts")

        kv_tensors = []
        for i in range(self._num_layers):
            k = raw.get(f"present_kv_{i}_k")
            v = raw.get(f"present_kv_{i}_v")
            if k is not None:
                kv_tensors.append(k)
            if v is not None:
                kv_tensors.append(v)

        split_kv = split_batched_kv(
            kv_tensors, self._original_past_lens,
            self._padded_past_len, self._seq,
        ) if kv_tensors else [[] for _ in self._slots]

        new_c2w = [raw.get(n) for n in self._c2w_state_output_names]
        split_c2w = []
        for row_idx in range(batch_size):
            split_c2w.append([
                t[row_idx : row_idx + 1] if t is not None else None
                for t in new_c2w
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
            split_kv=split_kv,
            split_c2w=split_c2w,
            codec_sum=codec_sum,
            updated_tc=updated_tc,
        )


@dataclass
class StepOutput:
    """Results from one decode step, already split per session."""
    slots: List[SlotKVState]
    eos_flags: List[bool]
    audio_chunks: List[Optional[bytes]]
    split_kv: List[List[torch.Tensor]]
    split_c2w: List[List[Optional[torch.Tensor]]]
    codec_sum: Optional[torch.Tensor] = None
    updated_tc: Optional[torch.Tensor] = None


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class Executor:
    """Manages TRT engines and CUDA streams for pipelined decode.

    CUDA stream layout:
        compute_stream: Talker decode + Code Predictor
        c2w_stream:     Code2Wav (can overlap with next Talker step)
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
        self._c2w_stream = torch.cuda.Stream(device=self._device)

        self._fused_engine: Optional[TRTEngine] = None
        self._embedding_weights = None  # EmbeddingWeights instance
        self._kv_pool: Optional[KVCachePool] = None

        self._c2w_state_input_names: list[str] = []
        self._c2w_state_output_names: list[str] = []
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
        """Detect c2w_* I/O names from the loaded TRT engine."""
        if self._fused_engine is None:
            return
        input_names, output_names = self._fused_engine.get_io_names()
        self._c2w_state_input_names = sorted(
            n for n in input_names if n.startswith("c2w_")
        )
        self._c2w_state_output_names = sorted(
            n for n in output_names if n.startswith("c2w_")
        )
        logger.info("C2W states: %d inputs, %d outputs",
                     len(self._c2w_state_input_names),
                     len(self._c2w_state_output_names))

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
        """Execute prefill for a single session.

        Args:
            slot: the GPU slot to store resulting KV cache
            prefill_embeds: [1, S, H] bfloat16
        """
        self._kv_pool.init_kv_tensors(slot)

        if self._fused_engine is None:
            slot.past_len = int(prefill_embeds.shape[1])
            slot.kv_tensors = []
            slot.c2w_states = []
            return

        batch = 1
        seq = int(prefill_embeds.shape[1])

        inputs = self._build_fused_inputs(
            input_embeds=prefill_embeds.to(self._config.dtype),
            slots=[slot],
            past_kv_tensors=None,
            past_seq_lens=None,
            use_dummy_kv=True,
        )

        out_names = self._build_output_names()

        with torch.cuda.stream(self._compute_stream):
            raw = self._fused_engine.infer(inputs, out_names, self._compute_stream)

        self._compute_stream.synchronize()

        kv_tensors = []
        for i in range(self._config.num_layers):
            k = raw[f"present_kv_{i}_k"][:, :, 1:, :].contiguous()
            v = raw[f"present_kv_{i}_v"][:, :, 1:, :].contiguous()
            kv_tensors.append(k)
            kv_tensors.append(v)

        slot.kv_tensors = kv_tensors
        slot.past_len = seq
        slot.frame_idx = 0
        slot.c2w_states = [raw[n] for n in self._c2w_state_output_names]
        slot.token_counts = raw.get("updated_token_counts",
            torch.zeros(1, self._config.codec_vocab_size,
                        device=self._device, dtype=torch.int64))

        codec_sum = raw.get("codec_sum")
        if codec_sum is not None:
            slot.next_embed = codec_sum

    # ------------------------------------------------------------------
    # Decode step (async / pipelined)
    # ------------------------------------------------------------------

    def launch_decode_step(self, slots: List[SlotKVState]) -> GPUFuture:
        """Launch one fused decode step for a batch.  Returns immediately.

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

        batched_past_kv, past_seq_lens = pad_talker_past_kv(
            [s.kv_tensors for s in slots],
            device=self._device,
            dtype=self._config.dtype,
        )
        padded_past_len = int(past_seq_lens.max().item()) if batch_size > 0 else 0

        inputs = self._build_fused_inputs(
            input_embeds=input_embeds,
            slots=slots,
            past_kv_tensors=batched_past_kv,
            past_seq_lens=past_seq_lens,
            use_dummy_kv=False,
        )

        out_names = self._build_output_names()

        with torch.cuda.stream(self._compute_stream):
            raw = self._fused_engine.infer(inputs, out_names, self._compute_stream)

        return GPUFuture(
            _compute_stream=self._compute_stream,
            _c2w_stream=self._c2w_stream,
            _raw=raw,
            _slots=slots,
            _original_past_lens=original_past_lens,
            _padded_past_len=padded_past_len,
            _seq=1,
            _num_layers=self._config.num_layers,
            _num_c2w_states=self._config.num_c2w_states,
            _c2w_state_output_names=self._c2w_state_output_names,
        )

    # ------------------------------------------------------------------
    # Input / output name builders
    # ------------------------------------------------------------------

    def _build_fused_inputs(
        self,
        input_embeds: torch.Tensor,
        slots: List[SlotKVState],
        past_kv_tensors: Optional[list[torch.Tensor]],
        past_seq_lens: Optional[torch.Tensor],
        use_dummy_kv: bool,
    ) -> Dict[str, torch.Tensor]:
        """Build the full input dict for the fused TRT engine."""
        batch = int(input_embeds.shape[0])
        seq = int(input_embeds.shape[1])

        if use_dummy_kv:
            if past_seq_lens is None:
                past_seq_lens = uniform_past_seq_lens(batch, 0, self._device)
            attn_bias = padded_attention_bias(
                past_seq_lens, seq, _FUSED_DUMMY_PAST_LEN,
                self._device, self._config.dtype,
            )
        else:
            padded_past_len = int(past_kv_tensors[0].shape[2]) if past_kv_tensors else 0
            attn_bias = padded_attention_bias(
                past_seq_lens, seq, padded_past_len,
                self._device, self._config.dtype,
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
            else torch.zeros(1, self._config.codec_vocab_size,
                             device=self._device, dtype=torch.int64)
            for s in slots
        ], dim=0)

        gumbel = torch.rand(batch, self._config.codec_vocab_size,
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

        c2w_past_len = int(slots[0].c2w_states[0].shape[2]) if (
            slots[0].c2w_states) else 0
        c2w_key_total = min(c2w_past_len + FUSED_CHUNK_T,
                            self._config.code2wav_sliding_window)
        if c2w_key_total <= 0:
            c2w_key_total = 1
        d["c2w_attention_bias"] = torch.zeros(
            batch, 1, FUSED_CHUNK_T, c2w_key_total,
            device=self._device, dtype=self._config.dtype,
        ).contiguous()

        if use_dummy_kv:
            for i in range(self._config.num_layers):
                dummy = torch.zeros(
                    batch, self._config.kv_heads, _FUSED_DUMMY_PAST_LEN,
                    self._config.head_dim,
                    device=self._device, dtype=self._config.dtype,
                )
                d[f"past_kv_{i}_k"] = dummy
                d[f"past_kv_{i}_v"] = dummy.clone()
        else:
            for i in range(self._config.num_layers):
                d[f"past_kv_{i}_k"] = past_kv_tensors[2 * i].contiguous()
                d[f"past_kv_{i}_v"] = past_kv_tensors[2 * i + 1].contiguous()

        if slots[0].c2w_states:
            for in_name, st in zip(self._c2w_state_input_names, slots[0].c2w_states):
                if batch == 1:
                    d[in_name] = st.contiguous()
                else:
                    batched_st = torch.cat(
                        [s.c2w_states[self._c2w_state_input_names.index(in_name)]
                         for s in slots], dim=0,
                    ).contiguous()
                    d[in_name] = batched_st

        return d

    def _build_output_names(self) -> List[str]:
        names = ["wav", "codec_sum", "full_codec", "logits", "updated_token_counts"]
        for i in range(self._config.num_layers):
            names.append(f"present_kv_{i}_k")
            names.append(f"present_kv_{i}_v")
        names.extend(self._c2w_state_output_names)
        return names

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Release GPU resources."""
        self._fused_engine = None
        self._kv_pool = None
        torch.cuda.empty_cache()
        logger.info("Executor shutdown")

"""
TalkerRunner: Pure TensorRT engine wrapper with manual KV Cache management.

Uses two engines produced by trtexec (no TRT-LLM):
  - talker_context.engine: prefill, outputs last_hidden, last_logits, present_kv_*
  - talker_decode_fused.engine: single decode step (Talker + CP + codec_sum), uses past_kv_*

context(inputs_embeds, position_ids) → (last_hidden, last_logits), fills KV cache.
decode_step(input_embeds, position_id) → (codec_sum, full_codec, logits), updates KV cache.

Runs inside Triton BLS (TensorRT + PyTorch, no tensorrt_llm).
"""

import logging
from pathlib import Path

import torch
import tensorrt as trt

logger = logging.getLogger("talker_runner")

TRT_TO_TORCH = {
    trt.float32: torch.float32,
    trt.float16: torch.float16,
    trt.bfloat16: torch.bfloat16,
    trt.int32: torch.int32,
    trt.int64: torch.int64,
    trt.bool: torch.bool,
}


def _load_engine(engine_path: Path, logger_trt=None):
    logger_trt = logger_trt or trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger_trt)
    with open(engine_path, "rb") as f:
        return runtime.deserialize_cuda_engine(f.read())


def _get_shape(tensor_name: str, engine: trt.ICudaEngine, is_shape_tensor: bool = False):
    if is_shape_tensor:
        return engine.get_tensor_shape(tensor_name)
    return engine.get_tensor_shape(tensor_name)


class TalkerRunner:
    """
    Pure TRT runner: context engine (prefill) + decode engine (fused step).
    Manual KV cache: filled by context(), updated each decode_step().
    """

    def __init__(self, engine_dir: str, device: int = 0):
        self.device = torch.device(f"cuda:{device}")
        self._seq_len = 0
        engine_dir = Path(engine_dir)

        ctx_engine_path = engine_dir / "talker_context.engine"
        dec_engine_path = engine_dir / "talker_decode_fused.engine"
        if not ctx_engine_path.exists():
            raise FileNotFoundError(f"Context engine not found: {ctx_engine_path}")
        if not dec_engine_path.exists():
            raise FileNotFoundError(f"Decode engine not found: {dec_engine_path}")

        trt_logger = trt.Logger(trt.Logger.WARNING)
        self._ctx_engine = _load_engine(ctx_engine_path, trt_logger)
        self._dec_engine = _load_engine(dec_engine_path, trt_logger)
        self._ctx_context = self._ctx_engine.create_execution_context()
        self._dec_context = self._dec_engine.create_execution_context()
        self.stream = torch.cuda.Stream(device=self.device)

        # Infer dimensions from context engine (last_hidden, present_kv_0_k)
        self.hidden_size = self._ctx_engine.get_tensor_shape("last_hidden")[-1]
        pk = self._ctx_engine.get_tensor_shape("present_kv_0_k")
        # present_kv_0_k: [B, num_kv_heads, S, head_dim]
        self.num_kv_heads = pk[1]
        self.head_dim = pk[3]
        self.num_layers = 0
        for i in range(self._ctx_engine.num_io_tensors):
            name = self._ctx_engine.get_tensor_name(i)
            if name and name.startswith("present_kv_") and name.endswith("_k"):
                idx = int(name.replace("present_kv_", "").replace("_k", ""))
                self.num_layers = max(self.num_layers, idx + 1)
        if self.num_layers == 0:
            self.num_layers = 28

        self.vocab_size = self._ctx_engine.get_tensor_shape("last_logits")[-1]
        self.max_batch_size = 8
        self.max_seq_len = 4096

        # Pre-allocate KV cache: list of (k, v) per layer, each [B, kv_heads, max_seq, head_dim]
        self._kv_k = []
        self._kv_v = []
        for _ in range(self.num_layers):
            self._kv_k.append(
                torch.zeros(
                    self.max_batch_size, self.num_kv_heads, self.max_seq_len, self.head_dim,
                    dtype=torch.bfloat16, device=self.device,
                )
            )
            self._kv_v.append(
                torch.zeros(
                    self.max_batch_size, self.num_kv_heads, self.max_seq_len, self.head_dim,
                    dtype=torch.bfloat16, device=self.device,
                )
            )

        logger.info(
            f"Pure TRT engines loaded: layers={self.num_layers}, H={self.hidden_size}, "
            f"kv_heads={self.num_kv_heads}, head_dim={self.head_dim}, vocab={self.vocab_size}"
        )

    def context(self, input_embeds: torch.Tensor, position_ids: torch.Tensor):
        """
        Prefill: run context engine, fill KV cache.

        Args:
            input_embeds: [B, S, H] bf16/fp32
            position_ids: [3, B, S] int64

        Returns:
            last_hidden: [B, 1, H], last_logits: [B, 1, V]
        """
        self.reset()
        B, S, H = input_embeds.shape
        self._seq_len = S

        inp_emb = input_embeds.to(dtype=torch.bfloat16, device=self.device)
        pos_ids = position_ids.to(device=self.device)
        if pos_ids.dtype != torch.int64:
            pos_ids = pos_ids.long()

        self._ctx_context.set_input_shape("input_embeds", tuple(inp_emb.shape))
        self._ctx_context.set_tensor_address("input_embeds", inp_emb.data_ptr())
        self._ctx_context.set_input_shape("position_ids", tuple(pos_ids.shape))
        self._ctx_context.set_tensor_address("position_ids", pos_ids.data_ptr())

        out_hidden = torch.empty(B, 1, H, dtype=torch.bfloat16, device=self.device)
        out_logits = torch.empty(B, 1, self.vocab_size, dtype=torch.float32, device=self.device)
        self._ctx_context.set_tensor_address("last_hidden", out_hidden.data_ptr())
        self._ctx_context.set_tensor_address("last_logits", out_logits.data_ptr())

        for i in range(self.num_layers):
            self._ctx_context.set_tensor_address(
                f"present_kv_{i}_k", self._kv_k[i].data_ptr()
            )
            self._ctx_context.set_tensor_address(
                f"present_kv_{i}_v", self._kv_v[i].data_ptr()
            )

        ok = self._ctx_context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        if not ok:
            raise RuntimeError("Context TRT engine execution failed")

        return out_hidden, out_logits

    def decode_step(self, input_embeds: torch.Tensor, position_id: torch.Tensor):
        """
        Fused decode step: 1 token → Talker + CP + codec_sum.

        Args:
            input_embeds: [B, 1, H]
            position_id: [3, B, 1] int64 (current position)

        Returns:
            codec_sum: [B, 1, H], full_codec: [B, 16], logits: [B, 1, V]
        """
        B = input_embeds.shape[0]
        inp = input_embeds.to(dtype=torch.bfloat16, device=self.device)
        pos = position_id.to(device=self.device)
        if pos.dtype != torch.int64:
            pos = pos.long()

        self._dec_context.set_input_shape("input_embeds", tuple(inp.shape))
        self._dec_context.set_tensor_address("input_embeds", inp.data_ptr())
        self._dec_context.set_input_shape("position_ids", tuple(pos.shape))
        self._dec_context.set_tensor_address("position_ids", pos.data_ptr())

        kv_seq = self._seq_len
        for i in range(self.num_layers):
            self._dec_context.set_input_shape(
                f"past_kv_{i}_k", (B, self.num_kv_heads, kv_seq, self.head_dim)
            )
            self._dec_context.set_input_shape(
                f"past_kv_{i}_v", (B, self.num_kv_heads, kv_seq, self.head_dim)
            )
            self._dec_context.set_tensor_address(f"past_kv_{i}_k", self._kv_k[i].data_ptr())
            self._dec_context.set_tensor_address(f"past_kv_{i}_v", self._kv_v[i].data_ptr())

        codec_sum = torch.empty(B, 1, self.hidden_size, dtype=torch.bfloat16, device=self.device)
        full_codec = torch.empty(B, 16, dtype=torch.int64, device=self.device)
        hidden = torch.empty(B, 1, self.hidden_size, dtype=torch.bfloat16, device=self.device)
        logits = torch.empty(B, 1, self.vocab_size, dtype=torch.float32, device=self.device)

        self._dec_context.set_tensor_address("codec_sum", codec_sum.data_ptr())
        self._dec_context.set_tensor_address("full_codec", full_codec.data_ptr())
        self._dec_context.set_tensor_address("hidden", hidden.data_ptr())
        self._dec_context.set_tensor_address("logits", logits.data_ptr())
        for i in range(self.num_layers):
            self._dec_context.set_tensor_address(
                f"present_kv_{i}_k", self._kv_k[i].data_ptr()
            )
            self._dec_context.set_tensor_address(
                f"present_kv_{i}_v", self._kv_v[i].data_ptr()
            )

        ok = self._dec_context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        if not ok:
            raise RuntimeError("Decode TRT engine execution failed")

        self._seq_len += 1
        return codec_sum, full_codec, logits

    @property
    def current_seq_len(self) -> int:
        return self._seq_len

    def reset(self):
        """Clear KV cache for next request."""
        for k in self._kv_k:
            k.zero_()
        for v in self._kv_v:
            v.zero_()
        self._seq_len = 0

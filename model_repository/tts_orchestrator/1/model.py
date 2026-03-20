"""
TTS Orchestrator — Triton Python BLS Backend.

Unified pipeline (same BLS for ONNX and TensorRT backends):
  1. Session init (Speaker Encoder / Speech Tokenizer via BLS)
  2. Prefill construction (in-process torch embedding; no BLS for embedders)
  3. BLS talker_unified: prefill (past_kv empty S_past=0) → codec_sum, full_codec, logits, KV
  4. Decode loop: BLS talker_unified(next_embed, position_id, KV) → codec_sum, full_codec, logits, updated KV
  5. Code2Wav (BLS) for chunked audio

Talker path uses torch + DLPack zero-copy (BF16 I/O); KV cache never leaves GPU.

Error isolation (10.5): per-request try/except; OOM and timeout handled; client disconnect
checked via response_sender.is_cancelled() in the decode loop.

Phase 3 multi-session: see session_manager.py (SessionManager, BatchScheduler, FlowState).
"""

import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

import triton_python_backend_utils as pb_utils

logger = logging.getLogger("tts_orchestrator")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
if _MODEL_DIR not in sys.path:
    sys.path.insert(0, _MODEL_DIR)

# Stateful code2wav: chunk_T=4 fixed; 37 state tensor shapes (batch=1, past_kv_len=0).
# Order matches code2wav_streaming.get_initial_state_shapes (16 KV + 17 conv + 4 transconv).
_CODE2WAV_STATE_SHAPES = [
    (1, 16, 0, 64), (1, 16, 0, 64), (1, 16, 0, 64), (1, 16, 0, 64),
    (1, 16, 0, 64), (1, 16, 0, 64), (1, 16, 0, 64), (1, 16, 0, 64),
    (1, 16, 0, 64), (1, 16, 0, 64), (1, 16, 0, 64), (1, 16, 0, 64),
    (1, 16, 0, 64), (1, 16, 0, 64), (1, 16, 0, 64), (1, 16, 0, 64),
    (1, 512, 2), (1, 1024, 6), (1, 1024, 6), (1, 1024, 6),
    (1, 768, 6), (1, 768, 18), (1, 768, 54), (1, 384, 6), (1, 384, 18), (1, 384, 54),
    (1, 192, 6), (1, 192, 18), (1, 192, 54), (1, 96, 6), (1, 96, 18), (1, 96, 54), (1, 96, 6),
    (1, 768, 8), (1, 384, 5), (1, 192, 4), (1, 96, 3),
]
CODE2WAV_CHUNK_T = 4
SAMPLES_PER_CODEC_FRAME = 1920

_MODEL_TYPE_ALLOWED_TASKS = {
    "base": {"voice_clone_icl", "voice_clone_xvec"},
    "custom_voice": {"custom_voice"},
    "voice_design": {"voice_design"},
}

class TritonPythonModel:

    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])
        params = self.model_config.get("parameters", {})

        self.variant = params.get("model_variant", {}).get("string_value", "unknown")
        weights_dir = params.get("weights_dir", {}).get("string_value", "")
        tokenizer_dir = params.get("tokenizer_dir", {}).get("string_value", "")

        model_dir = Path(_MODEL_DIR)
        if not weights_dir:
            weights_dir = str(model_dir / "weights")
        if not tokenizer_dir:
            tokenizer_dir = str(model_dir / "tokenizer")

        self.device_id = 0
        self.device = torch.device("cuda", self.device_id)
        self.max_steps = int(params.get("max_decode_steps", {}).get(
            "string_value", "2000"))
        # Stateful code2wav: fixed chunk_T=4; legacy params kept for config compatibility
        self.audio_chunk_threshold = int(params.get("audio_chunk_frames", {}).get(
            "string_value", "25"))
        self.first_chunk_frames = int(params.get("first_chunk_frames", {}).get(
            "string_value", "4"))
        self.request_timeout_sec = float(params.get("request_timeout_sec", {}).get(
            "string_value", "120"))

        from prefill_builder import EmbeddingWeights, PrefillBuilder, parse_task_type
        self._parse_task_type = parse_task_type

        logger.info(f"Loading weights from {weights_dir} ...")
        self.weights = EmbeddingWeights(weights_dir, device_id=self.device_id)
        self._tts_model_type = self.weights.config.get("tts_model_type", "unknown")
        self.num_layers = int(self.weights.config.get("talker_num_layers", 28))
        self.kv_heads = int(self.weights.config.get("talker_num_kv_heads", 8))
        talker_h = int(self.weights.config.get("talker_hidden_size", 2048))
        talker_heads = int(self.weights.config.get("talker_num_heads", 16))
        self.head_dim = talker_h // talker_heads if talker_heads else 128

        # tts_pad_embed already torch BF16 on GPU (from EmbeddingWeights)
        self._tts_pad_embed_torch = self.weights.tts_pad_embed

        logger.info(f"Loading tokenizer from {tokenizer_dir} ...")
        from lightweight_tokenizer import load_lightweight_tokenizer
        self.tokenizer = load_lightweight_tokenizer(tokenizer_dir)
        if self.tokenizer is None:
            raise RuntimeError(
                f"Lightweight tokenizer failed to load from {tokenizer_dir}. "
                "Ensure tokenizer.json (or vocab.json + merges.txt) is present."
            )
        logger.info("Tokenizer loaded (lightweight: tokenizers only)")

        self.prefill_builder = PrefillBuilder(self.weights, self.tokenizer)

        # Detect talker_unified backend and dtype: TRT (bf16/fp16/fp32) vs ONNX (FP32)
        repo_root = Path(_MODEL_DIR).resolve().parent.parent
        talker_config = repo_root / "talker_unified" / "config.pbtxt"
        self._talker_backend = "onnxruntime"
        self._talker_dtype = torch.float32
        if talker_config.exists():
            raw = talker_config.read_text()
            if "backend: \"tensorrt\"" in raw or 'backend: "tensorrt"' in raw:
                self._talker_backend = "tensorrt"
                # Infer TRT engine dtype from config (TYPE_BF16 / TYPE_FP16 / TYPE_FP32)
                if "TYPE_BF16" in raw:
                    self._talker_dtype = torch.bfloat16
                elif "TYPE_FP16" in raw:
                    self._talker_dtype = torch.float16
                else:
                    self._talker_dtype = torch.float32
            else:
                self._talker_dtype = torch.float32
        # Code2wav: infer dtype from config (TYPE_BF16 / TYPE_FP16 / TYPE_FP32).
        # TRT engine uses engine_dtype from .engine_dtype; ONNX uses FP32 from graph.
        # Env OVERRIDE_CODE2WAV_BF16=1 forces BF16 when detection fails in Docker/etc.
        code2wav_config = repo_root / "code2wav" / "config.pbtxt"
        self._code2wav_dtype = torch.float32
        if os.environ.get("OVERRIDE_CODE2WAV_BF16", "").strip() in ("1", "true", "yes"):
            self._code2wav_dtype = torch.bfloat16
            logger.info("[TTS Orchestrator] code2wav_dtype=BF16 (OVERRIDE_CODE2WAV_BF16 env)")
        elif code2wav_config.exists():
            raw = code2wav_config.read_text()
            if "TYPE_BF16" in raw:
                self._code2wav_dtype = torch.bfloat16
            elif "TYPE_FP16" in raw:
                self._code2wav_dtype = torch.float16
            elif "TYPE_FP32" in raw:
                self._code2wav_dtype = torch.float32
            elif "tensorrt" in raw.lower() and "backend" in raw:
                # TRT config without explicit float type → assume same as talker
                self._code2wav_dtype = self._talker_dtype
            elif self._talker_backend == "tensorrt":
                self._code2wav_dtype = self._talker_dtype
        elif self._talker_backend == "tensorrt":
            self._code2wav_dtype = self._talker_dtype
        logger.info(
            f"[TTS Orchestrator] Initialized: variant={self.variant}, num_layers={self.num_layers}, "
            f"talker_backend={self._talker_backend}, code2wav_dtype={self._code2wav_dtype}"
        )

    def execute(self, requests):
        for request in requests:
            response_sender = request.get_response_sender()
            try:
                self._handle_request(request, response_sender)
            except torch.cuda.OutOfMemoryError as e:
                logger.error(f"Request failed (OOM): {e}")
                self._send_error(response_sender, "out_of_memory: GPU OOM")
            except Exception as e:
                logger.error(f"Request failed: {e}")
                logger.error(traceback.format_exc())
                self._send_error(response_sender, str(e))
            finally:
                response_sender.send(
                    flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)
        return None

    def _validate_request(self, req: dict) -> None:
        """Validate request fields. Raises ValueError with clear message if invalid."""
        task_type_str = req.get("task_type", "voice_design")
        text = req.get("text", "")
        if not (text or "").strip():
            raise ValueError("Request field 'text' is required and must be non-empty")
        try:
            task_type = self._parse_task_type(task_type_str, req.get("x_vector_only", False))
        except ValueError as e:
            raise ValueError(f"Invalid task_type or parameters: {e}") from e
            
        allowed = _MODEL_TYPE_ALLOWED_TASKS.get(self._tts_model_type, set())
        if task_type.value not in allowed:
            raise ValueError(
                f"This instance (variant={self.variant}, type={self._tts_model_type}) "
                f"does not support task_type '{task_type.value}'. "
                f"Supported: {sorted(allowed)}"
            )
            
        if task_type_str and task_type_str.startswith("voice_clone"):
            ref_audio = req.get("ref_audio")
            if not ref_audio or not (ref_audio if isinstance(ref_audio, str) else "").strip():
                raise ValueError("Request field 'ref_audio' (base64) is required for voice_clone task_type")

    def _handle_request(self, request, response_sender):
        req_tensor = pb_utils.get_input_tensor_by_name(request, "request")
        req_json = req_tensor.as_numpy()[0].decode("utf-8")
        req = json.loads(req_json)

        if req.get("action") == "capabilities":
            caps = {
                "variant": self.variant,
                "tts_model_type": self._tts_model_type,
                "tts_model_size": self.weights.config.get("tts_model_size"),
                "supported_task_types": sorted(
                    _MODEL_TYPE_ALLOWED_TASKS.get(self._tts_model_type, set())
                ),
                "supported_languages": sorted(self.weights.codec_language_id.keys()),
                "supported_speakers": sorted(self.weights.spk_id_map.keys()),
                "max_decode_steps": self.max_steps,
                "engine_backend": self._talker_backend,
            }
            self._send_capabilities(response_sender, caps)
            return

        self._validate_request(req)

        task_type_str = req.get("task_type", "voice_design")
        text = req.get("text", "")
        language = req.get("language", "auto")
        speaker = req.get("speaker")
        instruct = req.get("instruct")
        x_vector_only = req.get("x_vector_only", False)

        task_type = self._parse_task_type(task_type_str, x_vector_only)
        logger.info(f"Request: task={task_type.value}, text='{text[:50]}...', lang={language}")

        spk_embedding = None
        ref_codes = None
        if task_type.value.startswith("voice_clone"):
            ref_audio_b64 = req.get("ref_audio")
            try:
                if ref_audio_b64:
                    t0 = time.perf_counter()
                    spk_embedding = self._bls_speaker_encoder(ref_audio_b64)
                    logger.debug(f"speaker_encoder: {(time.perf_counter() - t0) * 1000:.1f}ms")
                if task_type.value == "voice_clone_icl" and ref_audio_b64:
                    t0 = time.perf_counter()
                    ref_codes = self._bls_speech_tokenizer(ref_audio_b64)
                    logger.debug(f"speech_tokenizer_encoder: {(time.perf_counter() - t0) * 1000:.1f}ms")
            except Exception as e:
                raise RuntimeError(f"Voice clone audio processing failed: {e}") from e

        ref_text = req.get("ref_text")

        try:
            inputs_embeds, trailing_text = self.prefill_builder.build(
                task_type=task_type,
                text=text,
                language=language,
                speaker=speaker,
                instruct=instruct,
                spk_embedding=spk_embedding,
                ref_codes=ref_codes,
                ref_text=ref_text,
            )
        except Exception as e:
            raise RuntimeError(f"Prefill build failed: {e}") from e

        B, S, H = inputs_embeds.shape
        # TRT engine often built with max S_past=1024; cap prefill to avoid shape error
        max_prefill_len = 1024
        if S > max_prefill_len:
            logger.warning("Prefill length %d exceeds engine max %d, truncating", S, max_prefill_len)
            inputs_embeds = inputs_embeds[:, :max_prefill_len, :].contiguous()
            S = max_prefill_len
        request_start = time.monotonic()
        position_ids_1d = torch.arange(S, device=self.device, dtype=torch.int64)
        position_ids = position_ids_1d.reshape(1, 1, -1).expand(B, 3, S)

        # Prefill: unified talker (past_kv empty S_past=0)
        t0 = time.perf_counter()
        codec_sum, full_codec, logits, kv_tensors = self._bls_talker(
            inputs_embeds, position_ids
        )
        logger.debug(f"talker prefill: {(time.perf_counter() - t0) * 1000:.1f}ms")
        codec_eos_id = self.weights.codec_eos_id
        codec_frame_buffer = []  # list of [1, 16] full_codec tensors
        code2wav_states = self._create_code2wav_initial_states()
        frame_index = 0
        text_idx = 0

        # EOS check: use FP32 argmax to avoid BF16 precision issues (Phase 0)
        if int(logits[:, -1, :].float().argmax(dim=-1).item()) == codec_eos_id:
            logger.info("EOS at step 0")
            self._send_audio_chunk(response_sender, np.zeros(1, dtype=np.float32), is_final=True)
            return

        # Push first codec frame (stateful code2wav: decode in chunks of 4)
        codec_frame_buffer.append(self._full_codec_to_frame(full_codec))

        next_embed = (codec_sum + trailing_text[text_idx]).to(torch.bfloat16)
        text_idx += 1
        position_id = torch.full((B, 3, 1), S, device=self.device, dtype=torch.int64)

        # KV length guard: engine MAX_SEQ_LEN=1024; stop before overflow (Phase 0)
        max_kv_len = 1024 - 32
        for step in range(1, self.max_steps):
            if S + step > max_kv_len:
                logger.warning(
                    "KV cache approaching limit (%s + %s > %s), forcing EOS",
                    S, step, max_kv_len,
                )
                break
            if response_sender.is_cancelled():
                logger.info("Client disconnected, stopping generation")
                break
            if (time.monotonic() - request_start) > self.request_timeout_sec:
                logger.warning("Request timeout, stopping generation")
                self._send_error(response_sender, "request_timeout")
                return

            t0 = time.perf_counter()
            codec_sum, full_codec, logits, kv_tensors = self._bls_talker(
                next_embed, position_id, kv_tensors
            )
            logger.debug(f"talker step {step}: {(time.perf_counter() - t0) * 1000:.1f}ms")
            codec_frame_buffer.append(self._full_codec_to_frame(full_codec))

            # EOS check: use FP32 argmax to avoid BF16 precision issues (Phase 0)
            if int(logits[:, -1, :].float().argmax(dim=-1).item()) == codec_eos_id:
                logger.info(f"EOS at step {step}")
                break

            # Stateful code2wav: flush full chunks of 4 frames
            code2wav_states, frame_index = self._flush_code2wav_buffer(
                response_sender, codec_frame_buffer, code2wav_states, frame_index, is_final=False
            )

            text_add = trailing_text[text_idx] if text_idx < len(trailing_text) else self._tts_pad_embed_torch
            text_idx += 1
            next_embed = (codec_sum + text_add).to(torch.bfloat16)
            position_id = torch.full((B, 3, 1), S + step, device=self.device, dtype=torch.int64)

        # Final flush: decode remaining full chunks and any partial chunk (pad to 4)
        _, _ = self._flush_code2wav_buffer(
            response_sender, codec_frame_buffer, code2wav_states, frame_index, is_final=True
        )
        self._send_audio_chunk(response_sender, np.zeros(1, dtype=np.float32), is_final=True)

        logger.info("Generation complete")

    def _bls_talker(
        self,
        input_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        past_kv_tensors: list = None,
    ):
        """Unified BLS call to talker_unified. past_kv_tensors=None → prefill (empty S_past=0). Zero-copy via DLPack."""
        # TRT: BF16; ONNX: FP32 (set in initialize from talker_unified config.pbtxt)
        inp_emb = input_embeds.contiguous().to(self._talker_dtype)
        pos_ids = position_ids.contiguous()
        try:
            inputs = [
                pb_utils.Tensor.from_dlpack("input_embeds", inp_emb.contiguous()),
                pb_utils.Tensor.from_dlpack("position_ids", pos_ids.contiguous()),
            ]
        except Exception as e:
            logger.error(f"Error in from_dlpack input_embeds/position_ids: {e}")
            raise
        if past_kv_tensors is None:
            B = input_embeds.shape[0]
            # Empty past_kv: 0-sized tensors cause DLPack "not contiguous" errors. Use numpy+pb_utils.Tensor.
            # TRT expects BF16; numpy has no bf16. Use fp32 for 0-length - TRT may accept for empty.
            for i in range(self.num_layers):
                empty_k_np = np.empty((B, self.kv_heads, 0, self.head_dim), dtype=np.float32)
                empty_v_np = np.empty((B, self.kv_heads, 0, self.head_dim), dtype=np.float32)
                inputs.append(pb_utils.Tensor(f"past_kv_{i}_k", empty_k_np))
                inputs.append(pb_utils.Tensor(f"past_kv_{i}_v", empty_v_np))
        else:
            for i in range(self.num_layers):
                # Clone first to detach from any shared storage; ensure correct dtype for TRT.
                k = past_kv_tensors[2 * i].clone().to(device=self.device, dtype=self._talker_dtype).contiguous()
                v = past_kv_tensors[2 * i + 1].clone().to(device=self.device, dtype=self._talker_dtype).contiguous()
                try:
                    inputs.append(pb_utils.Tensor.from_dlpack(f"past_kv_{i}_k", k))
                    inputs.append(pb_utils.Tensor.from_dlpack(f"past_kv_{i}_v", v))
                except Exception as e:
                    logger.error(f"Error in from_dlpack k/v: {e}")
                    raise

        out_names = ["codec_sum", "full_codec", "hidden", "logits"]
        for i in range(self.num_layers):
            out_names.append(f"present_kv_{i}_k")
            out_names.append(f"present_kv_{i}_v")

        request = pb_utils.InferenceRequest(
            model_name="talker_unified",
            inputs=inputs,
            requested_output_names=out_names,
        )
        response = request.exec()
        if response.has_error():
            raise RuntimeError(f"talker_unified BLS error: {response.error().message()}")

        codec_sum = self._tensor_from_response_torch(response, "codec_sum")
        full_codec = self._tensor_from_response_torch(response, "full_codec")
        logits = self._tensor_from_response_torch(response, "logits")
        # TRT engine may return (3,1,H)/(3,16)/(3,1,V); slice to (1,...) for downstream
        if self._talker_backend == "tensorrt" and codec_sum.shape[0] == 3:
            codec_sum = codec_sum[:1]
            full_codec = full_codec[:1]
            logits = logits[:1]
        kv_tensors = []
        for i in range(self.num_layers):
            k = self._tensor_from_response_torch(response, f"present_kv_{i}_k")
            v = self._tensor_from_response_torch(response, f"present_kv_{i}_v")
            if self._talker_backend == "tensorrt" and k.shape[0] == 3:
                k, v = k[:1].clone(), v[:1].clone()
            # Force full copy + dtype: TRT returns BF16; fallback/triton path may yield FP32.
            # cpu().numpy() route guarantees we get correct dtype on device.
            k = torch.from_numpy(k.cpu().float().numpy()).to(
                device=self.device, dtype=self._talker_dtype
            ).contiguous()
            v = torch.from_numpy(v.cpu().float().numpy()).to(
                device=self.device, dtype=self._talker_dtype
            ).contiguous()
            kv_tensors.append(k)
            kv_tensors.append(v)

        return codec_sum, full_codec, logits, kv_tensors

    def _tensor_from_response_torch(self, response, name: str) -> torch.Tensor:
        """Get output tensor by name and return torch on GPU (zero-copy via DLPack when possible)."""
        t = pb_utils.get_output_tensor_by_name(response, name)
        if t.is_cpu():
            return torch.from_numpy(t.as_numpy()).to(self.device)
        try:
            return torch.from_dlpack(t)
        except Exception as e:
            logger.error(f"Error in torch.from_dlpack for {name}: {e}")
            return torch.from_numpy(t.as_numpy()).to(self.device)

    def _bls_speaker_encoder(self, ref_audio_b64: str) -> torch.Tensor:
        """Decode ref audio, compute mel, call speaker_encoder BLS. Returns [1, 1024] torch on GPU."""
        from audio_utils import (
            decode_audio_from_base64,
            resample_to_24k,
            compute_mel_spectrogram,
        )
        audio_np, sr = decode_audio_from_base64(ref_audio_b64)
        audio_24k = resample_to_24k(audio_np, sr)
        mel_np = compute_mel_spectrogram(audio_24k).astype(np.float32)
        inputs = [pb_utils.Tensor("mel", mel_np)]
        request = pb_utils.InferenceRequest(
            model_name="speaker_encoder",
            inputs=inputs,
            requested_output_names=["speaker_embedding"],
        )
        response = request.exec()
        if response.has_error():
            raise RuntimeError(
                f"speaker_encoder BLS error: {response.error().message()}"
            )
        return self._tensor_from_response_torch(response, "speaker_embedding")

    def _bls_speech_tokenizer(self, ref_audio_b64: str) -> torch.Tensor:
        """Decode ref audio, prepare waveform, call speech_tokenizer_encoder BLS. Returns [T_ref, 16] torch on GPU."""
        from audio_utils import (
            decode_audio_from_base64,
            resample_to_24k,
            prepare_waveform_tensor,
        )
        audio_np, sr = decode_audio_from_base64(ref_audio_b64)
        audio_24k = resample_to_24k(audio_np, sr)
        waveform = prepare_waveform_tensor(audio_24k)
        inputs = [pb_utils.Tensor("waveform", waveform)]
        request = pb_utils.InferenceRequest(
            model_name="speech_tokenizer_encoder",
            inputs=inputs,
            requested_output_names=["audio_codes"],
        )
        response = request.exec()
        if response.has_error():
            raise RuntimeError(
                f"speech_tokenizer_encoder BLS error: {response.error().message()}"
            )
        codes = self._tensor_from_response_torch(response, "audio_codes")
        return codes.squeeze(0).T

    def _full_codec_to_frame(self, full_codec: torch.Tensor) -> torch.Tensor:
        """Ensure full_codec from talker is [1, 16] for buffer. Always cast to int64:
        TRT/ORT may return full_codec as BF16; code2wav expects TYPE_INT64 for 'codes'."""
        t = full_codec.contiguous().to(torch.int64)
        if t.dim() == 3:
            t = t.squeeze(0)
        if t.dim() == 2 and t.shape[0] != 1:
            t = t.unsqueeze(0)
        return t.to(self.device)

    def _create_code2wav_initial_states(self):
        """Create 37 zero state tensors for stateful code2wav (GPU, correct dtype)."""
        return [
            torch.zeros(shape, device=self.device, dtype=self._code2wav_dtype)
            for shape in _CODE2WAV_STATE_SHAPES
        ]

    def _flush_code2wav_buffer(
        self,
        response_sender,
        codec_frame_buffer: list,
        code2wav_states: list,
        frame_index: int,
        is_final: bool,
    ):
        """Decode full chunks of 4 frames; if is_final, pad and decode remaining 1–3 frames. Mutates codec_frame_buffer. Returns (updated_states, updated_frame_index)."""
        while len(codec_frame_buffer) >= CODE2WAV_CHUNK_T:
            frames = codec_frame_buffer[:CODE2WAV_CHUNK_T]
            del codec_frame_buffer[:CODE2WAV_CHUNK_T]
            codes = torch.stack(frames, dim=-1)
            if codes.dim() == 2:
                codes = codes.unsqueeze(0)
            codes = codes.contiguous()
            cache_position = torch.arange(
                frame_index, frame_index + CODE2WAV_CHUNK_T, device=self.device, dtype=torch.int64
            )
            wav_np, code2wav_states = self._bls_code2wav_streaming(codes, cache_position, code2wav_states)
            self._send_audio_chunk(response_sender, wav_np, is_final=False)
            frame_index += CODE2WAV_CHUNK_T

        if is_final and len(codec_frame_buffer) > 0:
            n = len(codec_frame_buffer)
            codec_pad_id = self.weights.codec_pad_id
            pad_frames = [
                torch.full((1, 16), codec_pad_id, device=self.device, dtype=torch.int64)
                for _ in range(CODE2WAV_CHUNK_T - n)
            ]
            frames = codec_frame_buffer + pad_frames
            del codec_frame_buffer[:]
            codes = torch.stack(frames, dim=-1)
            if codes.dim() == 2:
                codes = codes.unsqueeze(0)
            codes = codes.contiguous()
            cache_position = torch.arange(
                frame_index, frame_index + CODE2WAV_CHUNK_T, device=self.device, dtype=torch.int64
            )
            wav_np, _ = self._bls_code2wav_streaming(codes, cache_position, code2wav_states)
            valid_samples = n * SAMPLES_PER_CODEC_FRAME
            self._send_audio_chunk(
                response_sender, wav_np[:valid_samples].astype(np.float32), is_final=False
            )
        return code2wav_states, frame_index

    def _bls_code2wav_streaming(
        self,
        codes: torch.Tensor,
        cache_position: torch.Tensor,
        state_tensors: list,
    ) -> tuple:
        """Stateful code2wav: codes [1, 16, 4], cache_position [4], 37 states -> wav [7680], 37 new states. Uses DLPack for GPU tensors."""
        # code2wav expects TYPE_INT64 for 'codes'; talker may return full_codec as BF16 in TRT mode.
        codes = codes.to(device=self.device, dtype=torch.int64).contiguous()
        try:
            inputs = [
                pb_utils.Tensor.from_dlpack("codes", codes.contiguous()),
                pb_utils.Tensor.from_dlpack("cache_position", cache_position.contiguous()),
            ]
        except Exception as e:
            logger.error(f"Error in from_dlpack codes/cache_position: {e}")
            raise
        state_input_names = []
        for i in range(8):
            state_input_names.append(f"past_kv_{i}_k")
            state_input_names.append(f"past_kv_{i}_v")
        for i in range(17):
            state_input_names.append(f"conv_state_{i}")
        for i in range(4):
            state_input_names.append(f"transconv_overlap_{i}")
        for i, t in enumerate(state_tensors):
            try:
                inputs.append(
                    pb_utils.Tensor.from_dlpack(state_input_names[i], t.contiguous().to(self._code2wav_dtype))
                )
            except Exception as e:
                logger.error(f"Error in from_dlpack state_tensors {state_input_names[i]}: {e}")
                raise

        out_names = ["wav"]
        for i in range(8):
            out_names.append(f"present_kv_{i}_k")
            out_names.append(f"present_kv_{i}_v")
        for i in range(17):
            out_names.append(f"new_conv_state_{i}")
        for i in range(4):
            out_names.append(f"new_transconv_overlap_{i}")

        request = pb_utils.InferenceRequest(
            model_name="code2wav",
            inputs=inputs,
            requested_output_names=out_names,
        )
        response = request.exec()
        if response.has_error():
            raise RuntimeError(f"code2wav BLS error: {response.error().message()}")

        wav_t = self._tensor_from_response_torch(response, "wav")
        wav_np = wav_t.cpu().float().numpy().flatten()
        new_states = []
        for i in range(8):
            new_states.append(self._tensor_from_response_torch(response, f"present_kv_{i}_k"))
            new_states.append(self._tensor_from_response_torch(response, f"present_kv_{i}_v"))
        for i in range(17):
            new_states.append(self._tensor_from_response_torch(response, f"new_conv_state_{i}"))
        for i in range(4):
            new_states.append(self._tensor_from_response_torch(response, f"new_transconv_overlap_{i}"))
        return wav_np, new_states

    def _send_audio_chunk(self, response_sender, audio: np.ndarray, is_final: bool = False):
        audio_tensor = pb_utils.Tensor("audio_chunk", audio.astype(np.float32))
        final_tensor = pb_utils.Tensor("is_final", np.array([is_final], dtype=bool))
        response = pb_utils.InferenceResponse(
            output_tensors=[audio_tensor, final_tensor])
        response_sender.send(response)

    def _send_capabilities(self, response_sender, caps: dict):
        caps_json = json.dumps(caps)
        # We must return audio_chunk (TYPE_FP32) and is_final (TYPE_BOOL) as defined in config.pbtxt
        # We can embed the JSON string in an error message, or we can add a new output tensor.
        # Since we cannot easily change the output signature dynamically, we will return it as an error message
        # with a special prefix, or we can just return it as a TritonError.
        # A cleaner way is to return it as a TritonError so the client can parse it.
        response = pb_utils.InferenceResponse(
            output_tensors=[
                pb_utils.Tensor("audio_chunk", np.zeros(1, dtype=np.float32)),
                pb_utils.Tensor("is_final", np.array([True], dtype=bool)),
            ],
            error=pb_utils.TritonError(f"CAPABILITIES:{caps_json}"),
        )
        try:
            response_sender.send(response)
        except Exception:
            pass

    def _send_error(self, response_sender, error_msg: str):
        audio = np.zeros(1, dtype=np.float32)
        response = pb_utils.InferenceResponse(
            output_tensors=[
                pb_utils.Tensor("audio_chunk", audio),
                pb_utils.Tensor("is_final", np.array([True], dtype=bool)),
            ],
            error=pb_utils.TritonError(error_msg),
        )
        try:
            response_sender.send(response)
        except Exception:
            pass

    def finalize(self):
        logger.info("[TTS Orchestrator] Finalized")

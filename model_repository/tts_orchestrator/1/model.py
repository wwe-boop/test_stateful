"""
TTS Orchestrator — Triton Python BLS Backend.

Unified pipeline (same BLS for ONNX and TensorRT backends):
  1. Session init (Speaker Encoder / Speech Tokenizer via BLS)
  2. Prefill construction (in-process torch embedding; no BLS for embedders)
  3. BLS talker_unified: prefill (past_kv dummy S_past=1) → codec_sum, full_codec, logits, KV
  4. Decode loop: BLS talker_unified(next_embed, position_id, KV) → codec_sum, full_codec, logits, updated KV
  5. Code2Wav (BLS) for chunked audio

Talker path uses torch + DLPack zero-copy (BF16 I/O); KV cache never leaves GPU.
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
        self.audio_chunk_threshold = int(params.get("audio_chunk_frames", {}).get(
            "string_value", "25"))
        self.first_chunk_frames = int(params.get("first_chunk_frames", {}).get(
            "string_value", "10"))

        from prefill_builder import EmbeddingWeights, PrefillBuilder, parse_task_type
        self._parse_task_type = parse_task_type

        logger.info(f"Loading weights from {weights_dir} ...")
        self.weights = EmbeddingWeights(weights_dir, device_id=self.device_id)
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
        logger.info(f"[TTS Orchestrator] Initialized: variant={self.variant}, num_layers={self.num_layers}")

    def execute(self, requests):
        for request in requests:
            response_sender = request.get_response_sender()
            try:
                self._handle_request(request, response_sender)
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
            self._parse_task_type(task_type_str, req.get("x_vector_only", False))
        except ValueError as e:
            raise ValueError(f"Invalid task_type or parameters: {e}") from e
        if task_type_str and task_type_str.startswith("voice_clone"):
            ref_audio = req.get("ref_audio")
            if not ref_audio or not (ref_audio if isinstance(ref_audio, str) else "").strip():
                raise ValueError("Request field 'ref_audio' (base64) is required for voice_clone task_type")

    def _handle_request(self, request, response_sender):
        req_tensor = pb_utils.get_input_tensor_by_name(request, "request")
        req_json = req_tensor.as_numpy()[0].decode("utf-8")
        req = json.loads(req_json)

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
        position_ids_1d = torch.arange(S, device=self.device, dtype=torch.int64)
        position_ids = position_ids_1d.reshape(1, 1, -1).expand(3, B, S)

        # Prefill: unified talker (past_kv dummy S_past=1)
        t0 = time.perf_counter()
        codec_sum, full_codec, logits, kv_tensors = self._bls_talker(
            inputs_embeds, position_ids
        )
        logger.debug(f"talker prefill: {(time.perf_counter() - t0) * 1000:.1f}ms")
        codec_eos_id = self.weights.codec_eos_id
        codec_buffer = [full_codec]
        text_idx = 0
        first_chunk_sent = False

        if int(logits[:, -1, :].argmax(dim=-1).item()) == codec_eos_id:
            logger.info("EOS at step 0")
            self._send_audio_chunk(response_sender, np.zeros(1, dtype=np.float32), is_final=True)
            return

        chunk_threshold = self.first_chunk_frames if not first_chunk_sent else self.audio_chunk_threshold
        if len(codec_buffer) >= chunk_threshold:
            audio = self._bls_code2wav(codec_buffer)
            self._send_audio_chunk(response_sender, audio, is_final=False)
            codec_buffer = []
            first_chunk_sent = True

        next_embed = (codec_sum + trailing_text[text_idx]).to(torch.bfloat16)
        text_idx += 1
        position_id = torch.full((3, B, 1), S, device=self.device, dtype=torch.int64)

        for step in range(1, self.max_steps):
            if response_sender.is_cancelled():
                logger.info("Client disconnected, stopping generation")
                break

            t0 = time.perf_counter()
            codec_sum, full_codec, logits, kv_tensors = self._bls_talker(
                next_embed, position_id, kv_tensors
            )
            logger.debug(f"talker step {step}: {(time.perf_counter() - t0) * 1000:.1f}ms")
            codec_buffer.append(full_codec)

            if int(logits[:, -1, :].argmax(dim=-1).item()) == codec_eos_id:
                logger.info(f"EOS at step {step}")
                break

            chunk_threshold = self.first_chunk_frames if not first_chunk_sent else self.audio_chunk_threshold
            if len(codec_buffer) >= chunk_threshold:
                audio = self._bls_code2wav(codec_buffer)
                self._send_audio_chunk(response_sender, audio, is_final=False)
                codec_buffer = []
                first_chunk_sent = True

            text_add = trailing_text[text_idx] if text_idx < len(trailing_text) else self._tts_pad_embed_torch
            text_idx += 1
            next_embed = (codec_sum + text_add).to(torch.bfloat16)
            position_id = torch.full((3, B, 1), S + step, device=self.device, dtype=torch.int64)

        if codec_buffer:
            audio = self._bls_code2wav(codec_buffer)
            self._send_audio_chunk(response_sender, audio, is_final=True)
        else:
            self._send_audio_chunk(response_sender, np.zeros(1, dtype=np.float32), is_final=True)

        logger.info("Generation complete")

    def _bls_talker(
        self,
        input_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        past_kv_tensors: list = None,
    ):
        """Unified BLS call to talker_unified (BF16 I/O). past_kv_tensors=None → prefill (dummy S_past=1). Zero-copy via DLPack."""
        # Inputs: GPU zero-copy via from_dlpack
        inp_emb = input_embeds.contiguous()
        pos_ids = position_ids.contiguous()
        inputs = [
            pb_utils.Tensor.from_dlpack("input_embeds", inp_emb),
            pb_utils.Tensor.from_dlpack("position_ids", pos_ids),
        ]
        if past_kv_tensors is None:
            B = input_embeds.shape[0]
            for i in range(self.num_layers):
                dummy_k = torch.zeros(
                    B, self.kv_heads, 1, self.head_dim,
                    dtype=torch.bfloat16, device=self.device,
                )
                dummy_v = torch.zeros(
                    B, self.kv_heads, 1, self.head_dim,
                    dtype=torch.bfloat16, device=self.device,
                )
                inputs.append(pb_utils.Tensor.from_dlpack(f"past_kv_{i}_k", dummy_k))
                inputs.append(pb_utils.Tensor.from_dlpack(f"past_kv_{i}_v", dummy_v))
        else:
            for i in range(self.num_layers):
                k = past_kv_tensors[2 * i].contiguous()
                v = past_kv_tensors[2 * i + 1].contiguous()
                inputs.append(pb_utils.Tensor.from_dlpack(f"past_kv_{i}_k", k))
                inputs.append(pb_utils.Tensor.from_dlpack(f"past_kv_{i}_v", v))

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
        kv_tensors = []
        for i in range(self.num_layers):
            kv_tensors.append(self._tensor_from_response_torch(response, f"present_kv_{i}_k"))
            kv_tensors.append(self._tensor_from_response_torch(response, f"present_kv_{i}_v"))

        return codec_sum, full_codec, logits, kv_tensors

    def _tensor_from_response_torch(self, response, name: str) -> torch.Tensor:
        """Get output tensor by name and return torch on GPU (zero-copy via DLPack when possible)."""
        t = pb_utils.get_output_tensor_by_name(response, name)
        if t.is_cpu():
            return torch.from_numpy(t.as_numpy()).to(self.device)
        try:
            return torch.from_dlpack(t)
        except (AttributeError, TypeError):
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

    def _bls_code2wav(self, codec_buffer: list) -> np.ndarray:
        """codec_buffer: list of torch [B, 16] int64 (full_codec from talker)."""
        stacked = torch.cat(codec_buffer, dim=0)
        codes = stacked.T.unsqueeze(0).cpu().numpy().astype(np.int64)

        inputs = [pb_utils.Tensor("codes", codes)]
        request = pb_utils.InferenceRequest(
            model_name="code2wav",
            inputs=inputs,
            requested_output_names=["wav"],
        )
        response = request.exec()
        if response.has_error():
            raise RuntimeError(f"Code2Wav BLS error: {response.error().message()}")

        wav = pb_utils.get_output_tensor_by_name(response, "wav")
        if wav.is_cpu():
            return wav.as_numpy().flatten()
        try:
            wav_t = torch.from_dlpack(wav)
            return wav_t.cpu().numpy().flatten()
        except (AttributeError, TypeError):
            return np.array(wav.as_numpy(), dtype=np.float32).flatten()

    def _send_audio_chunk(self, response_sender, audio: np.ndarray, is_final: bool = False):
        audio_tensor = pb_utils.Tensor("audio_chunk", audio.astype(np.float32))
        final_tensor = pb_utils.Tensor("is_final", np.array([is_final], dtype=bool))
        response = pb_utils.InferenceResponse(
            output_tensors=[audio_tensor, final_tensor])
        response_sender.send(response)

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

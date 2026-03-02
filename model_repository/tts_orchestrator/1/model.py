"""
TTS Orchestrator — Triton Python BLS Backend (Phase 3, Item 13).

Decoupled streaming model that orchestrates the full TTS pipeline:
  1. Session init (Speaker Encoder / Speech Tokenizer via BLS)
  2. Prefill construction (4 task types)
  3. Context (Pure TRT): prefill → hidden + logits, KV cache filled
  4. First step: standalone CP (BLS) + codec_sum → next_embed
  5. Decode loop: fused decode_step(codec_sum, full_codec, logits) + text_add → next_embed
  6. Code2Wav chunked decode → streaming audio output

Uses Pure TRT (talker_context + talker_decode_fused engines), no TRT-LLM.
"""

import json
import logging
import os
import sys
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

# Ensure local imports work
_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
if _MODEL_DIR not in sys.path:
    sys.path.insert(0, _MODEL_DIR)


class TritonPythonModel:

    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])
        params = self.model_config.get("parameters", {})

        self.variant = params.get("model_variant", {}).get("string_value", "unknown")
        weights_dir = params.get("weights_dir", {}).get("string_value", "")
        engine_dir = params.get("engine_dir", {}).get("string_value", "")
        tokenizer_dir = params.get("tokenizer_dir", {}).get("string_value", "")

        model_dir = Path(_MODEL_DIR)

        if not weights_dir:
            weights_dir = str(model_dir / "weights")
        if not engine_dir:
            # Engine dir: talker_context.engine + talker_decode_fused.engine (Pure TRT)
            engine_dir = str(model_dir / "engine")
        if not tokenizer_dir:
            tokenizer_dir = str(model_dir / "tokenizer")

        self.device = torch.device("cuda:0")
        self.max_steps = int(params.get("max_decode_steps", {}).get(
            "string_value", "2000"))
        self.audio_chunk_threshold = int(params.get("audio_chunk_frames", {}).get(
            "string_value", "25"))

        # Load embedding weights
        from prefill_builder import EmbeddingWeights, PrefillBuilder, parse_task_type
        self._parse_task_type = parse_task_type

        logger.info(f"Loading weights from {weights_dir} ...")
        self.weights = EmbeddingWeights(weights_dir, self.device)

        # Load tokenizer
        logger.info(f"Loading tokenizer from {tokenizer_dir} ...")
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_dir, trust_remote_code=True)
        except Exception as e:
            logger.warning(f"Tokenizer load failed ({e}), using fallback")
            self.tokenizer = None

        self.prefill_builder = PrefillBuilder(self.weights, self.tokenizer)

        # Load codec embedding sum (3D gather) — used only for first step before fused decode
        from codec_embedding_sum import CodecEmbeddingSum
        if self.weights.codec_embeddings_3d is not None:
            self.codec_emb_sum = CodecEmbeddingSum(
                self.weights.codec_embeddings_3d.to(self.device))
        else:
            logger.warning("codec_embeddings_3d.pt not found, "
                          "codec sum will use individual lookups")
            self.codec_emb_sum = None

        # Load Talker Runner
        logger.info(f"Loading Talker engine from {engine_dir} ...")
        try:
            from talker_runner import TalkerRunner
            self.talker = TalkerRunner(engine_dir, device=0)
        except Exception as e:
            logger.error(f"TalkerRunner init failed: {e}")
            logger.error(traceback.format_exc())
            self.talker = None

        logger.info(f"[TTS Orchestrator] Initialized: variant={self.variant}")

    def execute(self, requests):
        responses = []
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

    def _handle_request(self, request, response_sender):
        # Parse JSON request
        req_tensor = pb_utils.get_input_tensor_by_name(request, "request")
        req_json = req_tensor.as_numpy()[0].decode("utf-8")
        req = json.loads(req_json)

        task_type_str = req.get("task_type", "voice_design")
        text = req.get("text", "")
        language = req.get("language", "auto")
        speaker = req.get("speaker")
        instruct = req.get("instruct")
        x_vector_only = req.get("x_vector_only", False)

        task_type = self._parse_task_type(task_type_str, x_vector_only)
        logger.info(f"Request: task={task_type.value}, text='{text[:50]}...', "
                    f"lang={language}")

        # Session initialization: Speaker Encoder / Speech Tokenizer via BLS
        spk_embedding = None
        ref_codes = None

        if task_type.value.startswith("voice_clone"):
            ref_audio_b64 = req.get("ref_audio")
            if ref_audio_b64:
                spk_embedding = self._bls_speaker_encoder(ref_audio_b64)
            if task_type.value == "voice_clone_icl":
                ref_codes = self._bls_speech_tokenizer(ref_audio_b64)

        # Build prefill (need position_ids for Pure TRT context)
        inputs_embeds, trailing_text = self.prefill_builder.build(
            task_type=task_type,
            text=text,
            language=language,
            speaker=speaker,
            instruct=instruct,
            spk_embedding=spk_embedding,
            ref_codes=ref_codes,
        )

        if self.talker is None:
            raise RuntimeError("TalkerRunner not initialized")

        # Position IDs for context: [3, B, S] (3D RoPE, TTS: same value per dim)
        B, S, H = inputs_embeds.shape
        position_ids = torch.arange(S, device=self.device, dtype=torch.int64)
        position_ids = position_ids.unsqueeze(0).unsqueeze(0).expand(3, B, S)

        # Prefill (Pure TRT context engine: KV cache filled, returns last_hidden + last_logits)
        hidden, logits = self.talker.context(inputs_embeds, position_ids)

        codec_eos_id = self.weights.codec_eos_id
        codec_buffer = []
        text_idx = 0

        # First step: standalone CP (context engine does not include CP) + codec_sum
        codec_token_0 = logits[:, -1, :].argmax(dim=-1)
        if codec_token_0.item() == codec_eos_id:
            logger.info("EOS at step 0")
            self.talker.reset()
            self._send_audio_chunk(response_sender, np.zeros(1, dtype=np.float32), is_final=True)
            return

        cp_tokens = self._bls_code_predictor(hidden, codec_token_0)
        full_codec = torch.cat([
            codec_token_0.unsqueeze(0).unsqueeze(1),
            cp_tokens.view(1, -1),
        ], dim=1)
        codec_buffer.append(full_codec)

        if len(codec_buffer) >= self.audio_chunk_threshold:
            audio = self._bls_code2wav(codec_buffer)
            self._send_audio_chunk(response_sender, audio, is_final=False)
            codec_buffer = []

        if self.codec_emb_sum is not None:
            codec_sum = self.codec_emb_sum(full_codec).unsqueeze(1)
        else:
            codec_sum = self._codec_sum_fallback(full_codec)
        text_add = trailing_text[text_idx] if text_idx < len(trailing_text) else self.weights.tts_pad_embed
        text_idx += 1
        next_embed = (codec_sum + text_add).to(inputs_embeds.dtype)

        # Position for first decode step (S = context length)
        position_id = torch.full((3, B, 1), S, device=self.device, dtype=torch.int64)

        # Decode loop: fused engine returns codec_sum, full_codec, logits
        for step in range(1, self.max_steps):
            if response_sender.is_cancelled():
                logger.info("Client disconnected, stopping generation")
                break

            codec_sum, full_codec, logits = self.talker.decode_step(next_embed, position_id)
            codec_buffer.append(full_codec)

            if logits[:, -1, :].argmax(dim=-1).item() == codec_eos_id:
                logger.info(f"EOS at step {step}")
                break

            if len(codec_buffer) >= self.audio_chunk_threshold:
                audio = self._bls_code2wav(codec_buffer)
                self._send_audio_chunk(response_sender, audio, is_final=False)
                codec_buffer = []

            text_add = trailing_text[text_idx] if text_idx < len(trailing_text) else self.weights.tts_pad_embed
            text_idx += 1
            next_embed = (codec_sum + text_add).to(inputs_embeds.dtype)
            position_id = torch.full((3, B, 1), self.talker.current_seq_len, device=self.device, dtype=torch.int64)

        # Flush remaining codec
        if codec_buffer:
            audio = self._bls_code2wav(codec_buffer)
            self._send_audio_chunk(response_sender, audio, is_final=True)
        else:
            self._send_audio_chunk(response_sender, np.zeros(1, dtype=np.float32),
                                  is_final=True)

        self.talker.reset()
        logger.info(f"Generation complete: {step + 1} steps")

    def _codec_sum_fallback(self, full_codec: torch.Tensor) -> torch.Tensor:
        """Fallback codec_sum using individual embedding lookups."""
        B = full_codec.shape[0]
        H = self.weights.hidden_size
        result = torch.zeros(B, 1, H, device=self.device)
        with torch.no_grad():
            result += self.weights.codec_embedding(full_codec[:, 0]).unsqueeze(1)
        return result

    # ── BLS helpers ──

    def _bls_code_predictor(self, past_hidden: torch.Tensor,
                            codec_token_0: torch.Tensor) -> torch.Tensor:
        """Call Code Predictor model via BLS.

        Triton auto-fills max_batch_size from the ONNX batch dim, so inputs
        must include the batch dimension explicitly.
        """
        # past_hidden: [1, 1, H] → keep as [1, 1, H] (batch=1)
        past_hidden_np = past_hidden.float().cpu().numpy()
        # codec_token_0: [1] → reshape to [1, 1] (batch=1, scalar reshape)
        codec_0_np = codec_token_0.cpu().numpy().astype(np.int64).reshape(1, 1)

        inputs = [
            pb_utils.Tensor("past_hidden", past_hidden_np),
            pb_utils.Tensor("codec_token_0", codec_0_np),
        ]
        request = pb_utils.InferenceRequest(
            model_name="code_predictor",
            inputs=inputs,
            requested_output_names=["codec_tokens"],
        )
        response = request.exec()
        if response.has_error():
            raise RuntimeError(f"Code Predictor BLS error: {response.error().message()}")

        codec_tokens = pb_utils.get_output_tensor_by_name(response, "codec_tokens")
        if codec_tokens.is_cpu():
            return torch.from_numpy(codec_tokens.as_numpy()).to(self.device)
        else:
            ct = torch.utils.dlpack.from_dlpack(codec_tokens.to_dlpack())
            return ct.to(self.device)

    def _bls_speaker_encoder(self, ref_audio_b64: str) -> torch.Tensor:
        """Call Speaker Encoder via BLS (placeholder — needs audio preprocessing)."""
        logger.warning("Speaker Encoder BLS not yet implemented")
        return torch.zeros(1, self.weights.hidden_size, device=self.device)

    def _bls_speech_tokenizer(self, ref_audio_b64: str) -> torch.Tensor:
        """Call Speech Tokenizer via BLS (placeholder — needs audio preprocessing)."""
        logger.warning("Speech Tokenizer BLS not yet implemented")
        return None

    def _bls_code2wav(self, codec_buffer: list) -> np.ndarray:
        """Call Code2Wav model via BLS to decode codec tokens to audio.

        Code2Wav ONNX: input 'codes' [batch, 16, time], output 'wav' [batch, audio_len].
        Triton auto-filled batch dim, so codes shape = [batch=1, 16, T].
        """
        stacked = torch.cat(codec_buffer, dim=0)  # [T, 16]
        codes = stacked.T.unsqueeze(0).long().cpu().numpy()  # [1, 16, T]

        inputs = [
            pb_utils.Tensor("codes", codes),
        ]
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
        else:
            wav_t = torch.utils.dlpack.from_dlpack(wav.to_dlpack())
            return wav_t.cpu().numpy().flatten()

    def _send_audio_chunk(self, response_sender, audio: np.ndarray,
                          is_final: bool = False):
        """Send a streaming audio chunk response."""
        audio_tensor = pb_utils.Tensor("audio_chunk",
                                       audio.astype(np.float32))
        final_tensor = pb_utils.Tensor("is_final",
                                        np.array([is_final], dtype=bool))
        response = pb_utils.InferenceResponse(
            output_tensors=[audio_tensor, final_tensor])
        response_sender.send(response)

    def _send_error(self, response_sender, error_msg: str):
        """Send error response with empty audio."""
        audio = np.zeros(1, dtype=np.float32)
        audio_tensor = pb_utils.Tensor("audio_chunk", audio)
        final_tensor = pb_utils.Tensor("is_final", np.array([True], dtype=bool))
        response = pb_utils.InferenceResponse(
            output_tensors=[audio_tensor, final_tensor],
            error=pb_utils.TritonError(error_msg),
        )
        try:
            response_sender.send(response)
        except Exception:
            pass

    def finalize(self):
        if self.talker:
            self.talker.reset()
        logger.info("[TTS Orchestrator] Finalized")

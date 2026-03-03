"""
TTS Orchestrator — Triton Python BLS Backend.

Unified pipeline (same BLS for ONNX and TensorRT backends):
  1. Session init (Speaker Encoder / Speech Tokenizer via BLS)
  2. Prefill construction (in-process .pt text embedding)
  3. BLS talker_unified: prefill (past_kv dummy S_past=1) → codec_sum, full_codec, logits, KV
  4. Decode loop: BLS talker_unified(next_embed, position_id, KV) → codec_sum, full_codec, logits, updated KV
  5. Code2Wav (BLS) for chunked audio

Single talker_unified model handles both prefill and decode; KV cache passed via GPU zero-copy (dlpack).
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

        self.device = torch.device("cuda:0")
        self.max_steps = int(params.get("max_decode_steps", {}).get(
            "string_value", "2000"))
        self.audio_chunk_threshold = int(params.get("audio_chunk_frames", {}).get(
            "string_value", "25"))

        from prefill_builder import EmbeddingWeights, PrefillBuilder, parse_task_type
        self._parse_task_type = parse_task_type

        logger.info(f"Loading weights from {weights_dir} ...")
        self.weights = EmbeddingWeights(weights_dir, self.device)
        self.num_layers = int(self.weights.config.get("talker_num_layers", 28))
        self.kv_heads = int(self.weights.config.get("talker_num_kv_heads", 8))
        talker_h = int(self.weights.config.get("talker_hidden_size", 2048))
        talker_heads = int(self.weights.config.get("talker_num_heads", 16))
        self.head_dim = talker_h // talker_heads if talker_heads else 128

        logger.info(f"Loading tokenizer from {tokenizer_dir} ...")
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_dir, trust_remote_code=True)
        except Exception as e:
            logger.warning(f"Tokenizer load failed ({e}), using fallback")
            self.tokenizer = None

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

    def _handle_request(self, request, response_sender):
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
        logger.info(f"Request: task={task_type.value}, text='{text[:50]}...', lang={language}")

        spk_embedding = None
        ref_codes = None
        if task_type.value.startswith("voice_clone"):
            ref_audio_b64 = req.get("ref_audio")
            if ref_audio_b64:
                spk_embedding = self._bls_speaker_encoder(ref_audio_b64)
            if task_type.value == "voice_clone_icl":
                ref_codes = self._bls_speech_tokenizer(ref_audio_b64)

        inputs_embeds, trailing_text = self.prefill_builder.build(
            task_type=task_type,
            text=text,
            language=language,
            speaker=speaker,
            instruct=instruct,
            spk_embedding=spk_embedding,
            ref_codes=ref_codes,
        )

        B, S, H = inputs_embeds.shape
        position_ids = torch.arange(S, device=self.device, dtype=torch.int64)
        position_ids = position_ids.unsqueeze(0).unsqueeze(0).expand(3, B, S)

        # Prefill: unified talker (past_kv dummy S_past=1)
        codec_sum, full_codec, logits, kv_tensors = self._bls_talker(
            inputs_embeds, position_ids
        )
        codec_eos_id = self.weights.codec_eos_id
        codec_buffer = [full_codec]
        text_idx = 0

        if logits[:, -1, :].argmax(dim=-1).item() == codec_eos_id:
            logger.info("EOS at step 0")
            self._send_audio_chunk(response_sender, np.zeros(1, dtype=np.float32), is_final=True)
            return

        if len(codec_buffer) >= self.audio_chunk_threshold:
            audio = self._bls_code2wav(codec_buffer)
            self._send_audio_chunk(response_sender, audio, is_final=False)
            codec_buffer = []

        next_embed = (codec_sum + trailing_text[text_idx]).to(inputs_embeds.dtype)
        text_idx += 1
        position_id = torch.full((3, B, 1), S, device=self.device, dtype=torch.int64)

        for step in range(1, self.max_steps):
            if response_sender.is_cancelled():
                logger.info("Client disconnected, stopping generation")
                break

            codec_sum, full_codec, logits, kv_tensors = self._bls_talker(
                next_embed, position_id, kv_tensors
            )
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
        """Unified BLS call to talker_unified. past_kv_tensors=None → prefill (pass dummy S_past=1)."""
        inp_emb = input_embeds.cpu().numpy().astype(np.float32)
        pos_ids = position_ids.cpu().numpy().astype(np.int64)

        inputs = [
            pb_utils.Tensor("input_embeds", inp_emb),
            pb_utils.Tensor("position_ids", pos_ids),
        ]
        if past_kv_tensors is None:
            B = input_embeds.shape[0]
            dummy = np.zeros((B, self.kv_heads, 1, self.head_dim), dtype=np.float32)
            for i in range(self.num_layers):
                inputs.append(pb_utils.Tensor(f"past_kv_{i}_k", dummy.copy()))
                inputs.append(pb_utils.Tensor(f"past_kv_{i}_v", dummy.copy()))
        else:
            for i in range(self.num_layers):
                inputs.append(
                    pb_utils.Tensor.from_dlpack(
                        f"past_kv_{i}_k", past_kv_tensors[2 * i].to_dlpack()
                    )
                )
                inputs.append(
                    pb_utils.Tensor.from_dlpack(
                        f"past_kv_{i}_v", past_kv_tensors[2 * i + 1].to_dlpack()
                    )
                )

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

        codec_sum = self._tensor_from_response(response, "codec_sum")
        full_codec = self._tensor_from_response(response, "full_codec")
        logits = self._tensor_from_response(response, "logits")
        kv_tensors = []
        for i in range(self.num_layers):
            kv_tensors.append(self._tensor_from_response(response, f"present_kv_{i}_k"))
            kv_tensors.append(self._tensor_from_response(response, f"present_kv_{i}_v"))

        return codec_sum, full_codec, logits, kv_tensors

    def _tensor_from_response(self, response, name: str) -> torch.Tensor:
        """Get output tensor by name and convert to torch on self.device (GPU zero-copy when possible)."""
        t = pb_utils.get_output_tensor_by_name(response, name)
        if t.is_cpu():
            return torch.from_numpy(t.as_numpy()).to(self.device)
        return torch.utils.dlpack.from_dlpack(t.to_dlpack()).to(self.device)

    def _bls_speaker_encoder(self, ref_audio_b64: str) -> torch.Tensor:
        logger.warning("Speaker Encoder BLS not yet implemented")
        return torch.zeros(1, self.weights.hidden_size, device=self.device)

    def _bls_speech_tokenizer(self, ref_audio_b64: str):
        logger.warning("Speech Tokenizer BLS not yet implemented")
        return None

    def _bls_code2wav(self, codec_buffer: list) -> np.ndarray:
        stacked = torch.cat(codec_buffer, dim=0)
        codes = stacked.T.unsqueeze(0).long().cpu().numpy()

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
        wav_t = torch.utils.dlpack.from_dlpack(wav.to_dlpack())
        return wav_t.cpu().numpy().flatten()

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

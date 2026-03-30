"""
TTS Orchestrator — Triton Python BLS Backend.

Production pipeline (talker_code2wav_fused present):
  1. Speaker / speech_tokenizer_codec_fused (ICL) via BLS
  2. PrefillBuilder (in-process embeddings)
  3. BLS talker_code2wav_fused: prefill + each decode step → wav + KV + code2wav states

Legacy pipeline (talker_unified + code2wav only): chunked code2wav buffer (chunk_T=4).

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
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

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

from batch_decode_scheduler import (
    FusedDecodeTicket,
    group_by_c2w_past_len,
    pad_talker_past_kv,
    padded_attention_bias,
    uniform_past_seq_lens,
    zeros_attention_bias,
)
from text_segmenter import split_text_for_token_budget

# Legacy code2wav (chunk_T=4): default shapes for 8 decoder KV layers + 17 conv + 4 transconv.
# Fused path: triton_manifest.json must include code2wav_fused (export_09 + assemble).
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
FUSED_CHUNK_T = 1
# Prefill with no history: TRT/ORT may require a fixed min past_kv time dim; use one dummy slot
# and past_seq_lens=0 so padded_attention_bias masks it (-inf), logical past length remains 0.
_FUSED_DUMMY_PAST_LEN = 1

# Default c2w_* I/O for split-engine path only (talker_unified + code2wav).
_C2W_FUSED_DEFAULT_INPUT_NAMES = []
for _i in range(8):
    _C2W_FUSED_DEFAULT_INPUT_NAMES.append(f"c2w_past_kv_{_i}_k")
    _C2W_FUSED_DEFAULT_INPUT_NAMES.append(f"c2w_past_kv_{_i}_v")
for _i in range(17):
    _C2W_FUSED_DEFAULT_INPUT_NAMES.append(f"c2w_conv_state_{_i}")
for _i in range(4):
    _C2W_FUSED_DEFAULT_INPUT_NAMES.append(f"c2w_transconv_overlap_{_i}")

_C2W_FUSED_DEFAULT_OUTPUT_NAMES = []
for _i in range(8):
    _C2W_FUSED_DEFAULT_OUTPUT_NAMES.append(f"c2w_present_kv_{_i}_k")
    _C2W_FUSED_DEFAULT_OUTPUT_NAMES.append(f"c2w_present_kv_{_i}_v")
for _i in range(17):
    _C2W_FUSED_DEFAULT_OUTPUT_NAMES.append(f"c2w_new_conv_state_{_i}")
for _i in range(4):
    _C2W_FUSED_DEFAULT_OUTPUT_NAMES.append(f"c2w_new_transconv_overlap_{_i}")

_MODEL_TYPE_ALLOWED_TASKS = {
    "base": {"voice_clone_icl", "voice_clone_xvec"},
    "custom_voice": {"custom_voice"},
    "voice_design": {"voice_design"},
}


@dataclass
class _FusedBatchSession:
    response_sender: Any
    req: dict
    task_type: Any
    language: str
    speaker: Optional[str]
    instruct: Optional[str]
    spk_embedding: Optional[torch.Tensor]
    ref_codes: Optional[torch.Tensor]
    ref_codec_sum_vec: Optional[torch.Tensor]
    ref_text: Optional[str]
    text_segments: list[str]
    request_start: float
    segment_idx: int = 0
    kv_tensors: Optional[list] = None
    c2w_states: Optional[list] = None
    trailing_text: Optional[list] = None
    next_embed: Optional[torch.Tensor] = None
    text_idx: int = 0
    past_len: int = 0
    frame_idx: int = 0
    done: bool = False

class TritonPythonModel:
    def _is_codec_eos(
        self,
        logits: torch.Tensor,
        full_codec: torch.Tensor,
    ) -> bool:
        codec_eos_id = int(self.weights.codec_eos_id)
        logit_eos = int(logits[:, -1, :].float().argmax(dim=-1).item()) == codec_eos_id
        fc0 = int(full_codec[0, 0].item()) if full_codec.numel() > 0 else -1
        return logit_eos or (fc0 == codec_eos_id)

    def _is_logit_eos(self, logits: torch.Tensor) -> bool:
        codec_eos_id = int(self.weights.codec_eos_id)
        return int(logits[:, -1, :].float().argmax(dim=-1).item()) == codec_eos_id

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
        self.engine_max_prefill_len = int(params.get("engine_max_prefill_len", {}).get(
            "string_value", os.environ.get("ENGINE_MAX_PREFILL_LEN", "128")))
        self.engine_max_decode_len = int(params.get("engine_max_decode_len", {}).get(
            "string_value", os.environ.get("ENGINE_MAX_DECODE_LEN", "512")))
        self.rollover_margin = int(params.get("rollover_margin", {}).get(
            "string_value", os.environ.get("ROLLOVER_MARGIN", "64")))
        self.enable_text_rollover = params.get("enable_text_rollover", {}).get(
            "string_value",
            os.environ.get("ENABLE_TEXT_ROLLOVER", "1"),
        ).strip().lower() not in ("0", "false", "no")
        self.enable_prefix_kv_cache = params.get("enable_prefix_kv_cache", {}).get(
            "string_value",
            os.environ.get("ENABLE_PREFIX_KV_CACHE", "1"),
        ).strip().lower() not in ("0", "false", "no")
        self.prefix_kv_cache_max_entries = int(params.get("prefix_kv_cache_max_entries", {}).get(
            "string_value",
            os.environ.get("PREFIX_KV_CACHE_MAX_ENTRIES", "8")))
        self.code2wav_sliding_window = int(params.get("code2wav_sliding_window", {}).get(
            "string_value",
            os.environ.get("CODE2WAV_SLIDING_WINDOW", "72")))
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
        self._prefix_kv_cache: "OrderedDict[str, list[torch.Tensor]]" = OrderedDict()

        repo_root = Path(_MODEL_DIR).resolve().parent.parent
        self._triton_manifest: Optional[Dict[str, Any]] = None
        for _mf in (
            repo_root / "triton_manifest.json",
            repo_root / "tts_orchestrator" / "1" / "triton_manifest.json",
        ):
            if _mf.is_file():
                try:
                    with open(_mf, encoding="utf-8") as _f:
                        self._triton_manifest = json.load(_f)
                    logger.info(
                        f"[TTS Orchestrator] Loaded triton_manifest.json from {_mf.name}"
                    )
                    break
                except Exception as e:
                    logger.warning(
                        f"[TTS Orchestrator] Failed to load {_mf}: {e}"
                    )
        fused_cfg = repo_root / "talker_code2wav_fused" / "config.pbtxt"
        legacy_force = os.environ.get("USE_LEGACY_TALKER_CODE2WAV", "").strip().lower() in (
            "1", "true", "yes",
        )
        self._use_fused_decode = fused_cfg.exists() and not legacy_force
        self._speech_codec_fused_available = (
            repo_root / "speech_tokenizer_codec_fused" / "config.pbtxt"
        ).exists()

        def _dtype_from_pbtxt(path: Path) -> tuple:
            backend = "onnxruntime"
            dt = torch.float32
            if not path.exists():
                return backend, dt
            raw = path.read_text()
            if "tensorrt" in raw.lower() and "backend" in raw:
                backend = "tensorrt"
            if "TYPE_BF16" in raw:
                dt = torch.bfloat16
            elif "TYPE_FP16" in raw:
                dt = torch.float16
            elif "TYPE_FP32" in raw:
                dt = torch.float32
            return backend, dt

        self._talker_backend = "onnxruntime"
        self._talker_dtype = torch.float32
        self._code2wav_dtype = torch.float32

        if self._use_fused_decode:
            self._talker_backend, self._talker_dtype = _dtype_from_pbtxt(fused_cfg)
            self._code2wav_dtype = self._talker_dtype
            # Fused TRT path: disable prefix-KV cache until cache contract is fully
            # aligned with fused prefill semantics (avoids early-EOS regressions).
            self.enable_prefix_kv_cache = False
            # Manifest I/O dtype (export_09 / generate_triton_configs) must drive Python tensors:
            # config.pbtxt scan can miss BF16 if the on-disk file differs from assemble output, and
            # c2w_* TRT bindings require the same dtype as talker floats (see triton_io_float_dtype).
            if self._triton_manifest:
                raw_io = self._triton_manifest.get("triton_io_float_dtype") or self._triton_manifest.get(
                    "onnx_io_dtype"
                )
                if raw_io:
                    s = str(raw_io).lower().strip()
                    if s in ("bf16", "bfloat16"):
                        self._talker_dtype = torch.bfloat16
                        self._code2wav_dtype = torch.bfloat16
                    elif s in ("fp16", "float16"):
                        self._talker_dtype = torch.float16
                        self._code2wav_dtype = torch.float16
                    elif s in ("fp32", "float32", "float"):
                        self._talker_dtype = torch.float32
                        self._code2wav_dtype = torch.float32
            if self._talker_backend == "tensorrt" and os.environ.get("OVERRIDE_CODE2WAV_BF16", "").strip() in ("1", "true", "yes"):
                self._code2wav_dtype = torch.bfloat16
            logger.info(
                f"[TTS Orchestrator] fused pipeline: talker_code2wav_fused "
                f"backend={self._talker_backend}, talker_dtype={self._talker_dtype}, "
                f"code2wav_dtype={self._code2wav_dtype}"
            )
            if not self._triton_manifest or not isinstance(
                self._triton_manifest.get("code2wav_fused"), dict
            ):
                raise RuntimeError(
                    "Fused decode requires triton_manifest.json with code2wav_fused "
                    "(run export_09 and assemble; manifest is copied to repo root and tts_orchestrator/1/)"
                )
            lay = self._triton_manifest["code2wav_fused"]
            try:
                self._c2w_state_input_names = lay["c2w_state_input_names"]
                self._c2w_state_output_names = lay["c2w_state_output_names"]
                self._code2wav_state_shapes_fused = [
                    tuple(s) for s in lay["initial_state_shapes"]
                ]
                logger.info(
                    f"[TTS Orchestrator] fused c2w layout: "
                    f"decoder_layers={lay.get('num_code2wav_hidden_layers', '?')}, "
                    f"c2w_inputs={len(self._c2w_state_input_names)}"
                )
            except Exception as e:
                raise RuntimeError(
                    f"Invalid code2wav_fused in triton_manifest.json: {e}"
                ) from e
        else:
            talker_config = repo_root / "talker_unified" / "config.pbtxt"
            self._talker_backend, self._talker_dtype = _dtype_from_pbtxt(talker_config)
            code2wav_config = repo_root / "code2wav" / "config.pbtxt"
            if os.environ.get("OVERRIDE_CODE2WAV_BF16", "").strip() in ("1", "true", "yes"):
                self._code2wav_dtype = torch.bfloat16
                logger.info("[TTS Orchestrator] code2wav_dtype=BF16 (OVERRIDE_CODE2WAV_BF16 env)")
            elif code2wav_config.exists():
                _, self._code2wav_dtype = _dtype_from_pbtxt(code2wav_config)
                if self._code2wav_dtype == torch.float32 and self._talker_backend == "tensorrt":
                    self._code2wav_dtype = self._talker_dtype
            elif self._talker_backend == "tensorrt":
                self._code2wav_dtype = self._talker_dtype
            logger.info(
                f"[TTS Orchestrator] legacy pipeline: talker_unified + code2wav "
                f"talker_backend={self._talker_backend}"
            )
            self._c2w_state_input_names = list(_C2W_FUSED_DEFAULT_INPUT_NAMES)
            self._c2w_state_output_names = list(_C2W_FUSED_DEFAULT_OUTPUT_NAMES)
            self._code2wav_state_shapes_fused = list(_CODE2WAV_STATE_SHAPES)

        logger.info(
            f"[TTS Orchestrator] Initialized: variant={self.variant}, num_layers={self.num_layers}, "
            f"fused_decode={self._use_fused_decode}, speech_codec_fused={self._speech_codec_fused_available}, "
            f"code2wav_dtype={self._code2wav_dtype}, "
            f"engine_max_prefill_len={self.engine_max_prefill_len}, "
            f"engine_max_decode_len={self.engine_max_decode_len}, "
            f"rollover_margin={self.rollover_margin}, "
            f"text_rollover={self.enable_text_rollover}, "
            f"prefix_kv_cache={self.enable_prefix_kv_cache}, "
            f"prefix_kv_cache_max_entries={self.prefix_kv_cache_max_entries}, "
            f"code2wav_sliding_window={self.code2wav_sliding_window}"
        )

    def execute(self, requests):
        if self._use_fused_decode and len(requests) > 1:
            self._execute_fused_batched_requests(requests)
            return None
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

    def _execute_fused_batched_requests(self, requests):
        response_senders = [request.get_response_sender() for request in requests]
        sessions: list[_FusedBatchSession] = []
        try:
            for request, response_sender in zip(requests, response_senders):
                try:
                    session = self._prepare_fused_batch_session(request, response_sender)
                    if session is not None:
                        sessions.append(session)
                except torch.cuda.OutOfMemoryError as e:
                    logger.error(f"Request failed (OOM): {e}")
                    self._send_error(response_sender, "out_of_memory: GPU OOM")
                except Exception as e:
                    logger.error(f"Request failed during batched prepare: {e}")
                    logger.error(traceback.format_exc())
                    self._send_error(response_sender, str(e))

            active: list[_FusedBatchSession] = []
            for session in sessions:
                self._activate_next_segment_or_finish(session, active)

            while active:
                ready: list[FusedDecodeTicket[_FusedBatchSession]] = []
                still_active: list[_FusedBatchSession] = []
                for session in active:
                    if session.response_sender.is_cancelled():
                        logger.info("Client disconnected, dropping session from batch")
                        session.done = True
                        continue
                    if (time.monotonic() - session.request_start) > self.request_timeout_sec:
                        self._send_error(session.response_sender, "request_timeout")
                        session.done = True
                        continue
                    if (
                        session.next_embed is None
                        or session.kv_tensors is None
                        or session.c2w_states is None
                    ):
                        session.done = True
                        continue
                    c2w_past_len = int(session.c2w_states[0].shape[2]) if session.c2w_states else -1
                    ready.append(
                        FusedDecodeTicket(
                            payload=session,
                            talker_past_len=session.past_len,
                            c2w_past_len=c2w_past_len,
                        )
                    )
                    still_active.append(session)

                active = []
                for group in group_by_c2w_past_len(ready).values():
                    batch_sessions = [ticket.payload for ticket in group]
                    try:
                        self._run_fused_decode_batch_step(batch_sessions, active)
                    except torch.cuda.OutOfMemoryError as e:
                        for session in batch_sessions:
                            self._send_error(
                                session.response_sender,
                                f"out_of_memory: GPU OOM ({e})",
                            )
                            session.done = True
                    except Exception as e:
                        logger.error(f"Batched decode step failed: {e}")
                        logger.error(traceback.format_exc())
                        for session in batch_sessions:
                            self._send_error(
                                session.response_sender,
                                f"batched_decode_failed: {e}",
                            )
                            session.done = True

                for session in still_active:
                    if not session.done and session not in active and session.next_embed is not None:
                        active.append(session)
        finally:
            for response_sender in response_senders:
                try:
                    response_sender.send(
                        flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
                    )
                except Exception:
                    pass

    def _prepare_fused_batch_session(
        self,
        request,
        response_sender,
    ) -> Optional[_FusedBatchSession]:
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
                "engine_max_prefill_len": self.engine_max_prefill_len,
                "engine_max_decode_len": self.engine_max_decode_len,
                "rollover_margin": self.rollover_margin,
                "enable_text_rollover": self.enable_text_rollover,
                "enable_prefix_kv_cache": self.enable_prefix_kv_cache,
                "prefix_kv_cache_max_entries": self.prefix_kv_cache_max_entries,
                "code2wav_sliding_window": self.code2wav_sliding_window,
                "engine_backend": self._talker_backend,
                "use_fused_decode": self._use_fused_decode,
            }
            self._send_capabilities(response_sender, caps)
            return None

        self._validate_request(req)
        task_type_str = req.get("task_type", "voice_design")
        language = req.get("language", "auto")
        speaker = req.get("speaker")
        instruct = req.get("instruct")
        x_vector_only = req.get("x_vector_only", False)
        task_type = self._parse_task_type(task_type_str, x_vector_only)

        spk_embedding = None
        ref_codes = None
        ref_codec_sum_vec = None
        if task_type.value.startswith("voice_clone"):
            ref_audio_b64 = req.get("ref_audio")
            if ref_audio_b64:
                spk_embedding = self._bls_speaker_encoder(ref_audio_b64)
                if task_type.value == "voice_clone_icl":
                    if self._speech_codec_fused_available:
                        ref_codec_sum_vec = self._bls_speech_tokenizer_codec_fused(
                            ref_audio_b64
                        )
                    else:
                        ref_codes = self._bls_speech_tokenizer(ref_audio_b64)

        ref_text = req.get("ref_text")
        text_segments = self._plan_text_segments(
            text=req.get("text", ""),
            task_type=task_type,
            language=language,
            speaker=speaker,
            instruct=instruct,
            spk_embedding=spk_embedding,
            ref_codes=ref_codes,
            ref_codec_sum_vec=ref_codec_sum_vec,
            ref_text=ref_text,
        )
        return _FusedBatchSession(
            response_sender=response_sender,
            req=req,
            task_type=task_type,
            language=language,
            speaker=speaker,
            instruct=instruct,
            spk_embedding=spk_embedding,
            ref_codes=ref_codes,
            ref_codec_sum_vec=ref_codec_sum_vec,
            ref_text=ref_text,
            text_segments=text_segments,
            request_start=time.monotonic(),
        )

    def _activate_next_segment_or_finish(
        self,
        session: _FusedBatchSession,
        active_out: list[_FusedBatchSession],
    ) -> None:
        while session.segment_idx < len(session.text_segments):
            seg_text = session.text_segments[session.segment_idx]
            plan = self.prefill_builder.build_plan(
                task_type=session.task_type,
                text=seg_text,
                language=session.language,
                speaker=session.speaker,
                instruct=session.instruct,
                spk_embedding=session.spk_embedding,
                ref_codes=session.ref_codes,
                ref_codec_sum_vec=session.ref_codec_sum_vec,
                ref_text=session.ref_text,
            )
            (
                inputs_embeds,
                position_ids,
                effective_prompt_len,
                initial_past_kv_tensors,
                prefix_cache_used,
            ) = self._prepare_segment_prefill(plan)

            cache_pos = torch.zeros(
                inputs_embeds.shape[0],
                FUSED_CHUNK_T,
                device=self.device,
                dtype=torch.int64,
            )
            wav, codec_sum, _, logits, kv_tensors, c2w_states = (
                self._bls_talker_code2wav_fused(
                    inputs_embeds,
                    position_ids,
                    cache_pos,
                    initial_past_kv_tensors,
                    self._create_code2wav_initial_states(),
                )
            )
            self._send_fused_wav(session.response_sender, wav)
            if self._is_logit_eos(logits):
                logger.info("EOS at segment %d step 0", session.segment_idx)
                session.segment_idx += 1
                continue

            text_add = plan.trailing[0] if plan.trailing else self._tts_pad_embed_torch
            session.trailing_text = plan.trailing
            session.next_embed = (codec_sum + text_add).to(torch.float32)
            session.text_idx = 1
            session.kv_tensors = kv_tensors
            session.c2w_states = c2w_states
            session.past_len = effective_prompt_len
            session.frame_idx = 1
            active_out.append(session)
            if len(session.text_segments) > 1:
                logger.info(
                    "Activate segment %d/%d: chars=%d, prompt=%d, trailing=%d, prefix_cache=%s",
                    session.segment_idx + 1,
                    len(session.text_segments),
                    len(seg_text),
                    effective_prompt_len,
                    len(plan.trailing),
                    prefix_cache_used,
                )
            return

        session.done = True
        self._send_audio_chunk(
            session.response_sender,
            np.zeros(1, dtype=np.float32),
            is_final=True,
        )

    def _split_batched_kv_rows(
        self,
        batched_kv_tensors: list[torch.Tensor],
        new_past_lens: list[int],
    ) -> list[list[torch.Tensor]]:
        rows: list[list[torch.Tensor]] = []
        for row_idx, keep in enumerate(new_past_lens):
            row: list[torch.Tensor] = []
            for tensor in batched_kv_tensors:
                row.append(
                    tensor[row_idx : row_idx + 1, :, : int(keep), :].clone().contiguous()
                )
            rows.append(row)
        return rows

    def _run_fused_decode_batch_step(
        self,
        sessions: list[_FusedBatchSession],
        active_out: list[_FusedBatchSession],
    ) -> None:
        batched_input = torch.cat(
            [session.next_embed.to(self._talker_dtype) for session in sessions],
            dim=0,
        )
        batched_pos = torch.cat(
            [
                torch.full(
                    (1, 3, 1),
                    session.past_len,
                    device=self.device,
                    dtype=torch.int64,
                )
                for session in sessions
            ],
            dim=0,
        )
        batched_cache_pos = torch.cat(
            [
                torch.full(
                    (1, FUSED_CHUNK_T),
                    session.frame_idx,
                    device=self.device,
                    dtype=torch.int64,
                )
                for session in sessions
            ],
            dim=0,
        )
        batched_past_kv, past_seq_lens = pad_talker_past_kv(
            [session.kv_tensors for session in sessions],
            device=self.device,
            dtype=self._talker_dtype,
        )
        padded_past_len = int(past_seq_lens.max().item()) if len(sessions) > 0 else 0
        batched_attention = padded_attention_bias(
            past_seq_lens,
            seq=int(batched_input.shape[1]),
            padded_past_len=padded_past_len,
            device=self.device,
            dtype=self._talker_dtype,
        )

        batched_c2w_states = []
        for state_idx in range(len(sessions[0].c2w_states)):
            batched_c2w_states.append(
                torch.cat(
                    [
                        session.c2w_states[state_idx].to(
                            device=self.device,
                            dtype=self._code2wav_dtype,
                        )
                        for session in sessions
                    ],
                    dim=0,
                ).contiguous()
            )

        wav, codec_sum, _, logits, kv_tensors, c2w_states = (
            self._bls_talker_code2wav_fused(
                batched_input,
                batched_pos,
                batched_cache_pos,
                batched_past_kv,
                batched_c2w_states,
                attention_bias_override=batched_attention,
                past_seq_lens_override=past_seq_lens,
            )
        )
        new_past_lens = [session.past_len + int(batched_input.shape[1]) for session in sessions]
        split_kv = self._split_batched_kv_rows(kv_tensors, new_past_lens)
        split_c2w = []
        for row_idx in range(len(sessions)):
            row_states = [t[row_idx : row_idx + 1].clone().contiguous() for t in c2w_states]
            split_c2w.append(row_states)

        for row_idx, session in enumerate(sessions):
            row_logits = logits[row_idx : row_idx + 1]
            eos = self._is_logit_eos(row_logits)
            if not eos:
                self._send_fused_wav(session.response_sender, wav[row_idx : row_idx + 1])
                text_add = (
                    session.trailing_text[session.text_idx]
                    if session.text_idx < len(session.trailing_text)
                    else self._tts_pad_embed_torch
                )
                session.text_idx += 1
                session.next_embed = (
                    codec_sum[row_idx : row_idx + 1] + text_add
                ).to(torch.float32)
                session.kv_tensors = split_kv[row_idx]
                session.c2w_states = split_c2w[row_idx]
                session.past_len = new_past_lens[row_idx]
                session.frame_idx += 1
                active_out.append(session)
                continue

            session.segment_idx += 1
            session.kv_tensors = None
            session.c2w_states = None
            session.next_embed = None
            session.trailing_text = None
            self._activate_next_segment_or_finish(session, active_out)

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
                "engine_max_prefill_len": self.engine_max_prefill_len,
                "engine_max_decode_len": self.engine_max_decode_len,
                "rollover_margin": self.rollover_margin,
                "enable_text_rollover": self.enable_text_rollover,
                "enable_prefix_kv_cache": self.enable_prefix_kv_cache,
                "prefix_kv_cache_max_entries": self.prefix_kv_cache_max_entries,
                "code2wav_sliding_window": self.code2wav_sliding_window,
                "engine_backend": self._talker_backend,
                "use_fused_decode": self._use_fused_decode,
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
        ref_codec_sum_vec = None
        if task_type.value.startswith("voice_clone"):
            ref_audio_b64 = req.get("ref_audio")
            try:
                if ref_audio_b64:
                    t0 = time.perf_counter()
                    spk_embedding = self._bls_speaker_encoder(ref_audio_b64)
                    logger.debug(f"speaker_encoder: {(time.perf_counter() - t0) * 1000:.1f}ms")
                if task_type.value == "voice_clone_icl" and ref_audio_b64:
                    t0 = time.perf_counter()
                    if self._speech_codec_fused_available:
                        ref_codec_sum_vec = self._bls_speech_tokenizer_codec_fused(ref_audio_b64)
                        logger.debug(
                            f"speech_tokenizer_codec_fused: {(time.perf_counter() - t0) * 1000:.1f}ms"
                        )
                    else:
                        ref_codes = self._bls_speech_tokenizer(ref_audio_b64)
                        logger.debug(
                            f"speech_tokenizer_encoder: {(time.perf_counter() - t0) * 1000:.1f}ms"
                        )
            except Exception as e:
                raise RuntimeError(f"Voice clone audio processing failed: {e}") from e

        ref_text = req.get("ref_text")
        request_start = time.monotonic()

        text_segments = self._plan_text_segments(
            text=text,
            task_type=task_type,
            language=language,
            speaker=speaker,
            instruct=instruct,
            spk_embedding=spk_embedding,
            ref_codes=ref_codes,
            ref_codec_sum_vec=ref_codec_sum_vec,
            ref_text=ref_text,
        )

        for seg_idx, seg_text in enumerate(text_segments):
            if response_sender.is_cancelled():
                logger.info("Client disconnected before segment %d", seg_idx)
                return
            try:
                plan = self.prefill_builder.build_plan(
                    task_type=task_type,
                    text=seg_text,
                    language=language,
                    speaker=speaker,
                    instruct=instruct,
                    spk_embedding=spk_embedding,
                    ref_codes=ref_codes,
                    ref_codec_sum_vec=ref_codec_sum_vec,
                    ref_text=ref_text,
                )
            except Exception as e:
                raise RuntimeError(f"Prefill build failed: {e}") from e

            inputs_embeds, position_ids, effective_prompt_len, initial_past_kv_tensors, prefix_cache_used = (
                self._prepare_segment_prefill(plan)
            )
            trailing_text = plan.trailing
            batch = int(inputs_embeds.shape[0])
            send_terminal_marker = seg_idx == len(text_segments) - 1

            if len(text_segments) > 1:
                logger.info(
                    "Synthesize segment %d/%d: chars=%d, prefill=%d, trailing=%d, prefix_cache=%s",
                    seg_idx + 1,
                    len(text_segments),
                    len(seg_text),
                    effective_prompt_len,
                    len(trailing_text),
                    prefix_cache_used,
                )

            if self._use_fused_decode:
                self._generation_loop_fused(
                    response_sender,
                    inputs_embeds,
                    position_ids,
                    trailing_text,
                    effective_prompt_len,
                    batch,
                    request_start,
                    send_terminal_marker=send_terminal_marker,
                    initial_past_kv_tensors=initial_past_kv_tensors,
                )
                continue

            self._generation_loop_legacy(
                response_sender,
                inputs_embeds,
                position_ids,
                trailing_text,
                effective_prompt_len,
                batch,
                request_start,
                send_terminal_marker=send_terminal_marker,
                initial_past_kv_tensors=initial_past_kv_tensors,
            )

    def _prepare_segment_prefill(self, plan):
        full_prefill_embeds = plan.prefill_embeds
        full_prefill_len = int(full_prefill_embeds.shape[1])

        inputs_embeds = full_prefill_embeds
        effective_prompt_len = full_prefill_len
        initial_past_kv_tensors = None
        prefix_cache_used = False
        if (
            self.enable_prefix_kv_cache
            and plan.prefix_cache_key
            and plan.cacheable_prefix_embeds is not None
            and plan.request_prefill_embeds is not None
        ):
            cached_prefix_len = int(plan.cacheable_prefix_embeds.shape[1])
            request_prefill_len = int(plan.request_prefill_embeds.shape[1])
            if request_prefill_len <= self.engine_max_prefill_len:
                prefix_kv = self._get_or_build_prefix_kv(
                    plan.prefix_cache_key,
                    plan.cacheable_prefix_embeds,
                )
            else:
                prefix_kv = None
            if prefix_kv is not None:
                prefix_cache_used = True
                initial_past_kv_tensors = prefix_kv
                inputs_embeds = plan.request_prefill_embeds
                effective_prompt_len = cached_prefix_len + int(inputs_embeds.shape[1])
                logger.info(
                    "Prefix KV cache hit: key=%s, prefix_len=%d",
                    plan.prefix_cache_key,
                    cached_prefix_len,
                )

        if not prefix_cache_used and full_prefill_len > self.engine_max_prefill_len:
            logger.warning(
                "Segment prefill length %d exceeds engine max %d, truncating",
                full_prefill_len,
                self.engine_max_prefill_len,
            )
            inputs_embeds = full_prefill_embeds[:, : self.engine_max_prefill_len, :].contiguous()
            effective_prompt_len = self.engine_max_prefill_len

        batch = int(inputs_embeds.shape[0])
        seq = int(inputs_embeds.shape[1])
        if prefix_cache_used:
            position_ids_1d = torch.arange(
                effective_prompt_len - seq,
                effective_prompt_len,
                device=self.device,
                dtype=torch.int64,
            )
            position_ids = position_ids_1d.reshape(1, 1, -1).expand(batch, 3, seq)
        else:
            position_ids_1d = torch.arange(seq, device=self.device, dtype=torch.int64)
            position_ids = position_ids_1d.reshape(1, 1, -1).expand(batch, 3, seq)
        return inputs_embeds, position_ids, effective_prompt_len, initial_past_kv_tensors, prefix_cache_used

    def _get_cached_prefix_kv(self, key: str) -> Optional[list[torch.Tensor]]:
        hit = self._prefix_kv_cache.get(key)
        if hit is None:
            return None
        self._prefix_kv_cache.move_to_end(key)
        return [t.clone().contiguous() for t in hit]

    def _put_cached_prefix_kv(self, key: str, kv_tensors: list[torch.Tensor]) -> None:
        self._prefix_kv_cache[key] = [t.clone().contiguous() for t in kv_tensors]
        self._prefix_kv_cache.move_to_end(key)
        while len(self._prefix_kv_cache) > self.prefix_kv_cache_max_entries:
            old_key, _ = self._prefix_kv_cache.popitem(last=False)
            logger.info("Evict prefix KV cache entry: %s", old_key)

    def _get_or_build_prefix_kv(
        self,
        cache_key: str,
        prefix_embeds: torch.Tensor,
    ) -> Optional[list[torch.Tensor]]:
        cached = self._get_cached_prefix_kv(cache_key)
        if cached is not None:
            return cached

        prefix_len = int(prefix_embeds.shape[1])
        batch = int(prefix_embeds.shape[0])
        prefix_position_ids = torch.arange(
            prefix_len, device=self.device, dtype=torch.int64
        ).reshape(1, 1, -1).expand(batch, 3, prefix_len)

        try:
            if self._use_fused_decode:
                cache_pos = torch.zeros(batch, FUSED_CHUNK_T, device=self.device, dtype=torch.int64)
                _, _, _, _, kv_tensors, _ = self._bls_talker_code2wav_fused(
                    prefix_embeds,
                    prefix_position_ids,
                    cache_pos,
                    None,
                    self._create_code2wav_initial_states(),
                )
            else:
                _, _, _, kv_tensors = self._bls_talker(prefix_embeds, prefix_position_ids)
        except Exception as e:
            logger.warning("Prefix KV cache build failed for key=%s: %s", cache_key, e)
            return None

        self._put_cached_prefix_kv(cache_key, kv_tensors)
        logger.info("Built prefix KV cache entry: key=%s, prefix_len=%d", cache_key, prefix_len)
        return self._get_cached_prefix_kv(cache_key)

    def _segment_budget_for_task(self, task_type) -> int:
        if task_type.value == "voice_design":
            return max(16, self.engine_max_prefill_len - max(16, self.rollover_margin // 2))
        return max(16, self.engine_max_decode_len - max(16, self.rollover_margin))

    def _segment_within_limits(
        self,
        task_type,
        text: str,
        language: str,
        speaker: Optional[str],
        instruct: Optional[str],
        spk_embedding: Optional[torch.Tensor],
        ref_codes: Optional[torch.Tensor],
        ref_codec_sum_vec: Optional[torch.Tensor],
        ref_text: Optional[str],
    ) -> tuple[bool, int, int]:
        plan = self.prefill_builder.build_plan(
            task_type=task_type,
            text=text,
            language=language,
            speaker=speaker,
            instruct=instruct,
            spk_embedding=spk_embedding,
            ref_codes=ref_codes,
            ref_codec_sum_vec=ref_codec_sum_vec,
            ref_text=ref_text,
        )
        prefill_len = int(plan.prefill_embeds.shape[1])
        input_len = prefill_len
        if (
            self.enable_prefix_kv_cache
            and plan.prefix_cache_key
            and plan.request_prefill_embeds is not None
        ):
            input_len = int(plan.request_prefill_embeds.shape[1])
        trailing_len = int(len(plan.trailing))
        decode_budget = max(1, self.engine_max_decode_len - self.rollover_margin)
        total_budget = self.engine_max_prefill_len + self.engine_max_decode_len - max(16, self.rollover_margin // 2)
        within = input_len <= self.engine_max_prefill_len and prefill_len <= total_budget
        if task_type.value != "voice_design":
            within = within and trailing_len <= decode_budget
        return within, input_len, trailing_len

    def _plan_text_segments(
        self,
        text: str,
        task_type,
        language: str,
        speaker: Optional[str],
        instruct: Optional[str],
        spk_embedding: Optional[torch.Tensor],
        ref_codes: Optional[torch.Tensor],
        ref_codec_sum_vec: Optional[torch.Tensor],
        ref_text: Optional[str],
    ) -> list[str]:
        if not self.enable_text_rollover:
            return [text]

        coarse_budget = self._segment_budget_for_task(task_type)
        pending = split_text_for_token_budget(text, self.tokenizer, coarse_budget)
        if not pending:
            return [text]

        planned: list[str] = []
        while pending:
            seg_text = pending.pop(0)
            ok, prefill_len, trailing_len = self._segment_within_limits(
                task_type=task_type,
                text=seg_text,
                language=language,
                speaker=speaker,
                instruct=instruct,
                spk_embedding=spk_embedding,
                ref_codes=ref_codes,
                ref_codec_sum_vec=ref_codec_sum_vec,
                ref_text=ref_text,
            )
            if ok:
                planned.append(seg_text)
                continue

            refined_budget = max(16, coarse_budget // 2)
            refined = split_text_for_token_budget(seg_text, self.tokenizer, refined_budget)
            if len(refined) <= 1:
                logger.warning(
                    "Segment still exceeds safe limits but cannot be refined further: "
                    "chars=%d prefill=%d trailing=%d",
                    len(seg_text),
                    prefill_len,
                    trailing_len,
                )
                planned.append(seg_text)
                continue

            logger.info(
                "Split oversized segment: chars=%d prefill=%d trailing=%d -> %d subsegments",
                len(seg_text),
                prefill_len,
                trailing_len,
                len(refined),
            )
            pending = refined + pending

        if len(planned) > 1:
            logger.info("Long-text rollover planned: %d segments", len(planned))
        return planned

    def _generation_loop_fused(
        self,
        response_sender,
        inputs_embeds,
        position_ids,
        trailing_text,
        S,
        B,
        request_start,
        send_terminal_marker: bool = True,
        initial_past_kv_tensors=None,
    ):
        """Prefill + decode via talker_code2wav_fused (chunk_T=1 wav per step)."""
        c2w_states = self._create_code2wav_initial_states()
        cache_pos = torch.zeros(B, FUSED_CHUNK_T, device=self.device, dtype=torch.int64)

        t0 = time.perf_counter()
        wav, codec_sum, full_codec, logits, kv_tensors, c2w_states = (
            self._bls_talker_code2wav_fused(
                inputs_embeds, position_ids, cache_pos, initial_past_kv_tensors, c2w_states
            )
        )
        logger.debug(f"fused prefill: {(time.perf_counter() - t0) * 1000:.1f}ms")

        self._send_fused_wav(response_sender, wav)

        if self._is_logit_eos(logits):
            logger.info("EOS at step 0")
            if send_terminal_marker:
                self._send_audio_chunk(response_sender, np.zeros(1, dtype=np.float32), is_final=True)
            return

        text_idx = 0
        next_embed = (
            (codec_sum + trailing_text[text_idx]).to(torch.float32)
            if trailing_text
            else (codec_sum + self._tts_pad_embed_torch).to(torch.float32)
        )
        text_idx += 1
        position_id = torch.full((B, 3, 1), S, device=self.device, dtype=torch.int64)
        frame_idx = 1
        max_kv_len = self.engine_max_prefill_len + self.engine_max_decode_len - max(16, self.rollover_margin // 2)

        for step in range(1, self.max_steps):
            if S + step > max_kv_len:
                logger.warning("KV cache approaching limit, forcing EOS")
                break
            if response_sender.is_cancelled():
                logger.info("Client disconnected, stopping generation")
                break
            if (time.monotonic() - request_start) > self.request_timeout_sec:
                self._send_error(response_sender, "request_timeout")
                return

            cache_pos = torch.full(
                (B, FUSED_CHUNK_T), frame_idx, device=self.device, dtype=torch.int64
            )
            t0 = time.perf_counter()
            wav, codec_sum, full_codec, logits, kv_tensors, c2w_states = (
                self._bls_talker_code2wav_fused(
                    next_embed, position_id, cache_pos, kv_tensors, c2w_states
                )
            )
            logger.debug(f"fused step {step}: {(time.perf_counter() - t0) * 1000:.1f}ms")
            # Do not stream the EOS frame's wav: codec logits target EOS while Code2Wav still
            # runs on clamped/special indices, often audible as a click or buzz at the tail.
            if self._is_logit_eos(logits):
                logger.info(f"EOS at step {step}")
                break

            self._send_fused_wav(response_sender, wav)

            text_add = (
                trailing_text[text_idx]
                if text_idx < len(trailing_text)
                else self._tts_pad_embed_torch
            )
            text_idx += 1
            next_embed = (codec_sum + text_add).to(torch.float32)
            position_id = torch.full((B, 3, 1), S + step, device=self.device, dtype=torch.int64)
            frame_idx += 1

        if send_terminal_marker:
            self._send_audio_chunk(response_sender, np.zeros(1, dtype=np.float32), is_final=True)
        logger.info("Generation complete (fused)")

    def _send_fused_wav(self, response_sender, wav: torch.Tensor):
        w = wav
        if w.dim() == 3:
            w = w.reshape(-1)
        else:
            w = w.flatten()
        self._send_audio_chunk(response_sender, w.cpu().float().numpy(), is_final=False)

    def _generation_loop_legacy(
        self,
        response_sender,
        inputs_embeds,
        position_ids,
        trailing_text,
        S,
        B,
        request_start,
        send_terminal_marker: bool = True,
        initial_past_kv_tensors=None,
    ):
        t0 = time.perf_counter()
        codec_sum, full_codec, logits, kv_tensors = self._bls_talker(
            inputs_embeds, position_ids, initial_past_kv_tensors
        )
        logger.debug(f"talker prefill: {(time.perf_counter() - t0) * 1000:.1f}ms")
        codec_frame_buffer = []
        code2wav_states = self._create_code2wav_initial_states()
        frame_index = 0
        text_idx = 0

        if self._is_codec_eos(logits, full_codec):
            logger.info("EOS at step 0")
            if send_terminal_marker:
                self._send_audio_chunk(response_sender, np.zeros(1, dtype=np.float32), is_final=True)
            return

        codec_frame_buffer.append(self._full_codec_to_frame(full_codec))

        next_embed = (codec_sum + trailing_text[text_idx]).to(self._talker_dtype)
        text_idx += 1
        position_id = torch.full((B, 3, 1), S, device=self.device, dtype=torch.int64)

        max_kv_len = self.engine_max_prefill_len + self.engine_max_decode_len - max(16, self.rollover_margin // 2)
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

            if self._is_codec_eos(logits, full_codec):
                logger.info(f"EOS at step {step}")
                break

            code2wav_states, frame_index = self._flush_code2wav_buffer(
                response_sender, codec_frame_buffer, code2wav_states, frame_index, is_final=False
            )

            text_add = trailing_text[text_idx] if text_idx < len(trailing_text) else self._tts_pad_embed_torch
            text_idx += 1
            next_embed = (codec_sum + text_add).to(self._talker_dtype)
            position_id = torch.full((B, 3, 1), S + step, device=self.device, dtype=torch.int64)

        _, _ = self._flush_code2wav_buffer(
            response_sender, codec_frame_buffer, code2wav_states, frame_index, is_final=True
        )
        if send_terminal_marker:
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
        if pos_ids.dim() == 3:
            pos_ids = pos_ids.unsqueeze(-1).contiguous()
        expected_batch = int(input_embeds.shape[0])
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
        codec_sum = self._maybe_fix_trt_batch_axis(codec_sum, expected_batch)
        full_codec = self._maybe_fix_trt_batch_axis(full_codec, expected_batch)
        logits = self._maybe_fix_trt_batch_axis(logits, expected_batch)
        kv_tensors = []
        for i in range(self.num_layers):
            k = self._tensor_from_response_torch(response, f"present_kv_{i}_k")
            v = self._tensor_from_response_torch(response, f"present_kv_{i}_v")
            k = self._maybe_fix_trt_batch_axis(k, expected_batch)
            v = self._maybe_fix_trt_batch_axis(v, expected_batch)
            # Force full copy + dtype: TRT returns BF16; fallback/triton path may yield FP32.
            # cpu().numpy() route guarantees we get correct dtype on device.
            k = torch.from_numpy(k.cpu().float().numpy()).to(
                device=self.device, dtype=torch.float32
            ).contiguous()
            v = torch.from_numpy(v.cpu().float().numpy()).to(
                device=self.device, dtype=torch.float32
            ).contiguous()
            kv_tensors.append(k)
            kv_tensors.append(v)

        return codec_sum, full_codec, logits, kv_tensors

    def _bls_talker_code2wav_fused(
        self,
        input_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        cache_position: torch.Tensor,
        past_kv_tensors,
        c2w_states: list,
        attention_bias_override: Optional[torch.Tensor] = None,
        past_seq_lens_override: Optional[torch.Tensor] = None,
    ):
        """BLS talker_code2wav_fused: prefill or one decode step with chunk_T=1 wav."""
        inp_emb = input_embeds.contiguous().to(self._talker_dtype)
        pos_ids = position_ids.contiguous()
        if pos_ids.dim() == 3:
            pos_ids = pos_ids.unsqueeze(-1).contiguous()
        cache_pos = cache_position.to(device=self.device, dtype=torch.float32).contiguous()
        batch = int(input_embeds.shape[0])
        seq = int(input_embeds.shape[1])
        chunk_t = int(cache_pos.shape[1])
        use_dummy_past_kv = past_kv_tensors is None

        if use_dummy_past_kv:
            if past_seq_lens_override is None:
                past_seq_lens = uniform_past_seq_lens(batch, 0, self.device)
            else:
                past_seq_lens = past_seq_lens_override
            if attention_bias_override is None:
                attention_bias = padded_attention_bias(
                    past_seq_lens,
                    seq,
                    _FUSED_DUMMY_PAST_LEN,
                    self.device,
                    self._talker_dtype,
                )
            else:
                attention_bias = attention_bias_override
        else:
            past_len = int(past_kv_tensors[0].shape[2])
            past_seq_lens = past_seq_lens_override
            if past_seq_lens is None:
                past_seq_lens = uniform_past_seq_lens(batch, past_len, self.device)
            if attention_bias_override is None:
                attention_bias = zeros_attention_bias(
                    batch=batch,
                    seq=seq,
                    past_len=past_len,
                    device=self.device,
                    dtype=self._talker_dtype,
                )
            else:
                attention_bias = attention_bias_override
        try:
            inputs = [
                pb_utils.Tensor.from_dlpack("input_embeds", inp_emb),
                pb_utils.Tensor.from_dlpack("position_ids", pos_ids),
                pb_utils.Tensor.from_dlpack("attention_bias", attention_bias.contiguous()),
                pb_utils.Tensor.from_dlpack("cache_position", cache_pos),
            ]
        except Exception as e:
            logger.error(
                f"fused: from_dlpack input_embeds/position_ids/attention_bias/"
                f"cache_position: {e}"
            )
            raise

        c2w_past_len = int(c2w_states[0].shape[2]) if c2w_states else 0
        c2w_key_total = min(c2w_past_len + chunk_t, self.code2wav_sliding_window)
        if c2w_key_total <= 0:
            c2w_key_total = 1
        c2w_attention_bias = torch.zeros(
            (batch, 1, chunk_t, c2w_key_total),
            device=self.device,
            dtype=self._code2wav_dtype,
        )
        try:
            inputs.append(pb_utils.Tensor.from_dlpack("c2w_attention_bias", c2w_attention_bias.contiguous()))
        except Exception as e:
            logger.error(f"fused: from_dlpack c2w_attention_bias: {e}")
            raise

        if use_dummy_past_kv:
            for i in range(self.num_layers):
                dummy_k = torch.zeros(
                    (batch, self.kv_heads, _FUSED_DUMMY_PAST_LEN, self.head_dim),
                    device=self.device,
                    dtype=self._talker_dtype,
                )
                dummy_v = torch.zeros(
                    (batch, self.kv_heads, _FUSED_DUMMY_PAST_LEN, self.head_dim),
                    device=self.device,
                    dtype=self._talker_dtype,
                )
                try:
                    inputs.append(pb_utils.Tensor.from_dlpack(f"past_kv_{i}_k", dummy_k))
                    inputs.append(pb_utils.Tensor.from_dlpack(f"past_kv_{i}_v", dummy_v))
                except Exception as e:
                    logger.error(f"fused: from_dlpack dummy past_kv {i}: {e}")
                    raise
        else:
            for i in range(self.num_layers):
                k = past_kv_tensors[2 * i].clone().to(
                    device=self.device, dtype=self._talker_dtype
                ).contiguous()
                v = past_kv_tensors[2 * i + 1].clone().to(
                    device=self.device, dtype=self._talker_dtype
                ).contiguous()
                try:
                    inputs.append(pb_utils.Tensor.from_dlpack(f"past_kv_{i}_k", k))
                    inputs.append(pb_utils.Tensor.from_dlpack(f"past_kv_{i}_v", v))
                except Exception as e:
                    logger.error(f"fused: from_dlpack talker past_kv {i}: {e}")
                    raise

        if len(c2w_states) != len(self._c2w_state_input_names):
            raise RuntimeError(
                f"c2w_states length {len(c2w_states)} != expected {len(self._c2w_state_input_names)}"
            )
        for in_name, st in zip(self._c2w_state_input_names, c2w_states):
            self._append_state_tensor_input(
                inputs, in_name, st, self._code2wav_dtype, "fused"
            )

        out_names = ["wav", "codec_sum", "full_codec", "logits"]
        for i in range(self.num_layers):
            out_names.append(f"present_kv_{i}_k")
            out_names.append(f"present_kv_{i}_v")
        out_names.extend(self._c2w_state_output_names)

        request = pb_utils.InferenceRequest(
            model_name="talker_code2wav_fused",
            inputs=inputs,
            requested_output_names=out_names,
        )
        response = request.exec()
        if response.has_error():
            raise RuntimeError(
                f"talker_code2wav_fused BLS error: {response.error().message()}"
            )

        wav = self._tensor_from_response_torch(response, "wav")
        codec_sum = self._tensor_from_response_torch(response, "codec_sum")
        full_codec = self._tensor_from_response_torch(response, "full_codec")
        logits = self._tensor_from_response_torch(response, "logits")
        wav = self._maybe_fix_trt_batch_axis(wav, batch)
        codec_sum = self._maybe_fix_trt_batch_axis(codec_sum, batch)
        full_codec = self._maybe_fix_trt_batch_axis(full_codec, batch)
        logits = self._maybe_fix_trt_batch_axis(logits, batch)

        kv_tensors = []
        for i in range(self.num_layers):
            k = self._tensor_from_response_torch(response, f"present_kv_{i}_k")
            v = self._tensor_from_response_torch(response, f"present_kv_{i}_v")
            k = self._maybe_fix_trt_batch_axis(k, batch)
            v = self._maybe_fix_trt_batch_axis(v, batch)
            k = torch.from_numpy(k.cpu().float().numpy()).to(
                device=self.device, dtype=torch.float32
            ).contiguous()
            v = torch.from_numpy(v.cpu().float().numpy()).to(
                device=self.device, dtype=torch.float32
            ).contiguous()
            if use_dummy_past_kv and k.shape[2] > 0:
                # Drop the dummy prefill slot so downstream decode sees true past_len=0 semantics.
                k = k[:, :, 1:, :].contiguous()
                v = v[:, :, 1:, :].contiguous()
            kv_tensors.append(k)
            kv_tensors.append(v)

        new_c2w = []
        for out_name in self._c2w_state_output_names:
            t = self._tensor_from_response_torch(response, out_name)
            t = self._maybe_fix_trt_batch_axis(t, batch)
            t = self._clip_code2wav_state_window(
                out_name,
                t.to(device=self.device, dtype=torch.float32).contiguous(),
            )
            new_c2w.append(t)

        return wav, codec_sum, full_codec, logits, kv_tensors, new_c2w

    def _maybe_fix_trt_batch_axis(
        self,
        tensor: torch.Tensor,
        expected_batch: int,
    ) -> torch.Tensor:
        if self._talker_backend != "tensorrt":
            return tensor
        if tensor.dim() == 0 or tensor.shape[0] == expected_batch:
            return tensor
        if expected_batch == 1 and tensor.shape[0] == 3:
            return tensor[:1].clone()
        return tensor

    def _append_state_tensor_input(
        self,
        inputs: list,
        name: str,
        tensor: torch.Tensor,
        dtype: torch.dtype,
        log_prefix: str,
    ) -> None:
        t = tensor.contiguous().to(device=self.device, dtype=dtype).contiguous()
        if 0 in t.shape:
            # Code2wav cold-start uses c2w past_kv with T=0 (see SlidingWindowKVCache.get_seq_length).
            # Do NOT pad T to 1 here: that would make S_past==1 and break causal math in the fused graph.
            # Talker dummy past (T=1 + past_seq_lens=0 + bias) is talker-only; c2w is separate.
            # 0-numel: prefer zeros_like (stable layout); some PyTorch builds still fail DLPack on 0-numel tensors.
            z = torch.zeros_like(t, dtype=dtype, device=self.device).contiguous()
            try:
                inputs.append(pb_utils.Tensor.from_dlpack(name, z))
            except Exception as e:
                logger.warning(
                    "%s: from_dlpack 0-size %s failed (%s); using NumPy empty (0 payload bytes)",
                    log_prefix,
                    name,
                    e,
                )
                shape = tuple(int(dim) for dim in t.shape)
                inputs.append(pb_utils.Tensor(name, np.empty(shape, dtype=np.float32)))
            return
        try:
            inputs.append(pb_utils.Tensor.from_dlpack(name, t))
        except Exception as e:
            logger.error(f"{log_prefix}: from_dlpack {name}: {e}")
            raise

    def _clip_code2wav_state_window(
        self,
        name: str,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        if tensor.dim() < 4:
            return tensor
        if "past_kv" not in name and "present_kv" not in name:
            return tensor
        max_kv_t = self.code2wav_sliding_window
        # Fused TRT profile keeps c2w past_kv max at sliding_window-1 so that
        # c2w_attention_bias key_total = past_kv + chunk_t stays within window.
        if name.startswith("c2w_"):
            max_kv_t = max(1, self.code2wav_sliding_window - 1)
        if tensor.shape[2] <= max_kv_t:
            return tensor
        return tensor[:, :, -max_kv_t:, :].contiguous()

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

    def _bls_speech_tokenizer_codec_fused(self, ref_audio_b64: str) -> torch.Tensor:
        """Fused speech tokenizer + ref codec sum (ICL). Returns [1, 1, H] on GPU."""
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
            model_name="speech_tokenizer_codec_fused",
            inputs=inputs,
            requested_output_names=["ref_codec_sum_vec"],
        )
        response = request.exec()
        if response.has_error():
            raise RuntimeError(
                f"speech_tokenizer_codec_fused BLS error: {response.error().message()}"
            )
        return self._tensor_from_response_torch(response, "ref_codec_sum_vec")

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
        """Zero state tensors for code2wav / fused c2w (GPU, correct dtype)."""
        shapes = (
            self._code2wav_state_shapes_fused
            if self._use_fused_decode
            else _CODE2WAV_STATE_SHAPES
        )
        return [
            torch.zeros(shape, device=self.device, dtype=self._code2wav_dtype)
            for shape in shapes
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
                frame_index, frame_index + CODE2WAV_CHUNK_T, device=self.device, dtype=torch.float32
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
                frame_index, frame_index + CODE2WAV_CHUNK_T, device=self.device, dtype=torch.float32
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
        cache_position = cache_position.to(device=self.device, dtype=torch.float32).contiguous()
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
            self._append_state_tensor_input(
                inputs,
                state_input_names[i],
                t,
                self._code2wav_dtype,
                "code2wav",
            )

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
            new_states.append(
                self._clip_code2wav_state_window(
                    f"present_kv_{i}_k",
                    self._tensor_from_response_torch(response, f"present_kv_{i}_k"),
                )
            )
            new_states.append(
                self._clip_code2wav_state_window(
                    f"present_kv_{i}_v",
                    self._tensor_from_response_torch(response, f"present_kv_{i}_v"),
                )
            )
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

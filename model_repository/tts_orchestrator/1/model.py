"""
TTS Orchestrator — Triton Python BLS Backend (Continuous Batching, Phase 2).

Architecture: vLLM backend pattern (decoupled + max_batch_size=0 + background threads).

Three-thread model:
  1. Triton execute() thread: parse request, create session, enqueue, return None immediately.
  2. Engine thread (AsyncIO event loop): continuous decode loop — drain new sessions,
     prefill (interleaved), batch decode, flow control, EOS/timeout handling.
  3. Response thread: dequeue (audio_chunk, response_sender) and send to client.

Production pipeline (fused only):
  PrefillBuilder (in-process embeddings) -> BLS talker_code2wav_fused (prefill + decode + wav).

Streaming text input:
  Gateway sends init/append_text/text_complete actions. FlowState: IDLE / ACTIVE / DONE.
  Streaming init with empty text uses UNASSIGNED_SLOT until first text arrives.
  RatioTracker: EMA for decode_steps/text_tokens; dynamic segment split when KV budget tight.
  No PAUSED state — pad until EOS or request timeout.
"""

import asyncio
import json
import logging
import os
import queue
import sys
import threading
import time
import traceback
from collections import OrderedDict
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
from ratio_tracker import RatioTracker
from session_manager import (
    UNASSIGNED_SLOT,
    BatchScheduler,
    FlowState,
    SessionManager,
    TTSSession,
    generate_session_id,
)
from decode_fsm import DecodePhase, DecodeSessionFSM
import mlfq_scheduler as mlfq
from text_segmenter import split_text_for_token_budget

SAMPLES_PER_CODEC_FRAME = 1920
FUSED_CHUNK_T = 1
_FUSED_DUMMY_PAST_LEN = 1

_MODEL_TYPE_ALLOWED_TASKS = {
    "base": {"voice_clone_icl", "voice_clone_xvec"},
    "custom_voice": {"custom_voice"},
    "voice_design": {"voice_design"},
}


class TritonPythonModel:

    # ── EOS helpers ──

    def _is_logit_eos(self, logits: torch.Tensor) -> bool:
        codec_eos_id = int(self.weights.codec_eos_id)
        return int(logits[:, -1, :].float().argmax(dim=-1).item()) == codec_eos_id

    def _is_codec_eos(self, full_codec: torch.Tensor) -> bool:
        codec_eos_id = int(self.weights.codec_eos_id)
        return int(full_codec[:, 0].item()) == codec_eos_id

    def _build_sampling_inputs(
        self,
        batch: int,
        token_counts: torch.Tensor,
    ) -> tuple:
        """Build engine-side sampling inputs: token_counts, gumbel_noise, temperature, penalty."""
        if self._do_sample:
            u = torch.rand(batch, 50, device=self.device, dtype=torch.float32).clamp(1e-8, 1.0)
            gumbel_noise = -torch.log(-torch.log(u))
        else:
            gumbel_noise = torch.zeros(batch, 50, device=self.device, dtype=torch.float32)

        temperature = torch.full(
            (batch, 1), self._temperature if self._do_sample else 1.0,
            device=self.device, dtype=torch.float32,
        )
        penalty = torch.full(
            (batch, 1), self._repetition_penalty,
            device=self.device, dtype=torch.float32,
        )
        return token_counts.contiguous(), gumbel_noise, temperature, penalty

    # ── Lifecycle ──

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
        self._max_pad_steps = int(params.get("max_pad_steps", {}).get(
            "string_value", os.environ.get("MAX_PAD_STEPS", "500")))
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
            os.environ.get("PREFIX_KV_CACHE_MAX_ENTRIES", "16")))
        self._ratio_tracker = RatioTracker(
            initial=float(params.get("ratio_initial", {}).get(
                "string_value", os.environ.get("RATIO_INITIAL", "8.0"))),
            alpha=float(params.get("ratio_alpha", {}).get(
                "string_value", os.environ.get("RATIO_ALPHA", "0.1"))),
            overflow_alpha=float(params.get("ratio_overflow_alpha", {}).get(
                "string_value", os.environ.get("RATIO_OVERFLOW_ALPHA", "0.5"))),
            min_ratio=float(params.get("ratio_min", {}).get(
                "string_value", os.environ.get("RATIO_MIN", "2.0"))),
            max_ratio=float(params.get("ratio_max", {}).get(
                "string_value", os.environ.get("RATIO_MAX", "15.0"))),
        )
        self.code2wav_sliding_window = int(params.get("code2wav_sliding_window", {}).get(
            "string_value",
            os.environ.get("CODE2WAV_SLIDING_WINDOW", "72")))
        self._temperature = float(params.get("temperature", {}).get(
            "string_value", os.environ.get("TEMPERATURE", "0.9")))
        self._repetition_penalty = float(params.get("repetition_penalty", {}).get(
            "string_value", os.environ.get("REPETITION_PENALTY", "1.05")))
        self._do_sample = params.get("do_sample", {}).get(
            "string_value",
            os.environ.get("DO_SAMPLE", "1"),
        ).strip().lower() not in ("0", "false", "no")
        self.request_timeout_sec = float(params.get("request_timeout_sec", {}).get(
            "string_value", "120"))
        self.max_batch_slots = int(params.get("max_batch_slots", {}).get(
            "string_value", os.environ.get("MAX_BATCH_SLOTS", "8")))
        self._segment_token_budget = int(params.get("segment_token_budget", {}).get(
            "string_value", os.environ.get("SEGMENT_TOKEN_BUDGET", "300")))
        self._silence_max_n = int(params.get("silence_max_n", {}).get(
            "string_value", os.environ.get("SILENCE_MAX_N", "80")))
        self._min_pad_steps_before_silence_abort = int(params.get("min_pad_steps_before_silence_abort", {}).get(
            "string_value", os.environ.get("MIN_PAD_STEPS_BEFORE_SILENCE_ABORT", "4")))
        self._engine_global_step = 0

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

        self._tts_pad_embed_torch = self.weights.tts_pad_embed
        self._vocab_size = int(self.weights.config.get("talker_vocab_size", 3072))

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
        if not fused_cfg.exists():
            raise RuntimeError(
                "talker_code2wav_fused config.pbtxt not found. "
                "This orchestrator requires the fused pipeline."
            )
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

        self._talker_backend, self._talker_dtype = _dtype_from_pbtxt(fused_cfg)
        self._code2wav_dtype = self._talker_dtype

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

        # Session management
        self._session_mgr = SessionManager()
        self._scheduler = BatchScheduler(
            max_slots=self.max_batch_slots, max_queue_size=32)

        # Session queue: execute() thread -> engine thread
        self._session_queue: queue.Queue[TTSSession] = queue.Queue()
        # Response queue: engine thread -> response thread
        self._response_queue: queue.Queue = queue.Queue()

        # Engine thread
        self._engine_ready = threading.Event()
        self._engine_thread = threading.Thread(
            target=self._engine_thread_entry, daemon=True)
        self._engine_thread.start()
        self._engine_ready.wait(timeout=10)

        # Response thread
        self._response_thread = threading.Thread(
            target=self._response_loop, daemon=True)
        self._response_thread.start()

        logger.info(
            f"[TTS Orchestrator] Initialized: variant={self.variant}, num_layers={self.num_layers}, "
            f"speech_codec_fused={self._speech_codec_fused_available}, "
            f"code2wav_dtype={self._code2wav_dtype}, "
            f"engine_max_prefill_len={self.engine_max_prefill_len}, "
            f"engine_max_decode_len={self.engine_max_decode_len}, "
            f"rollover_margin={self.rollover_margin}, "
            f"text_rollover={self.enable_text_rollover}, "
            f"code2wav_sliding_window={self.code2wav_sliding_window}, "
            f"max_batch_slots={self.max_batch_slots}, "
            f"ratio_ema={self._ratio_tracker.ema}, "
            f"do_sample={self._do_sample}, temp={self._temperature}, "
            f"penalty={self._repetition_penalty}"
        )

    # ── Execute (producer — enqueue and return immediately) ──

    def execute(self, requests):
        for request in requests:
            response_sender = request.get_response_sender()
            try:
                req_tensor = pb_utils.get_input_tensor_by_name(request, "request")
                req_json = req_tensor.as_numpy()[0].decode("utf-8")
                req = json.loads(req_json)

                action = req.get("action", "synthesize")

                if action == "capabilities":
                    self._handle_capabilities(response_sender)
                    response_sender.send(
                        flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)
                    continue

                if action == "append_text":
                    self._handle_append_text(req, response_sender)
                    continue

                if action == "text_complete":
                    self._handle_text_complete(req, response_sender)
                    continue

                # action == "init" or "synthesize" (backward compat)
                session = self._create_session(req, response_sender)
                if session is not None:
                    self._session_queue.put(session)

            except Exception as e:
                logger.error(f"Request failed in execute: {e}")
                logger.error(traceback.format_exc())
                self._enqueue_error(response_sender, str(e))
                self._enqueue_final(response_sender)
        return None

    def _handle_capabilities(self, response_sender):
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
            "code2wav_sliding_window": self.code2wav_sliding_window,
            "engine_backend": self._talker_backend,
            "max_batch_slots": self.max_batch_slots,
            "ratio_ema": self._ratio_tracker.ema,
        }
        caps_json = json.dumps(caps)
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

    def _handle_append_text(self, req: dict, response_sender):
        session_id = req.get("session_id")
        text = req.get("text", "")
        if not session_id:
            self._enqueue_error(response_sender, "append_text requires session_id")
            self._enqueue_final(response_sender)
            return
        session = self._session_mgr.get(session_id)
        if session is None:
            self._enqueue_error(response_sender, f"session {session_id} not found")
            self._enqueue_final(response_sender)
            return
        if text:
            session.append_text(text)
        response_sender.send(flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)

    def _handle_text_complete(self, req: dict, response_sender):
        session_id = req.get("session_id")
        if not session_id:
            self._enqueue_error(response_sender, "text_complete requires session_id")
            self._enqueue_final(response_sender)
            return
        session = self._session_mgr.get(session_id)
        if session is None:
            self._enqueue_error(response_sender, f"session {session_id} not found")
            self._enqueue_final(response_sender)
            return
        session.mark_text_complete()
        response_sender.send(flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)

    def _create_session(
        self, req: dict, response_sender
    ) -> Optional[TTSSession]:
        self._validate_request(req)

        session_id = req.get("session_id") or generate_session_id()
        text = req.get("text", "")
        is_streaming = req.get("action") == "init"
        # Streaming init with no text: defer GPU slot until first text arrives.
        if is_streaming and not (text or "").strip():
            slot_id = UNASSIGNED_SLOT
        else:
            slot_id = self._scheduler.try_allocate_slot()
            if slot_id is None:
                if not self._scheduler.enqueue(None):
                    self._enqueue_error(response_sender, "server_busy: all slots full and queue full")
                    self._enqueue_final(response_sender)
                    return None
                evictable = self._scheduler.find_evictable(self._session_mgr)
                if evictable is not None:
                    self._evict_session(evictable, "evicted: slot needed for new request")
                    slot_id = self._scheduler.try_allocate_slot()
                if slot_id is None:
                    self._enqueue_error(response_sender, "server_busy: no free slot")
                    self._enqueue_final(response_sender)
                    return None
            self._scheduler.assign_slot(slot_id, session_id)

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

        text = req.get("text", "")
        ref_text = req.get("ref_text")

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
        ) if text.strip() else []

        session = TTSSession(
            session_id=session_id,
            slot_id=slot_id,
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
            is_streaming=is_streaming,
        )
        mlfq.on_session_created(session)
        if not is_streaming and text.strip():
            session.mark_text_complete()
        self._session_mgr.add(session)
        return session

    def _allocate_slot_for_session(self, session: TTSSession) -> bool:
        """Assign a batch slot when a deferred streaming session first needs GPU."""
        if session.slot_id != UNASSIGNED_SLOT:
            return True
        slot_id = self._scheduler.try_allocate_slot()
        if slot_id is None:
            if not self._scheduler.enqueue(None):
                return False
            evictable = self._scheduler.find_evictable(self._session_mgr)
            if evictable is not None:
                self._evict_session(evictable, "evicted: slot needed for new request")
                slot_id = self._scheduler.try_allocate_slot()
            if slot_id is None:
                return False
        self._scheduler.assign_slot(slot_id, session.session_id)
        session.slot_id = slot_id
        return True

    def _validate_request(self, req: dict) -> None:
        task_type_str = req.get("task_type", "voice_design")
        text = req.get("text", "")
        action = req.get("action", "synthesize")
        if action in ("init",) and not (text or "").strip():
            pass
        elif action not in ("init",) and not (text or "").strip():
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

    def _dynamic_silence_limit(self, remaining_steps: int) -> int:
        max_n = self._silence_max_n
        if remaining_steps <= 0:
            return 0
        return max(1, min(max_n, int(remaining_steps * max_n / 200.0)))

    @staticmethod
    def _snap_cut_to_punct(text: str, raw_cut: int) -> int:
        """Snap *raw_cut* backward to just after the nearest L1 punctuation.

        Searches text[:raw_cut] from right to left for '。！？!?\\n'.
        Returns the position right after the punctuation (so head = text[:pos]).
        Falls back to raw_cut if no punctuation is found in the first half.
        """
        search_end = raw_cut
        search_start = max(0, raw_cut // 3)
        for i in range(search_end - 1, search_start - 1, -1):
            if text[i] in "。！？!?\n":
                return i + 1
        return raw_cut

    def _split_segment_at_text_idx(self, session: TTSSession, row_logits: torch.Tensor):
        """Split current segment: head = consumed text, tail = remainder (new segment)."""
        offsets = session.trailing_token_char_offsets or []
        seg = session.current_segment_text or ""
        trailing_len = len(session.trailing_text or [])
        if offsets and session.text_idx < len(offsets):
            cut_char = offsets[session.text_idx]
        else:
            approx_ratio = len(seg) / max(trailing_len, 1)
            cut_char = min(len(seg), int(session.text_idx * approx_ratio))
        head = seg[:cut_char].strip()
        tail = seg[cut_char:].strip()
        try:
            session.last_segment_codec_tail = row_logits[:, -1, :].detach().cpu()
        except Exception:
            session.last_segment_codec_tail = None
        if tail:
            session.text_segments.insert(session.segment_idx + 1, tail)
        if head:
            session.text_segments[session.segment_idx] = head
        mlfq.on_segment_boundary(session)
        session.segment_idx += 1
        session.reset_decode_state()
        self._activate_next_segment(session)
        return cut_char, len(head), len(tail)

    def _snapshot_session_checkpoint(self, session: TTSSession, codec_sum: torch.Tensor) -> None:
        if session.kv_tensors is None or session.c2w_states is None:
            return
        session.kv_checkpoint = [t.clone().contiguous() for t in session.kv_tensors]
        session.c2w_checkpoint = [t.clone().contiguous() for t in session.c2w_states]
        session.checkpoint_past_len = int(session.past_len)
        session.checkpoint_codec_sum = codec_sum.clone().contiguous()

    def _apply_kv_rollback(self, session: TTSSession) -> None:
        """Restore talker KV + code2wav to post-prefill snapshot; keep text_idx."""
        if (
            session.kv_checkpoint is None
            or session.c2w_checkpoint is None
            or session.checkpoint_codec_sum is None
        ):
            logger.error("KV rollback missing checkpoint for session %s", session.session_id)
            return
        session.kv_tensors = [t.clone().contiguous() for t in session.kv_checkpoint]
        session.c2w_states = [t.clone().contiguous() for t in session.c2w_checkpoint]
        session.past_len = int(session.checkpoint_past_len)
        session.frame_idx = 1
        session.pad_consecutive_silence = 0
        session.token_counts = torch.zeros(
            1, self._vocab_size, device=self.device, dtype=torch.int64,
        )
        tr = session.trailing_text or []
        if session.fsm:
            session.fsm.on_kv_rollback_done(
                session.checkpoint_past_len, self._ratio_tracker.ema,
                trailing_len=len(tr),
            )
        if not tr:
            session.next_embed = None
            return
        ti = max(0, min(session.text_idx - 1, len(tr) - 1))
        text_add = tr[ti]
        session.next_embed = (session.checkpoint_codec_sum + text_add).to(torch.float32)

    # ── Engine thread (consumer — continuous decode loop) ──

    def _engine_thread_entry(self):
        asyncio.run(self._run_engine())

    async def _run_engine(self):
        self._engine_shutdown = asyncio.Event()
        self._engine_loop = asyncio.get_running_loop()
        self._engine_ready.set()
        logger.info("[Engine] Decode loop started")

        try:
            while not self._engine_shutdown.is_set():
                did_work = False

                # 1. Drain new sessions from queue
                new_sessions: list[TTSSession] = []
                while True:
                    try:
                        session = self._session_queue.get_nowait()
                        new_sessions.append(session)
                    except queue.Empty:
                        break

                # 2. Process new sessions (transition PENDING → IDLE or ACTIVE)
                for session in new_sessions:
                    if session.flow_state != FlowState.PENDING:
                        continue
                    if session.text_segments:
                        if not self._allocate_slot_for_session(session):
                            self._enqueue_error(session.response_sender, "server_busy: no free slot")
                            self._finish_session(session)
                            continue
                        try:
                            self._activate_next_segment(session)
                            did_work = True
                        except Exception as e:
                            logger.error(f"[Engine] Prefill failed for {session.session_id}: {e}")
                            logger.error(traceback.format_exc())
                            self._enqueue_error(session.response_sender, str(e))
                            self._finish_session(session)
                    else:
                        session.flow_state = FlowState.IDLE
                        logger.info(f"[Engine] Session {session.session_id} IDLE (awaiting text)")

                # 3. IDLE sessions: text arrival -> plan + prefill
                for session in self._session_mgr.get_idle():
                    self._try_activate_waiting_session(session)
                    if session.flow_state == FlowState.ACTIVE:
                        did_work = True

                # 4. ACTIVE streaming: extend trailing in-flight (KV-continuous)
                for session in list(self._session_mgr.get_active()):
                    if session.has_pending_text():
                        chunks = session.drain_text_buffer()
                        extra = "".join(chunks)
                        if extra.strip():
                            if session.is_streaming and session.trailing_text is not None:
                                include_eos = session.text_complete
                                new_embeds = self.prefill_builder.build_trailing_embeds(
                                    extra, include_eos=include_eos,
                                )
                                if new_embeds:
                                    base = len(session.current_segment_text)
                                    session.current_segment_text += extra
                                    ext_off = self.prefill_builder.build_trailing_char_offsets(
                                        extra, include_eos=include_eos,
                                    )
                                    session.trailing_token_char_offsets.extend(
                                        [base + int(o) for o in ext_off]
                                    )
                                    session.trailing_text.extend(new_embeds)
                                    if session.fsm and session.fsm.phase == DecodePhase.PHASE_B:
                                        session.fsm.trailing_char_offsets = (
                                            session.trailing_token_char_offsets
                                        )
                                        session.fsm.segment_text = session.current_segment_text
                                        session.fsm.restart_phase_a_after_extend(
                                            session.past_len, self._ratio_tracker.ema,
                                            trailing_len=len(session.trailing_text or []),
                                        )
                                    elif session.fsm:
                                        session.fsm.trailing_char_offsets = (
                                            session.trailing_token_char_offsets
                                        )
                                        session.fsm.segment_text = session.current_segment_text
                                    if include_eos:
                                        session._eos_injected = True
                                    logger.info(
                                        "[Engine] Session %s trailing extended +%d tokens "
                                        "(total=%d, eos=%s)",
                                        session.session_id, len(new_embeds),
                                        len(session.trailing_text), include_eos,
                                    )
                            else:
                                session.text_segments.extend(
                                    self._plan_text_segments(
                                        text=extra,
                                        task_type=session.task_type,
                                        language=session.language,
                                        speaker=session.speaker,
                                        instruct=session.instruct,
                                        spk_embedding=session.spk_embedding,
                                        ref_codes=session.ref_codes,
                                        ref_codec_sum_vec=session.ref_codec_sum_vec,
                                        ref_text=session.ref_text,
                                    )
                                )
                                logger.info(
                                    f"[Engine] Session {session.session_id} extended segments (+planned)"
                                )

                    # Inject eos when text_complete is signaled for a streaming
                    # session whose trailing was built without eos.
                    if (
                        session.is_streaming
                        and session.text_complete
                        and session.trailing_text is not None
                        and not getattr(session, '_eos_injected', False)
                    ):
                        session.trailing_text.append(
                            self.prefill_builder.w.tts_eos_embed.clone()
                        )
                        session.trailing_token_char_offsets.append(
                            len(session.current_segment_text)
                        )
                        if session.fsm:
                            session.fsm.trailing_char_offsets = session.trailing_token_char_offsets
                        session._eos_injected = True
                        logger.info(
                            "[Engine] Injected eos for session %s (trailing=%d)",
                            session.session_id, len(session.trailing_text),
                        )

                # 5. Batch decode all ACTIVE sessions
                generating = self._session_mgr.get_active()
                if generating:
                    did_work = True
                    timed_out = []
                    cancelled = []
                    ready: list[TTSSession] = []
                    for session in generating:
                        if session.response_sender.is_cancelled():
                            cancelled.append(session)
                            continue
                        if (time.monotonic() - session.request_start) > self.request_timeout_sec:
                            timed_out.append(session)
                            continue
                        if session.next_embed is None or session.kv_tensors is None or session.c2w_states is None:
                            continue
                        ready.append(session)

                    for session in cancelled:
                        logger.info(f"[Engine] Client disconnected: {session.session_id}")
                        self._finish_session(session)
                    for session in timed_out:
                        self._enqueue_error(session.response_sender, "request_timeout")
                        self._finish_session(session)

                    if ready:
                        self._engine_global_step += 1
                        mlfq.global_aging(
                            self._session_mgr.get_active(),
                            self._engine_global_step,
                        )
                        ready = mlfq.order_ready_sessions(
                            ready,
                            self._engine_global_step,
                        )
                        tickets = [
                            FusedDecodeTicket(
                                payload=session,
                                talker_past_len=session.past_len,
                                c2w_past_len=int(session.c2w_states[0].shape[2]) if session.c2w_states else -1,
                            )
                            for session in ready
                        ]
                        for group in group_by_c2w_past_len(tickets).values():
                            batch_sessions = [ticket.payload for ticket in group]
                            try:
                                self._run_fused_decode_batch_step(batch_sessions)
                            except torch.cuda.OutOfMemoryError as e:
                                for s in batch_sessions:
                                    self._enqueue_error(s.response_sender, f"out_of_memory: {e}")
                                    self._finish_session(s)
                            except Exception as e:
                                logger.error(f"[Engine] Batch decode failed: {e}")
                                logger.error(traceback.format_exc())
                                for s in batch_sessions:
                                    self._enqueue_error(s.response_sender, f"decode_failed: {e}")
                                    self._finish_session(s)

                # 6. Idle timeout (streaming sessions with no text yet)
                for session in list(self._session_mgr.get_idle()):
                    if session.is_idle_timed_out():
                        logger.warning(f"[Engine] Session {session.session_id} idle timeout")
                        self._enqueue_error(session.response_sender, "idle_timeout: no text received")
                        self._finish_session(session)

                if not did_work:
                    await asyncio.sleep(0.001)

        except Exception as e:
            logger.error(f"[Engine] Fatal error: {e}")
            logger.error(traceback.format_exc())
        finally:
            for session in list(self._session_mgr.get_all()):
                self._enqueue_error(session.response_sender, "engine_shutdown")
                self._finish_session(session)
            logger.info("[Engine] Decode loop stopped")

    def _try_activate_waiting_session(self, session: TTSSession):
        """IDLE session: accumulate text, plan segments, allocate slot, prefill.

        If the session went IDLE with KV preserved (streaming pause), resume
        the decode loop directly by injecting new trailing text embeddings
        instead of doing a full re-prefill.
        """
        if session.flow_state != FlowState.IDLE:
            return

        has_pending = session.has_pending_text()
        has_segments = bool(session.text_segments and session.segment_idx < len(session.text_segments))

        # --- Streaming resume path: KV preserved, inject new trailing ---
        if session.has_preserved_kv and (has_pending or session.text_complete):
            if has_pending:
                chunks = session.drain_text_buffer()
                new_text = "".join(chunks)
                if new_text.strip():
                    include_eos = session.text_complete
                    new_trailing = self.prefill_builder.build_trailing_embeds(
                        new_text, include_eos=include_eos,
                    )
                    if new_trailing:
                        self._snapshot_session_checkpoint(session, session.last_codec_sum)
                        text_add = new_trailing[0]
                        session.trailing_text = new_trailing
                        session.text_idx = 1
                        session.current_segment_text = new_text.strip()
                        session.trailing_token_char_offsets = (
                            self.prefill_builder.build_trailing_char_offsets(
                                new_text.strip(), include_eos=include_eos,
                            )
                        )
                        session.fsm = DecodeSessionFSM(
                            engine_max_decode_len=self.engine_max_decode_len,
                            rollover_margin=self.rollover_margin,
                            max_pad_steps=self._max_pad_steps,
                        )
                        session.fsm.on_prefill_done(
                            session.checkpoint_past_len,
                            session.trailing_token_char_offsets,
                            session.current_segment_text,
                            self._ratio_tracker.ema,
                            trailing_len=len(session.trailing_text or []),
                        )
                        session.next_embed = (
                            session.last_codec_sum + text_add
                        ).to(torch.float32)
                        session.last_codec_sum = None
                        session.flow_state = FlowState.ACTIVE
                        logger.info(
                            "Streaming resume: sid=%s ACTIVE (KV preserved, "
                            "past_len=%d, new_trailing=%d, eos=%s)",
                            session.session_id, session.past_len,
                            len(new_trailing), include_eos,
                        )
                        return
                elif session.text_complete:
                    pass  # fall through to text_complete-only path below
                else:
                    return

            if session.text_complete:
                # No new text but text_complete signaled: inject eos and resume
                self._snapshot_session_checkpoint(session, session.last_codec_sum)
                eos_trailing = [self.prefill_builder.w.tts_eos_embed.clone()]
                text_add = eos_trailing[0]
                session.trailing_text = eos_trailing
                session.text_idx = 1
                session.trailing_token_char_offsets = [0]
                session.current_segment_text = session.current_segment_text or ""
                session.fsm = DecodeSessionFSM(
                    engine_max_decode_len=self.engine_max_decode_len,
                    rollover_margin=self.rollover_margin,
                    max_pad_steps=self._max_pad_steps,
                )
                session.fsm.on_prefill_done(
                    session.checkpoint_past_len,
                    session.trailing_token_char_offsets,
                    session.current_segment_text,
                    self._ratio_tracker.ema,
                    trailing_len=len(session.trailing_text or []),
                )
                session.next_embed = (
                    session.last_codec_sum + text_add
                ).to(torch.float32)
                session.last_codec_sum = None
                session.flow_state = FlowState.ACTIVE
                logger.info(
                    "Streaming resume (eos only): sid=%s ACTIVE (past_len=%d)",
                    session.session_id, session.past_len,
                )
                return
            return

        # Session with preserved KV but no text yet: stay IDLE until text arrives.
        if session.has_preserved_kv:
            return

        # --- Normal path: plan segments and prefill ---
        if not has_segments and not has_pending:
            return

        if not has_segments:
            chunks = session.drain_text_buffer()
            if not chunks:
                return
            full_text = "".join(chunks)
            if not full_text.strip():
                return
            session.text_segments = self._plan_text_segments(
                text=full_text,
                task_type=session.task_type,
                language=session.language,
                speaker=session.speaker,
                instruct=session.instruct,
                spk_embedding=session.spk_embedding,
                ref_codes=session.ref_codes,
                ref_codec_sum_vec=session.ref_codec_sum_vec,
                ref_text=session.ref_text,
            )
            if not session.text_segments:
                return
        elif has_pending:
            chunks = session.drain_text_buffer()
            extra = "".join(chunks)
            if extra.strip():
                session.text_segments.extend(
                    self._plan_text_segments(
                        text=extra,
                        task_type=session.task_type,
                        language=session.language,
                        speaker=session.speaker,
                        instruct=session.instruct,
                        spk_embedding=session.spk_embedding,
                        ref_codes=session.ref_codes,
                        ref_codec_sum_vec=session.ref_codec_sum_vec,
                        ref_text=session.ref_text,
                    )
                )

        if not self._allocate_slot_for_session(session):
            self._enqueue_error(session.response_sender, "server_busy: no free slot")
            self._finish_session(session)
            return

        try:
            self._activate_next_segment(session)
        except Exception as e:
            logger.error(f"[Engine] Prefill failed for waiting session {session.session_id}: {e}")
            logger.error(traceback.format_exc())
            self._enqueue_error(session.response_sender, str(e))
            self._finish_session(session)

    def _activate_next_segment(self, session: TTSSession):
        """Prefill the next segment and transition to ACTIVE."""
        while session.segment_idx < len(session.text_segments):
            seg_text = session.text_segments[session.segment_idx]
            is_last_known_segment = (session.segment_idx == len(session.text_segments) - 1)
            streaming_may_continue = session.is_streaming and not session.text_complete
            include_eos = not (streaming_may_continue and is_last_known_segment)
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
                include_eos=include_eos,
            )
            if plan.warnings:
                for w_msg in plan.warnings:
                    self._enqueue_warning(session.response_sender, w_msg)

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
            init_tc = torch.zeros(1, self._vocab_size, device=self.device, dtype=torch.int64)
            tc, gn, temp, pen = self._build_sampling_inputs(1, init_tc)
            wav, codec_sum, full_codec, logits, updated_tc, kv_tensors, c2w_states = (
                self._bls_talker_code2wav_fused(
                    inputs_embeds,
                    position_ids,
                    cache_pos,
                    tc, gn, temp, pen,
                    initial_past_kv_tensors,
                    self._create_code2wav_initial_states(),
                )
            )
            self._enqueue_fused_wav(session.response_sender, wav)
            if self._is_codec_eos(full_codec):
                logger.info("EOS at segment %d step 0", session.segment_idx)
                session.segment_idx += 1
                continue

            text_add = plan.trailing[0] if plan.trailing else self._tts_pad_embed_torch
            session.trailing_text = plan.trailing
            session.current_segment_text = seg_text
            session.next_embed = (codec_sum + text_add).to(torch.float32)
            session.token_counts = updated_tc
            session.text_idx = 1
            session.kv_tensors = kv_tensors
            session.c2w_states = c2w_states
            session.past_len = effective_prompt_len
            session.segment_start_past_len = effective_prompt_len
            session._eos_injected = include_eos
            session.frame_idx = 1
            session.flow_state = FlowState.ACTIVE
            session.prefilled = True
            self._snapshot_session_checkpoint(session, codec_sum)
            session.trailing_token_char_offsets = list(plan.trailing_token_char_offsets or [])
            session.fsm = DecodeSessionFSM(
                engine_max_decode_len=self.engine_max_decode_len,
                rollover_margin=self.rollover_margin,
                max_pad_steps=self._max_pad_steps,
            )
            session.fsm.on_prefill_done(
                session.checkpoint_past_len,
                session.trailing_token_char_offsets,
                seg_text,
                self._ratio_tracker.ema,
                trailing_len=len(plan.trailing),
            )
            logger.info(
                "Activate segment %d/%d: sid=%s chars=%d, prompt=%d, trailing=%d, prefix_cache=%s",
                session.segment_idx + 1,
                len(session.text_segments),
                session.session_id,
                len(seg_text),
                effective_prompt_len,
                len(plan.trailing),
                prefix_cache_used,
            )
            return

        # All current segments consumed — check if more text arrived
        if not session.text_complete and session.has_pending_text():
            extra = "".join(session.drain_text_buffer())
            if extra.strip():
                session.text_segments.extend(
                    self._plan_text_segments(
                        text=extra,
                        task_type=session.task_type,
                        language=session.language,
                        speaker=session.speaker,
                        instruct=session.instruct,
                        spk_embedding=session.spk_embedding,
                        ref_codes=session.ref_codes,
                        ref_codec_sum_vec=session.ref_codec_sum_vec,
                        ref_text=session.ref_text,
                    )
                )
                if session.segment_idx < len(session.text_segments):
                    return self._activate_next_segment(session)

        if not session.text_complete:
            session.flow_state = FlowState.IDLE
            session.reset_decode_state()
            logger.info(
                "All current segments consumed, session %s back to IDLE (awaiting more text)",
                session.session_id,
            )
            return

        self._enqueue_audio_chunk(
            session.response_sender,
            np.zeros(1, dtype=np.float32),
            is_final=True,
        )
        self._finish_session(session)

    def _run_fused_decode_batch_step(self, sessions: list[TTSSession]) -> None:
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

        batch_size = len(sessions)
        batched_tc = torch.cat(
            [
                (session.token_counts if session.token_counts is not None
                 else torch.zeros(1, self._vocab_size, device=self.device, dtype=torch.int64))
                for session in sessions
            ],
            dim=0,
        )
        batched_tc, batched_gn, batched_temp, batched_pen = self._build_sampling_inputs(
            batch_size, batched_tc,
        )
        wav, codec_sum, full_codec, logits, updated_tc, kv_tensors, c2w_states = (
            self._bls_talker_code2wav_fused(
                batched_input,
                batched_pos,
                batched_cache_pos,
                batched_tc,
                batched_gn,
                batched_temp,
                batched_pen,
                batched_past_kv,
                batched_c2w_states,
                attention_bias_override=batched_attention,
                past_seq_lens_override=past_seq_lens,
            )
        )
        seq = int(batched_input.shape[1])
        new_past_lens = [session.past_len + seq for session in sessions]
        original_past_lens = [session.past_len for session in sessions]

        split_kv = self._split_batched_kv_rows(
            kv_tensors, original_past_lens, padded_past_len, seq,
        )
        split_c2w = []
        for row_idx in range(len(sessions)):
            row_states = [t[row_idx : row_idx + 1].clone().contiguous() for t in c2w_states]
            split_c2w.append(row_states)

        max_kv_len = self.engine_max_decode_len

        for row_idx, session in enumerate(sessions):
            row_logits = logits[row_idx : row_idx + 1]
            row_fc = full_codec[row_idx : row_idx + 1]
            eos = self._is_codec_eos(row_fc)
            row_wav = wav[row_idx : row_idx + 1]
            row_cs = codec_sum[row_idx : row_idx + 1]
            new_len = new_past_lens[row_idx]
            fsm = session.fsm
            tr = session.trailing_text or []
            trailing_len = len(tr)

            def _end_segment_naturally() -> None:
                total_steps = new_len - session.segment_start_past_len
                tt = max(1, min(session.text_idx, trailing_len))
                self._ratio_tracker.update(total_steps, tt)
                try:
                    session.last_segment_codec_tail = row_logits[:, -1, :].detach().cpu()
                except Exception:
                    session.last_segment_codec_tail = None
                mlfq.on_segment_boundary(session)
                session.segment_idx += 1
                session.reset_decode_state()
                self._activate_next_segment(session)

            # Hard KV overflow
            if new_len > max_kv_len:
                total_steps = new_len - session.segment_start_past_len
                tt = max(1, min(session.text_idx, trailing_len))
                orig_text_idx_kv = session.text_idx
                self._ratio_tracker.update_overflow(total_steps, tt)
                cut, h, t = self._split_segment_at_text_idx(session, row_logits)
                logger.warning(
                    "KV overflow sid=%s steps=%d text_idx=%d/%d — "
                    "split at char %d (head=%d tail=%d)",
                    session.session_id, total_steps, orig_text_idx_kv, trailing_len,
                    cut, h, t,
                )
                mlfq.on_decode_step_done(session)
                continue

            # Codec EOS handling
            if eos:
                in_phase_a = fsm and fsm.phase == DecodePhase.PHASE_A
                if in_phase_a and session.text_idx < trailing_len:
                    session._spurious_eos_count += 1
                    max_spurious = max(10, trailing_len // 2)
                    if session._spurious_eos_count > max_spurious:
                        logger.warning(
                            "Too many spurious Codec EOS (%d) in Phase A sid=%s "
                            "text_idx=%d/%d — forcing Phase B",
                            session._spurious_eos_count,
                            session.session_id, session.text_idx, trailing_len,
                        )
                        if fsm:
                            fsm.enter_phase_b("spurious_eos_limit")
                    else:
                        if session._spurious_eos_count <= 3:
                            logger.info(
                                "Ignoring spurious Codec EOS in Phase A sid=%s "
                                "text_idx=%d/%d frame=%d (count=%d)",
                                session.session_id, session.text_idx, trailing_len,
                                session.frame_idx, session._spurious_eos_count,
                            )
                        eos = False

            if eos:
                total_steps = new_len - session.segment_start_past_len
                tt = max(1, min(session.text_idx, trailing_len))
                if session.text_idx < trailing_len:
                    orig_text_idx = session.text_idx
                    fsm_phase = fsm.phase.value if fsm else "no_fsm"
                    self._ratio_tracker.update(total_steps, tt)
                    cut, h, t = self._split_segment_at_text_idx(session, row_logits)
                    logger.info(
                        "Codec EOS with unconsumed trailing sid=%s text_idx=%d/%d steps=%d "
                        "fsm_phase=%s frame=%d — split at char %d (head=%d tail=%d)",
                        session.session_id, orig_text_idx, trailing_len,
                        total_steps, fsm_phase, session.frame_idx,
                        cut, h, t,
                    )
                    mlfq.on_decode_step_done(session)
                    continue
                logger.info(
                    "Segment EOS sid=%s steps=%d text_idx=%d trailing_len=%d",
                    session.session_id, total_steps, session.text_idx, trailing_len,
                )
                _end_segment_naturally()
                mlfq.on_decode_step_done(session)
                continue

            if fsm and fsm.phase == DecodePhase.PHASE_A and session.text_idx >= trailing_len:
                fsm.enter_phase_b("trailing_exhausted")

            if (
                session.text_idx < trailing_len
                and (fsm is None or fsm.phase == DecodePhase.PHASE_A)
            ):
                text_add = tr[session.text_idx]
                session.text_idx += 1
                if fsm:
                    fsm.note_phase_a_step()
                    if fsm.should_enter_phase_b_after_consuming_token(
                        session.text_idx, trailing_len,
                    ):
                        fsm.enter_phase_b("fsm_threshold")
                self._enqueue_fused_wav(session.response_sender, row_wav)
            elif session.text_idx >= trailing_len and not session.text_complete:
                self._enqueue_fused_wav(session.response_sender, row_wav)
                session.last_codec_sum = row_cs.clone()
                self._snapshot_session_checkpoint(session, row_cs)
                session.token_counts = updated_tc[row_idx : row_idx + 1].clone()
                session.kv_tensors = split_kv[row_idx]
                session.c2w_states = split_c2w[row_idx]
                session.past_len = new_len
                session.frame_idx += 1
                session.flow_state = FlowState.IDLE
                session.next_embed = None
                logger.info(
                    "Streaming pause: sid=%s IDLE with KV preserved (past_len=%d, frame=%d)",
                    session.session_id, session.past_len, session.frame_idx,
                )
                mlfq.on_decode_step_done(session)
                continue
            else:
                if fsm and not fsm.phase_b_eos_injected:
                    text_add = self.prefill_builder.w.tts_eos_embed.clone()
                    fsm.phase_b_eos_injected = True
                else:
                    text_add = self._tts_pad_embed_torch
                pad_steps = session.frame_idx - trailing_len
                has_unconsumed = session.text_idx < trailing_len

                rms = float(row_wav.float().pow(2).mean().sqrt().item())
                is_silent = rms < 5e-4
                if is_silent:
                    session.pad_consecutive_silence += 1
                else:
                    session.pad_consecutive_silence = 0

                self._enqueue_fused_wav(session.response_sender, row_wav)

                rem_kv = max_kv_len - new_len
                pad_silence_limit = self._dynamic_silence_limit(rem_kv)
                pad_mature = pad_steps >= self._min_pad_steps_before_silence_abort
                silence_abort = pad_mature and session.pad_consecutive_silence > pad_silence_limit
                timeout_abort = pad_steps > self._max_pad_steps
                should_rollover = timeout_abort or silence_abort
                if should_rollover:
                    reason = (
                        "PAD_SILENCE" if silence_abort else "PAD_TIMEOUT"
                    )
                    logger.warning(
                        "Pad phase %s for session %s: pad_steps=%d silence_run=%d "
                        "(limit=%d, silence_limit=%d, rem_kv=%d)",
                        reason, session.session_id, pad_steps,
                        session.pad_consecutive_silence,
                        self._max_pad_steps, pad_silence_limit, rem_kv,
                    )
                    total_steps = new_len - session.segment_start_past_len
                    tt = max(1, trailing_len)
                    self._ratio_tracker.update_overflow(total_steps, tt)
                    if has_unconsumed:
                        cut, h, t = self._split_segment_at_text_idx(session, row_logits)
                        logger.warning(
                            "Pad %s split sid=%s at char %d (head=%d tail=%d)",
                            reason, session.session_id, cut, h, t,
                        )
                        mlfq.on_decode_step_done(session)
                        continue
                    try:
                        session.last_segment_codec_tail = row_logits[:, -1, :].detach().cpu()
                    except Exception:
                        session.last_segment_codec_tail = None
                    mlfq.on_segment_boundary(session)
                    session.segment_idx += 1
                    session.reset_decode_state()
                    self._activate_next_segment(session)
                    mlfq.on_decode_step_done(session)
                    continue

            session.next_embed = (row_cs + text_add).to(torch.float32)
            session.token_counts = updated_tc[row_idx : row_idx + 1].clone()
            session.kv_tensors = split_kv[row_idx]
            session.c2w_states = split_c2w[row_idx]
            session.past_len = new_len
            session.frame_idx += 1
            mlfq.on_decode_step_done(session)

    def _split_batched_kv_rows(
        self,
        batched_kv_tensors: list[torch.Tensor],
        original_past_lens: list[int],
        padded_past_len: int,
        seq: int,
    ) -> list[list[torch.Tensor]]:
        """Extract per-session KV from padded batch output.

        present_kv layout: [real_data(padded_past_len) | new_kv(seq)]
        For a session with original_past_len < padded_past_len, the range
        [original_past_len : padded_past_len] is padding zeros and must be
        removed. The correct output is cat([:original_past_len], [padded_past_len:]).
        """
        rows: list[list[torch.Tensor]] = []
        for row_idx, orig_pl in enumerate(original_past_lens):
            row: list[torch.Tensor] = []
            needs_depad = orig_pl < padded_past_len
            for tensor in batched_kv_tensors:
                t = tensor[row_idx : row_idx + 1]
                if needs_depad:
                    real_past = t[:, :, :orig_pl, :]
                    new_part = t[:, :, padded_past_len : padded_past_len + seq, :]
                    t = torch.cat([real_past, new_part], dim=2).contiguous()
                else:
                    t = t[:, :, : orig_pl + seq, :].clone().contiguous()
                row.append(t)
            rows.append(row)
        return rows

    def _finish_session(self, session: TTSSession):
        """Mark session done, release slot, send FINAL flag."""
        session.flow_state = FlowState.DONE
        if session.slot_id >= 0:
            self._scheduler.release_slot(session.slot_id)
        self._session_mgr.remove(session.session_id)
        self._enqueue_final(session.response_sender)

    def _evict_session(self, session: TTSSession, reason: str):
        logger.warning(f"[Engine] Evicting session {session.session_id}: {reason}")
        self._enqueue_error(session.response_sender, reason)
        self._finish_session(session)

    # ── Response thread ──

    def _response_loop(self):
        logger.info("[Response] Thread started")
        while True:
            item = self._response_queue.get()
            if item is None:
                break
            response_sender, response, flags = item
            try:
                response_sender.send(response, flags)
            except Exception as e:
                logger.error(f"[Response] Send failed: {e}")
        logger.info("[Response] Thread stopped")

    def _enqueue_audio_chunk(self, response_sender, audio: np.ndarray, is_final: bool = False):
        audio_tensor = pb_utils.Tensor("audio_chunk", audio.astype(np.float32))
        final_tensor = pb_utils.Tensor("is_final", np.array([is_final], dtype=bool))
        response = pb_utils.InferenceResponse(
            output_tensors=[audio_tensor, final_tensor])
        self._response_queue.put((response_sender, response, 0))

    def _enqueue_fused_wav(self, response_sender, wav: torch.Tensor):
        w = wav.reshape(-1) if wav.dim() == 3 else wav.flatten()
        self._enqueue_audio_chunk(response_sender, w.cpu().float().numpy(), is_final=False)

    def _enqueue_warning(self, response_sender, warning_msg: str):
        audio = np.zeros(1, dtype=np.float32)
        response = pb_utils.InferenceResponse(
            output_tensors=[
                pb_utils.Tensor("audio_chunk", audio),
                pb_utils.Tensor("is_final", np.array([False], dtype=bool)),
                pb_utils.Tensor("warning", np.array([warning_msg], dtype=object)),
            ],
        )
        self._response_queue.put((response_sender, response, 0))

    def _enqueue_error(self, response_sender, error_msg: str):
        audio = np.zeros(1, dtype=np.float32)
        response = pb_utils.InferenceResponse(
            output_tensors=[
                pb_utils.Tensor("audio_chunk", audio),
                pb_utils.Tensor("is_final", np.array([True], dtype=bool)),
            ],
            error=pb_utils.TritonError(error_msg),
        )
        self._response_queue.put((response_sender, response, 0))

    def _enqueue_final(self, response_sender):
        self._response_queue.put(
            (response_sender, None, pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL))

    # ── Prefill / segment planning ──

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
            cache_pos = torch.zeros(batch, FUSED_CHUNK_T, device=self.device, dtype=torch.int64)
            init_tc = torch.zeros(batch, self._vocab_size, device=self.device, dtype=torch.int64)
            tc, gn, temp, pen = self._build_sampling_inputs(batch, init_tc)
            _, _, _, _, _, kv_tensors, _ = self._bls_talker_code2wav_fused(
                prefix_embeds,
                prefix_position_ids,
                cache_pos,
                tc, gn, temp, pen,
                None,
                self._create_code2wav_initial_states(),
            )
        except Exception as e:
            logger.warning("Prefix KV cache build failed for key=%s: %s", cache_key, e)
            return None

        self._put_cached_prefix_kv(cache_key, kv_tensors)
        logger.info("Built prefix KV cache entry: key=%s, prefix_len=%d", cache_key, prefix_len)
        return self._get_cached_prefix_kv(cache_key)

    def _segment_budget_for_task(self, task_type) -> int:
        if task_type.value == "voice_design":
            return max(16, self.engine_max_prefill_len - max(16, self.rollover_margin // 2))
        remaining = max(1, self.engine_max_decode_len - max(4, self.rollover_margin // 4))
        phase_a_cap = max(8, int(remaining / (self._ratio_tracker.ema + 1.0)))
        safe_budget = min(self._segment_token_budget, phase_a_cap)
        return max(16, safe_budget)

    def _segment_within_limits(
        self,
        task_type,
        text: str,
        language: str,
        speaker: Optional[str],
        instruct: Optional[str],
        spk_embedding,
        ref_codes,
        ref_codec_sum_vec,
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
        within = input_len <= self.engine_max_prefill_len
        return within, input_len, trailing_len

    def _plan_text_segments(
        self,
        text: str,
        task_type,
        language: str,
        speaker: Optional[str],
        instruct: Optional[str],
        spk_embedding,
        ref_codes,
        ref_codec_sum_vec,
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
            previews = [
                f"[{i+1}] {len(s)}ch: {repr(s[:20])}...{repr(s[-15:])}"
                for i, s in enumerate(planned)
            ]
            logger.info(
                "Long-text rollover planned: %d segments\n  %s",
                len(planned),
                "\n  ".join(previews),
            )
        return planned

    # ── BLS inference calls (fused pipeline only) ──

    def _bls_talker_code2wav_fused(
        self,
        input_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        cache_position: torch.Tensor,
        token_counts: torch.Tensor,
        gumbel_noise: torch.Tensor,
        temperature: torch.Tensor,
        penalty: torch.Tensor,
        past_kv_tensors,
        c2w_states: list,
        attention_bias_override: Optional[torch.Tensor] = None,
        past_seq_lens_override: Optional[torch.Tensor] = None,
    ):
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
                pb_utils.Tensor.from_dlpack("token_counts", token_counts.contiguous()),
                pb_utils.Tensor.from_dlpack("gumbel_noise", gumbel_noise.contiguous()),
                pb_utils.Tensor.from_dlpack("temperature", temperature.contiguous()),
                pb_utils.Tensor.from_dlpack("penalty", penalty.contiguous()),
                pb_utils.Tensor.from_dlpack("cache_position", cache_pos),
            ]
        except Exception as e:
            logger.error(
                f"fused: from_dlpack input_embeds/position_ids/attention_bias/"
                f"token_counts/gumbel_noise/temperature/penalty/cache_position: {e}"
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

        out_names = ["wav", "codec_sum", "full_codec", "logits", "updated_token_counts"]
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
        updated_tc = self._tensor_from_response_torch(response, "updated_token_counts")
        wav = self._maybe_fix_trt_batch_axis(wav, batch)
        codec_sum = self._maybe_fix_trt_batch_axis(codec_sum, batch)
        full_codec = self._maybe_fix_trt_batch_axis(full_codec, batch)
        logits = self._maybe_fix_trt_batch_axis(logits, batch)
        updated_tc = self._maybe_fix_trt_batch_axis(updated_tc, batch)

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

        return wav, codec_sum, full_codec, logits, updated_tc, kv_tensors, new_c2w

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
        if name.startswith("c2w_"):
            max_kv_t = max(1, self.code2wav_sliding_window - 1)
        if tensor.shape[2] <= max_kv_t:
            return tensor
        return tensor[:, :, -max_kv_t:, :].contiguous()

    def _tensor_from_response_torch(self, response, name: str) -> torch.Tensor:
        t = pb_utils.get_output_tensor_by_name(response, name)
        if t.is_cpu():
            return torch.from_numpy(t.as_numpy()).to(self.device)
        try:
            return torch.from_dlpack(t)
        except Exception as e:
            logger.error(f"Error in torch.from_dlpack for {name}: {e}")
            return torch.from_numpy(t.as_numpy()).to(self.device)

    def _create_code2wav_initial_states(self):
        return [
            torch.zeros(shape, device=self.device, dtype=self._code2wav_dtype)
            for shape in self._code2wav_state_shapes_fused
        ]

    # ── BLS sub-model calls (speaker encoder, speech tokenizer) ──

    def _bls_speaker_encoder(self, ref_audio_b64: str) -> torch.Tensor:
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

    # ── Finalize ──

    def finalize(self):
        logger.info("[TTS Orchestrator] Finalizing...")
        if hasattr(self, '_engine_loop') and self._engine_loop is not None:
            self._engine_loop.call_soon_threadsafe(self._engine_shutdown.set)
        if hasattr(self, '_engine_thread') and self._engine_thread is not None:
            self._engine_thread.join(timeout=10)
        if hasattr(self, '_response_queue'):
            self._response_queue.put(None)
        if hasattr(self, '_response_thread') and self._response_thread is not None:
            self._response_thread.join(timeout=5)
        logger.info("[TTS Orchestrator] Finalized")

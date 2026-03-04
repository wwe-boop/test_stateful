"""
PrefillBuilder: construct inputs_embeds for Talker Backbone prefill.

Supports 4 task types (architecture.md §10.4):
  - VOICE_DESIGN:     role + instruct + tag + bos + first_text
  - CUSTOM_VOICE:     role + instruct + tag + spk(codec) + bos + first_text
  - VOICE_CLONE_XVEC: role + tag + spk_embed + bos + first_text
  - VOICE_CLONE_ICL:  role + tag + spk_embed + bos + ICL(ref) + first_text

In-process torch: text/codec embeddings via loaded .pt weights on GPU (BF16).
All tensor ops use torch; no BLS for embedders. Tokenizer remains lightweight (tokenizers lib).
"""

import json
import logging
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger("prefill_builder")


class ResizeMLP(nn.Module):
    """Text projection: linear_fc1 -> silu -> linear_fc2 (matches Qwen3TTSTalkerResizeMLP)."""

    def __init__(self, input_size: int, intermediate_size: int, output_size: int, bias: bool = True):
        super().__init__()
        self.linear_fc1 = nn.Linear(input_size, intermediate_size, bias=bias)
        self.linear_fc2 = nn.Linear(intermediate_size, output_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(torch.nn.functional.silu(self.linear_fc1(x)))


class TaskType(Enum):
    VOICE_CLONE_ICL = "voice_clone_icl"
    VOICE_CLONE_XVEC = "voice_clone_xvec"
    CUSTOM_VOICE = "custom_voice"
    VOICE_DESIGN = "voice_design"


def parse_task_type(task_type_str: str, x_vector_only: bool = False) -> TaskType:
    if task_type_str == "voice_clone":
        return TaskType.VOICE_CLONE_XVEC if x_vector_only else TaskType.VOICE_CLONE_ICL
    elif task_type_str == "custom_voice":
        return TaskType.CUSTOM_VOICE
    elif task_type_str == "voice_design":
        return TaskType.VOICE_DESIGN
    else:
        raise ValueError(f"Unknown task_type: {task_type_str}")


class EmbeddingWeights:
    """Holds config and torch-loaded weights for prefill (BF16 on CUDA).

    Loads .pt files: text_embedding, text_projection, codec_embedding (talker),
    special_embeddings, codec_embeddings_3d (optional).
    Provides text_embed() and codec_embed() for in-process lookup.
    """

    def __init__(self, weights_dir: str, device_id: int = 0):
        weights_dir = Path(weights_dir)
        with open(weights_dir / "config.json") as f:
            self.config = json.load(f)

        self.variant = self.config["variant"]
        self.hidden_size = self.config["talker_hidden_size"]
        self.text_hidden_size = self.config.get("talker_text_hidden_size", self.hidden_size)
        self.vocab_size = self.config["talker_vocab_size"]
        self.codec_eos_id = self.config["codec_eos_token_id"]
        self.codec_bos_id = self.config["codec_bos_id"]
        self.codec_pad_id = self.config["codec_pad_id"]
        self.codec_language_id = self.config.get("codec_language_id", {})
        self.spk_id_map = self.config.get("spk_id", {})

        self.device = torch.device("cuda", device_id)
        dtype = torch.bfloat16

        # Text embedding: nn.Embedding
        text_emb_path = weights_dir / "text_embedding.pt"
        if not text_emb_path.exists():
            raise FileNotFoundError(
                f"text_embedding.pt not found in {weights_dir}. Run export_06_embeddings.py."
            )
        text_emb_sd = torch.load(text_emb_path, map_location=self.device, weights_only=True)
        weight = text_emb_sd["weight"]
        num_embeddings, text_emb_dim = weight.shape[0], weight.shape[1]
        self.text_embedding = nn.Embedding(num_embeddings, text_emb_dim).to(
            device=self.device, dtype=dtype
        )
        self.text_embedding.load_state_dict(
            {k: v.to(device=self.device, dtype=dtype) for k, v in text_emb_sd.items()}
        )
        self.text_embedding.eval()

        # Text projection: ResizeMLP (text_hidden_size -> hidden_size)
        text_proj_path = weights_dir / "text_projection.pt"
        if not text_proj_path.exists():
            raise FileNotFoundError(
                f"text_projection.pt not found in {weights_dir}. Run export_06_embeddings.py."
            )
        text_proj_sd = torch.load(text_proj_path, map_location=self.device, weights_only=True)
        # linear_fc1: [intermediate, input], linear_fc2: [output, intermediate]
        in_size = text_proj_sd["linear_fc1.weight"].shape[1]
        mid_size = text_proj_sd["linear_fc1.weight"].shape[0]
        out_size = text_proj_sd["linear_fc2.weight"].shape[0]
        self.text_projection = ResizeMLP(in_size, mid_size, out_size, bias=True).to(
            device=self.device, dtype=dtype
        )
        self.text_projection.load_state_dict(
            {k: v.to(device=self.device, dtype=dtype) for k, v in text_proj_sd.items()}
        )
        self.text_projection.eval()

        # Codec embedding (talker only)
        codec_path = weights_dir / "codec_embeddings.pt"
        if not codec_path.exists():
            raise FileNotFoundError(
                f"codec_embeddings.pt not found in {weights_dir}. Run export_06_embeddings.py."
            )
        codec_data = torch.load(codec_path, map_location=self.device, weights_only=True)
        talker_sd = codec_data["talker_codec_embedding"]
        c_weight = talker_sd["weight"]
        codec_vocab, codec_dim = c_weight.shape[0], c_weight.shape[1]
        self.codec_embedding = nn.Embedding(codec_vocab, codec_dim).to(
            device=self.device, dtype=dtype
        )
        self.codec_embedding.load_state_dict(
            {k: v.to(device=self.device, dtype=dtype) for k, v in talker_sd.items()}
        )
        self.codec_embedding.eval()

        # Special embeddings [1, 1, H]
        special_path = weights_dir / "special_embeddings.pt"
        if not special_path.exists():
            raise FileNotFoundError(
                f"special_embeddings.pt not found in {weights_dir}. Run export_06_embeddings.py."
            )
        special = torch.load(special_path, map_location=self.device, weights_only=True)
        self.tts_pad_embed = special["tts_pad_embed"].to(device=self.device, dtype=dtype)
        self.tts_bos_embed = special["tts_bos_embed"].to(device=self.device, dtype=dtype)
        self.tts_eos_embed = special["tts_eos_embed"].to(device=self.device, dtype=dtype)

        # 3D codec embeddings for ICL (optional)
        path_3d = weights_dir / "codec_embeddings_3d.pt"
        self.codec_embeddings_3d = None
        if path_3d.exists():
            self.codec_embeddings_3d = torch.load(
                path_3d, map_location=self.device, weights_only=True
            ).to(device=self.device, dtype=dtype)

        logger.info(
            f"Weights loaded: variant={self.variant}, "
            f"hidden={self.hidden_size}, vocab={self.vocab_size}"
        )

    def text_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        """token_ids [B, S] int64 -> [B, S, H] bfloat16."""
        with torch.no_grad():
            emb = self.text_embedding(token_ids)
            return self.text_projection(emb)

    def codec_embed(self, codec_ids: torch.Tensor) -> torch.Tensor:
        """codec_ids [B, S] int64 -> [B, S, H] bfloat16."""
        with torch.no_grad():
            return self.codec_embedding(codec_ids)


class PrefillBuilder:
    """
    Build prefill inputs_embeds for different task types.

    Uses EmbeddingWeights.text_embed/codec_embed (in-process torch); all tensor ops are torch.
    """

    def __init__(self, weights: EmbeddingWeights, tokenizer: Any):
        self.w = weights
        self.tokenizer = tokenizer

    def build(
        self,
        task_type: TaskType,
        text: str,
        language: str = "auto",
        speaker: Optional[str] = None,
        instruct: Optional[str] = None,
        spk_embedding: Optional[torch.Tensor] = None,
        ref_codes: Optional[torch.Tensor] = None,
        ref_text: Optional[str] = None,
    ) -> tuple:
        """
        Build prefill inputs_embeds and trailing_text_hidden queue.

        Returns:
            (inputs_embeds, trailing_text_hidden) where:
              inputs_embeds: torch [1, S_prefill, H] bfloat16 on GPU
              trailing_text_hidden: list of torch [1, 1, H] for decode
        """
        w = self.w
        device = w.device

        # Tokenize: tokenizer returns numpy [1, S]
        assistant_text = f"<|im_start|>assistant\n{text}<|im_end|>"
        input_ids_np = self.tokenizer(assistant_text, return_tensors="pt")["input_ids"]
        if isinstance(input_ids_np, np.ndarray):
            pass
        else:
            input_ids_np = np.asarray(input_ids_np, dtype=np.int64)
        if input_ids_np.ndim == 1:
            input_ids_np = input_ids_np.reshape(1, -1)
        input_ids = torch.as_tensor(input_ids_np, device=device, dtype=torch.int64)

        # Role embed: first 3 tokens
        role_embed = w.text_embed(input_ids[:, :3])

        # Tag: think/language codec tokens
        if language == "auto":
            tag_ids = torch.tensor(
                [[w.codec_pad_id, w.codec_bos_id, w.codec_pad_id]],
                device=device,
                dtype=torch.int64,
            )
        else:
            lang_id = w.codec_language_id.get(language, w.codec_pad_id)
            tag_ids = torch.tensor(
                [[w.codec_pad_id, w.codec_bos_id, lang_id, w.codec_pad_id]],
                device=device,
                dtype=torch.int64,
            )
        tag_codec_embed = w.codec_embed(tag_ids)
        bos_codec_embed = w.codec_embed(
            torch.tensor([[w.codec_bos_id]], device=device, dtype=torch.int64)
        )

        # Instruct embed (for CUSTOM_VOICE / VOICE_DESIGN)
        instruct_embed = None
        if instruct and task_type in (TaskType.CUSTOM_VOICE, TaskType.VOICE_DESIGN):
            instruct_ids_np = self.tokenizer(instruct, return_tensors="pt")["input_ids"]
            if not isinstance(instruct_ids_np, np.ndarray):
                instruct_ids_np = np.asarray(instruct_ids_np, dtype=np.int64)
            if instruct_ids_np.ndim == 1:
                instruct_ids_np = instruct_ids_np.reshape(1, -1)
            instruct_ids = torch.as_tensor(instruct_ids_np, device=device, dtype=torch.int64)
            instruct_embed = w.text_embed(instruct_ids)

        # Speaker embed
        speaker_embed = None
        if task_type == TaskType.CUSTOM_VOICE and speaker:
            spk_id_val = w.spk_id_map.get(speaker.lower())
            if spk_id_val is not None:
                speaker_embed = w.codec_embed(
                    torch.tensor([[spk_id_val]], device=device, dtype=torch.int64)
                )
        elif task_type in (TaskType.VOICE_CLONE_ICL, TaskType.VOICE_CLONE_XVEC):
            if spk_embedding is not None:
                speaker_embed = spk_embedding.reshape(1, 1, -1)

        # Codec layer: tag + [speaker] + bos
        if speaker_embed is not None:
            codec_layer = torch.cat(
                [tag_codec_embed, speaker_embed, bos_codec_embed], dim=1
            )
        else:
            codec_layer = torch.cat([tag_codec_embed, bos_codec_embed], dim=1)

        # Text layer: pad alignment + tts_bos
        n_codec = codec_layer.shape[1]
        pad_count = max(n_codec - 2, 0)
        text_layer = torch.cat(
            [
                w.tts_pad_embed.expand(1, pad_count, w.hidden_size),
                w.tts_bos_embed,
            ],
            dim=1,
        )
        dual_track = text_layer + codec_layer[:, :-1]

        if instruct_embed is not None:
            base_prefill = torch.cat([role_embed, instruct_embed, dual_track], dim=1)
        else:
            base_prefill = torch.cat([role_embed, dual_track], dim=1)

        # First text token
        if input_ids.shape[1] > 3:
            first_text_embed = w.text_embed(input_ids[:, 3:4])
        else:
            first_text_embed = w.tts_pad_embed
        first_text_with_bos = first_text_embed + codec_layer[:, -1:]

        # ICL (voice_clone_icl)
        if (
            task_type == TaskType.VOICE_CLONE_ICL
            and ref_codes is not None
            and w.codec_embeddings_3d is not None
        ):
            T_ref, _ = ref_codes.shape
            g_idx = torch.arange(16, device=device, dtype=torch.int64).reshape(1, -1)
            g_idx = g_idx.expand(T_ref, 16)
            codec_sum_vec = w.codec_embeddings_3d[
                g_idx, ref_codes, :
            ].sum(dim=(0, 1), keepdim=True)
            codec_embed_icl = torch.cat([bos_codec_embed, codec_sum_vec], dim=1)
            if ref_text and ref_text.strip():
                ref_assistant = f"<|im_start|>assistant\n{ref_text.strip()}<|im_end|>"
                ref_ids_np = self.tokenizer(ref_assistant, return_tensors="pt")[
                    "input_ids"
                ]
                if not isinstance(ref_ids_np, np.ndarray):
                    ref_ids_np = np.asarray(ref_ids_np, dtype=np.int64)
                if ref_ids_np.ndim == 1:
                    ref_ids_np = ref_ids_np.reshape(1, -1)
                ref_ids = torch.as_tensor(ref_ids_np, device=device, dtype=torch.int64)
                ref_id = ref_ids[:, 3:-2] if ref_ids.shape[1] > 5 else ref_ids[:, :0]
            else:
                ref_id = input_ids[:, :0]
            text_id = (
                input_ids[:, 3:-5] if input_ids.shape[1] > 8 else input_ids[:, 3:4]
            )
            text_embed_icl = w.text_embed(torch.cat([ref_id, text_id], dim=1))
            text_embed_icl = torch.cat([text_embed_icl, w.tts_eos_embed], dim=1)
            text_lens = text_embed_icl.shape[1]
            codec_lens = codec_embed_icl.shape[1]
            if text_lens > codec_lens:
                icl_embed = text_embed_icl[:, :codec_lens] + codec_embed_icl
                trailing = [
                    text_embed_icl[:, i : i + 1] for i in range(codec_lens, text_lens)
                ]
            else:
                pad_len = codec_lens - text_lens
                if pad_len > 0:
                    pads = [w.tts_pad_embed] * pad_len
                    text_embed_icl = torch.cat([text_embed_icl] + pads, dim=1)
                icl_embed = text_embed_icl + codec_embed_icl
                trailing = [w.tts_pad_embed]
            prefill = torch.cat(
                [base_prefill, first_text_with_bos, icl_embed], dim=1
            )
        else:
            prefill = torch.cat([base_prefill, first_text_with_bos], dim=1)
            trailing = []
            if input_ids.shape[1] > 4:
                end_idx = max(4, input_ids.shape[1] - 5)
                for i in range(4, end_idx):
                    t_embed = w.text_embed(input_ids[:, i : i + 1])
                    trailing.append(t_embed)
            trailing.append(w.tts_eos_embed)

        logger.info(
            f"Prefill built: task={task_type.value}, "
            f"shape={tuple(prefill.shape)}, trailing={len(trailing)}"
        )
        return prefill, trailing

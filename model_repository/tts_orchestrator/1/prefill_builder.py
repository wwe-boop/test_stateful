"""
PrefillBuilder: construct inputs_embeds for Talker Backbone prefill.

Supports 4 task types (architecture.md §10.4):
  - VOICE_DESIGN:     role + instruct + tag + bos + first_text
  - CUSTOM_VOICE:     role + instruct + tag + spk(codec) + bos + first_text
  - VOICE_CLONE_XVEC: role + tag + spk_embed + bos + first_text
  - VOICE_CLONE_ICL:  role + tag + spk_embed + bos + ICL(ref) + first_text

All embedding layers are reconstructed from exported .pt weights
(no PyTorch model dependency at runtime).
"""

import json
import logging
from enum import Enum
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger("prefill_builder")


class _ResizeMLP(nn.Module):
    """Mirrors Qwen3TTSTalkerResizeMLP: fc1 → silu → fc2."""

    def __init__(self, input_size, intermediate_size, output_size, bias=True):
        super().__init__()
        self.linear_fc1 = nn.Linear(input_size, intermediate_size, bias=bias)
        self.linear_fc2 = nn.Linear(intermediate_size, output_size, bias=bias)

    def forward(self, x):
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
    """Holds all exported embedding weights needed for prefill construction."""

    def __init__(self, weights_dir: str, device: torch.device,
                 dtype: torch.dtype = torch.float32):
        weights_dir = Path(weights_dir)
        self.device = device
        self.dtype = dtype

        config_path = weights_dir / "config.json"
        with open(config_path) as f:
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

        # Text embedding + projection
        text_emb_sd = torch.load(weights_dir / "text_embedding.pt",
                                 map_location=device, weights_only=True)
        self.text_embedding = nn.Embedding(
            text_emb_sd["weight"].shape[0],
            text_emb_sd["weight"].shape[1],
        ).to(device)
        self.text_embedding.load_state_dict(text_emb_sd)
        self.text_embedding.eval()

        text_proj_sd = torch.load(weights_dir / "text_projection.pt",
                                  map_location=device, weights_only=True)
        # ResizeMLP: linear_fc1 → act → linear_fc2
        fc1_w = text_proj_sd["linear_fc1.weight"]
        fc2_w = text_proj_sd["linear_fc2.weight"]
        self.text_projection = _ResizeMLP(
            fc1_w.shape[1], fc1_w.shape[0], fc2_w.shape[0],
            bias="linear_fc1.bias" in text_proj_sd,
        ).to(device)
        self.text_projection.load_state_dict(text_proj_sd)
        self.text_projection.eval()

        # Codec embedding (talker only — for tag/bos/spk lookups)
        codec_data = torch.load(weights_dir / "codec_embeddings.pt",
                                map_location=device, weights_only=True)
        talker_sd = codec_data["talker_codec_embedding"]
        self.codec_embedding = nn.Embedding(
            talker_sd["weight"].shape[0],
            talker_sd["weight"].shape[1],
        ).to(device)
        self.codec_embedding.load_state_dict(talker_sd)
        self.codec_embedding.eval()

        # 3D stacked codec embeddings (for decode loop codec_sum)
        path_3d = weights_dir / "codec_embeddings_3d.pt"
        self.codec_embeddings_3d = None
        if path_3d.exists():
            self.codec_embeddings_3d = torch.load(
                path_3d, map_location=device, weights_only=True)

        # Special embeddings (pre-computed tts_pad, tts_bos, tts_eos)
        special = torch.load(weights_dir / "special_embeddings.pt",
                             map_location=device, weights_only=True)
        self.tts_pad_embed = special["tts_pad_embed"].to(device)   # [1, 1, H]
        self.tts_bos_embed = special["tts_bos_embed"].to(device)   # [1, 1, H]
        self.tts_eos_embed = special["tts_eos_embed"].to(device)   # [1, 1, H]

        # Codec head (for logits → hidden inversion if needed)
        codec_head_sd = torch.load(weights_dir / "codec_head.pt",
                                   map_location=device, weights_only=True)
        in_f = codec_head_sd["weight"].shape[1]
        out_f = codec_head_sd["weight"].shape[0]
        self.codec_head = nn.Linear(in_f, out_f, bias="bias" in codec_head_sd)
        self.codec_head.load_state_dict(codec_head_sd)
        self.codec_head.to(device).eval()

        # Code predictor lm_heads (for standalone CP fallback)
        cp_heads_path = weights_dir / "code_predictor_lm_heads.pt"
        self.cp_lm_heads = None
        if cp_heads_path.exists():
            self.cp_lm_heads = torch.load(
                cp_heads_path, map_location=device, weights_only=True)

        logger.info(f"Weights loaded: variant={self.variant}, "
                    f"hidden={self.hidden_size}, vocab={self.vocab_size}")

    def text_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        """token_ids [B, S] -> text_proj(text_embedding(ids)) [B, S, H]"""
        with torch.no_grad():
            return self.text_projection(self.text_embedding(token_ids))

    def codec_embed(self, codec_ids: torch.Tensor) -> torch.Tensor:
        """codec_ids [B, S] or [S] -> [B, S, H] codec embedding lookup."""
        with torch.no_grad():
            if codec_ids.dim() == 1:
                codec_ids = codec_ids.unsqueeze(0)
            return self.codec_embedding(codec_ids)


class PrefillBuilder:
    """
    Build prefill inputs_embeds for different task types.

    The prefill layout follows architecture.md §4.1 / §10.4:
    - Text layer (upper track): text_proj(text_embed(token_ids))
    - Codec layer (lower track): codec_embed(codec_ids)
    - Combined: text_layer + codec_layer (element-wise add, dual-track)
    """

    def __init__(self, weights: EmbeddingWeights, tokenizer):
        self.w = weights
        self.tokenizer = tokenizer
        self.device = weights.device

    def build(self, task_type: TaskType, text: str, language: str = "auto",
              speaker: Optional[str] = None,
              instruct: Optional[str] = None,
              spk_embedding: Optional[torch.Tensor] = None,
              ref_codes: Optional[torch.Tensor] = None,
              ref_text: Optional[str] = None) -> tuple:
        """
        Build prefill inputs_embeds and trailing_text_hidden queue.

        Returns:
            (inputs_embeds, trailing_text_hidden) where:
              inputs_embeds: [1, S_prefill, H]
              trailing_text_hidden: list of [1, 1, H] tensors for decode
        """
        w = self.w
        cfg = w.config

        # Tokenize the full text (assistant format)
        assistant_text = f"<|im_start|>assistant\n{text}<|im_end|>"
        input_ids = self.tokenizer(assistant_text, return_tensors="pt")["input_ids"].to(self.device)

        # Role embed: first 3 tokens (<|im_start|> assistant \n)
        role_embed = w.text_embed(input_ids[:, :3])  # [1, 3, H]

        # Tag: think/language codec tokens
        if language == "auto":
            codec_nothink_id = w.codec_pad_id  # placeholder
            tag_ids = torch.tensor([w.codec_pad_id, w.codec_bos_id, w.codec_pad_id],
                                   device=self.device, dtype=torch.long)
        else:
            lang_id = w.codec_language_id.get(language, w.codec_pad_id)
            tag_ids = torch.tensor([w.codec_pad_id, w.codec_bos_id, lang_id, w.codec_pad_id],
                                   device=self.device, dtype=torch.long)

        tag_codec_embed = w.codec_embed(tag_ids)  # [1, 3-4, H]
        bos_codec_embed = w.codec_embed(
            torch.tensor([w.codec_bos_id], device=self.device))  # [1, 1, H]

        # Instruct embed (for CUSTOM_VOICE / VOICE_DESIGN)
        instruct_embed = None
        if instruct and task_type in (TaskType.CUSTOM_VOICE, TaskType.VOICE_DESIGN):
            instruct_ids = self.tokenizer(instruct, return_tensors="pt")["input_ids"].to(self.device)
            instruct_embed = w.text_embed(instruct_ids)  # [1, S_ins, H]

        # Speaker embed
        speaker_embed = None
        if task_type == TaskType.CUSTOM_VOICE and speaker:
            spk_id_val = w.spk_id_map.get(speaker.lower())
            if spk_id_val is not None:
                speaker_embed = w.codec_embed(
                    torch.tensor([spk_id_val], device=self.device))  # [1, 1, H]
        elif task_type in (TaskType.VOICE_CLONE_ICL, TaskType.VOICE_CLONE_XVEC):
            if spk_embedding is not None:
                speaker_embed = spk_embedding.view(1, 1, -1).to(self.device)

        # Codec layer: tag + [speaker] + bos
        if speaker_embed is not None:
            codec_layer = torch.cat([tag_codec_embed, speaker_embed, bos_codec_embed], dim=1)
        else:
            codec_layer = torch.cat([tag_codec_embed, bos_codec_embed], dim=1)

        # Text layer: pad alignment + tts_bos
        n_codec = codec_layer.shape[1]
        pad_count = n_codec - 2  # tag+spk positions use pad
        text_layer = torch.cat([
            w.tts_pad_embed.expand(-1, max(pad_count, 0), -1),
            w.tts_bos_embed,
        ], dim=1)  # [1, n_codec-1, H]

        # Base prefill: role + (text_layer + codec_layer[:-1])
        dual_track = text_layer + codec_layer[:, :-1]

        if instruct_embed is not None:
            base_prefill = torch.cat([role_embed, instruct_embed, dual_track], dim=1)
        else:
            base_prefill = torch.cat([role_embed, dual_track], dim=1)

        # First text token
        if input_ids.shape[1] > 3:
            first_text_embed = w.text_embed(input_ids[:, 3:4])  # [1, 1, H]
        else:
            first_text_embed = w.tts_pad_embed
        first_text_with_bos = first_text_embed + codec_layer[:, -1:]

        prefill = torch.cat([base_prefill, first_text_with_bos], dim=1)

        # Trailing text hidden (for decode loop text injection)
        trailing = []
        if input_ids.shape[1] > 4:
            # Remaining text tokens (excluding last 5 = eos + template tokens)
            end_idx = max(4, input_ids.shape[1] - 5)
            for i in range(4, end_idx):
                t_embed = w.text_embed(input_ids[:, i:i+1])  # [1, 1, H]
                trailing.append(t_embed)
        trailing.append(w.tts_eos_embed)

        logger.info(f"Prefill built: task={task_type.value}, "
                    f"shape={tuple(prefill.shape)}, trailing={len(trailing)}")
        return prefill, trailing

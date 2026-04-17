#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor
from transformers.generation.logits_process import TemperatureLogitsWarper, TopKLogitsWarper

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "third_party" / "Qwen3-TTS"))
sys.path.insert(0, str(REPO_ROOT))

from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration
from engine.frontend.spliter.tokenizer import load_lightweight_tokenizer
from engine.backend.prefill import OFFICIAL_ASSISTANT_FMT


def sample_like_hf(logits, history_ids, *, top_k, temperature, repetition_penalty, suppress_tokens):
    scores = logits.float().clone()
    if suppress_tokens:
        scores[..., suppress_tokens] = -1e9
    if repetition_penalty != 1.0:
        scores = RepetitionPenaltyLogitsProcessor(repetition_penalty)(history_ids, scores)
    if temperature != 1.0:
        scores = TemperatureLogitsWarper(temperature)(history_ids, scores)
    if top_k > 0:
        scores = TopKLogitsWarper(top_k)(history_ids, scores)
    probs = torch.softmax(scores, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1), scores


def tensor_stat(x):
    if x is None:
        return None
    x = x.detach()
    return {
        "shape": list(x.shape),
        "norm": float(x.float().norm().cpu()),
        "mean": float(x.float().mean().cpu()),
        "min": float(x.float().min().cpu()),
        "max": float(x.float().max().cpu()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--speaker", default="vivian")
    ap.add_argument("--language", default="auto")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--steps", type=int, default=12)
    args = ap.parse_args()

    model_dir = str(REPO_ROOT / "workspace/models/Qwen3-TTS-12Hz-1.7B-CustomVoice")
    device = torch.device("cuda:0")
    model = Qwen3TTSForConditionalGeneration.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="eager")
    model.eval()
    talker = model.talker

    tokenizer = load_lightweight_tokenizer(model_dir)
    assistant_text = OFFICIAL_ASSISTANT_FMT.format(text=args.text)
    out = tokenizer(assistant_text, return_tensors="pt")
    input_id = torch.as_tensor(out["input_ids"], device=device, dtype=torch.long)
    if input_id.dim() == 1:
        input_id = input_id.unsqueeze(0)

    # Build EXACT official high-level prefill/trailing tensors by copying code literally.
    language = args.language
    speaker = args.speaker
    if language.lower() == 'auto':
        language_id = None
    else:
        language_id = talker.config.codec_language_id[language.lower()]
    if (language.lower() in ['chinese', 'auto'] and speaker and talker.config.spk_is_dialect[speaker.lower()] != False):
        dialect = talker.config.spk_is_dialect[speaker.lower()]
        language_id = talker.config.codec_language_id[dialect]

    tts_bos_embed, tts_eos_embed, tts_pad_embed = talker.text_projection(
        talker.get_text_embeddings()(
            torch.tensor([[model.config.tts_bos_token_id, model.config.tts_eos_token_id, model.config.tts_pad_token_id]], device=device, dtype=input_id.dtype)
        )
    ).chunk(3, dim=1)

    spk_id = talker.config.spk_id[speaker.lower()]
    speaker_embed = talker.get_input_embeddings()(torch.tensor(spk_id, device=device, dtype=input_id.dtype))

    if language_id is None:
        codec_prefill_list = [[talker.config.codec_nothink_id, talker.config.codec_think_bos_id, talker.config.codec_think_eos_id]]
    else:
        codec_prefill_list = [[talker.config.codec_think_id, talker.config.codec_think_bos_id, language_id, talker.config.codec_think_eos_id]]

    codec_input_0 = talker.get_input_embeddings()(torch.tensor(codec_prefill_list, device=device, dtype=input_id.dtype))
    codec_input_1 = talker.get_input_embeddings()(torch.tensor([[talker.config.codec_pad_id, talker.config.codec_bos_id]], device=device, dtype=input_id.dtype))
    codec_input = torch.cat([codec_input_0, speaker_embed.view(1,1,-1), codec_input_1], dim=1)

    role = talker.text_projection(talker.get_text_embeddings()(input_id[:, :3]))
    pre_codec = torch.cat((tts_pad_embed.expand(-1, codec_input.shape[1] - 2, -1), tts_bos_embed), dim=1) + codec_input[:, :-1]
    talker_input_embed = torch.cat((role, pre_codec), dim=1)
    talker_input_embed = torch.cat([
        talker_input_embed,
        talker.text_projection(talker.get_text_embeddings()(input_id[:, 3:4])) + codec_input[:, -1:]
    ], dim=1)
    trailing_text_hidden = torch.cat((talker.text_projection(talker.get_text_embeddings()(input_id[:, 4:-5])), tts_eos_embed), dim=1)

    original_len = talker_input_embed.shape[1]
    talker_attention_mask = torch.ones(1, original_len, device=device, dtype=torch.long)
    prefill_position_ids = talker_attention_mask.long().cumsum(-1) - 1
    prefill_position_ids.masked_fill_(talker_attention_mask == 0, 1)
    prefill_cache_position = torch.arange(original_len, device=device, dtype=torch.long)

    suppress_tokens = [i for i in range(talker.config.vocab_size - 1024, talker.config.vocab_size) if i not in (talker.config.codec_eos_token_id,)]

    history = torch.empty((1,0), device=device, dtype=torch.long)
    recs = []
    torch.manual_seed(args.seed)

    # Step 0: prefill call
    with torch.no_grad():
        out0 = talker.forward(
            input_ids=None,
            attention_mask=talker_attention_mask,
            position_ids=prefill_position_ids,
            inputs_embeds=talker_input_embed,
            use_cache=True,
            cache_position=prefill_cache_position,
            generation_step=None,
            trailing_text_hidden=trailing_text_hidden,
            tts_pad_embed=tts_pad_embed,
            subtalker_dosample=True,
            subtalker_top_k=50,
            subtalker_top_p=1.0,
            subtalker_temperature=0.9,
            return_dict=True,
        )
    tok0, scores0 = sample_like_hf(out0.logits[:, -1, :], history, top_k=50, temperature=0.9, repetition_penalty=1.05, suppress_tokens=suppress_tokens)
    history = torch.cat([history, tok0.unsqueeze(1)], dim=1)
    recs.append({
        'step': 0,
        'phase': 'prefill',
        'generation_step_in': None,
        'generation_step_out': int(out0.generation_step),
        'sampled_codec0': int(tok0.item()),
        'trailing_len': int(trailing_text_hidden.shape[1]),
        'input_embeds': tensor_stat(talker_input_embed),
        'past_hidden': tensor_stat(out0.past_hidden),
        'codec_ids': None if out0.hidden_states[1] is None else out0.hidden_states[1].detach().cpu().tolist(),
    })

    current = out0
    for step in range(1, args.steps):
        attention_mask = torch.ones(1, original_len + step, device=device, dtype=torch.long)
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        position_ids = position_ids[:, -1:].clone()
        cache_position = torch.tensor([original_len + step - 1], device=device, dtype=torch.long)
        with torch.no_grad():
            outn = talker.forward(
                input_ids=tok0.unsqueeze(1),
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=current.past_key_values,
                use_cache=True,
                cache_position=cache_position,
                generation_step=current.generation_step,
                trailing_text_hidden=current.trailing_text_hidden,
                tts_pad_embed=current.tts_pad_embed,
                past_hidden=current.past_hidden,
                subtalker_dosample=True,
                subtalker_top_k=50,
                subtalker_top_p=1.0,
                subtalker_temperature=0.9,
                return_dict=True,
            )
        tok0, scores = sample_like_hf(outn.logits[:, -1, :], history, top_k=50, temperature=0.9, repetition_penalty=1.05, suppress_tokens=suppress_tokens)
        history = torch.cat([history, tok0.unsqueeze(1)], dim=1)
        phase = 'text' if current.generation_step < current.trailing_text_hidden.shape[1] else 'pad'
        recs.append({
            'step': step,
            'phase': phase,
            'generation_step_in': int(current.generation_step),
            'generation_step_out': int(outn.generation_step),
            'sampled_codec0': int(tok0.item()),
            'input_ids': history[0, -1].item(),
            'position_ids': position_ids.detach().cpu().tolist(),
            'cache_position': cache_position.detach().cpu().tolist(),
            'past_hidden': tensor_stat(outn.past_hidden),
            'codec_ids': None if outn.hidden_states[1] is None else outn.hidden_states[1].detach().cpu().tolist(),
        })
        current = outn

    out_path = REPO_ROOT / 'workspace' / 'official_step_trace.json'
    out_path.write_text(json.dumps(recs, ensure_ascii=False, indent=2))
    print('saved', out_path)
    for r in recs:
        print(r['step'], r['phase'], 'gen', r['generation_step_in'], '->', r['generation_step_out'], 'codec0', r['sampled_codec0'], 'codec_ids', r['codec_ids'])


if __name__ == '__main__':
    main()

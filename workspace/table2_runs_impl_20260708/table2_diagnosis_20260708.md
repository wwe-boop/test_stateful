# Table 2 CER Diagnosis

Date: 2026-07-08

Scope: Table 2 only, `test-prosody-mini`, remote repo `/home/zehan/workspace/Qwen3-TTS-Triton`.

## Current Table 2 Status

The measured Table 2 remains:

- `stateless_once`: CER `11.13%±0.09%`
- `stateful_stream`: CER `10.86%±0.12%`
- `acoustic_tail_only`: CER `11.40%±0.85%`
- `kv_tail_only`: CER `27.01%±0.55%`
- `tail_kv_pause_recovery`: CER `28.24%±0.23%`
- `full_steadystream`: not measured, because no C4 trained checkpoint was found

The C2/C3 rows should be read as diagnostic prototypes, not acceptable final systems.

## Why CER Became Bad

The high CER is real generation failure, not table aggregation or ASR noise.

Smoke sample `seed_42/prosody_mini_001` shows:

- `stateful_stream` and `acoustic_tail_only`: ASR includes the final sentence and CER is `2.27%`.
- `kv_tail_only`: ASR stops before `听起来很费时间，但香气层次反而更干净。`, CER `21.59%`.
- `tail_kv_pause_recovery`: ASR also misses the final sentence or only produces residue, CER about `20-24%`.

Ablations performed:

- Resetting inherited `token_counts` did not fix CER.
- Sweeping `kv_tail_tokens` over `16/32/64/128/256/384` changed `audio_steps`, but ASR still missed the second sentence for C2.
- Prepending cached CustomVoice prefix before the KV tail (`kv_prepend_prefix=true`) also did not recover CER; several settings early-EOSed.

Likely root cause:

Current C2 restores a cropped Talker KV tail across segment boundaries, but this cropped KV does not have a verified sink/prefix/position contract. It may improve boundary F0/energy because it injects recent acoustic/prosodic context, but it destabilizes semantic continuation and EOS behavior. C3 inherits the same C2 failure, so pause recovery can reduce pause error while CER remains bad.

## Does The Plan Explain Training?

Yes, `steadystream_plan_v2.md` section 3 explains the intended C4 training direction, but it is not a ready-to-run command.

The plan requires:

- 50-300h long recordings for 2-3 target speakers.
- Official `Qwen3-TTS/finetuning/prepare_data.py` and `sft_12hz.py --speaker_name` as the base path.
- A modified continuation dataset format that keeps real boundary silence in previous segment codes and records `pause_ms` / `punct_class`.
- Continuation SFT with bounded-history dropout, optional lookahead, and loss only on speech tokens.
- LoRA first, then full SFT, with WER/SIM gates before accepting C4.

Current blocker:

No ready C4 dataset path or trained C4 checkpoint was found in this repo. The official fine-tuning scripts exist in the parent repo, but they are single-speaker fine-tuning scaffolding; C4 still needs the continuation JSONL/collate changes and suitable long-audio data.

## Code/Diagnostic Changes Added

Added explicit diagnostic switches in `engine/backend/engine_loop.py`:

- `kv_inherit_token_counts=false` can reset token counts for ablation.
- `kv_prepend_prefix=true` can test cached prefix + KV tail composition.

Both are opt-in; default C2 behavior remains the original measured prototype.

Validation:

- `/home/zehan/miniforge3/envs/qwen3-tts/bin/python -m pytest -q tests/unit/test_engine_loop_pipeline.py -q`
- Result: `19 passed`

Relevant smoke artifacts:

- Remote: `workspace/table2_smoke_cer_20260708.json`
- Remote: `workspace/table2_smoke_prefixtail_sweep_cer_20260708.json`
- Local audio: `outputs/steadystream_table2_c2fix_audio/`

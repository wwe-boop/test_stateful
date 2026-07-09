# C4 Speaker 001 Data Inventory

Date: 2026-07-10
Scope: Phase 0 inventory for `steadystream_c4_execution_playbook_20260710.md`.

## Summary

Speaker `001` has usable short-form data on 5090, but no discoverable real continuous long recordings. The C4 route should therefore use synthetic full-passage generation as the primary data source, with the existing 001 short-form dataset only as reference/context, not as the main continuation-training source.

## Locations Checked

- `/home/train/tts`: no `.wav`, `.flac`, `.mp3`, or training `.jsonl` manifests beyond model/config files.
- `/home/train/tts/qwen3-tts/trained/zehan/0701_trained_model`: checkpoint only.
- `/home/train/tts/qwen3-tts/trained/zehan/spk001_60min_lr4e7_bs4_ep3_epoch2_20260701`: checkpoint only.
- `/home/zehan/workspace/datasets/cosyvoice2_train_raw_single_ref_20260520`: 001 short-form synthesized/cleaned dataset.
- `/home/zehan/workspace/WashDataset`: one `clean_speech.wav` plus processed mp3 fragments; not enough evidence as 0701 training source.
- `/home/zehan/workspace/instrcutTtsEval/*001*`: eval/generation outputs, not raw training recordings.

## 001 Short-Form Dataset Found

Path: `/home/zehan/workspace/datasets/cosyvoice2_train_raw_single_ref_20260520`

Important manifests:

- `manifests/input_source.jsonl`: 2689 source text rows, speaker `customer_service_clone_3000`, source paths under `/x2robot_v2/...`.
- `manifests/synth_manifest.jsonl`: 2689 synthesized rows, speaker `001`.
- `manifests/asr_manifest.jsonl`: 2689 ASR checked rows.
- `manifests/clean_manifest.jsonl`: 2673 kept rows.
- `manifests/reject_manifest.jsonl`: 16 rejected rows.

Stats from `audio_clean/*.wav`:

| metric | value |
|---|---:|
| clean wav count | 2673 |
| total duration | 7.873 h |
| mean duration | 10.60 s |
| median duration | 8.68 s |
| p90 duration | 19.32 s |
| p95 duration | 20.84 s |
| p99 duration | 23.68 s |
| max duration | 28.88 s |
| files >30 s | 0 |
| files >60 s | 0 |

Existing summary files report:

- `synth_ok=2689`, `synth_fail=0`.
- `clean=2673`, `reject=16`, `clean_ratio=0.99405`.
- `duration_total_sec=28496.88`, `duration_avg_sec=10.598`, `duration_max_sec=28.88`.

## Missing Original Continuous Data

`input_source.jsonl` references original paths such as:

`/x2robot_v2/zehan/Qwen3-TTS-Train/workspaces/customer_service_clone_3000_audio_json_20260519/audio/001_short.wav`

But `/x2robot_v2` is not mounted on 5090, so the original source audio/json workspace is unavailable from this host. The available local dataset is already short-form synthesized/cleaned audio, not continuous raw recording.

## Local Audio Pulled For Listening

Pulled to local Codex workspace:

`outputs/c4_phase0_inventory_audio/`

Files:

- `0.wav`: short sample, 4.36 s.
- `2.wav`: long-ish sample, 17.44 s.
- `2015.wav`: longest clean sample, 28.88 s.

## Route Decision

No continuous long recordings were found. Proceed with playbook Phase 1 synthetic full-passage generation as the main C4 data route.

Use the found `cosyvoice2_train_raw_single_ref_20260520` dataset only as:

- evidence that speaker `001` has a healthy short-form domain,
- optional source of style/domain text snippets,
- optional fallback for single-sentence replay/regularization,
- not as the main C4 continuation dataset because it has no >30 s continuous recordings and no original paragraph-level continuity.

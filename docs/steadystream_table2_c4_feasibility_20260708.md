# SteadyStream Table 2 C4 Training Feasibility Note

Date: 2026-07-08

Scope: Table 2 only, remote repo `/home/zehan/workspace/Qwen3-TTS-Triton`.

## Current Table 2 State

The first five Table 2 rows have been measured on `test-prosody-mini`
with three seeds:

| Variant | CER |
|---|---:|
| Stateless | 11.13% +/- 0.09% |
| Existing stateful Triton stream | 10.86% +/- 0.12% |
| C1 acoustic tail prototype | 11.40% +/- 0.85% |
| C2 KV/token tail prototype | 27.01% +/- 0.55% |
| C1+C2+C3 prototype | 28.24% +/- 0.23% |
| Full SteadyStream C4 | Not measured |

The C2 and C1+C2+C3 rows are diagnostic prototypes, not acceptable final
systems. The full SteadyStream row must remain blank until a continuation
trained checkpoint exists. Using the base engine for that row would hide
the train-test mismatch that C4 is supposed to solve.

## Why CER Regressed

The CER jump is a real generation failure rather than an ASR or aggregation
artifact. Smoke audio for `seed_42/prosody_mini_001` shows that stateful
streaming and C1 include the final sentence, while the C2 prototype often
stops before the final sentence:

`听起来很费时间，但香气层次反而更干净。`

The likely root cause is that the current C2 path restores a cropped Talker
KV tail without a verified sink, CustomVoice prefix, position-id, and EOS
contract. This can preserve some boundary prosody, which explains better
F0 and energy numbers, but it destabilizes semantic continuation. C3 then
inherits the C2 failure: pause correction can improve pause deviation, but
it cannot repair early EOS or missing text.

## What The Plan Requires For C4

`steadystream_plan_v2.md` section 3 defines C4 as continuation SFT, not as
a direct invocation of the official single-sentence fine-tuning command.
The required training shape is:

- 50-300h long continuous recordings for 2-3 target speakers.
- Strict source isolation between training and `test-prosody`.
- Real boundary silence retained at the end of the previous segment codes,
  capped at 600 ms.
- Per-boundary `pause_ms` and `punct_class`.
- Multi-segment sequence assembly:
  `[CustomVoice prefix] + sum(text_k + codes_k + boundary_token)`.
- Loss only on speech codec tokens.
- Bounded-history dropout and optional lookahead to match the C2 inference
  format.

The parent repo `/home/zehan/workspace/Qwen3-TTS/finetuning` currently
contains the official single-speaker, single-utterance path:

`audio + text + ref_audio -> prepare_data.py -> audio_codes -> sft_12hz.py`

That path is useful as a base, but its current dataset/collate builds one
text segment followed by one audio-code segment and EOS. It does not yet
build the multi-segment continuation sequence needed for C4.

## Current Blocker

No ready C4 continuation dataset or trained C4 checkpoint was found under
the checked remote paths:

- `/home/zehan/workspace/Qwen3-TTS-Triton`
- `/home/zehan/workspace/Qwen3-TTS`
- `/x2robot_v2/zehan/Qwen3-TTS-EasyFinetuning`
- `/mnt/nas/zbl-nas-1/zehan`

GPU state also argues against starting an unverified training run:

- GPU0 is occupied by the vLLM Qwen3-30B service.
- GPU1 is occupied by the Qwen3-TTS engine service.

Therefore the responsible Table 2 action is to keep the C4 row blank, record
the blocker, and prepare validation tooling for the moment a real long-audio
manifest is provided.

## Added Readiness Check

`scripts/python/build_c4_continuation_manifest.py` validates the expected
C4 continuation JSONL schema. It checks:

- Top-level `sample_id`, `speaker_name`, `language`, and `segments`.
- At least two segments per sample by default.
- Per-segment non-empty `text`.
- Per-segment `pause_ms` in `[0, 600]` ms by default.
- Per-segment known `punct_class`.
- Optional training-ready `codes` validation with `--require-codes`.
- Optional exact source-overlap guard with `--eval-source-list`.

Example for a training-ready manifest:

```bash
python scripts/python/build_c4_continuation_manifest.py \
  --input-jsonl /path/to/c4_continuation_with_codes.jsonl \
  --require-codes \
  --output-summary workspace/c4_manifest_summary.json
```

This script does not invent data and does not launch training. It only
answers whether the manifest is shaped enough to justify modifying the
fine-tuning collate and starting a C4 LoRA run.

## API-Synthetic Smoke Data

API synthesis can be used to create a diagnostic C4 smoke dataset, but it
must not be treated as the final C4 training source for Table 2. If the API
is the same base Qwen3-TTS engine under test, synthetic training becomes
self-distillation and can reinforce the same early-EOS and KV-contract
failure that caused the C2 CER regression.

The supported smoke path is:

```bash
python scripts/python/synthesize_c4_smoke_dataset.py \
  --input-jsonl workspace/datasets/test-prosody-mini-smoke.jsonl \
  --out-dir workspace/c4_synthetic_smoke_20260708_api1 \
  --endpoint 127.0.0.1:50051 \
  --limit 1
```

This writes:

- `c4_synthetic_smoke_manifest.jsonl`: no-code continuation manifest.
- `prepare_data_input.jsonl`: flat segment rows for official `prepare_data.py`.
- `wav/<sample_id>/segment_*.wav`: per-segment synthetic audio with boundary
  silence appended to the previous segment.
- `wav/<sample_id>/full.wav`: concatenated listening copy.
- `summary.json`: synthesis trace and audio durations.

The generated manifest passes schema validation without `--require-codes`.
Generating a training-ready manifest still requires audio tokenizer codes.
The remote host cannot reach HuggingFace directly, but the tokenizer can be
downloaded via the mirror:

```bash
HF_ENDPOINT=https://hf-mirror.com python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Qwen/Qwen3-TTS-Tokenizer-12Hz",
    local_dir="workspace/hf_models/Qwen3-TTS-Tokenizer-12Hz",
)
PY
```

Then extract codes with the parent official script:

```bash
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
python /home/zehan/workspace/Qwen3-TTS/finetuning/prepare_data.py \
  --device cuda:1 \
  --tokenizer_model_path /home/zehan/workspace/Qwen3-TTS-Triton/workspace/hf_models/Qwen3-TTS-Tokenizer-12Hz \
  --input_jsonl workspace/c4_synthetic_smoke_20260708_api1/prepare_data_input_abs.jsonl \
  --output_jsonl workspace/c4_synthetic_smoke_20260708_api1/prepare_data_with_codes.jsonl
```

Attach the flat `audio_codes` back to the continuation manifest:

```bash
python scripts/python/attach_c4_codes_from_prepare.py \
  --manifest-jsonl workspace/c4_synthetic_smoke_20260708_api1/c4_synthetic_smoke_manifest.jsonl \
  --prepared-jsonl workspace/c4_synthetic_smoke_20260708_api1/prepare_data_with_codes.jsonl \
  --output-jsonl workspace/c4_synthetic_smoke_20260708_api1/c4_synthetic_smoke_manifest_with_codes.jsonl
```

The one-sample API smoke run now validates with `--require-codes`:
8 coded segments, 247 code frames, and 0 schema issues. This is still smoke
data only, not final C4 evidence.

## Continuation Batch Dry-Run

Before modifying the training loop, run a layout dry-run:

```bash
python scripts/python/build_c4_continuation_batch.py \
  --manifest-jsonl workspace/c4_synthetic_smoke_20260708_api1/c4_synthetic_smoke_manifest_with_codes.jsonl \
  --output-summary workspace/c4_synthetic_smoke_20260708_api1/continuation_batch_summary.json
```

The builder creates the same tensor families as the official collate
(`input_ids`, `codec_ids`, `codec_mask`, `codec_0_labels`, masks), but lays
all segments in one sequence and masks loss only on speech codec tokens.
For now, segment boundaries are represented by `codec_eos_token_id` and are
not trained as loss targets. A learned `<bnd>` token still requires a model
vocabulary/config change.

Training status:

- `peft` and `bitsandbytes` are not installed in the current `qwen3-tts`
  environment, so LoRA/QLoRA is not available yet.
- The official `finetuning/sft_12hz.py` path is full-parameter training.
- GPU1 has roughly 11 GB free while the TTS engine is resident, so a direct
  1.7B full-parameter C4 run is not safe without freeing/moving services or
  adding LoRA support.

The 0.6B Base model can be downloaded through the same mirror for a
no-backward compatibility smoke:

```bash
HF_ENDPOINT=https://hf-mirror.com python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    local_dir="workspace/hf_models/Qwen3-TTS-12Hz-0.6B-Base",
)
PY

CUDA_VISIBLE_DEVICES=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
python scripts/python/run_c4_forward_smoke.py \
  --model-dir workspace/hf_models/Qwen3-TTS-12Hz-0.6B-Base \
  --manifest-jsonl workspace/c4_synthetic_smoke_20260708_api1/c4_synthetic_smoke_manifest_with_codes.jsonl \
  --output-summary workspace/c4_synthetic_smoke_20260708_api1/c4_forward_smoke_0p6b_summary.json
```

This is not a training run. It only verifies that the continuation batch is
accepted by a real Qwen3-TTS forward pass before adding LoRA or freeing GPUs
for an actual C4 update.

The forward smoke uses `talker.text_projection` when text and codec embedding
dimensions differ. This matters for 0.6B, where text embeddings are 2048-D and
codec embeddings are 1024-D; direct addition, as in the current official SFT
snippet, is not shape-compatible for this smoke path.

Observed 0.6B smoke result on the one-sample synthetic manifest:

- batch shape: `[1, 339, 2]`
- codec/loss positions: `247`
- talker loss: `13.598655`
- sub-talker loss: `11.102696`
- combined no-backward loss: `16.929464`
- CUDA reserved memory during smoke: about `2.56 GB`

A one-step frozen-backbone train smoke can be run without PEFT:

```bash
CUDA_VISIBLE_DEVICES=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
python scripts/python/run_c4_train_step_smoke.py \
  --model-dir workspace/hf_models/Qwen3-TTS-12Hz-0.6B-Base \
  --manifest-jsonl workspace/c4_synthetic_smoke_20260708_api1/c4_synthetic_smoke_manifest_with_codes.jsonl \
  --save-trainable-state workspace/c4_synthetic_smoke_20260708_api1/c4_train_step_smoke_state.pt \
  --output-summary workspace/c4_synthetic_smoke_20260708_api1/c4_train_step_smoke_summary.json
```

This freezes the backbone and trains only `talker.codec_head` plus
`talker.code_predictor.lm_head` for one optimizer step. It is only an
autograd/optimizer smoke test, not a useful C4 checkpoint.

Observed one-step smoke result:

- trainable params: `34,603,008 / 914,643,008` (`3.7832%`)
- combined loss before step: `16.929464`
- grad norm before clipping: `14.5625`
- saved trainable state: `67 MB`
- CUDA reserved memory during step: about `2.56 GB`

A package-free LoRA smoke can be run without installing `peft`:

```bash
CUDA_VISIBLE_DEVICES=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
python scripts/python/run_c4_lora_smoke_train.py \
  --model-dir workspace/hf_models/Qwen3-TTS-12Hz-0.6B-Base \
  --manifest-jsonl workspace/c4_synthetic_smoke_20260708_api1/c4_synthetic_smoke_manifest_with_codes.jsonl \
  --steps 5 --rank 4 --alpha 8 --lr 1e-4 \
  --save-adapter workspace/c4_synthetic_smoke_20260708_api1/c4_lora_smoke_adapter.pt \
  --output-summary workspace/c4_synthetic_smoke_20260708_api1/c4_lora_smoke_summary.json
```

This injects LoRA into matching `self_attn.q_proj` and `self_attn.v_proj`
modules, freezes the base weights, and saves only adapter tensors. It is a
closer rehearsal for C4 LoRA than the head-only smoke, but still uses one
synthetic sample and therefore is not a Table 2 C4 model.

Observed package-free LoRA smoke result:

- matched modules: `66`
- trainable params: `675,840 / 915,318,848` (`0.073837%`)
- loss over 5 steps: `16.929464 -> 16.068239`
- saved adapter: `1.4 MB`
- CUDA reserved memory during run: about `4.32 GB`

## Five-Sample API-Synthetic C4 Smoke

The one-sample smoke was extended to a five-sample synthetic continuation set
generated from DashScope text and the local TTS API. One generated text row used
speaker `Ethan`, which is not supported by the current Triton config, so the
smoke copy maps it to `ryan` and preserves the original value in
`metadata.original_speaker`.

Remote paths:

- Text manifest:
  `workspace/datasets/c4-synthetic-train-smoke5.supported.jsonl`
- Synthesized audio and C4 artifacts:
  `workspace/c4_synthetic_smoke5_20260708/`
- Training-ready continuation manifest:
  `workspace/c4_synthetic_smoke5_20260708/c4_synthetic_smoke_manifest_with_codes.jsonl`

Validation result:

- samples: `5`
- segments: `42`
- coded segments: `42`
- code frames: `1,831`
- speech from codes: `146.48 s`
- speakers: `ryan=1`, `serena=2`, `vivian=2`
- punct classes: `colon=2`, `comma=14`, `exclamation=3`,
  `period=19`, `question=1`, `semicolon=3`
- schema issues: `0`

The five-sample continuation batch dry-run accepts all rows at once:

- batch shape: `[5, 633, 2]`
- sequence lengths: `[340, 463, 633, 510, 484]`
- codec/loss positions: `1,831`
- attention tokens: `2,430`

`run_c4_lora_smoke_train.py` now supports multiple manifest rows by cycling
rows with batch size 1. This avoids mixing different speaker embeddings inside
one tensor batch while still validating that the multi-sample manifest, per-row
reference mel, continuation layout, optimizer, and adapter save path all work.

Observed five-sample package-free LoRA smoke result on the 0.6B base model:

- sample strategy: `cycle_rows_batch_size_1`
- manifest rows: `5`
- steps: `10` (each sample seen twice)
- matched modules: `66`
- trainable params: `675,840 / 915,318,848` (`0.073837%`)
- aggregate codec/loss positions: `1,831`
- loss over 10 steps:
  `[16.529987, 16.487158, 13.633871, 15.783941, 15.756899,
  15.764313, 15.568721, 12.864022, 14.712641, 14.752712]`
- first step: `prosody_mini_2001`, combined loss `16.529987`,
  grad norm `9.8125`
- last step: `prosody_mini_2005`, combined loss `14.752712`,
  grad norm `11.6875`
- saved adapter:
  `workspace/c4_synthetic_smoke5_20260708/c4_lora_smoke5_adapter.pt`
  (`1.4 MB`)

This closes the C4 plumbing gap from API synthesis to official tokenizer codes
to continuation collation to LoRA optimization. It still must not be reported as
the full SteadyStream C4 row, because the data are synthetic self-distillation
from the same base TTS service and the adapter has not been deployed or measured
with the Table 2 inference components.

## Next Training Step Once Data Exists

1. Run the readiness check on the real continuation JSONL.
2. Reject the dataset if it has missing codes, missing pause fields, source
   overlap with evaluation data, or too little total speech duration.
3. Implement a continuation dataset/collate in the parent fine-tuning repo
   that shares the CustomVoice prefix builder with the Triton inference path.
4. Add single-sample token-layout assertions before any long training run.
5. Start a small LoRA smoke run first. Only after CER/WER, SIM, and audio
   spot checks pass should the checkpoint be used to fill the Table 2 C4 row.

# SteadyStream Table 2 E47 Delivery

Scope: Table 2 only. This delivery reports the current runtime C4 ICL key-variant result on `speaker=001`, not Table 3/4.

## Run Contract

| Item | Value |
|---|---|
| Dataset | `workspace/datasets/test-prosody-mini_speaker001.jsonl` first 20 rows |
| Seed | `42` |
| Runtime | 5090 custom TRT service, gRPC `127.0.0.1:50071` |
| Checkpoint | merged C4 interleaved LoRA200 custom checkpoint |
| Input mode | `token` |
| Boundary policy | `force_text_chunk_boundary=true`, proxy boundary disallowed |
| ASR/CER | Paraformer-zh via `workspace/table2_asr_batch.py` |

## Delivered Table

| Row | Variant | What it does | Exact boundary | CER mean | Pause deviation | Current decision |
|---|---|---|---:|---:|---:|---|
| Baseline online | `stateful_stream` | normal token streaming, no explicit history ICL | pass 20/20 | 3.1026% | 98.295ms | healthy online baseline |
| Offline reference | `offline_full` | synthesize full text once | proxy only | 4.8454% | 91.137ms | reference only; ASR sensitive on digit-heavy rows |
| C1+C3 transition | `acoustic_tail_pause_recovery` | acoustic tail carry + pause recovery, no KV/ICL semantic history | pass 20/20 | **2.5031%** | 137.790ms | safe transition/control row |
| C4 ICL | `c4_icl_prefill` | runtime ICL layout: previous text+codes as prefill, current text decoded from pad channel | pass 20/20 | 3.1100% | 100.820ms | recommended semantic SteadyStream row |
| C4 ICL + C3 | `c4_icl_prefill_c3` | C4 ICL plus boundary pause recovery | pass 20/20 | 4.2417% | **21.875ms** | pause-optimized backup |

## Interpretation

The C4 ICL route is now validated at eval20 scale: `c4_icl_prefill` matches `stateful_stream` CER while preserving exact designed boundaries and avoiding the old KV-tail historical repetition. C3 is useful, but it is a tradeoff: `c4_icl_prefill_c3` cuts pause deviation by about 79ms versus pure ICL, while CER rises by about 1.13pp, mainly on digit/English/ID-heavy rows.

Recommendation for the Table 2 SteadyStream row: use `c4_icl_prefill` when CER/text correctness is primary; report `c4_icl_prefill_c3` as a pause-optimized variant if boundary pause quality is the priority.

## Artifacts

| Artifact | Path |
|---|---|
| 5090 generation output | `workspace/table2_c4_runtime_icl_eval20_e47_20260710/` |
| 4090 ASR/CER JSON | `workspace/table2_c4_runtime_icl_eval20_e47_20260710/table2_cer_key5.json` |
| Progress report | `docs/steadystream_table2_progress_report_20260708.md` |
| Local listen pack | `/Users/liuzehan/Documents/Codex/2026-07-08/ssh-4090-host-home-zehan-workspace/outputs/table2_c4_runtime_icl_eval20_e47_20260710/listen/` |

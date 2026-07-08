# SteadyStream C4 Dataset Sourcing Report

Date: 2026-07-08

Scope: find public or gated-public datasets suitable for the C4 continuation SFT row in SteadyStream Table 2. The target is not ordinary short-utterance TTS fine-tuning; C4 needs multi-segment continuation samples with enough natural context to recover boundary pause/prosody and bounded history behavior.

## Executive decision

Use **WenetSpeech4TTS Premium** as the first C4 pilot dataset. It is Mandarin, TTS-oriented, quality-filtered, has word timestamps, and was explicitly processed by adjusting segment boundaries, merging by speaker similarity/pause duration, enhancing audio quality, and removing speaker mixing within a segment. This is the closest public dataset I found to the continuation SFT shape required by `steadystream_plan_v2.md`.

Use **Emilia ZH** as the second source or augmentation pool. It is much larger and diverse, but needs a pilot inspection to prove that adjacent `Wxxxxxx` records under the same speaker preserve usable continuity.

Use **AISHELL-3** only for smoke tests and code-path validation. It is clean and easy to download, but mostly short independent TTS utterances, so it cannot be treated as final C4 evidence.

## Candidate ranking

| Priority | Dataset | Why it fits C4 | Main risk | Decision |
|---:|---|---|---|---|
| P0 | WenetSpeech4TTS Premium | Mandarin TTS data; 945h Premium subset; word timestamps; source comes from YouTube/podcasts; boundary and speaker-mixing cleanup already done | Gated HF access and non-commercial/copyright terms; audio is 16 kHz and should be resampled for Qwen tokenizer if needed | First pilot and most likely C4 training source |
| P0/P1 | Emilia ZH | About 49.9k hours Chinese in Emilia; diverse podcasts/interviews/audiobooks; JSON metadata has id, text, duration, speaker, language, dnsmos | Gated HF access; huge scale; adjacent item continuity must be verified | Second pilot / augmentation source |
| P2 | AISHELL-3 | Clean Mandarin TTS corpus; Apache 2.0; direct OpenSLR download works from 4090 | Short independent utterances; no natural long-context boundary | Smoke only, not final C4 |
| P2 | LibriTTS | English TTS corpus; neighboring sentence context can be extracted | English, not the Mandarin Table 2 target | Optional continuation mechanics test |
| P3 | MAGICDATA Mandarin Read Speech | 755h Mandarin transcripts, direct OpenSLR source | Read speech and CC BY-NC-ND; not TTS-continuation friendly | Avoid for final C4 |

## Remote access checks on 4090-Host

Direct Hugging Face is blocked from 4090 with `Network is unreachable`, but `hf-mirror.com` works.

Verified with `https://hf-mirror.com/api/datasets/...`:

| Repo | hf-mirror API | Gated | Files observed | Useful files observed |
|---|---|---:|---:|---|
| `Wenetspeech4TTS/WenetSpeech4TTS` | OK | auto | 137 | `Premium/` has 11 files; also `filelists/`, `DNSMOS_P808Scores/`, `Testset/` |
| `amphion/Emilia-Dataset` | OK | auto | 4345 | `Emilia/ZH/` has 920 tar shards; `Emilia-YODAS/ZH/` has 9 tar shards |

OpenSLR direct downloads are reachable from 4090:

| Dataset | Probe result |
|---|---|
| AISHELL-3 | `https://www.openslr.org/resources/93/data_aishell3.tgz` returns HTTP 200, size 19,057,141,777 bytes |
| LibriTTS dev-clean | `https://www.openslr.org/resources/60/dev-clean.tar.gz` returns HTTP 200, size 1,291,469,655 bytes |

## How to turn WenetSpeech4TTS into C4 data

WenetSpeech4TTS should be transformed into continuation JSONL as follows:

1. Download only `Premium` first, not the full 972 GB repository.
2. Parse `filelists/Premium_filelist.lst` to locate wav/txt pairs.
3. Parse each `.txt`: it contains utterance text and word timestamps.
4. Group segments by source id and same-speaker cluster inferred from IDs such as `X0000000028_244085533_S00092-S00094`.
5. Sort grouped segments by segment index and build 4-12 segment continuation samples.
6. Derive `punct_class` from transcript-ending punctuation.
7. Derive `pause_ms` from word timestamps and/or VAD on boundary audio; cap at 600 ms as required by the plan.
8. Resample to the Qwen tokenizer expected rate if needed, then run parent `prepare_data.py` to extract audio codes.
9. Feed codes into the existing C4 manifest validator and continuation batch builder.

Expected output schema remains:

```json
{"sample_id":"...","speaker_name":"...","language":"zh","source_id":"...","segments":[{"text":"...","wav":"...","pause_ms":240,"punct_class":"comma","codes":[...]}]}
```

## How to turn Emilia ZH into C4 data

Emilia should be used after a small pilot inspection:

1. Stream or download only `Emilia/ZH/*.tar` shards.
2. Read WebDataset records: each audio is paired with JSON metadata containing `id`, `text`, `duration`, `speaker`, `language`, and `dnsmos`.
3. Filter `language == zh`, `dnsmos >= 3.0` or stricter, and duration in a stable range such as 3-15 seconds.
4. Group by `speaker`, then sort by `Wxxxxxx` in ids like `ZH_B00000_S00000_W000001`.
5. Listen to adjacent windows from 20 groups before trusting it as true continuation data.
6. Use only groups that sound like real neighboring speech, not unrelated clips assigned to the same speaker.

## Download commands for the first pilot

These commands require the Hugging Face account to accept the gated dataset terms first. On 4090, use the mirror endpoint.

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_TOKEN=<token_with_accepted_dataset_access>

/home/zehan/miniforge3/envs/qwen3-tts/bin/python - <<PY
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Wenetspeech4TTS/WenetSpeech4TTS",
    repo_type="dataset",
    local_dir="workspace/datasets/raw/WenetSpeech4TTS",
    allow_patterns=[
        "Premium/Premium_md5check.txt",
        "Premium/WenetSpeech4TTS_Premium_0.tar.gz",
        "filelists/Premium_filelist.lst",
        "DNSMOS_P808Scores/*Premium*",
        "Testset/*",
    ],
    resume_download=True,
)
PY
```

If gated access is not ready, run AISHELL-3 only as a smoke fallback:

```bash
mkdir -p workspace/datasets/raw/AISHELL-3
cd workspace/datasets/raw/AISHELL-3
wget -c https://www.openslr.org/resources/93/data_aishell3.tgz
```

## Immediate next engineering task

Add `scripts/python/build_c4_manifest_wenetspeech4tts.py` after the first Premium tar is downloaded. The script should parse WenetSpeech4TTS filelists/txt timestamps, group adjacent source segments, compute boundary `pause_ms`/`punct_class`, and emit the same continuation manifest consumed by the existing C4 validator.

Do not use AISHELL-3 or synthetic API data as the final C4 Table 2 evidence. They are only for proving the training code path.

## Sources checked

- WenetSpeech4TTS Hugging Face dataset card: https://huggingface.co/datasets/Wenetspeech4TTS/WenetSpeech4TTS
- Emilia Hugging Face dataset card: https://huggingface.co/datasets/amphion/Emilia-Dataset
- AISHELL-3 OpenSLR SLR93: https://www.openslr.org/93/
- MAGICDATA Mandarin Chinese Read Speech Corpus OpenSLR SLR68: https://www.openslr.org/68/
- LibriTTS OpenSLR SLR60: https://www.openslr.org/60/

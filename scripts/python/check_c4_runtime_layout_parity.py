#!/usr/bin/env python3
"""Check C4 continuation batch layout against runtime full-current prefill shape.

The check is intentionally symbolic: it does not load the model or compare
floating-point embeddings. It verifies that the C4 training collate contains
the exact token-slot order needed by SteadyStream full-current inference:

    prefix + text1 + text_eos + codec_bos + codes1 + codec_eos
           + text2 + text_eos + codec_bos

The optional runtime prefix length lets us record whether the deployed
runtime's cacheable prefix has the same length as the C4 collate prefix.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.python.build_c4_continuation_batch import (  # noqa: E402
    build_continuation_batch,
    load_special_ids,
    read_jsonl,
)

OFFICIAL_ASSISTANT_FMT = "<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
OFFICIAL_INSTRUCT_FMT = "<|im_start|>user\n{instruct}<|im_end|>\n"


def _span_len(span: list[int]) -> int:
    return int(span[1]) - int(span[0])


def _encode_text_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"]
    if hasattr(encoded, "reshape"):
        return [int(x) for x in encoded.reshape(-1).tolist()]
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return [int(x) for x in encoded]


def _normalize_language_name(language: str) -> str:
    return str(language or "auto").strip().lower().replace(" ", "_")


def runtime_prefix_token_pairs(
    *,
    tokenizer: Any,
    config: dict[str, Any],
    language: str,
    speaker: str | None,
    instruct: str | None,
) -> list[list[int]]:
    talker = config.get("talker_config") or config
    tts_pad = int(config["tts_pad_token_id"])
    tts_bos = int(config["tts_bos_token_id"])
    codec_pad = int(talker["codec_pad_id"])
    codec_bos = int(talker["codec_bos_id"])
    codec_nothink = int(talker["codec_nothink_id"])
    codec_think_bos = int(talker["codec_think_bos_id"])
    codec_think_eos = int(talker["codec_think_eos_id"])
    codec_think = int(talker.get("codec_think_id", 2154))
    codec_language_id = {str(k).lower(): int(v) for k, v in (talker.get("codec_language_id") or {}).items()}
    spk_id = {str(k).lower(): int(v) for k, v in (talker.get("spk_id") or {}).items()}
    spk_is_dialect = {
        str(k).lower(): v
        for k, v in (talker.get("spk_is_dialect") or {}).items()
    }

    prefix_pairs: list[list[int]] = []
    if instruct is not None and instruct != "":
        for token_id in _encode_text_ids(tokenizer, OFFICIAL_INSTRUCT_FMT.format(instruct=instruct)):
            prefix_pairs.append([int(token_id), 0])

    assistant_empty_ids = _encode_text_ids(tokenizer, OFFICIAL_ASSISTANT_FMT.format(text=""))
    if len(assistant_empty_ids) < 3:
        raise ValueError("assistant prompt template produced fewer than 3 role tokens")
    for token_id in assistant_empty_ids[:3]:
        prefix_pairs.append([int(token_id), 0])

    language_key = _normalize_language_name(language)
    language_id = None
    if language_key != "auto":
        if language_key not in codec_language_id:
            raise ValueError(f"unknown language for runtime prefix: {language!r}")
        language_id = codec_language_id[language_key]
    speaker_key = str(speaker or "").lower()
    dialect_key = spk_is_dialect.get(speaker_key)
    if language_key in {"chinese", "auto"} and dialect_key:
        language_id = codec_language_id[str(dialect_key).lower()]

    if language_id is None:
        codec_prefix = [codec_nothink, codec_think_bos, codec_think_eos]
    else:
        codec_prefix = [codec_think, codec_think_bos, int(language_id), codec_think_eos]
    if speaker_key:
        if speaker_key not in spk_id:
            raise ValueError(f"unknown speaker for runtime prefix: {speaker!r}")
        codec_prefix.append(spk_id[speaker_key])
    codec_prefix.extend([codec_pad, codec_bos])

    # PrefillBuilder zips text slots with codec slots except the trailing codec_bos.
    for index, codec_id in enumerate(codec_prefix[:-1]):
        text_id = tts_bos if index == len(codec_prefix) - 2 else tts_pad
        prefix_pairs.append([int(text_id), int(codec_id)])
    return prefix_pairs


def check_layout_parity(
    rows: list[dict[str, Any]],
    *,
    tokenizer: Any,
    special_ids: Any,
    config: dict[str, Any],
    max_segments: int | None = None,
    runtime_prefix_len: int | None = None,
    runtime_language: str | None = None,
    runtime_speaker: str | None = None,
    runtime_instruct: str | None = None,
) -> dict[str, Any]:
    batch, layouts = build_continuation_batch(
        rows,
        tokenizer=tokenizer,
        special_ids=special_ids,
        max_segments=max_segments,
    )
    checks: list[dict[str, Any]] = []

    for row_index, layout in enumerate(layouts):
        segments = layout["segments"]
        if len(segments) < 2:
            checks.append(
                {
                    "sample_id": layout["sample_id"],
                    "status": "fail",
                    "reason": "need at least two segments for continuation parity",
                }
            )
            continue

        history = segments[0]
        current = segments[1]
        c4_prefix_len = int(history["text_span"][0])
        c4_prefix_pairs = batch["input_ids"][row_index, :c4_prefix_len, :].long().tolist()
        runtime_prefix_pairs = None
        runtime_content_matches_c4 = None
        runtime_content_mismatches: list[dict[str, Any]] = []
        if runtime_language is not None:
            runtime_prefix_pairs = runtime_prefix_token_pairs(
                tokenizer=tokenizer,
                config=config,
                language=runtime_language,
                speaker=runtime_speaker,
                instruct=runtime_instruct,
            )
            runtime_content_matches_c4 = runtime_prefix_pairs == c4_prefix_pairs
            if not runtime_content_matches_c4:
                max_len = max(len(runtime_prefix_pairs), len(c4_prefix_pairs))
                for idx in range(max_len):
                    got = runtime_prefix_pairs[idx] if idx < len(runtime_prefix_pairs) else None
                    expected = c4_prefix_pairs[idx] if idx < len(c4_prefix_pairs) else None
                    if got != expected:
                        runtime_content_mismatches.append(
                            {"pos": idx, "runtime": got, "c4": expected}
                        )
                    if len(runtime_content_mismatches) >= 8:
                        break
        history_text_len = _span_len(history["text_span"])
        current_text_len = _span_len(current["text_span"])
        history_codec_frames = int(history["codec_frames"])
        expected_current_codec_bos = (
            c4_prefix_len
            + history_text_len
            + 1  # history text EOS
            + 1  # history codec BOS
            + history_codec_frames
            + 1  # history boundary codec EOS
            + current_text_len
            + 1  # current text EOS
        )
        expected_prefill_len = expected_current_codec_bos + 1
        prefix_matches_runtime = (
            None if runtime_prefix_len is None else c4_prefix_len == runtime_prefix_len
        )

        failures: list[str] = []
        if int(history["text_eos"]) != int(history["text_span"][1]):
            failures.append("history_text_eos_not_after_text")
        if int(history["codec_bos"]) != int(history["text_eos"]) + 1:
            failures.append("history_codec_bos_not_after_text_eos")
        if int(history["codec_span"][0]) != int(history["codec_bos"]) + 1:
            failures.append("history_codec_span_not_after_bos")
        if int(history["boundary_codec_eos"]) != int(history["codec_span"][1]):
            failures.append("history_boundary_eos_not_after_codes")
        if int(current["text_span"][0]) != int(history["boundary_codec_eos"]) + 1:
            failures.append("current_text_not_after_history_boundary")
        if int(current["codec_bos"]) != expected_current_codec_bos:
            failures.append("current_codec_bos_mismatch")
        if runtime_content_matches_c4 is False:
            failures.append("runtime_prefix_content_mismatch")

        status = "pass" if not failures else "fail"
        if status == "pass" and prefix_matches_runtime is False:
            status = "warn"

        checks.append(
            {
                "sample_id": layout["sample_id"],
                "row_index": row_index,
                "status": status,
                "failures": failures,
                "c4_prefix_len": c4_prefix_len,
                "runtime_prefix_len": runtime_prefix_len,
                "prefix_matches_runtime": prefix_matches_runtime,
                "runtime_prefix_content_checked": runtime_language is not None,
                "runtime_prefix_content_matches_c4": runtime_content_matches_c4,
                "runtime_prefix_len_from_content": (
                    len(runtime_prefix_pairs) if runtime_prefix_pairs is not None else None
                ),
                "runtime_prefix_mismatches_head": runtime_content_mismatches,
                "history_text_tokens_including_eos": history_text_len + 1,
                "history_codec_frames": history_codec_frames,
                "current_text_tokens_including_eos": current_text_len + 1,
                "current_codec_bos": int(current["codec_bos"]),
                "expected_current_codec_bos": expected_current_codec_bos,
                "runtime_full_current_prefill_len_if_same_prefix": expected_prefill_len,
                "c4_current_codec_start": int(current["codec_span"][0]),
            }
        )

    statuses = [item["status"] for item in checks]
    overall = "pass"
    if any(status == "fail" for status in statuses):
        overall = "fail"
    elif any(status == "warn" for status in statuses):
        overall = "warn"

    return {
        "overall_status": overall,
        "batch_shape": list(batch["input_ids"].shape),
        "runtime_prefix_len": runtime_prefix_len,
        "n_rows": len(rows),
        "checks": checks,
        "note": (
            "warn means the C4 continuation order is internally consistent, "
            "but the C4 collate prefix length differs from the runtime "
            "cacheable prefix length supplied by --runtime-prefix-len."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-jsonl", type=Path, required=True)
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=Path("workspace/model_repository/tts_orchestrator/1/tokenizer"),
    )
    parser.add_argument(
        "--config-json",
        type=Path,
        default=Path("workspace/model_repository/tts_orchestrator/1/tokenizer/config.json"),
    )
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--max-segments", type=int, default=2)
    parser.add_argument("--runtime-prefix-len", type=int)
    parser.add_argument("--runtime-language")
    parser.add_argument("--runtime-speaker")
    parser.add_argument("--runtime-instruct")
    parser.add_argument("--output-summary", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from transformers import AutoTokenizer

    rows = read_jsonl(args.manifest_jsonl)
    if args.limit:
        rows = rows[: args.limit]
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_dir,
        trust_remote_code=True,
        fix_mistral_regex=True,
    )
    config = json.loads(args.config_json.read_text(encoding="utf-8"))
    summary = check_layout_parity(
        rows,
        tokenizer=tokenizer,
        special_ids=load_special_ids(args.config_json),
        config=config,
        max_segments=args.max_segments,
        runtime_prefix_len=args.runtime_prefix_len,
        runtime_language=args.runtime_language,
        runtime_speaker=args.runtime_speaker,
        runtime_instruct=args.runtime_instruct,
    )
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.output_summary:
        args.output_summary.parent.mkdir(parents=True, exist_ok=True)
        args.output_summary.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 1 if summary["overall_status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())

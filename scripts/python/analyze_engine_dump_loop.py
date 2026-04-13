#!/usr/bin/env python3
"""Brute-force loop analyzer for sequential engine dumps.

This script ignores engine scheduling/planning and only checks whether the
state carried from one dumped step into the next dumped step is correct.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
import torch.nn.functional as F


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip()).strip("._-") or "unknown"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dump-dir", required=True, help="Directory that contains .pt dump files")
    p.add_argument("--session-key", required=True, help="Full session key, e.g. longtext-medium-greedy:0")
    p.add_argument("--report-json", default=None, help="Optional output JSON report path")
    p.add_argument("--max-steps", type=int, default=0, help="Limit number of dumps to inspect (0 = all)")
    return p.parse_args()


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.shape != b.shape:
        return 0.0
    if a.numel() == 0:
        return 1.0
    af = a.reshape(-1).float()
    bf = b.reshape(-1).float()
    an = float(af.norm().item())
    bn = float(bf.norm().item())
    if an == 0.0 and bn == 0.0:
        return 1.0
    if an == 0.0 or bn == 0.0:
        return 0.0
    return float(F.cosine_similarity(af.unsqueeze(0), bf.unsqueeze(0)).item())


def _tensor_stats(expected: torch.Tensor, actual: torch.Tensor) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "shape_expected": list(expected.shape),
        "shape_actual": list(actual.shape),
    }
    if expected.shape != actual.shape:
        out["match"] = False
        out["reason"] = "shape_mismatch"
        return out
    if expected.dtype in (torch.int32, torch.int64, torch.int16, torch.int8, torch.uint8, torch.bool):
        neq = expected.to(torch.int64) != actual.to(torch.int64)
        out["match"] = bool(not neq.any())
        out["num_mismatch"] = int(neq.sum().item())
        return out

    eq = torch.eq(expected, actual)
    finite_mask = torch.isfinite(expected) & torch.isfinite(actual)
    if finite_mask.any():
        delta = (expected[finite_mask].float() - actual[finite_mask].float()).abs()
        out["max_abs"] = float(delta.max().item())
        out["mean_abs"] = float(delta.mean().item())
        out["cosine"] = _cosine(expected[finite_mask], actual[finite_mask])
    else:
        out["max_abs"] = 0.0
        out["mean_abs"] = 0.0
        out["cosine"] = 1.0
    out["match"] = bool(torch.equal(expected, actual))
    out["num_mismatch"] = int((~eq).sum().item())
    return out


def _load_dump(path: Path) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu")


def _find_row(payload: Dict[str, Any], session_key: str) -> int:
    sessions = payload["metadata"].get("slot_session_ids", [])
    if session_key not in sessions:
        return -1
    return int(sessions.index(session_key))


def _slice_batch_tensor(payload: Dict[str, Any], tensor: torch.Tensor, row_idx: int) -> torch.Tensor:
    batch = len(payload["metadata"].get("slot_session_ids", []))
    if tensor.ndim > 0 and tensor.shape[0] == batch:
        return tensor[row_idx : row_idx + 1]
    return tensor


def _get_tensor(entry: Dict[str, Any], section: str, key: str) -> torch.Tensor:
    payload = entry["payload"]
    row_idx = int(entry["row_idx"])
    tensor = payload[section][key]
    return _slice_batch_tensor(payload, tensor, row_idx)


def _iter_session_files(dump_dir: Path, session_key: str) -> Iterable[Path]:
    safe_session = _safe_name(session_key)
    session_timeline = dump_dir / "sessions" / safe_session / "timeline.jsonl"
    if session_timeline.is_file():
        seen: set[str] = set()
        with session_timeline.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                dump_file = str(row.get("dump_file") or "").strip()
                if not dump_file or dump_file in seen:
                    continue
                seen.add(dump_file)
                path = dump_dir / dump_file
                if path.is_file():
                    yield path
        return

    for path in sorted(dump_dir.glob("*.pt")):
        yield path


def _append_with_window(past: torch.Tensor, delta: torch.Tensor, max_past: int) -> torch.Tensor:
    out = torch.cat([past, delta], dim=3)
    if out.shape[3] > max_past:
        out = out[:, :, :, -max_past:, :]
    return out.contiguous()


def _expected_talker_next(cur: Dict[str, Any], nxt: Dict[str, Any]) -> torch.Tensor:
    meta = cur["payload"]["metadata"]
    row = int(cur["row_idx"])
    cur_real_len = int(meta["original_talker_past_lens"][row])
    if meta["stage"] == "prefill":
        real_next = _get_tensor(cur, "outputs", "talker_new_kv").contiguous()
    else:
        cur_padded = _get_tensor(cur, "inputs", "talker_past_kv")
        cur_real = cur_padded[:, :, :, :cur_real_len, :]
        real_next = torch.cat(
            [cur_real, _get_tensor(cur, "outputs", "talker_new_kv")],
            dim=3,
        ).contiguous()
    next_padded_len = int(_get_tensor(nxt, "inputs", "talker_past_kv").shape[3])
    if real_next.shape[3] < next_padded_len:
        pad = next_padded_len - real_next.shape[3]
        real_next = torch.nn.functional.pad(real_next, (0, 0, 0, pad))
    return real_next.contiguous()


def _expected_c2w_next(cur: Dict[str, Any], nxt: Dict[str, Any]) -> torch.Tensor:
    meta = cur["payload"]["metadata"]
    row = int(cur["row_idx"])
    cfg = meta["config"]
    max_past = int(cfg["c2w_sliding_window"]) - 1
    if meta["stage"] == "prefill":
        real_next = _get_tensor(cur, "outputs", "c2w_new_kv")
    else:
        cur_real_len = int(meta["slot_c2w_len_before"][row])
        cur_padded = _get_tensor(cur, "inputs", "c2w_past_kv")
        if cur_real_len > 0:
            cur_real = cur_padded[:, :, :, -cur_real_len:, :]
        else:
            cur_real = cur_padded[:, :, :, :0, :]
        real_next = _append_with_window(
            cur_real,
            _get_tensor(cur, "outputs", "c2w_new_kv"),
            max_past,
        )
    next_padded_len = int(_get_tensor(nxt, "inputs", "c2w_past_kv").shape[3])
    if real_next.shape[3] < next_padded_len:
        pad = next_padded_len - real_next.shape[3]
        real_next = torch.nn.functional.pad(real_next, (0, 0, pad, 0))
    return real_next.contiguous()


def _check_position_ids(entry: Dict[str, Any]) -> Dict[str, Any]:
    payload = entry["payload"]
    row_idx = int(entry["row_idx"])
    pos = _get_tensor(entry, "inputs", "position_ids")
    meta = payload["metadata"]
    expected_start = int(meta["slot_past_len_before"][row_idx])
    actual_start = int(pos[0, 0, 0, 0].item())
    return {
        "expected_start": expected_start,
        "actual_start": actual_start,
        "match": expected_start == actual_start,
    }


def _check_cache_position(prev: Dict[str, Any], nxt: Dict[str, Any]) -> Dict[str, Any]:
    prev_cp = int(_get_tensor(prev, "inputs", "cache_position")[0, 0].item())
    next_cp = int(_get_tensor(nxt, "inputs", "cache_position")[0, 0].item())
    return {
        "expected_next": prev_cp + 1,
        "actual_next": next_cp,
        "match": (prev_cp + 1) == next_cp,
    }


def _build_expected_mask(
    *,
    batch: int,
    seq: int,
    padded_len: int,
    valid_len: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    total = padded_len + seq
    expected = torch.zeros(batch, 1, seq, total, dtype=dtype)
    if padded_len > 0 and valid_len < padded_len:
        expected[:, :, :, valid_len:padded_len] = float("-inf")
    if seq > 1:
        causal = torch.triu(
            torch.full((seq, seq), float("-inf"), dtype=dtype),
            diagonal=1,
        )
        expected[:, :, :, padded_len : padded_len + seq] += causal.unsqueeze(0).unsqueeze(0)
    return expected


def _check_mask(entry: Dict[str, Any], *, key: str, padded_len: int, valid_len: int) -> Dict[str, Any]:
    bias = _get_tensor(entry, "inputs", key)
    expected = _build_expected_mask(
        batch=int(bias.shape[0]),
        seq=int(bias.shape[2]),
        padded_len=int(padded_len),
        valid_len=int(valid_len),
        dtype=bias.dtype,
    )
    return {
        "pad_cols": max(0, padded_len - valid_len),
        **_tensor_stats(expected, bias),
    }


def _pairwise_checks(cur: Dict[str, Any], nxt: Dict[str, Any]) -> Dict[str, Any]:
    cur_meta = cur["payload"]["metadata"]
    nxt_meta = nxt["payload"]["metadata"]
    cur_row = int(cur["row_idx"])
    nxt_row = int(nxt["row_idx"])

    result: Dict[str, Any] = {
        "current_dump_id": int(cur_meta["dump_id"]),
        "next_dump_id": int(nxt_meta["dump_id"]),
        "current_stage": cur_meta["stage"],
        "next_stage": nxt_meta["stage"],
        "session": cur_meta["slot_session_ids"][cur_row],
        "talker_kv": _tensor_stats(_expected_talker_next(cur, nxt), _get_tensor(nxt, "inputs", "talker_past_kv")),
        "c2w_kv": _tensor_stats(_expected_c2w_next(cur, nxt), _get_tensor(nxt, "inputs", "c2w_past_kv")),
        "token_counts": _tensor_stats(
            _get_tensor(cur, "outputs", "updated_token_counts"),
            _get_tensor(nxt, "inputs", "token_counts"),
        ),
        "cache_position": _check_cache_position(cur, nxt),
        "next_position_ids": _check_position_ids(nxt),
    }

    conv_pairs = zip(
        cur_meta.get("c2w_conv_output_names", []),
        nxt_meta.get("c2w_conv_input_names", []),
    )
    result["conv_states"] = {
        f"{out_name}->{in_name}": _tensor_stats(
            _get_tensor(cur, "outputs", out_name),
            _get_tensor(nxt, "inputs", in_name),
        )
        for out_name, in_name in conv_pairs
    }
    trans_pairs = zip(
        cur_meta.get("c2w_transconv_output_names", []),
        nxt_meta.get("c2w_transconv_input_names", []),
    )
    result["transconv_states"] = {
        f"{out_name}->{in_name}": _tensor_stats(
            _get_tensor(cur, "outputs", out_name),
            _get_tensor(nxt, "inputs", in_name),
        )
        for out_name, in_name in trans_pairs
    }
    return result


def main() -> None:
    args = _parse_args()
    dump_dir = Path(args.dump_dir).expanduser().resolve()
    files = list(_iter_session_files(dump_dir, args.session_key))
    if not files:
        raise SystemExit(f"No dumps found for session_key={args.session_key} under {dump_dir}")
    if args.max_steps > 0:
        files = files[: args.max_steps]

    files = sorted(files, key=lambda p: int(p.name.split("_", 1)[0]))

    print("Loop Dump")
    print("=========")
    print(f"dump_dir: {dump_dir}")
    print(f"session_key: {args.session_key}")
    print(f"num_files: {len(files)}")
    print()

    report: Dict[str, Any] = {
        "dump_dir": str(dump_dir),
        "session_key": args.session_key,
        "files": [str(p) for p in files],
        "initial": {},
        "pairs": [],
    }

    entries: list[Dict[str, Any]] = []
    for path in files:
        payload = _load_dump(path)
        row_idx = _find_row(payload, args.session_key)
        if row_idx < 0:
            continue
        entries.append({"path": path, "payload": payload, "row_idx": row_idx})

    if not entries:
        raise SystemExit(f"No payloads in {dump_dir} contain session_key={args.session_key}")

    first = entries[0]
    first_meta = first["payload"]["metadata"]
    first_row = int(first["row_idx"])
    report["initial"] = {
        "dump_id": int(first_meta["dump_id"]),
        "stage": first_meta["stage"],
        "position_ids": _check_position_ids(first),
        "talker_attention_bias": _check_mask(
            first,
            key="attention_bias",
            padded_len=int(_get_tensor(first, "inputs", "talker_past_kv").shape[3]),
            valid_len=int(first_meta["original_talker_past_lens"][first_row]),
        ),
        "c2w_attention_bias": _check_mask(
            first,
            key="c2w_attention_bias",
            padded_len=int(_get_tensor(first, "inputs", "c2w_past_kv").shape[3]),
            valid_len=int(first_meta["slot_c2w_len_before"][first_row]),
        ),
    }
    print(
        f"initial dump={first_meta['dump_id']} stage={first_meta['stage']} "
        f"position_ids_ok={report['initial']['position_ids']['match']} "
        f"talker_mask_ok={report['initial']['talker_attention_bias']['match']} "
        f"c2w_mask_ok={report['initial']['c2w_attention_bias']['match']}"
    )

    cur = first
    for nxt in entries[1:]:
        pair = _pairwise_checks(cur, nxt)
        report["pairs"].append(pair)

        conv_ok = all(v.get("match", False) for v in pair["conv_states"].values())
        trans_ok = all(v.get("match", False) for v in pair["transconv_states"].values())
        print(
            f"{pair['current_dump_id']:06d}->{pair['next_dump_id']:06d} "
            f"talker={pair['talker_kv'].get('match')} "
            f"c2w={pair['c2w_kv'].get('match')} "
            f"tc={pair['token_counts'].get('match')} "
            f"cp={pair['cache_position'].get('match')} "
            f"pos={pair['next_position_ids'].get('match')} "
            f"conv={conv_ok} trans={trans_ok}"
        )
        cur = nxt

    if args.report_json:
        report_path = Path(args.report_json).expanduser().resolve()
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print()
        print(f"Wrote report: {report_path}")


if __name__ == "__main__":
    main()

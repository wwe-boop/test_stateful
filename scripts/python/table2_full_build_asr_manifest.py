#!/usr/bin/env python3
"""Build an ASR manifest for Table 2 CER evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def join_segments(segments: object) -> str:
    if not isinstance(segments, list):
        return ""
    return "".join(str(seg).strip() for seg in segments)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", default="workspace/table2_runs")
    parser.add_argument("--out", default="workspace/table2_runs/table2_asr_manifest.json")
    parser.add_argument("--container-runs-root", default="/tmp/table2_runs_asr")
    parser.add_argument(
        "--variants",
        default=(
            "stateless_once,stateful_stream,acoustic_tail_only,"
            "kv_tail_only,tail_kv_pause_recovery,full_steadystream"
        ),
    )
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    rows = []
    for result_path in sorted(runs_root.glob("seed_*/*/results.json")):
        sample_dir = result_path.parent
        data = json.loads(result_path.read_text(encoding="utf-8"))
        ref_text = join_segments(data.get("segments"))
        if not ref_text:
            raise SystemExit(f"Missing segments/ref text in {result_path}")
        rel_sample_dir = sample_dir.relative_to(runs_root)
        for variant in variants:
            wav_name = f"{variant}.wav"
            wav_path = sample_dir / wav_name
            if not wav_path.exists():
                raise SystemExit(f"Missing wav: {wav_path}")
            rows.append(
                {
                    "key": f"{data['seed']}:{data['sample_id']}:{variant}",
                    "sample_id": data["sample_id"],
                    "seed": int(data["seed"]),
                    "speaker": data.get("speaker"),
                    "variant": variant,
                    "reference": ref_text,
                    "wav_path": str(Path(args.container_runs_root) / rel_sample_dir / wav_name),
                }
            )

    payload = {
        "runs_root": str(runs_root),
        "container_runs_root": args.container_runs_root,
        "variants": variants,
        "n_rows": len(rows),
        "rows": rows,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} ASR rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

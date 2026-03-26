#!/usr/bin/env python3
"""
Emit trtexec --minShapes / --optShapes / --maxShapes for talker_code2wav_fused.onnx.
Print three lines: MIN, OPT, MAX (comma-separated, no spaces).
"""

from __future__ import annotations

import sys


def c2w_state_specs(bmax: str, n_c2w_layers: int = 8):
    specs = []
    for i in range(n_c2w_layers):
        specs.append((f"past_kv_{i}_k", "1x16x1x64", "1x16x4x64", f"{bmax}x16x72x64"))
        specs.append((f"past_kv_{i}_v", "1x16x1x64", "1x16x4x64", f"{bmax}x16x72x64"))
    conv = [
        ("conv_state_0", "1x512x2"),
        ("conv_state_1", "1x1024x6"),
        ("conv_state_2", "1x1024x6"),
        ("conv_state_3", "1x1024x6"),
        ("conv_state_4", "1x768x6"),
        ("conv_state_5", "1x768x18"),
        ("conv_state_6", "1x768x54"),
        ("conv_state_7", "1x384x6"),
        ("conv_state_8", "1x384x18"),
        ("conv_state_9", "1x384x54"),
        ("conv_state_10", "1x192x6"),
        ("conv_state_11", "1x192x18"),
        ("conv_state_12", "1x192x54"),
        ("conv_state_13", "1x96x6"),
        ("conv_state_14", "1x96x18"),
        ("conv_state_15", "1x96x54"),
        ("conv_state_16", "1x96x6"),
    ]
    for name, s in conv:
        rest = s[2:]  # drop "1x"
        specs.append((name, s, s, f"{bmax}x{rest}"))
    tc = [
        ("transconv_overlap_0", "1x768x8"),
        ("transconv_overlap_1", "1x384x5"),
        ("transconv_overlap_2", "1x192x4"),
        ("transconv_overlap_3", "1x96x3"),
    ]
    for name, s in tc:
        rest = s[2:]
        specs.append((name, s, s, f"{bmax}x{rest}"))
    return specs


def main():
    if len(sys.argv) < 6:
        print(
            "Usage: trt_fused_talk_c2w_profiles.py H KV_HEADS HEAD_DIM NUM_LAYERS MAX_BATCH "
            "[MAX_INPUT_LEN] [MAX_SEQ_LEN] [NUM_C2W_DECODER_LAYERS]",
            file=sys.stderr,
        )
        sys.exit(1)
    H, KV, HD, NL, Bmax = sys.argv[1:6]
    max_in = sys.argv[6] if len(sys.argv) > 6 else "128"
    max_seq = sys.argv[7] if len(sys.argv) > 7 else "512"
    n_c2w = int(sys.argv[8]) if len(sys.argv) > 8 else 8
    nl = int(NL)
    Bopt = "1"
    opt_spast = "128"

    parts_min = [
        f"input_embeds:1x1x{H}",
        f"position_ids:1x3x1",
        "attention_bias:1x1x1x1",
        "past_seq_lens:1",
        "cache_position:1x1",
        "c2w_attention_bias:1x1x1x2",
    ]
    parts_opt = [
        f"input_embeds:{Bopt}x1x{H}",
        f"position_ids:{Bopt}x3x1",
        f"attention_bias:{Bopt}x1x1x{int(opt_spast) + 1}",
        f"past_seq_lens:{Bopt}",
        f"cache_position:{Bopt}x1",
        f"c2w_attention_bias:{Bopt}x1x1x5",
    ]
    parts_max = [
        f"input_embeds:{Bmax}x{max_in}x{H}",
        f"position_ids:{Bmax}x3x{max_in}",
        f"attention_bias:{Bmax}x1x{max_in}x{int(max_seq) + int(max_in)}",
        f"past_seq_lens:{Bmax}",
        f"cache_position:{Bmax}x1",
        f"c2w_attention_bias:{Bmax}x1x1x73",
    ]

    for i in range(nl):
        parts_min.append(f"past_kv_{i}_k:1x{KV}x0x{HD}")
        parts_min.append(f"past_kv_{i}_v:1x{KV}x0x{HD}")
        parts_opt.append(f"past_kv_{i}_k:{Bopt}x{KV}x{opt_spast}x{HD}")
        parts_opt.append(f"past_kv_{i}_v:{Bopt}x{KV}x{opt_spast}x{HD}")
        parts_max.append(f"past_kv_{i}_k:{Bmax}x{KV}x{max_seq}x{HD}")
        parts_max.append(f"past_kv_{i}_v:{Bmax}x{KV}x{max_seq}x{HD}")

    for name, smin, sopt, smax in c2w_state_specs(Bmax, n_c2w_layers=n_c2w):
        parts_min.append(f"c2w_{name}:{smin}")
        parts_opt.append(f"c2w_{name}:{sopt}")
        parts_max.append(f"c2w_{name}:{smax}")

    print(",".join(parts_min))
    print(",".join(parts_opt))
    print(",".join(parts_max))


if __name__ == "__main__":
    main()

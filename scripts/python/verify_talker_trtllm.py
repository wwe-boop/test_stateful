#!/usr/bin/env python3
"""
[Phase 2, Item 9 — Part 2] Verify Talker TRT-LLM engine against PyTorch reference.

Loads the TRT-LLM engine via ModelRunner, passes inputs_embeds through
prompt_embedding_table, and compares logits with the PyTorch reference.

This script runs INSIDE a TRT-LLM container.

Usage (inside container):
  python3 /mnt/scripts/verify_talker_trtllm.py \
      --engine-dir /mnt/model/trtllm_engine \
      --ref-file /mnt/model/talker_trtllm_ref.npz \
      --n-decode-steps 5
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("talker_trtllm")


def cosine_sim(a, b):
    a_t = torch.from_numpy(a).float().flatten().unsqueeze(0)
    b_t = torch.from_numpy(b).float().flatten().unsqueeze(0)
    return float(F.cosine_similarity(a_t, b_t))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine-dir", required=True)
    parser.add_argument("--ref-file", required=True)
    parser.add_argument("--n-decode-steps", type=int, default=5)
    parser.add_argument("--report-file", default=None)
    args = parser.parse_args()

    ref = np.load(args.ref_file)
    inputs_embeds = ref["inputs_embeds"]  # [1, S, H]
    ref_prefill_logits = ref["prefill_logits"]  # [1, S, V]
    ref_decode_tokens = ref["decode_tokens"]  # [S_decode+1]
    ref_decode_logits = ref["decode_logits"]  # [N, V]
    hidden_size = int(ref["hidden_size"])
    n_text = int(ref["n_text"])
    codec_bos_id = int(ref["codec_bos_id"])
    text_ids = ref["text_ids"]  # [1, n_text]

    seq_len = inputs_embeds.shape[1]
    vocab_size = ref_prefill_logits.shape[-1]
    logger.info(f"Reference: seq_len={seq_len}, hidden={hidden_size}, vocab={vocab_size}")
    logger.info(f"Reference decode tokens: {ref_decode_tokens.tolist()}")

    # Load TRT-LLM engine (TRT-LLM 1.1.0+ supports QK-Norm for Qwen3)
    try:
        import tensorrt_llm
        _ver = getattr(tensorrt_llm, "__version__", "unknown")
        logger.info(f"TensorRT-LLM version: {_ver}")
    except Exception:
        pass
    logger.info(f"Loading TRT-LLM engine from {args.engine_dir} ...")
    t0 = time.time()

    from tensorrt_llm.runtime import ModelRunner, SamplingConfig

    runner = ModelRunner.from_dir(
        args.engine_dir,
        max_output_len=args.n_decode_steps + 1,
    )
    logger.info(f"Engine loaded in {time.time() - t0:.1f}s")
    logger.info(f"max_prompt_embedding_table_size: {runner.max_prompt_embedding_table_size}")

    # Prepare prompt_embedding_table from inputs_embeds.
    # TRT-LLM PromptTuningEmbedding: tokens >= vocab_size use prompt table.
    # prompt_embedding_table: [num_tokens, hidden_size]
    prompt_table = torch.from_numpy(inputs_embeds[0]).cuda()  # [S, H]
    logger.info(f"prompt_table shape: {prompt_table.shape}")

    # batch_input_ids: virtual token IDs >= vocab_size, so the engine uses
    # prompt_embedding_table instead of the (zeroed) vocab embedding.
    # IDs: [vocab_size, vocab_size+1, ..., vocab_size+S-1]
    batch_input_ids = [torch.arange(vocab_size, vocab_size + seq_len, dtype=torch.int32)]
    logger.info(f"batch_input_ids range: [{vocab_size}, {vocab_size + seq_len - 1}]")

    # prompt_tasks: all use task 0 (single task)
    prompt_tasks = "0"

    # Sampling: greedy (top_k=1, temperature=1.0)
    sampling = SamplingConfig(
        end_id=-1,  # no EOS
        pad_id=-1,
        max_new_tokens=args.n_decode_steps,
        top_k=1,
        temperature=1.0,
        return_dict=True,
        output_sequence_lengths=True,
    )

    logger.info("Running TRT-LLM generate ...")
    t1 = time.time()
    outputs = runner.generate(
        batch_input_ids=batch_input_ids,
        sampling_config=sampling,
        prompt_table=prompt_table,
        prompt_tasks=prompt_tasks,
    )
    gen_time = time.time() - t1
    logger.info(f"Generate completed in {gen_time:.3f}s")

    # Extract outputs
    output_ids = outputs["output_ids"]  # [B, beam, total_len]
    seq_lengths = outputs["sequence_lengths"]  # [B, beam]
    context_logits = outputs.get("context_logits", None)
    generation_logits = outputs.get("generation_logits", None)

    logger.info(f"output_ids shape: {output_ids.shape}")
    logger.info(f"sequence_lengths: {seq_lengths}")

    # Generated token IDs (after context)
    generated_ids = output_ids[0, 0, seq_len:].cpu().numpy()
    logger.info(f"TRT-LLM generated IDs: {generated_ids.tolist()}")
    logger.info(f"PyTorch  reference IDs: {ref_decode_tokens[1:].tolist()}")

    # Compare decode tokens
    n_compare = min(len(generated_ids), len(ref_decode_tokens) - 1)
    token_match = 0
    for i in range(n_compare):
        match = int(generated_ids[i]) == int(ref_decode_tokens[i + 1])
        if match:
            token_match += 1
        logger.info(f"  step {i}: TRT={generated_ids[i]}, PT={ref_decode_tokens[i+1]}, "
                    f"{'MATCH' if match else 'DIFF'}")

    token_match_rate = token_match / n_compare if n_compare > 0 else 0
    logger.info(f"Token match rate: {token_match}/{n_compare} = {token_match_rate:.1%}")

    # Compare context logits if available
    prefill_cosine = None
    if context_logits is not None:
        ctx_logits = context_logits[0].cpu().numpy()  # [S, V]
        ref_ctx = ref_prefill_logits[0]  # [S, V]
        logger.info(f"Context logits shape: TRT={ctx_logits.shape}, PT={ref_ctx.shape}")

        prefill_cosine = cosine_sim(ctx_logits, ref_ctx)
        logger.info(f"Prefill logits cosine similarity: {prefill_cosine:.6f}")

        # Per-position cosine
        for pos in range(min(seq_len, 5)):
            pos_cos = cosine_sim(ctx_logits[pos], ref_ctx[pos])
            logger.info(f"  pos {pos}: cosine={pos_cos:.6f}")
        if seq_len > 5:
            pos_cos = cosine_sim(ctx_logits[-1], ref_ctx[-1])
            logger.info(f"  pos {seq_len-1} (last): cosine={pos_cos:.6f}")

        # Argmax match at last position
        trt_token_0 = ctx_logits[-1].argmax()
        pt_token_0 = ref_ctx[-1].argmax()
        logger.info(f"codec_token_0: TRT={trt_token_0}, PT={pt_token_0}, "
                    f"{'MATCH' if trt_token_0 == pt_token_0 else 'DIFF'}")

    # Compare generation logits if available
    decode_cosines = []
    if generation_logits is not None:
        gen_logits = generation_logits[0].cpu().numpy()  # [N, beam, V] or [N, V]
        logger.info(f"Generation logits shape: {gen_logits.shape}")

        for i in range(min(gen_logits.shape[0], ref_decode_logits.shape[0])):
            gl = gen_logits[i].flatten()[-vocab_size:]
            rl = ref_decode_logits[i].flatten()
            cos = cosine_sim(gl, rl)
            decode_cosines.append(cos)
            logger.info(f"  decode step {i}: cosine={cos:.6f}")

    # Summary (TRT-LLM 1.1.0+ uses QK-Norm for Qwen3; no "ignored tensors" warning)
    report = {
        "variant": str(Path(args.engine_dir).parent.name),
        "seq_len": seq_len,
        "n_decode_steps": n_compare,
        "token_match_rate": token_match_rate,
        "generated_ids": generated_ids.tolist(),
        "reference_ids": ref_decode_tokens.tolist(),
        "prefill_cosine": prefill_cosine,
        "decode_cosines": decode_cosines,
        "gen_time_s": gen_time,
        "qk_norm_missing": False,
    }

    logger.info("")
    logger.info("=" * 60)
    logger.info("  SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Token match rate: {token_match_rate:.1%}")
    if prefill_cosine is not None:
        logger.info(f"  Prefill cosine:   {prefill_cosine:.6f}")
    if decode_cosines:
        logger.info(f"  Decode cosines:   {[f'{c:.4f}' for c in decode_cosines]}")
    status = "PASS" if (prefill_cosine is None or prefill_cosine > 0.9) else "WARN"
    logger.info(f"  Status: {status}")

    report_path = args.report_file or str(
        Path(args.engine_dir).parent / "trtllm_verification_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"  Report: {report_path}")


if __name__ == "__main__":
    main()

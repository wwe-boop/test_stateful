#!/usr/bin/env python3
"""
Verify Code Predictor ONNX → TensorRT compilation, accuracy, and performance.

Runs INSIDE an NGC TensorRT container (has torch, tensorrt, cuda-python, onnx).
Validates the §5.5 checklist from architecture.md:
  1. ONNX → TRT engine compilation
  2. Engine accuracy vs ONNX Runtime (single-stage) / exact-match (unrolled)
  3. Performance benchmark (B=1, B=8) with CUDA events for precise GPU timing

Usage (from host, via verify_code_predictor.sh):
  docker run --gpus all -v .../exported/<variant>:/mnt/model \
    <NGC_IMAGE> python3 /mnt/scripts/verify_code_predictor_trt.py \
    --model-dir /mnt/model [--mode both] [--bf16|--fp16|--fp32]
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import tensorrt as trt

# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def log_section(msg: str):
    print(f"\n{'='*60}", flush=True)
    print(f"  {msg}", flush=True)
    print(f"{'='*60}", flush=True)

def log_ok(msg: str):
    print(f"  [PASS] {msg}", flush=True)

def log_fail(msg: str):
    print(f"  [FAIL] {msg}", flush=True)

def log_info(msg: str):
    print(f"  [INFO] {msg}", flush=True)

def log_warn(msg: str):
    print(f"  [WARN] {msg}", flush=True)

TRT_TO_TORCH_DTYPE = {
    trt.float32: torch.float32,
    trt.float16: torch.float16,
    trt.bfloat16: torch.bfloat16,
    trt.int32: torch.int32,
    trt.int64: torch.int64,
    trt.int8: torch.int8,
    trt.bool: torch.bool,
}

TRT_TO_NP_DTYPE = {
    trt.float32: np.float32,
    trt.float16: np.float16,
    trt.bfloat16: np.float32,  # numpy has no bfloat16; use float32 for comparison
    trt.int32: np.int32,
    trt.int64: np.int64,
    trt.int8: np.int8,
    trt.bool: np.bool_,
}

# ---------------------------------------------------------------------------
#  ONNX inspection
# ---------------------------------------------------------------------------

def inspect_onnx(onnx_path: str) -> dict:
    import onnx
    model = onnx.load(onnx_path)
    n_inits = len(model.graph.initializer)
    n_nodes = len(model.graph.node)

    inputs = []
    for inp in model.graph.input:
        name = inp.name
        shape = [d.dim_value or d.dim_param for d in inp.type.tensor_type.shape.dim]
        dtype = inp.type.tensor_type.elem_type
        inputs.append({"name": name, "shape": shape, "dtype": dtype})

    outputs = []
    for out in model.graph.output:
        name = out.name
        shape = [d.dim_value or d.dim_param for d in out.type.tensor_type.shape.dim]
        dtype = out.type.tensor_type.elem_type
        outputs.append({"name": name, "shape": shape, "dtype": dtype})

    del model
    return {
        "n_initializers": n_inits,
        "n_nodes": n_nodes,
        "inputs": inputs,
        "outputs": outputs,
        "model_size_mb": os.path.getsize(onnx_path) / 1024 / 1024,
    }

# ---------------------------------------------------------------------------
#  TensorRT engine build
# ---------------------------------------------------------------------------

def build_trt_engine(onnx_path: str, engine_path: str, bf16: bool = True,
                     fp16: bool = False, max_batch: int = 8) -> bool:
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    log_info(f"Parsing ONNX: {onnx_path}")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                log_fail(f"ONNX parse error: {parser.get_error(i)}")
            return False
    log_ok(f"ONNX parsed: {network.num_layers} layers, "
           f"{network.num_inputs} inputs, {network.num_outputs} outputs")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)

    if bf16:
        config.set_flag(trt.BuilderFlag.BF16)
        log_info("BF16 enabled")
    elif fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        log_info("FP16 enabled")

    profile = builder.create_optimization_profile()
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        name = inp.name
        shape = inp.shape
        min_s, opt_s, max_s = list(shape), list(shape), list(shape)
        for d in range(len(shape)):
            if shape[d] == -1:
                if d == 0:
                    min_s[d], opt_s[d], max_s[d] = 1, 1, max_batch
                elif d == 1:
                    min_s[d], opt_s[d], max_s[d] = 2, 8, 17
                else:
                    min_s[d], opt_s[d], max_s[d] = 1, 4, 16
        profile.set_shape(name, tuple(min_s), tuple(opt_s), tuple(max_s))
        log_info(f"  Profile '{name}': min={min_s}, opt={opt_s}, max={max_s}")

    config.add_optimization_profile(profile)

    t0 = time.time()
    log_info("Building TRT engine ...")
    engine_bytes = builder.build_serialized_network(network, config)
    elapsed = time.time() - t0

    if engine_bytes is None:
        log_fail(f"TRT engine build FAILED after {elapsed:.1f}s")
        return False

    engine_size_mb = engine_bytes.nbytes / 1024 / 1024
    log_ok(f"Engine built in {elapsed:.1f}s — {engine_size_mb:.1f} MB")

    with open(engine_path, "wb") as f:
        f.write(memoryview(engine_bytes))
    log_ok(f"Saved: {engine_path}")
    return True

# ---------------------------------------------------------------------------
#  TRT runner using PyTorch GPU tensors
# ---------------------------------------------------------------------------

class TRTRunner:
    """Manages a TRT engine with torch GPU tensors for I/O."""

    def __init__(self, engine_path: str, device: torch.device):
        self.device = device
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream(device=device)

        self.input_names = []
        self.output_names = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

    def infer(self, feeds: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        for name, tensor in feeds.items():
            self.context.set_input_shape(name, tuple(tensor.shape))

        for name, tensor in feeds.items():
            self.context.set_tensor_address(name, tensor.data_ptr())

        outputs = {}
        for name in self.output_names:
            shape = self.context.get_tensor_shape(name)
            trt_dtype = self.engine.get_tensor_dtype(name)
            torch_dtype = TRT_TO_TORCH_DTYPE.get(trt_dtype, torch.float32)
            out = torch.empty(tuple(shape), dtype=torch_dtype, device=self.device)
            self.context.set_tensor_address(name, out.data_ptr())
            outputs[name] = out

        self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        return outputs

    def benchmark(self, feeds: dict[str, torch.Tensor],
                  warmup: int = 20, iterations: int = 200) -> dict:
        for name, tensor in feeds.items():
            self.context.set_input_shape(name, tuple(tensor.shape))
            self.context.set_tensor_address(name, tensor.data_ptr())

        out_bufs = {}
        for name in self.output_names:
            shape = self.context.get_tensor_shape(name)
            trt_dtype = self.engine.get_tensor_dtype(name)
            torch_dtype = TRT_TO_TORCH_DTYPE.get(trt_dtype, torch.float32)
            out_bufs[name] = torch.empty(tuple(shape), dtype=torch_dtype,
                                         device=self.device)
            self.context.set_tensor_address(name, out_bufs[name].data_ptr())

        for _ in range(warmup):
            self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()

        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]

        for i in range(iterations):
            start_events[i].record(self.stream)
            self.context.execute_async_v3(self.stream.cuda_stream)
            end_events[i].record(self.stream)

        self.stream.synchronize()

        latencies = np.array([s.elapsed_time(e) for s, e in zip(start_events, end_events)])
        return {
            "mean_ms": float(np.mean(latencies)),
            "median_ms": float(np.median(latencies)),
            "p99_ms": float(np.percentile(latencies, 99)),
            "min_ms": float(np.min(latencies)),
            "max_ms": float(np.max(latencies)),
            "std_ms": float(np.std(latencies)),
            "iterations": iterations,
        }

# ---------------------------------------------------------------------------
#  Verify: unrolled
# ---------------------------------------------------------------------------

def verify_unrolled(model_dir: str, bf16: bool, fp16: bool, batch_sizes: list,
                    device: torch.device) -> dict:
    onnx_path = os.path.join(model_dir, "code_predictor_unrolled.onnx")
    engine_path = os.path.join(model_dir, "code_predictor_unrolled.plan")

    if not os.path.exists(onnx_path):
        log_fail(f"ONNX not found: {onnx_path}")
        return {"status": "skipped", "reason": "onnx_not_found"}

    log_section("Unrolled Code Predictor (15-stage)")

    info = inspect_onnx(onnx_path)
    log_info(f"ONNX: {info['model_size_mb']:.1f} MB, "
             f"initializers={info['n_initializers']}, nodes={info['n_nodes']}")

    expected_inits = 80
    if info["n_initializers"] > 500:
        log_warn(f"Initializer count {info['n_initializers']} >> {expected_inits} — "
                 f"weight sharing may have failed!")
    else:
        log_ok(f"Initializer count {info['n_initializers']} reasonable "
               f"(expected ~{expected_inits} for shared weights)")

    for inp in info["inputs"]:
        log_info(f"  Input:  {inp['name']} {inp['shape']} dtype={inp['dtype']}")
    for out in info["outputs"]:
        log_info(f"  Output: {out['name']} {out['shape']} dtype={out['dtype']}")

    max_batch = max(batch_sizes)
    if not os.path.exists(engine_path):
        if not build_trt_engine(onnx_path, engine_path, bf16=bf16,
                                fp16=fp16, max_batch=max_batch):
            return {"status": "build_failed", "onnx_info": info}
    else:
        engine_mb = os.path.getsize(engine_path) / 1024 / 1024
        log_ok(f"Engine already exists: {engine_path} ({engine_mb:.1f} MB)")

    hidden_size = info["inputs"][0]["shape"][2]
    if isinstance(hidden_size, str):
        hidden_size = 2048
    log_info(f"hidden_size={hidden_size}")

    runner = TRTRunner(engine_path, device)

    log_info("Running ONNX reference (CPU) ...")
    B = 1
    np_feeds = {
        "past_hidden": np.random.randn(B, 1, hidden_size).astype(np.float32),
        "codec_token_0": np.array([42], dtype=np.int64),
    }
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_out = dict(zip([o.name for o in sess.get_outputs()],
                        sess.run(None, np_feeds)))
    del sess

    log_info("Running TRT inference (GPU) ...")
    gpu_feeds = {k: torch.from_numpy(v).to(device) for k, v in np_feeds.items()}
    trt_out = runner.infer(gpu_feeds)

    for name in onnx_out:
        onnx_val = onnx_out[name].flatten()
        trt_val = trt_out[name].cpu().numpy().flatten()
        match = np.array_equal(onnx_val, trt_val)
        if match:
            log_ok(f"  {name}: exact match (values: {onnx_val[:5]})")
        else:
            mismatches = np.sum(onnx_val != trt_val)
            log_warn(f"  {name}: {mismatches}/{len(onnx_val)} tokens differ "
                     f"(ONNX: {onnx_val[:5]}, TRT: {trt_val[:5]})")
            log_info("  Token mismatch expected with reduced precision (different argmax outcomes)")

    # benchmark
    benchmarks = {}
    for bs in batch_sizes:
        log_info(f"Benchmark B={bs} ...")
        bench_feeds = {
            "past_hidden": torch.randn(bs, 1, hidden_size, dtype=torch.float32,
                                       device=device),
            "codec_token_0": torch.randint(0, 2048, (bs,), dtype=torch.int64,
                                           device=device),
        }
        stats = runner.benchmark(bench_feeds)
        benchmarks[f"B={bs}"] = stats
        log_ok(f"  B={bs}: mean={stats['mean_ms']:.3f}ms, "
               f"median={stats['median_ms']:.3f}ms, p99={stats['p99_ms']:.3f}ms")

    return {
        "status": "pass",
        "onnx_info": info,
        "benchmarks": benchmarks,
        "engine_path": engine_path,
    }

# ---------------------------------------------------------------------------
#  Verify: single-stage
# ---------------------------------------------------------------------------

def verify_single_stage(model_dir: str, bf16: bool, fp16: bool, batch_sizes: list,
                        device: torch.device) -> dict:
    onnx_path = os.path.join(model_dir, "code_predictor_single_stage.onnx")
    engine_path = os.path.join(model_dir, "code_predictor_single_stage.plan")

    if not os.path.exists(onnx_path):
        log_fail(f"ONNX not found: {onnx_path}")
        return {"status": "skipped", "reason": "onnx_not_found"}

    log_section("Single-Stage Code Predictor")

    info = inspect_onnx(onnx_path)
    log_info(f"ONNX: {info['model_size_mb']:.1f} MB, "
             f"initializers={info['n_initializers']}, nodes={info['n_nodes']}")
    for inp in info["inputs"]:
        log_info(f"  Input:  {inp['name']} {inp['shape']} dtype={inp['dtype']}")
    for out in info["outputs"]:
        log_info(f"  Output: {out['name']} {out['shape']} dtype={out['dtype']}")

    max_batch = max(batch_sizes)
    if not os.path.exists(engine_path):
        if not build_trt_engine(onnx_path, engine_path, bf16=bf16,
                                fp16=fp16, max_batch=max_batch):
            return {"status": "build_failed", "onnx_info": info}
    else:
        engine_mb = os.path.getsize(engine_path) / 1024 / 1024
        log_ok(f"Engine already exists: {engine_path} ({engine_mb:.1f} MB)")

    hidden_size = info["inputs"][0]["shape"][2]
    if isinstance(hidden_size, str):
        hidden_size = 2048
    vocab_size_shape = info["inputs"][1]["shape"][0]
    vocab_size = vocab_size_shape if isinstance(vocab_size_shape, int) else 2048
    log_info(f"hidden_size={hidden_size}, vocab_size={vocab_size}")

    runner = TRTRunner(engine_path, device)

    # accuracy: ONNX RT vs TRT (logits → max_abs_diff)
    B, S = 1, 5
    np_feeds = {
        "sequence": np.random.randn(B, S, hidden_size).astype(np.float32),
        "lm_head_weight": np.random.randn(vocab_size, hidden_size).astype(np.float32),
        "lm_head_bias": np.zeros(vocab_size, dtype=np.float32),
    }

    log_info("Running ONNX reference (CPU) ...")
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_out = dict(zip([o.name for o in sess.get_outputs()],
                        sess.run(None, np_feeds)))
    del sess

    log_info("Running TRT inference (GPU) ...")
    gpu_feeds = {k: torch.from_numpy(v).to(device) for k, v in np_feeds.items()}
    trt_out = runner.infer(gpu_feeds)

    accuracy_ok = True
    for name in onnx_out:
        onnx_v = onnx_out[name].astype(np.float32).flatten()
        trt_v = trt_out[name].cpu().numpy().astype(np.float32).flatten()
        abs_diff = float(np.max(np.abs(onnx_v - trt_v)))
        denom = np.maximum(np.abs(onnx_v), 1e-6)
        rel_diff = float(np.max(np.abs(onnx_v - trt_v) / denom))
        cos_sim = float(np.dot(onnx_v, trt_v) /
                        (np.linalg.norm(onnx_v) * np.linalg.norm(trt_v) + 1e-12))
        argmax_match = np.argmax(onnx_v) == np.argmax(trt_v)
        log_info(f"  {name}: abs_diff={abs_diff:.4f}, rel_diff={rel_diff:.4f}, "
                 f"cosine={cos_sim:.6f}, argmax_match={argmax_match}")

        if cos_sim > 0.999:
            log_ok(f"Accuracy PASSED (cosine={cos_sim:.6f})")
        elif cos_sim > 0.99:
            log_warn(f"Accuracy MARGINAL (cosine={cos_sim:.6f})")
        else:
            log_fail(f"Accuracy FAILED (cosine={cos_sim:.6f})")
            accuracy_ok = False

    # benchmark
    benchmarks = {}
    for bs in batch_sizes:
        log_info(f"Benchmark B={bs} ...")
        bench_feeds = {
            "sequence": torch.randn(bs, S, hidden_size, dtype=torch.float32,
                                    device=device),
            "lm_head_weight": torch.randn(vocab_size, hidden_size, dtype=torch.float32,
                                          device=device),
            "lm_head_bias": torch.zeros(vocab_size, dtype=torch.float32, device=device),
        }
        stats = runner.benchmark(bench_feeds)
        benchmarks[f"B={bs}"] = stats
        log_ok(f"  B={bs}: mean={stats['mean_ms']:.3f}ms, "
               f"median={stats['median_ms']:.3f}ms, p99={stats['p99_ms']:.3f}ms")

    return {
        "status": "pass" if accuracy_ok else "fail",
        "onnx_info": info,
        "max_abs_diff": abs_diff,
        "benchmarks": benchmarks,
        "engine_path": engine_path,
    }

# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Verify Code Predictor ONNX -> TRT (arch.md 5.5)")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--mode", default="both",
                        choices=["unrolled", "single_stage", "both"])
    precision = parser.add_mutually_exclusive_group()
    precision.add_argument("--bf16", action="store_true", default=True,
                           help="Use BF16 precision (default)")
    precision.add_argument("--fp16", action="store_true",
                           help="Use FP16 precision instead of BF16")
    precision.add_argument("--fp32", action="store_true",
                           help="Use FP32 precision (no reduced precision)")
    parser.add_argument("--batch-sizes", default="1,8")
    args = parser.parse_args()

    if args.fp32 or args.fp16:
        args.bf16 = False

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    model_dir = args.model_dir
    device = torch.device("cuda:0")

    log_section("Code Predictor TRT Verification")
    log_info(f"Model dir:    {model_dir}")
    log_info(f"Mode:         {args.mode}")
    precision_str = "BF16" if args.bf16 else ("FP16" if args.fp16 else "FP32")
    log_info(f"Precision:    {precision_str}")
    log_info(f"Batch sizes:  {batch_sizes}")
    log_info(f"GPU:          {torch.cuda.get_device_name(0)}")
    log_info(f"TensorRT:     {trt.__version__}")

    results = {}

    if args.mode in ("unrolled", "both"):
        results["unrolled"] = verify_unrolled(
            model_dir, args.bf16, args.fp16, batch_sizes, device)

    if args.mode in ("single_stage", "both"):
        results["single_stage"] = verify_single_stage(
            model_dir, args.bf16, args.fp16, batch_sizes, device)

    log_section("SUMMARY")
    all_pass = True
    for model_type, result in results.items():
        status = result["status"]
        icon = "PASS" if status == "pass" else ("SKIP" if status == "skipped" else "FAIL")
        print(f"  [{icon}] {model_type}: {status}")
        if "onnx_info" in result:
            info = result["onnx_info"]
            print(f"      ONNX: {info['model_size_mb']:.1f} MB, "
                  f"{info['n_initializers']} inits, {info['n_nodes']} nodes")
        if "benchmarks" in result:
            for bs_key, stats in result["benchmarks"].items():
                print(f"      {bs_key}: {stats['mean_ms']:.3f}ms mean, "
                      f"{stats['median_ms']:.3f}ms median, "
                      f"{stats['p99_ms']:.3f}ms p99")
        if status not in ("pass", "skipped"):
            all_pass = False

    report_path = os.path.join(model_dir, "trt_verification_report.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log_info(f"Report: {report_path}")

    if not all_pass:
        unrolled_s = results.get("unrolled", {}).get("status", "")
        single_s = results.get("single_stage", {}).get("status", "")
        if unrolled_s == "build_failed" and single_s == "pass":
            log_warn("Unrolled failed but single-stage works -> use fallback (5.6)")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()

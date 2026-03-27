# TensorRT Myelin 最小复现（dim_count=4 vs stride_order=3）

这个复现包用于复现以下错误（TRT 10.15.1）：

`MyelinCheckException: tensor.cpp:852: CHECK_EQ(dim_count(), stride_order().size()) failed. LHS: 4 RHS: 3`

并同时给出一个只改一个条件的通过对照组。

## 环境

- GPU: NVIDIA（已验证 RTX 5090）
- Docker image: `nvcr.io/nvidia/tritonserver:26.02-py3`
- TensorRT: `10.15.1`（镜像内 `/usr/src/tensorrt/bin/trtexec`）
- Python env: `conda env qwen3-tts`（用于导出 ONNX）

## 1. 生成复现 ONNX

```bash
python scripts/python/repro_myelin_stride_mismatch/build_repro_onnx.py \
  --out-root /tmp/myelin_repro_case
```

输出两份 ONNX（都裁剪为仅 `wav` 输出）：

- fail: `/tmp/myelin_repro_case/dynamic_state/tokenizer/code2wav_decoder_wav_only.onnx`
- pass: `/tmp/myelin_repro_case/static_state_8/tokenizer/code2wav_decoder_wav_only.onnx`

## 2. 运行 fail + pass

```bash
bash scripts/python/repro_myelin_stride_mismatch/run_repro.sh both /tmp/myelin_repro_case
```

也可单独跑：

```bash
bash scripts/python/repro_myelin_stride_mismatch/run_repro.sh fail /tmp/myelin_repro_case
bash scripts/python/repro_myelin_stride_mismatch/run_repro.sh pass /tmp/myelin_repro_case
```

日志位置：

- fail 日志: `/tmp/myelin_repro_case/logs/fail.log`
- pass 日志: `/tmp/myelin_repro_case/logs/pass.log`

## 3. 预期现象

### Fail case（dynamic state batch）

应出现关键行：

- `ForeignNode[/Slice_1.../Clip]`
- `MyelinCheckException: tensor.cpp:852`
- `LHS: 4`
- `RHS: 3`

并最终构建失败（`Could not find any implementation for node {ForeignNode[/Slice_1.../Clip]}`）。

### Pass case（static state batch=8）

应出现：

- `&&&& PASSED TensorRT.trtexec`

即仅把 `conv_state_* / transconv_overlap_*` 改为固定 batch=8 后可构建通过。

## 4. 变量对照（关键）

- 失败组：`conv_state_* / transconv_overlap_*` batch 为动态（min=1, max=8）。
- 通过组：上述 state batch 固定为 `8`。
- 其他输入 profile 保持一致（`codes/cache_position/c2w_attention_bias/past_kv_*`）。

## 5. 手工 PyTorch 小图搜索（<=50 节点）

如果你要给 NVIDIA 提交“手工搭建的小图”而不是业务图裁剪版，可用下面脚本自动搜索：

```bash
python scripts/python/repro_myelin_stride_mismatch/search_pytorch_repro.py \
  --out-root /tmp/myelin_pytorch_search \
  --max-nodes 50 \
  --max-trials 64
```

输出：

- `summary.json`: `/tmp/myelin_pytorch_search/summary.json`
- 每个 trial 的 ONNX: `/tmp/myelin_pytorch_search/onnx/`
- 每个 trial 的 TRT 日志: `/tmp/myelin_pytorch_search/logs/`

说明：

- 该脚本会批量构造手工 PyTorch module，导出 ONNX 后 `onnxsim`，仅对 `<=50` 节点候选跑 `trtexec`。
- 一旦命中 `MyelinCheckException: tensor.cpp:852` 会提前停止，并在终端打印命中的 trial 与路径。

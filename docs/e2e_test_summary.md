# 端到端测试数据汇总

## 1. 测试范围

E2E 测试位于 `tests/e2e/test_e2e.py`，通过 Triton gRPC 调用 `tts_orchestrator`，覆盖场景与错误处理（对应架构 T3.1、T3.3、T3.4）。

| 用例 ID | 测试项 | 说明 |
|--------|--------|------|
| **T3.1a** | voice_design | 纯文本 + task_type=voice_design，校验流式音频与首包/总耗时 |
| **T3.1b** | custom_voice | speaker + instruct，校验多 chunk 输出 |
| **T3.1c** | voice_clone_xvec | 需 ref_audio + x_vector_only（当前默认 skip，缺 speaker_encoder） |
| **T3.1d** | voice_clone_icl | 需 ref_audio + ref_text（当前默认 skip） |
| **T3.3a** | error_empty_text | 空 text → 服务返回错误 |
| **T3.3b** | error_invalid_task_type | 非法 task_type → 错误 |
| **T3.3c** | error_voice_clone_no_ref | voice_clone 无 ref_audio → 错误 |
| **T3.3d** | error_voice_clone_bad_base64 | ref_audio 非法 base64 → 错误 |
| **T3.4** | first_chunk_latency | 首包延迟与总耗时（目标见下） |

## 2. 指标定义

| 指标 | 含义 | 目标（架构/生产） |
|------|------|-------------------|
| **first_chunk_latency** | 请求发出到收到第一个 audio_chunk 的时间 | 生产 &lt; 200ms；架构参考 ~76ms（prefill+10 步 decode+code2wav） |
| **total_sec** | 单次请求从发起到收到 is_final=True 的总时间 | 与文本长度和 decode 步数相关 |
| **chunks** | 收到的音频 chunk 数量 | ≥ 1 |
| **samples** | 总采样点数（float32, 24kHz） | 与生成时长一致 |

## 3. 最近一次完整运行结果（参考）

- **环境**: Triton 容器 `qwen3-tts-triton:latest`，model_repository 为 ONNX 模式、design-1.7b。
- **结果**: **7 passed, 2 skipped**（voice_clone 两条因无 speaker_encoder 跳过）。
- **总耗时**: ~12 分钟（含 3 条实际 TTS 推理，单条数十秒量级）。
- **断言**: 首包延迟 &lt; 15s（CI 放宽）；生产目标仍为 &lt; 200ms。

## 4. 如何获取具体数值

运行测试并打开 `-s` 查看 print 输出，即可看到各次推理的首包/总耗时与采样数：

```bash
# 确保 Triton 已起：bash scripts/bash/build_triton.sh run
cd /path/to/Qwen3-TTS-Triton
python -m pytest tests/e2e/test_e2e.py -v -s
```

输出中会出现类似：

- `[E2E T3.1a] voice_design first_chunk_s=... total_s=... samples=...`
- `[E2E T3.4] first_chunk_latency_ms=... total_ms=... chunks=... samples=...`

将 `first_chunk_s` × 1000 或 `first_chunk_latency_ms` 与 200ms 对比即可评估首响是否达标。

## 5. 与架构目标对照

| 项目 | 架构目标 | 说明 |
|------|----------|------|
| 首包延迟（TTS 部分） | ~76ms（prefill ~20ms + 10 步 decode ~41ms + code2wav ~15ms） | 当前为 ONNX 后端；TRT 可进一步逼近 |
| 单步 decode | ~2.6ms (B=1) / ~2.8ms (B=8) | 需单独 benchmark talker_unified |
| 首包 chunk | 10 帧 → ~19200 samples @ 24kHz | 由 `first_chunk_frames` 控制 |

## 6. 已知限制

- **voice_clone**：需部署 `speaker_encoder` 且提供有效 ref_audio，当前 CI 不跑，可本地用真实音频手动测。
- **并发/压测**：当前用例为单请求顺序执行，不测多路并发与 GPU 利用率；高利用率需多路并发或单独压测脚本。

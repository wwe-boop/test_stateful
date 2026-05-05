# Qwen3-TTS Triton

*让我们像播放音频一样播放文本！*

## 引言

Qwen3-TTS Triton 是一个 **工程预览版** 项目：把官方 Qwen3-TTS PyTorch 权重导出为 ONNX/TensorRT 运行时，并围绕 Triton/standalone engine 做 token 级流式 TTS、模型 fuse、前端分词、prefix cache、连续批处理和 WebUI 性能展示。

这个项目的目标不是把所有 Qwen3-TTS 官方路径一次性包装成生产服务，而是开放一条已经高度优化、可复现、可继续验证的工程链路，让社区能一起把它打磨成可靠的开源推理系统。

## 当前定位

**请先读这一段。**

- 当前建议的 v0.1 稳定范围是 `custom-1.7b` / `custom_voice` 路径。
- `design-1.7b` / `voice_design` 处于实验状态：代码里已有部分路径，但不应当对外承诺完全测通。
- `base` 语音克隆和 ICL 语音克隆仍是计划项：prefill/任务类型里能看到分支，但 standalone 引擎还没有完整打通 ref audio preprocessing、spk embedding/ref codes 注入和端到端验证。
- 流式模式仍可能出现幻觉、重复、漏读、插入未提供内容、长文本不稳定等问题。请把当前版本当作工程预览或研究预览，不要直接用于生产内容生成。
- README、WebUI 和 benchmark 会优先使用中文说明。中文口径稳定后再整理英文版。

## 性能声明

项目里提到的低延迟数字是有条件结果，不是通用承诺：

- `13ms TTFT`：最低观测值，依赖指定硬件、warm engine、prefix/cache 命中、单路请求、特定 engine profile 和本地链路。
- `180ms 128-stream avg TTFT`：并发压测口径，需要明确硬件、cache、输入文本、profile、采样参数和客户端测量方式。
- WebUI 默认会在 live Triton/engine 不可用时展示 fixture trace/metrics，但不会补 synthetic beep 音频。只有结果 source 标记为 `live_triton`、`live_engine_websocket` 或 `live_official_pytorch` 且带 `audio` 字段时，才代表可回放的实时合成音频。

详细 benchmark 口径见 [docs/zh/benchmark_methodology.md](docs/zh/benchmark_methodology.md)。

## 能力状态

| 路径 | 当前状态 | 开源口径 |
| --- | --- | --- |
| `custom-1.7b` / `custom_voice` | 优先稳定 | v0.1 推荐路径，WebUI 和 demo 默认围绕它展示 |
| `design-1.7b` / `voice_design` | 实验 | 可保留代码和导出入口，但需要标注未充分测通 |
| `base-1.7b` / x-vector voice clone | 计划中 | 当前不应标为可用；需要 ref audio 预处理和端到端测试 |
| `icl` voice clone | 计划中 | 当前不应标为可用；需要 ref audio/ref text/code 注入链路 |
| `0.6b` variants | 未作为 v0.1 主线 | 可保留导出/下载入口，发布前需要单独验证 |

## 快速开始

```bash
git clone --recursive https://github.com/user/Qwen3-TTS-Triton.git
cd Qwen3-TTS-Triton

# 推荐先跑 custom-1.7b
bash scripts/bash/autorun.sh all -m custom-1.7b
```

三阶段流程：

```text
Phase A: setup_env.sh
  下载模型、安装环境、导出 ONNX/weights/manifest

Phase B: build_engines.sh
  在 NGC 容器里用 trtexec 编译 TensorRT engine，并把 engine profile 写回 triton_manifest.json

Phase C: deploy.sh / compose.sh
  以 standalone engine、engine Docker 或 Triton gateway 启动服务
```

## 构建参数

`autorun.sh` 现在会透传关键 engine profile 参数：

```bash
bash scripts/bash/autorun.sh build -m custom-1.7b \
  --max-batch-size 64 \
  --max-input-len 128 \
  --max-seq-len 512 \
  --dtype bf16
```

这些值会写入 `workspace/exported/<variant>/triton_manifest.json` 的 `engine_profile` 字段。runtime 启动时如果请求的 batch/seq 超过 profile，会直接报错，避免 silent clamp 或运行时才暴露 TensorRT shape 问题。

常用参数：

```text
Phase B:
  --max-batch-size <N>          TensorRT profile 最大 batch
  --max-input-len <N>           prefill/input 最大 token 长度
  --max-seq-len <N>             KV cache 最大 sequence 长度
  --dtype bf16|fp16|fp32|fp8    TensorRT build precision
  --triton-io-float-dtype <T>   TensorRT/Triton float I/O dtype，默认等于 --dtype
  --target-driver <ver>         按部署机 NVIDIA driver 选择 NGC 镜像

Phase C:
  --gateway standalone|triton|engine-docker
  --engine-mode trt|onnx
  --runtime-max-batch-size <N>  runtime scheduler batch 上限
  --runtime-max-seq-len <N>     runtime scheduler seq 上限
```

示例：

```bash
# 构建较小 profile，便于低显存机器验证
bash scripts/bash/autorun.sh build -m custom-1.7b \
  --max-batch-size 16 --max-input-len 96 --max-seq-len 384

# runtime 使用不得超过 manifest 里记录的 profile
bash scripts/bash/autorun.sh deploy -m custom-1.7b \
  --gateway standalone \
  --runtime-max-batch-size 16 \
  --runtime-max-seq-len 384
```

## 部署方式

### Standalone

本机 Python 运行 `engine.server`，适合调试 engine、协议和 WebSocket/gRPC：

```bash
bash scripts/bash/deploy.sh run --gateway standalone --variant custom-1.7b
```

默认端口：

- gRPC: `localhost:50051`
- WebSocket: `ws://localhost:50052/v1/ws`
- HTTP capabilities: `http://localhost:50052/v1/capabilities`
- HTTP health: `http://localhost:8080/health`

### Engine Docker

独立 engine 容器挂载 `workspace/`，不把模型和 engine 烘进镜像：

```bash
bash scripts/bash/deploy.sh run --gateway engine-docker --variant custom-1.7b
```

如果你正在频繁改 engine 代码，不建议反复重建镜像。使用 compose 的开发覆盖层或 watch：

```bash
# 代码以 bind mount 方式进入容器，适合开发
bash scripts/bash/compose.sh up --gateway engine --variant custom-1.7b --dev

# 或使用 Docker Compose watch，同步代码并重启服务
bash scripts/bash/compose.sh watch --gateway engine --variant custom-1.7b
```

长期部署时可以使用普通镜像；开发期用 `--dev`/`watch`，把“环境层”和“代码层”分开。

### Triton

组装 `workspace/model_repository` 并启动 Triton：

```bash
bash scripts/bash/compose.sh prepare --gateway triton --variant custom-1.7b --engine-mode trt
bash scripts/bash/compose.sh up --gateway triton --variant custom-1.7b
```

也可以通过高层入口：

```bash
bash scripts/bash/deploy.sh run --gateway triton --variant custom-1.7b --engine-mode trt
```

## WebUI Demo

WebUI 分为三个工程展示板块：

- `Text Player`：把文本按 engine decode step 播放出来。一个音频 chunk 对应一个 step；前半段是 text token，进入 flush 后会直接显示 `PAD` step，不隐藏模型真实工作过程。合成完成后 slider 会 seek 实际 WAV 音频。
- `Performance PK`：用同一段文本跑官方离线 API、官方 online-text API、裸 engine 和 Triton server，对齐请求开始时间后同步回放。官方两行只展示两个时间：`TTFT approx` 和 `Full wav ready`。`TTFT approx` 定义为官方 API 内第一个生成 code0 的时间减去请求开始时间；PK 图里官方捕获到的完整 WAV 从这个近似 TTFT 点开始回放。
- `Concurrency`：用短文本跑多路合成，统计 TTFT 分布和吞吐；默认请求 live Triton 并保存每路真实音频，点击 lane 可以回放该路合成结果。模拟并发只作为显式 fallback，不附带假音频。

首次体验建议用一键 demo 入口，WebUI dev server、Demo API 和 Triton 都由这个 launcher 启动/复用：

```bash
bash scripts/demo/start_webui_demo.sh --variant custom-1.7b
```

如果希望 `Performance PK` 四路都有真实 trace/audio，但机器显存不足以同时常驻官方 PyTorch、裸 engine 和 Triton，使用顺序实测采集模式：

```bash
bash scripts/demo/start_webui_demo.sh --variant custom-1.7b --collect-live-race
```

这个模式会在 WebUI 启动前依次测官方 PyTorch、裸 engine、Triton：每个 backend 单独占用 GPU，采集完成后写入 `workspace/demo_traces/race_default.json` 和 `workspace/demo_audio/*.wav`，最后默认留下 Triton 供 WebUI 使用。页面里的 PK 进度条是采集结果的同步回放，不要求四个 backend 同时在线。

WebUI 的 `Collect isolated PK` 按钮也会触发同一条顺序采集链路：Demo API 在本机启动采集脚本，通过 WebSocket 把日志推到页面，完成后自动加载新的 trace/audio。`Probe live endpoints` 只用于检查已经在线的服务；如果 official/bare engine 没有启动，它会诚实地显示 warning。

注意：官方 `generate_custom_voice(non_streaming_mode=False)` 在上游 docstring 中说明的是模拟 streaming text input，并不会暴露真正的 audio chunk streaming。WebUI 因此把官方 public API 的真实完成时间保留为 `Full wav ready`，并额外用 hook 获取 `TTFT approx`：第一个生成 code0 出现时间减去请求开始时间。流式 text 模式下，这个点应接近“第一个文本 token 进入 backbone 后输出第一个 code0”；离线模式下，这个点应是完整文本输入后开始音频 code 生成时输出第一个 code0。PK 回放从 `TTFT approx` 开始播放捕获到的完整 WAV。

官方 PyTorch baseline 的模型加载不计入测量时间；采集脚本会先加载模型，再对 offline/streaming 两种模式各跑 1 次不计入指标的 warmup，最后才记录正式请求。需要调整 warmup 可用 `QWEN_DEMO_OFFICIAL_WARMUP_ROUNDS` / `QWEN_DEMO_OFFICIAL_WARMUP_TEXT`，或在 `collect_live_race_sequential.sh` 上使用 `--official-warmup-rounds` / `--official-warmup-text`。

WebUI demo 的 Triton active decode slots 默认按 128 路展示设置为 `TRITON_MAX_BATCH_SLOTS=128`。如果复用的是已经在跑的 Triton 容器，launcher 会读取该容器实际的 `MAX_BATCH_SLOTS`；比如容器仍是 64 slots，那么 128 路并发面板会明确显示 64 active / 64 queued，后 64 路 TTFT 会包含排队等待，不应解读为模型单路首包慢。

```bash
# 显式指定 Triton active decode slots
bash scripts/demo/start_webui_demo.sh --variant custom-1.7b --triton-slots 128
```

也可以从 WebUI 目录走 npm 脚本：

```bash
npm --prefix webui run demo -- --variant custom-1.7b
```

浏览器页面本身不能直接启动本机 Docker/Python 进程，因此自动启动逻辑放在本地 launcher 里。裸 engine 不默认和 Triton 同时启动，因为两者可能争抢 GPU/KV cache；需要把裸 engine 也纳入 `Performance PK` live 测量时显式加 `--with-engine`。

```bash
bash scripts/demo/start_webui_demo.sh --variant custom-1.7b --with-engine
```

官方 PyTorch baseline 默认关闭，避免首次打开 WebUI 就加载另一份大模型。需要现场跑官方离线/官方 streaming baseline 时加：

```bash
bash scripts/demo/start_webui_demo.sh --variant custom-1.7b --official-live
```

如果某个 live backend 不可用，WebUI 会展示 fixture timing/trace 并明确 warning；音频按钮只会在该 backend 捕获到真实 waveform bytes 时启用，不再用嘟声占位。

要一次性尝试官方 PyTorch baseline、裸 engine 和 Triton 的 4-way live PK，可以用：

```bash
bash scripts/demo/start_webui_demo.sh --variant custom-1.7b --full-live-pk
```

这个模式会同时占用更多 GPU 显存；如果出现 OOM 或 KV cache 竞争，WebUI 会保留可复现 warning，而不是用假数据补齐。

也可以只刷新发布 trace，不启动 WebUI：

```bash
bash scripts/demo/collect_live_race_sequential.sh --variant custom-1.7b
```

`collect_live_race_sequential.sh` 会明确停止/启动 compose 管理的 engine/Triton 来释放显存；某个 backend 测不到时默认保留 fixture timing 并记录 warning，不会补假音频。需要 CI 或发布前强校验时加 `--strict`。

多路合成默认走 live Triton。需要只看前端布局或离线演示指标时，可以显式关闭 live lane audio：

```bash
bash scripts/demo/start_webui_demo.sh --variant custom-1.7b --simulated-concurrency
```

```bash
# Terminal 1
python -m demo_api --host 0.0.0.0 --port 7860

# Terminal 2
cd webui
npm install
npm run dev
```

打开 `http://localhost:5173`。

Docker Compose demo profile：

```bash
bash scripts/bash/compose.sh up --gateway triton --variant custom-1.7b
docker compose --profile demo up --build demo-api webui
```

刷新发布 trace：

```bash
python scripts/demo/collect_demo_traces.py \
  --live-triton \
  --triton-grpc localhost:8001 \
  --output workspace/demo_traces/race_default.json
```

如果直接手动启动 Demo API，live concurrency 默认开启；需要模拟模式时显式设为 0：

```bash
QWEN_DEMO_ENABLE_LIVE_CONCURRENCY=0 python -m demo_api --port 7860
```

### Text Player 口径

`Text Player` 消费 engine/Triton 已有的 `text_token`、`first_audio_chunk`、`audio_chunk` 和 `segment_end` 事件。若 audio chunk 已带 `phase/token_idx/decode_step/chunk_ms` metadata，WebUI 直接使用逐 chunk metadata；否则用 `segment_end.meta.text_tokens` 和 `segment_end.meta.audio_steps` 构造 decode step：前 `text_tokens` 个 step 标记为 token，剩余 step 标记为 `PAD` flush。这个推导不会掩盖 flush。

Text Player 的 slider 和 inline token/PAD 都只 seek 已经生成的 WAV 音频，不向 engine 发起 rollback。精确“音频播放到哪个字/词”的语义后续仍建议接 streaming ASR 或 alignment 模型；这里展示的是 engine decode-step 工作过程。

## 流式协议

standalone engine 同时支持 gRPC 和 WebSocket。WebSocket 控制帧示例：

```json
{"type":"start","session_id":"demo","config":{"task_type":"custom_voice","speaker":"Serena"}}
{"type":"text","text":"你好，世界。"}
{"type":"end"}
```

服务端返回：

- JSON event frame：协议事件、文本 token、边界提交、完成事件等。
- Binary frame：PCM audio chunk，音频格式由 start/event 元数据声明。

## 项目结构

```text
Qwen3-TTS-Triton/
├── engine/                     # standalone engine、frontend、scheduler、TRT executor
├── demo_api/                   # WebUI demo API、fixture/live trace、音频回放
├── webui/                      # Vite/React WebUI
├── scripts/
│   ├── bash/                   # autorun/setup/build/deploy/compose 脚本
│   ├── export/                 # PyTorch -> ONNX/manifest 导出脚本
│   └── python/                 # manifest/config/profile 工具
├── docs/
│   ├── zh/                     # 中文优先文档
│   └── architecture.md
└── workspace/                  # 模型、导出产物、engine、trace；默认 gitignored
```

## 中文文档

- [已知限制与风险](docs/zh/known_limitations.md)
- [Benchmark 方法](docs/zh/benchmark_methodology.md)
- [部署说明](docs/zh/deployment.md)
- [路线图](docs/zh/roadmap.md)
- [架构说明](docs/architecture.md)

## 许可证

本项目基于 [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) 进行工程化部署与优化。模型权重和上游代码的许可证请以 QwenLM 官方仓库为准。

# 部署说明

本文档说明当前推荐的部署方式、profile 参数和开发期容器策略。

## 推荐路径

v0.1 推荐先使用：

```bash
bash scripts/bash/autorun.sh all -m custom-1.7b
```

首次部署建议分阶段执行，便于定位问题：

```bash
bash scripts/bash/autorun.sh setup -m custom-1.7b
bash scripts/bash/autorun.sh build -m custom-1.7b
bash scripts/bash/autorun.sh deploy -m custom-1.7b --gateway standalone
```

## Phase A: 导出

Phase A 下载模型、安装依赖、导出 ONNX/weights/manifest。

```bash
bash scripts/bash/autorun.sh setup -m custom-1.7b
```

常用参数：

```text
--source auto|hf|modelscope
--skip-models
--skip-deps
--skip-export
--target-driver <driver>
```

## Phase B: 构建 TensorRT engine

Phase B 在 NGC 容器里运行 trtexec，并把实际 profile 写入 manifest：

```bash
bash scripts/bash/autorun.sh build -m custom-1.7b \
  --max-batch-size 64 \
  --max-input-len 128 \
  --max-seq-len 512 \
  --dtype bf16
```

写入位置：

```text
workspace/exported/custom-1.7b/triton_manifest.json
```

关键字段：

```json
{
  "engine_profile": {
    "engine_mode": "trt",
    "engine_dtype": "bf16",
    "triton_io_float_dtype": "bf16",
    "max_batch_size": 64,
    "max_input_len": 128,
    "max_seq_len": 512,
    "builder_image": "nvcr.io/nvidia/tritonserver:26.02-py3"
  }
}
```

runtime 的 batch/seq 不能超过这里的 profile。需要更大 batch 或更长文本时，重新跑 Phase B。

## Phase C: 启动服务

### Standalone

```bash
bash scripts/bash/deploy.sh run \
  --gateway standalone \
  --variant custom-1.7b \
  --max-batch 32 \
  --max-seq-len 512
```

端点：

- gRPC: `localhost:50051`
- WebSocket: `ws://localhost:50052/v1/ws`
- capabilities: `http://localhost:50052/v1/capabilities`
- health: `http://localhost:8080/health`

### Engine Docker

```bash
bash scripts/bash/deploy.sh run \
  --gateway engine-docker \
  --variant custom-1.7b
```

如果使用统一入口，等价命令是：

```bash
bash scripts/bash/autorun.sh deploy --gateway engine-docker -m custom-1.7b
```

这个模式使用 engine 镜像作为固定应用层：镜像内包含 TensorRT/Python
运行时、`/app/engine` 引擎代码、默认 `/app/engine.yaml` 和启动脚本；
运行时只读挂载和 Triton 相同的模型包
`workspace/model_repository/tts_orchestrator/1`。普通镜像里已经包含
`/app/engine` 代码，因此代码更新后需要重新构建并发布镜像。

本机 standalone gateway 也使用同一个模型包，只是由本机 Python 承载
`engine.server`。也就是说，standalone、engine-docker 和 Triton 的模型产物
都是 `model_repository/tts_orchestrator/1`，差异只在运行时进程和容器层。

生产环境推荐把“应用镜像、模型产物、部署配置”拆成三层：

- 应用镜像：`qwen3-engine:<tag>`，包含依赖和引擎代码，不挂载源码目录。
- 模型产物：统一的 Triton-compatible `model_repository`。engine 和 Triton
  都从 `/models/tts_orchestrator/1` 读取 `runtime/`、`weights/`、
  `tokenizer/` 和 manifest。engine-docker 当前要求 `runtime/model.plan`
  这种 `trt` 包。
- 部署配置：端口、batch、seq_len、session、speaker 默认值等，通过
  `ENGINE_CONFIG` 指向只读挂载的 YAML，或通过 `ENGINE_*` 环境变量覆盖。

示例：

```bash
ENGINE_CONFIG=/etc/qwen3-tts/engine.yaml \
ENGINE_CONFIG_FILE=/srv/qwen3/config/engine.yaml \
MODEL_REPO_DIR=/srv/qwen3/model_repository \
bash scripts/bash/compose.sh up --gateway engine --variant custom-1.7b
```

如果直接写 compose volume，可挂载：

```yaml
volumes:
  - /srv/qwen3/model_repository:/models:ro
  - /srv/qwen3/config/engine.yaml:/etc/qwen3-tts/engine.yaml:ro
environment:
  ENGINE_CONFIG: /etc/qwen3-tts/engine.yaml
```

开发期推荐改用：

```bash
bash scripts/bash/compose.sh up --gateway engine --variant custom-1.7b --dev
```

或：

```bash
bash scripts/bash/compose.sh watch --gateway engine --variant custom-1.7b
```

这样可以把环境层和代码层拆开，避免每次改 Python 代码都重新构建依赖镜像。

### Triton

```bash
bash scripts/bash/compose.sh prepare \
  --gateway triton \
  --variant custom-1.7b \
  --engine-mode trt

bash scripts/bash/compose.sh up \
  --gateway triton \
  --variant custom-1.7b
```

Triton 默认端口：

- HTTP: `localhost:8000`
- gRPC: `localhost:8001`
- Metrics: `localhost:8002`

## WebUI

本地运行：

```bash
python -m demo_api --host 0.0.0.0 --port 7860

cd webui
npm install
npm run dev
```

Compose 运行：

```bash
bash scripts/bash/compose.sh up --gateway triton --variant custom-1.7b
docker compose --profile demo up --build demo-api webui
```

## 常见问题

### runtime max_seq_len 超过 profile

错误类似：

```text
runtime max_seq_len=1024 exceeds engine profile max_seq_len=512
```

解决方式：

- 降低 `--max-seq-len` / `ENGINE_SCHEDULER_MAX_SEQ_LEN`。
- 或重新构建 engine：`bash scripts/bash/build_engines.sh --variant custom-1.7b --max-seq-len 1024`。

### dtype 不匹配

如果 Triton 报 `TYPE_FP32` / `TYPE_BF16` 之类错误，确保：

- Phase B 的 `--dtype` 和 `--triton-io-float-dtype` 符合目标。
- `triton_manifest.json` 已被 Phase B 更新。
- 重新 assemble model_repository。

### TensorRT plan 无法反序列化

TensorRT plan 与 runtime 版本强绑定。更换 TensorRT/NGC image 后需要重新构建 engine。

### WebUI 显示 fixture fallback

说明 live Triton 或 live engine 当前不可达。检查：

- Triton gRPC 端口是否是 `localhost:8001`。
- demo API 的 `QWEN_DEMO_TRITON_GRPC` 是否正确。
- standalone engine WebSocket 是否是 `localhost:50052`。

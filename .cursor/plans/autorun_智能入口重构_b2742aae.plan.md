---
name: Autorun 智能入口重构
overview: 将当前 `autorun.sh`（仅 Phase A）重构为三阶段全流程智能入口脚本，支持子命令、交互式模型选择、参数覆盖，并将原 Phase A 逻辑抽取到 `setup_env.sh`。
todos:
  - id: rename-autorun
    content: 将 autorun.sh 原内容移入 setup_env.sh，更新文件头注释
    status: completed
  - id: create-status-lib
    content: 新建 lib/status.sh，实现 detect_phase_{a,b,c}_status / detect_available_variants / print_status_summary
    status: completed
  - id: update-tools
    content: 在 tools.sh 中添加 source lib/status.sh
    status: completed
  - id: rewrite-autorun
    content: 重写 autorun.sh：子命令解析 + 交互式引导 + 参数转发 + 三阶段串联
    status: completed
  - id: update-rules
    content: 更新 shell-scripts.mdc 入口脚本文档
    status: completed
isProject: false
---

# Autorun 智能入口重构

## 现状分析

当前三阶段各有独立入口，用户需按顺序手动调用：

- `autorun.sh` — Phase A: 环境 + 导出 (9 步)
- `build_engines.sh` — Phase B: TRT-LLM 引擎编译
- `build_triton.sh <cmd>` — Phase C: Triton 部署

`autorun.sh` 名称暗示"一键全自动"，但实际只覆盖 Phase A，名不副实。

## 重构方案

### 核心思路

```
autorun.sh       → 智能入口（串联 A→B→C，子命令/交互）
setup_env.sh     → 原 autorun.sh 的 Phase A 逻辑（重命名）
build_engines.sh → 不变
build_triton.sh  → 不变
```

### 1. 重命名：`autorun.sh` -> `setup_env.sh`

将当前 `autorun.sh` 的全部内容原样移动到 `[setup_env.sh](scripts/bash/setup_env.sh)`，仅修改文件头注释。这样 Phase A 仍可独立使用。

### 2. 新建 `autorun.sh` — 三阶段智能入口

新 `autorun.sh` 作为统一入口，支持两种使用模式：

**模式 A：子命令模式**（精确控制）

```bash
autorun.sh setup      # Phase A only
autorun.sh build      # Phase B only  
autorun.sh deploy     # Phase C only (= build_triton.sh run)
autorun.sh all        # A → B → C 全流程
autorun.sh status     # 查看当前状态
```

**模式 B：零参数交互模式**（新手友好）

```bash
autorun.sh            # 无参数 → 交互式引导
autorun.sh base-1.7b  # 第一个参数是模型名 → 全流程部署该模型
```

### 3. 交互式引导流程（无参数时）

```
┌─────────────────────────────────────────────┐
│  Qwen3-TTS Triton — Intelligent Launcher    │
│                                             │
│  Detected:                                  │
│    GPU: NVIDIA A100 (driver 550.90)         │
│    CUDA: 12.4  Docker: OK                   │
│    Phase A: ✓ (base-1.7b exported)          │
│    Phase B: ✗ (no engines)                  │
│    Phase C: ✗ (not deployed)                │
│                                             │
│  What would you like to do?                 │
│                                             │
│  [1] Full pipeline (setup → build → deploy) │
│  [2] Setup environment (Phase A)            │
│  [3] Build engines (Phase B)                │
│  [4] Deploy Triton (Phase C)                │
│  [5] Resume from last incomplete step       │
│  [6] Show detailed status                   │
│                                             │
│  Enter choice [1-6]:                        │
└─────────────────────────────────────────────┘
```

### 4. 状态检测函数

新增 `lib/status.sh`，提供阶段完成度检测：

- `detect_phase_a_status` — 检查 `workspace/exported/` 目录中的 ONNX/checkpoint/weights 是否存在
- `detect_phase_b_status` — 检查 `workspace/exported/*/trtllm_engine/*.engine` 是否存在
- `detect_phase_c_status` — 检查 Triton 容器是否运行（`docker ps`）
- `detect_available_variants` — 列出已下载 / 已导出 / 已编译的变体
- `print_status_summary` — 彩色状态面板

### 5. 完整参数列表

```bash
autorun.sh [command] [model_variant] [options]

Commands:
  all                 Full pipeline: setup → build → deploy (default if model given)
  setup               Phase A only (environment + export)
  build               Phase B only (TRT-LLM engines)
  deploy              Phase C only (Triton server)
  status              Show pipeline status
  stop                Stop Triton server

Options:
  --variant, -m       Model variant (base-1.7b, custom-1.7b, ...)
  --skip-setup        Skip Phase A (assume env ready)
  --skip-build        Skip Phase B (use ONNX fallback)
  --skip-deploy       Skip Phase C
  --dry-run           Show what would be done
  --yes, -y           Skip confirmations (non-interactive)
  --help, -h          Show help
  
  # Phase A options (forwarded to setup_env.sh)
  --python VERSION    Python version (default: 3.10)
  --env-name NAME     Virtual env name (default: qwen3-tts)
  --source SOURCE     Model source (auto|hf|modelscope)
  --skip-models       Skip model download
  --skip-deps         Skip dependency install
  --skip-export       Skip model export
  
  # Phase B options (forwarded to build_engines.sh)
  --max-batch-size N  TRT-LLM max batch (default: 8)
  --image IMAGE       Override NGC container image
  --dtype DTYPE       Engine precision (default: bfloat16)

  # Phase C options (forwarded to build_triton.sh)
  --grpc-port PORT    gRPC port (default: 8001)
  --http-port PORT    HTTP port (default: 8000)
```

### 6. 文件改动清单


| 文件                                | 操作                                 |
| --------------------------------- | ---------------------------------- |
| `scripts/bash/autorun.sh`         | **重写** — 新的智能入口                    |
| `scripts/bash/setup_env.sh`       | **新建** — 原 autorun.sh 内容移入         |
| `scripts/bash/lib/status.sh`      | **新建** — 阶段状态检测函数                  |
| `scripts/bash/tools.sh`           | **修改** — 添加 `source lib/status.sh` |
| `.cursor/rules/shell-scripts.mdc` | **更新** — 入口脚本表中 autorun 描述         |


`build_engines.sh`、`build_triton.sh`、`download_models.sh` 均不需要改动，由新 `autorun.sh` 通过调用它们来串联。

### 7. 串联机制

新 `autorun.sh` 的 `cmd_all` 函数核心逻辑：

```bash
cmd_all() {
    log_step "Phase A: Environment Setup & Model Export"
    bash "${SCRIPT_DIR}/setup_env.sh" $SETUP_ARGS || exit 1
    
    log_step "Phase B: TRT-LLM Engine Build"
    bash "${SCRIPT_DIR}/build_engines.sh" $BUILD_ARGS || exit 1
    
    log_step "Phase C: Triton Deployment"
    bash "${SCRIPT_DIR}/build_triton.sh" run $DEPLOY_ARGS || exit 1
}
```

各阶段通过 `bash` 子进程调用而非 `source`，保证隔离性，每阶段失败立即停止并提示用户可从断点恢复。
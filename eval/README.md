# SteadyStream Eval Package

**状态**：`v1.0.0-alpha` - E0 冻结候选版本

这个目录实现了 `steadystream_plan_v2.md` §4 中定义的所有评测指标。

## 核心模块

| 模块 | 功能 | 对应章节 |
|---|---|---|
| `boundary_metrics.py` | F0 和能量跳变测量，含 VAD 门控和浊音帧过滤 | §4.1, §4.2 |
| `excess_metrics.py` | 超额跳变（减自然基线）计算 | §4.3 |
| `pause_metrics.py` | 边界停顿偏差测量 | §4.3 |
| `sim_delta.py` | 说话人相似度损失（SIM Δ） | §4.4 |
| `latency_metrics.py` | FASL 和 RTF 测量 | §4.5 |
| `text_metrics.py` | CER 计算 | §4.6 |
| `acceptance_test.py` | §4.7 验收测试套件 | §4.7 |

## 核心修复（E0.1）

修复了 E1 预跑中发现的三大测量伪影：

### 问题 1：F0 跳变物理不可能值（23.1 st ≈ 两个八度）
**根因**：分析窗落入静音/清音段，`librosa.pyin` 返回 NaN 或极端值

**修复**：
- 加入 `vad_gate()` - 能量阈值 -40 dB + 最小语音帧比例 30%
- 加入浊音帧过滤 - F0 仅从浊音帧（voiced ratio ≥ 40%）计算几何平均
- 不满足条件时返回 `None`，不强行输出无效值

### 问题 2：能量跳变物理不可能值（max 58.4 dB）
**根因**：左右窗口一侧为静音（RMS ≈ 0），另一侧为语音，计算 `|rms_db(right) - rms_db(left)|` 时产生巨大差值

**修复**：
- 左右窗口分别通过 VAD 门控
- 仅当两侧都通过时才计算能量跳变
- 单侧失败时该边界的能量跳变记为 `None`

### 问题 3：三套脚本口径不一致
**根因**：
- `steadystream_boundary_probe.py` 使用 250ms 窗口测量边界跳变
- smoke 测试使用句级均值差（非边界口径）
- 各自独立实现，参数不同

**修复**：
- 统一到 `eval/` 包，所有链路 import 同一实现
- 参数标准化：左右各 250ms 窗口（§4.1/§4.2）
- 明确标注测量来源：`boundary_source = "exact_event" | "proxy_proportional"`

## 新增功能（E0.2）

### 超额口径（减自然基线）

**动机**（§9.1 结论 3）：离线合成的"边界代理值"为 4.6 st / 6.0 dB，说明自然边界本就有数个 semitone 的合理起伏。绝对跳变值无法区分"自然韵律变化"与"状态断裂"。

**实现**：
```python
from eval import NaturalBoundaryReference, compute_excess_metrics

# 1. 从自然语音样本构建参考分布
natural_ref = NaturalBoundaryReference.from_samples(natural_samples)
# 按标点分桶：comma, period, question, exclamation
# 每桶 ≥300 边界（§4.3 要求）

# 2. 对合成音频计算超额跳变
excess_metrics = compute_excess_metrics(
    boundary_metrics,
    natural_ref,
    punct_classes=["comma", "period", ...]
)

# 超额跳变 = max(0, 测量值 - P75_自然)
```

**验收标准**：离线合成的超额跳变均值应 <1 st / <2 dB（§4.7 准则 5）

## 验收标准（§4.7）

运行 `acceptance_test.py` 检查以下条件：

1. ✅ **F0 跳变范围合理性**：自然参考 P50 ∈ [1, 4] st，P90 ∈ [3, 8] st
2. ✅ **能量跳变范围合理性**：自然参考 P50 ∈ [1, 5] dB，P90 ∈ [3, 10] dB
3. ✅ **VAD 覆盖率**：非静音边界 F0 测量成功率 ≥80%
4. ✅ **口径一致性**：同一音频重复测量 3 次，变异系数 (CV) <5%
5. ✅ **超额口径有效性**：离线合成的超额跳变均值 <1 st / <2 dB
6. ✅ **停顿检测有效性**：自然参考逗号 P50 ~200ms，句号 P50 ~400ms

**全部通过 → 冻结 eval 包 → 在 test-prosody-mini 上复测 E1 各系统**

## 使用示例

### 基础边界测量

```python
import soundfile as sf
from eval import measure_boundary_metrics, summarize_boundary_metrics

audio, sr = sf.read("stateless.wav")
boundaries = [12000, 24000, 36000]  # sample indices

metrics = measure_boundary_metrics(audio, sr, boundaries)
summary = summarize_boundary_metrics(metrics)

print(f"F0 jump mean: {summary['f0_jump_mean_st']} st")
print(f"Energy jump mean: {summary['energy_jump_mean_db']} dB")
print(f"F0 coverage: {summary['f0_coverage']}")  # 测量成功率
```

### 超额指标（表 2 主口径）

```python
from eval import NaturalBoundaryReference, compute_excess_metrics

# 加载冻结的自然参考分布
with open("natural_reference_frozen_v1.json") as f:
    ref_data = json.load(f)
natural_ref = NaturalBoundaryReference.from_dict(ref_data)

# 计算超额跳变
punct_classes = ["comma", "period", "comma", "period"]
excess = compute_excess_metrics(metrics, natural_ref, punct_classes)
excess_summary = summarize_excess_metrics(excess)

print(f"Excess F0 mean: {excess_summary['excess_f0_mean_st']} st")
```

### 停顿偏差

```python
from eval import measure_pause_metrics, summarize_pause_metrics

expected_pauses_ms = [220, 460, 200, 480]  # 从标点类别查表得到
pause_metrics = measure_pause_metrics(audio, sr, boundaries, expected_pauses_ms)
pause_summary = summarize_pause_metrics(pause_metrics)

print(f"Pause deviation mean: {pause_summary['pause_deviation_mean_ms']} ms")
```

## E0 待办清单

- [x] E0.1: 修复 boundary_metrics.py 加入 VAD 门控和浊音帧过滤
- [x] E0.2: 实现超额口径（减自然基线）
- [ ] E0.3: 构建 test-prosody-mini (50条) 和自然边界参考集
- [ ] E0.4: 通过 §4.7 验收标准
- [ ] E0.5: 在 mini 集复测 E1 各系统获得首批正式数字

## 下一步

1. 构建 test-prosody-mini（50 条多标点句子，LLM 生成 + 人工审校）
2. 收集自然边界参考集（≥20 说话人，按标点分桶后每桶 ≥300 边界）
3. 运行 `acceptance_test.py` 验收
4. 通过后冻结 eval 包为 v1.0.0，所有后续改动走版本号 + 全量复测

## 依赖

- librosa >= 0.10.0 (F0 estimation via pyin)
- numpy
- soundfile
- torch, torchaudio (sim_delta 模块，可选)

冻结条件：参考 `steadystream_plan_v2.md` 中的 `§4.7` 与 `§6 E0`。

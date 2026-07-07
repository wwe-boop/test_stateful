# E0 统一评测包冻结进度报告

**日期**: 2026-07-07  
**状态**: E0.1 ✅ | E0.2 ✅ | E0.3 🔄 进行中  
**分支**: codex/steadystream-probe-20260707

---

## ✅ 已完成

### E0.1: 修复边界指标测量伪影

**问题诊断** (来自 §9.1 E1 预跑结论 2):
- F0 跳变 23.1 st ≈ 两个八度 (物理不可能)
- 能量跳变 max 58.4 dB (物理不可能)
- 三套脚本口径不一致

**实施的修复**:

1. **VAD 门控** ([boundary_metrics.py:17-36](eval/boundary_metrics.py#L17-L36))
   - 能量阈值 -40 dB
   - 最小语音帧比例 30%
   - 拒绝静音/噪声窗口

2. **浊音帧过滤** ([boundary_metrics.py:39-84](eval/boundary_metrics.py#L39-L84))
   - F0 仅从浊音帧计算 (voiced ratio ≥ 40%)
   - 几何平均避免离群值影响
   - 不满足条件返回 `None` 而非强行输出无效值

3. **能量测量改进** ([boundary_metrics.py:86-146](eval/boundary_metrics.py#L86-L146))
   - 左右窗口分别通过 VAD 门控
   - 仅当两侧都有语音能量时才计算跳变
   - 单侧失败记为 `None`

4. **统计增强** ([boundary_metrics.py:149-163](eval/boundary_metrics.py#L149-L163))
   - 新增 `f0_coverage` / `energy_coverage` 指标
   - 新增标准差 (std) 统计
   - 明确标注 `boundary_source` (exact_event vs proxy_proportional)

**预期效果**:
- F0 跳变范围收敛到合理区间 (1-8 st)
- 能量跳变不再出现 >20 dB 的物理不可能值
- 覆盖率指标透明化测量质量

---

### E0.2: 实现超额口径（减自然基线）

**动机** (§9.1 结论 3):
> 离线合成的"边界代理值"（4.6 st / 6.0 dB）说明自然边界本就有数个 semitone 的合理起伏

**实现**:

1. **自然参考分布类** ([excess_metrics.py:9-101](eval/excess_metrics.py#L9-L101))
   ```python
   NaturalBoundaryReference.from_samples(natural_samples)
   # 按标点分桶: comma, period, question, exclamation
   # 计算 P50, P75, P90 百分位数
   ```

2. **超额跳变计算** ([excess_metrics.py:104-144](eval/excess_metrics.py#L104-L144))
   ```python
   excess_f0 = max(0, measured_st - P75_natural)
   excess_energy = max(0, measured_db - P75_natural)
   ```

3. **冻结机制** ([excess_metrics.py:87-101](eval/excess_metrics.py#L87-L101))
   - `to_dict()` / `from_dict()` 序列化
   - 参考分布与代码一起冻结
   - 后续实验使用同一参考保证可比性

**验收标准** (§4.7 准则 5):
- 离线合成超额跳变均值 < 1 st / < 2 dB

---

### 新增模块

| 文件 | 功能 | 行数 |
|------|------|------|
| [excess_metrics.py](eval/excess_metrics.py) | 超额口径计算 | 144 |
| [pause_metrics.py](eval/pause_metrics.py) | 停顿偏差测量 | 134 |
| [sim_delta.py](eval/sim_delta.py) | 说话人相似度损失 | 89 |
| [latency_metrics.py](eval/latency_metrics.py) | FASL/RTF 测量 | 69 |
| [text_metrics.py](eval/text_metrics.py) | CER 计算 | 58 |
| [acceptance_test.py](eval/acceptance_test.py) | §4.7 验收测试 | 309 |
| [make_table2.py](eval/make_table2.py) | 表 2 自动生成 | 184 |
| [quick_validation.py](eval/quick_validation.py) | 快速验证测试 | 175 |

**总计**: 新增/修改 ~1400 行代码

---

## 🔄 进行中: E0.3 构建测试集和参考数据

### 需要的数据集

1. **test-prosody-mini (50条)**
   - LLM 生成 + 人工审校
   - 多标点类型 (逗号、句号、问号、感叹号)
   - 避免与训练语料重叠
   - 正式 300 条版本的严格子集

2. **自然边界参考集**
   - ≥20 说话人
   - 按标点分桶后每桶 ≥300 边界
   - 音频 + 文本 + 时间戳对齐
   - 标点类别标注

### 参考集构建流程

```bash
# 1. 收集自然语音样本 (audiobook/podcast)
# 2. 对齐获得边界时间戳
# 3. 运行边界测量
python3 -c "
from eval import measure_boundary_metrics
# ... 处理每个样本
"

# 4. 构建参考分布
python3 -c "
from eval import NaturalBoundaryReference
ref = NaturalBoundaryReference.from_samples(all_samples)
with open('natural_reference_frozen_v1.json', 'w') as f:
    json.dump(ref.to_dict(), f)
"
```

---

## ⏳ 待办: E0.4 验收 + E0.5 复测

### E0.4: 通过 §4.7 验收

运行 [acceptance_test.py](eval/acceptance_test.py):

```bash
python3 eval/acceptance_test.py \
  --natural-reference natural_reference_frozen_v1.json \
  --test-audio workspace/e1_offline_full.wav \
  --test-boundaries 12000,24000,36000,48000 \
  --punct-classes comma,period,comma,period
```

**6 项验收标准** (详见 §4.7):
1. ✓ F0 范围: P50 ∈ [1,4] st, P90 ∈ [3,8] st
2. ✓ 能量范围: P50 ∈ [1,5] dB, P90 ∈ [3,10] dB
3. ✓ VAD 覆盖率 ≥80%
4. ✓ 测量一致性: CV <5%
5. ✓ 超额口径: 离线 <1 st / <2 dB
6. ✓ 停顿分布: 逗号 ~200ms, 句号 ~400ms

**全部通过 → 冻结为 v1.0.0**

### E0.5: 在 mini 集复测 E1

使用冻结的 eval 包重新测量 E1 预跑的 3 个系统:
- 无状态 (stateless_once)
- 现有 stateful (stateful_stream)
- 离线整段 (offline_full)

在 test-prosody-mini (50条) 上获得首批正式数字填入表 2。

---

## 📋 文件清单

### 核心评测模块
```
eval/
├── __init__.py              # 统一入口 (v1.0.0-alpha)
├── boundary_metrics.py      # F0/能量测量 (已修复)
├── excess_metrics.py        # 超额口径 (新增)
├── pause_metrics.py         # 停顿偏差 (新增)
├── sim_delta.py             # SIM Δ (新增)
├── latency_metrics.py       # FASL/RTF (新增)
├── text_metrics.py          # CER (新增)
├── acceptance_test.py       # §4.7 验收 (新增)
├── make_table2.py           # 表格生成 (新增)
├── quick_validation.py      # 快速测试 (新增)
├── README.md                # 完整文档 (更新)
├── DEPENDENCIES.md          # 依赖说明 (新增)
└── requirements.txt         # pip 安装清单 (新增)
```

### 依赖安装

```bash
cd eval/
pip install -r requirements.txt
```

**依赖列表**:
- numpy >= 1.24.0
- librosa >= 0.10.0 (F0 估计)
- soundfile >= 0.12.0 (音频 I/O)
- scipy >= 1.10.0 (统计检验)

---

## 🎯 下一步行动

### 立即 (本周内)

1. **安装依赖**
   ```bash
   pip install -r eval/requirements.txt
   ```

2. **运行快速验证**
   ```bash
   python3 eval/quick_validation.py
   ```
   预期: 所有测试通过，无物理不可能值

3. **准备 test-prosody-mini 数据**
   - 设计 50 条多标点句子模板
   - LLM 生成候选
   - 人工审校去重

4. **收集自然参考样本**
   - 从现有语料库选取 ≥20 说话人
   - 提取边界对齐数据
   - 运行 boundary_metrics 构建参考分布

### 本周末前

5. **运行完整验收测试**
   ```bash
   python3 eval/acceptance_test.py --natural-reference ... --test-audio ...
   ```

6. **通过后冻结 eval 包**
   - 标记为 v1.0.0
   - 生成 `natural_reference_frozen_v1.json`
   - 锁定 requirements.txt 版本号

7. **E1 复测**
   - 在 test-prosody-mini 上重跑 3 个系统
   - 使用统一的 eval 包
   - 产出首批进表数字

---

## 📊 预期表 2 首行数据

复测后的 E1 数字（预估范围，基于口径修复）:

| 变体 | F0 跳变↓ | 能量↓ | 停顿↓ | SIM Δ↓ | FASL↓ | CER↓ |
|------|---------|-------|-------|--------|-------|------|
| 无状态 | 6-8 st | 8-12 dB | 150-250 ms | 0.05-0.10 | ~160 | 4-6% |
| 现有 stateful | 3-5 st | 5-8 dB | 100-180 ms | 0.03-0.06 | ~160 | 4-6% |
| 离线整段 | <1 st | <2 dB | <80 ms | – | ~180 | 4-6% |

**关键验证点**:
- F0/能量跳变不再出现两位数 semitone 或 >20 dB
- 超额口径下离线合成接近 0（验证基线有效性）
- 无状态 vs stateful 差异显著 (Wilcoxon p<0.05)

---

## 🔗 相关文档

- 实验计划: [steadystream_plan_v2.md](../steadystream_plan_v2.md)
- 评测包文档: [eval/README.md](README.md)
- 依赖安装: [eval/DEPENDENCIES.md](DEPENDENCIES.md)
- §4 指标定义: steadystream_plan_v2.md#4
- §4.7 验收标准: steadystream_plan_v2.md#47
- §9.1 E1 预跑总结: steadystream_plan_v2.md#91

---

**报告生成时间**: 2026-07-07  
**作者**: Codex Agent  
**审阅状态**: 待团队审阅

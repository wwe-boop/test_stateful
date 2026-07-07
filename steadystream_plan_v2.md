# SteadyStream：Qwen3-TTS CustomVoice 模拟流式跨片段一致性实验计划

版本：v2.1 ｜ 更新：2026-07-07 ｜ 周期：10 周（W1 起点 = 2026-07-07 当周）｜ 基座：Qwen3-TTS-12Hz-1.7B-CustomVoice / 官方微调定制说话人（QwenLM/Qwen3-TTS）

**当前状态**：E1 基线预跑完成（分支 `codex/steadystream-e1-20260707`，未改任何源码）✅ ｜ 核心假设获初步实验支持 ✅ ｜ 发现 Triton 已有 stateful clause stream 路径，实施主路径调整为 Triton 落地 ⚠ ｜ **阻塞项：三套评测脚本口径不一致且存在测量伪影，统一 eval 包冻结前所有数字不得进表**（详见 §9 进度日志与 §6 E0）

---

## 0. 目标与最终交付

### 0.1 最终交付表格（表 2 模板）

**表 2：有界继承状态下的跨片段边界稳定性（test-prosody）**

| 变体 | F0 跳变↓ | 能量↓ | 停顿↓ | SIM Δ↓ | FASL↓ | CER↓ |
|---|---|---|---|---|---|---|
| 无状态 | – | – | – | – | – | – |
| 现有 stateful（Triton）† | – | – | – | – | – | – |
| 仅声学尾 | – | – | – | – | – | – |
| 仅 KV/token 尾 | – | – | – | – | – | – |
| 尾 +KV+ 暂停恢复 | – | – | – | – | – | – |
| 完整 SteadyStream | – | – | – | – | – | – |

† v2.1 新增行：Triton 服务现有的 stateful clause stream 路径，作为"服务端现状"对照——E1 预跑显示其边界指标介于无状态与离线之间（旧口径下能量跳变 20.3→13.5 dB），说明已隐式携带部分状态；其具体机制待解剖（E1.5），解剖结论决定该行在组件坐标系中的位置及"仅声学尾"行的定义是否需要调整（若其真流式链路已天然保持解码器状态，则 C1 只剩 tail token 注入部分）。

单位与口径：F0 跳变（semitone）、能量（dB）、停顿（ms 偏差）、SIM Δ（相似度损失，×100）、FASL（ms）、CER（%）。所有格 = 3 种子均值±std；每列最优加粗，对"无状态"行做 Wilcoxon 显著性标记（* p<0.05, ** p<0.01）。指标计算全部定义于 §4，测试集 test-prosody 定义于 §5。

### 0.2 配套表格（同批产出）

- **表 1（主结果）**：无状态 / 完整 SteadyStream / 整段离线 topline × {test-prosody, test-long, test-general}，列 = 表 2 六指标 + 主观（边界 ABX、CMOS）；
- **表 3（预算消融）**：完整 SteadyStream 下声学尾长 T_tail 与 KV 滑窗宽 W 的扫描；
- **表 4（并发压测，v2.2 新增）**：1/8/16/32/64/128 路泊松到达重放固定 trace；列 = FASL(VAD) / Jitter / 卡顿率 / TTFT / 显存；>1 s 停顿子集单独报告；PAD 基线对照见 §6 E9；
- **表 5（instruct 消融，CustomVoice 特有，原表 4）**：instruct 注入方式 × 中途切换场景。

### 0.3 场景与核心假设

场景：CustomVoice 能力（speaker token 定身份 + `instruct` 风格控制，接口 `generate_custom_voice(text, language, speaker, instruct)`；官方微调产出的 `speaker_name` 走同一接口），文本按子句切分、逐片段合成、顺序拼接播出的模拟流式。

假设：跨片段跳变可分解为若干**可独立继承的状态**——声学尾、LM 的 KV/token 历史、边界停顿——逐项继承逐项收敛；三者之上再叠加 continuation 后训练弥合 train-test mismatch，即得完整 SteadyStream。表 2 的行序即该假设的验证路径。已有工作先验：无界上下文长文本会漂移崩溃（interleaved 方案 WER 达 70.97%），无边界标记的朴素滑窗说话人相似度从 0.57 崩到 0.22；边界感知后训练 + 有界滑窗 prompt 可将长文本 WER 从 71.0% 降到 4.8%——故 KV 继承必须有界、边界必须显式、训练不可省。

---

## 1. 方法：SteadyStream 组件化设计

### 1.1 四个组件

| 组件 | 名称 | 内容 | 继承的状态 |
|---|---|---|---|
| **C1** | 声学尾（acoustic tail） | ① 解码器（12Hz 因果 ConvNet）各层卷积 buffer 跨片段延续；② 上一片段末尾 T_tail（默认 1s，12.5Hz 即 ~13 帧 × 16 层码本）语音 token 以 re-prefill 方式注入下一片段前缀 | 波形级连续性 + 短程声学惯性 |
| **C2** | KV/token 尾 | LM 的 `past_key_values` 跨片段保留，有界管理：sink（speaker + instruct + 系统前缀 KV 永驻）+ 滑窗（最近 W 秒语音 token 及对应文本的 KV，默认 W=10s），整句粒度淘汰；dual-track / MTP / sub-talker 的 cache 按语音帧对齐裁剪 | 中长程韵律/风格记忆 |
| **C3** | 暂停恢复（pause recovery） | 边界停顿的建模与恢复：(a) 训练数据在片段边界**保留真实句间静音的语音 token**（不切除），使模型学会自然生成边界停顿；(b) 推理端停顿校正器——按标点类别 c 查自然参考分布得目标停顿 d̂(c)，对生成的边界静音做裁剪/补齐（容差 ±80ms 内不动），杜绝硬拼接的"零停顿"与状态继承下的"双停顿" | 边界时长结构 |
| **C4** | continuation 后训练 | 多片段续写格式 SFT：`[CustomVoice前缀] ⊕ Σ_k(<t>text_k<eot> ⊕ codes_k ⊕ <bnd>)`，`<bnd>` 显式边界 token；历史 drop 增强（0.5 概率仅留 sink+近 1–3 句，模拟 C2 有界形态）；lookahead 注入（30% 样本附下一句前 3–8 词）；loss 仅施于语音 token | 训推格式对齐 |

### 1.2 表 2 行 ↔ 组件映射（严格定义，代码中以配置开关实现）

| 表 2 行 | 组件 | 说明 |
|---|---|---|
| 无状态 | ∅ | 逐片段独立 `generate_custom_voice`，仅共享 speaker/instruct 参数；解码器状态每句重置 |
| 现有 stateful（Triton） | 未知（待 E1.5 解剖） | Triton 服务自带的 stateful clause stream 原样运行，不做任何改动；解剖后在此列补注其实际等价组件（可能 ⊂ C1∪C2） |
| 仅声学尾 | C1 | 无 KV 复用：每句重新 prefill（前缀 + 声学尾 token + 当前文本）；若 E1.5 确认 Triton 真流式已保持解码器状态，本行定义收窄为"解码器状态 + tail token 注入" |
| 仅 KV/token 尾 | C2 | KV 有界保留；解码器状态每句重置、不注入声学尾 token（C1 关） |
| 尾 +KV+ 暂停恢复 | C1+C2+C3 | 推理侧全开，基座模型未训 |
| 完整 SteadyStream | C1+C2+C3+C4 | 推理侧全开 + 后训练模型（推理格式与训练格式对齐） |

另设两个隐藏对照不进表 2 正文（进附录）：整段离线 topline（表 1 用）；C1+C2（无 C3），用于剥离停顿恢复的独立贡献。

### 1.3 正交开关（默认值进表 2，扫描进表 3/4）

(a) lookahead 词数：默认 3（C4 行）/ 0（其余行）；(b) 边界首 5 个语音 token 温度 0.5：全行统一开（属采样协议而非继承状态）；(c) instruct 注入：默认仅 sink 驻留，消融进表 4。

---

## 2. 推理引擎实现（承载表 2 全部行）

新建 `steadystream/session.py`，外挂驱动 `qwen_tts/core/models/modeling_qwen3_tts.py` 的 forward（不改官方源码）：

```python
class SteadyStreamSession:
    def __init__(self, model, codec, speaker, language, instruct="",
                 use_acoustic_tail=True, tail_sec=1.0,        # C1
                 use_kv=True, window_sec=10.0,                # C2
                 use_pause_recovery=True,                     # C3
                 lookahead_words=3):
        ids = build_customvoice_prefix(speaker, language, instruct)  # 由官方 processor 导出模板并对拍
        self.kv = prefill(model, ids); self.sink_len = len(ids)
        self.dec_state = codec.init_streaming_state() if use_acoustic_tail else None
        self.tail_codes = []           # C1 的语音 token 尾
        self.history = []              # 淘汰记账

    def synth_segment(self, text, punct_class):
        prefix_extra = self.tail_codes if (C1 and not C2) else []   # C1-only 走 re-prefill 注入
        ids = build_incremental_text(text, lookahead)
        self.kv = incremental_prefill(self.model, prefix_extra + ids,
                                      self.kv if C2 else refresh_prefix())
        codes = autoregressive_decode(...)                          # 首5 token 低温
        wav, self.dec_state = self.codec.decode_streaming(codes, self.dec_state) \
                              if C1 else self.codec.decode(codes), None
        wav = pause_corrector(wav, punct_class) if C3 else wav      # 查 d̂(c) 裁剪/补齐边界静音
        self.tail_codes = codes[-tail_frames:]
        self._evict(sink=self.sink_len, window=W)                   # C2 有界淘汰（整句粒度）
        return wav
```

工程检查清单（沿用 v1.1，此处只列关键）：CustomVoice 前缀 token 排布从官方 processor 导出 input_ids 对拍单测；dual-track/MTP cache 形状逐层核对、按帧对齐裁剪；position id 默认逻辑连续（物理重编号作对照，W4 前定型）；每句 `max_new_tokens = 语速先验×字数×1.6` 硬上限；解码器流式化单测——整段离线 vs 分块流式逐样本误差 < 1e-3；FASL 打点嵌入 session（见 §4.5）。

### 2.1 Triton 落地路线（v2.1 新增，实施主路径）

**背景**：远端 5090-Host 存在常驻 `engine.server`（占 ~29.9GB 显存），同卡本地直载完整模型不可行；且 Triton 已有 stateful clause stream 能力，E1 预跑证实其有效。故 C1/C2/C3 的实现主路径调整为 **`Qwen3-TTS-Triton` 仓库内落地**（单独本地分支），本地 `steadystream/session.py` 引擎降级为机制验证与对拍工具。

**前置确认（E1.5 解剖，阻塞 C2 排期）**：
1. **LM backend 类型**：Python backend 包 HF/自研推理 → 可完整实现 C2（跨请求持有 `past_key_values` + sink 保留 + 整句粒度淘汰 + position id 策略可控）；TensorRT-LLM 类引擎 → KV 细粒度操纵受引擎 API 限制，C2 可能降档为"session 级 prompt/prefix 复用"，需在解剖报告中明确可实现档位；
2. **现有 stateful 携带的状态**：跨句携带的是 KV、文本/token 历史、还是仅解码器 buffer；是否使用 Triton sequence batching 机制；
3. **codec 是否已分块流式解码**：若是，C1 的解码器状态部分已天然满足，表 2"仅声学尾"行按 §1.2 注收窄。

**实现要点**：
- session 路由：多副本/多实例下同一会话请求必须落到持有其 cache 的实例（sequence_id + direct scheduling）；
- 显存预算：每 session 滑窗上限 × 峰值并发数，在 29.9GB 常驻之外单独核算，超预算时按 LRU 驱逐整个 session（驱逐后该会话降级为 re-prefill 路径并打日志）；
- C1/C2/C3 在 Triton backend 内同样以配置开关实现，保证表 2 各行"只差一个变量"的可比性；
- 本地引擎与 Triton 实现之间做输出对拍（同 seed 同输入，语音 token 序列一致性抽检），防止两套实现语义漂移。

---

## 3. 数据与训练（产出 C4 模型）

### 3.1 数据（CustomVoice 主线，沿用 v1.1 §5 并做两处修订）

主线：目标说话人长篇连续录音 50–300h × 2–3 人，走官方 `finetuning/`（`prepare_data.py` 抽 codes → `sft_12hz.py --speaker_name`）；预置音色（Vivian 等）只参与推理侧各行评测。7 步管线不变（VAD → 双 ASR 交叉验证 → 子句切分 → 说话人质控 → DNSMOS≥3.0 → 官方脚本抽 token 按时间戳切跨度（±2 帧容忍）→ 组装 JSONL）。

**修订 1（服务 C3）**：切分时片段边界的真实静音**保留在前一片段的 codes 尾部**（上限 600ms，超出截断），并在 JSONL 记录 `pause_ms` 与标点类别 `punct_class`——这是暂停恢复的训练信号，也是自然停顿分布 d̂(c) 的统计来源。

**修订 2（服务表 2 的 test-prosody）**：评测集按 §5 重新定义，训练源与评测源按节目/书目严格隔离。

### 3.2 训练配置（沿用 v1.1 §6，摘要）

LoRA 先行（r=32–64，lr 1e-4，2–3 epoch，50–100h）→ 达标后全参 SFT（lr 自 2e-6 起扫，8×A100，混 10% 官方原格式单句重放）；门禁：seed-tts-eval WER 劣化 ≤0.3 绝对值、SIM 劣化 ≤0.01，instruct 跟随抽测不退化；每次改 dataset/collate 做单样本 token 排布断言，且与 §2 推理前缀构造共用同一函数做一致性单测。已知风险：官方微调脚本对小数据/不当配置会训崩（社区有 15h 数据训出纯噪声的先例），50h 子集先行验证。

---

## 4. 指标定义与计算（表 2 六列，逐一给出可实现口径）

通用记号：一条测试样本含片段 s_1…s_n，拼接点（边界）b_1…b_{n-1}；边界两侧各取 200ms 分析窗（跳过 C3 校正后的静音段，取静音外最近的有声 200ms）；音频统一 24kHz。聚合规则：边界级 → 样本级取均值 → 测试集级取均值±std（3 种子）。所有脚本落在 `eval/` 目录，指标代码在 W2 结束冻结版本。

### 4.1 F0 跳变（semitone，↓）

对每个边界 b_j：
1. pyworld harvest 提取两侧窗的 F0，仅保留浊音帧（浊音帧占比 <30% 的窗，该边界跳过 F0 指标）；
2. 半音域中值滤波去 octave error；
3. `ΔF0_j = | mean(logF0_after) − mean(logF0_before) | × 12/ln2`（semitone）；
4. **报告超额跳变**：`excess_j = max(0, ΔF0_j − P50_natural(punct_class_j))`，其中 P50_natural 来自自然边界参考集（20 名说话人真实长录音，按标点类别分桶统计）——直接报原始 ΔF0 会惩罚自然的句间起伏，超额口径只惩罚"比真人边界更跳"的部分；
5. 表格值 = 全部有效边界 excess 的均值。辅助报告（进附录）：|ΔF0| 分布对自然分布的 Wasserstein 距离。

### 4.2 能量（dB，↓）

同窗口：`ΔE_j = | 20·log10(RMS_after / RMS_before) |`，同样报超额口径 `max(0, ΔE_j − P50_natural(c_j))`；表格值 = 均值。RMS 在去直流、A 计权后计算，避免低频噪声污染。

### 4.3 停顿（ms，↓）

衡量边界停顿的**时长异常**：
1. 用能量阈值 + silero-vad 双判据测每个边界的实际静音时长 d_j（拼接 crossfade 区计入）；
2. 目标停顿 d̂(c_j) = 自然参考集中该标点类别的中位数（逗号/分号/句号/问叹号分桶；换 instruct 边界单独一桶）；
3. `pause_dev_j = | d_j − d̂(c_j) |`；表格值 = 均值。
4. 辅助报告：异常停顿率 = pause_dev > 150ms 的边界占比（零停顿硬拼接与双停顿都会被此项捕获）。
注意：C3 关闭的行（前三行）此列预期显著偏大，这正是表 2 要呈现的对比；C3 开启的行此列同时检验校正器本身是否引入新伪影（校正后仍需过 4.4/4.1）。

### 4.4 SIM Δ（相似度损失 ×100，↓）

CustomVoice 无 3s ref，参照 = 该音色 60s held-out 注册集的说话人嵌入（ECAPA 与 WavLM-TDNN 双模型，表格报 ECAPA，WavLM 进附录）：
1. `SIM_sys` = 每条样本整条音频 vs 注册集嵌入的余弦相似度，测试集均值；
2. `SIM_topline` = 整段离线合成（同文本同 speaker 同 instruct）的同口径相似度；
3. **`SIMΔ = (SIM_topline − SIM_sys) × 100`**——度量"流式化造成的音色损失"，与绝对相似度解耦（不同音色绝对值不可比）；
4. 辅助报告：句间跳变 `1 − min_j cos(e_j, e_{j+1})`（相邻片段嵌入）与逐句漂移斜率（第 k 句 vs 注册集相似度对 k 回归）。

### 4.5 FASL（First-Audio-Sample Latency，ms，↓）

每个片段从**其文本可用时刻** t_text(k)（模拟流式中由切分器发出该子句的时刻）到该片段**首个音频样本产出**时刻 t_audio(k) 的延迟：`FASL_k = t_audio(k) − t_text(k)`。
- 打点位置：session 内部，t_audio 取解码器吐出首帧 PCM 的墙钟时间（含 prefill、含 C3 校正器耗时）；
- 表格值 = 全片段均值；辅助报告 P95 与"首片段 FASL"（对齐官方 97ms 首包口径）、FASL 对句序 k 的曲线（验证 C2 有界后延迟恒定，re-prefill 类变体线性增长）；
- 测量环境固定：1×A100、bf16、`generate_config.json` 默认采样参数、batch=1、CUDA Graph/torch.compile 关闭（或全行统一开启），预热 5 条后计时。
- **服务化口径（v2.1 新增，Triton/gRPC 路径为主报口径）**：走服务后端到端延迟包含网络与队列排队，故 FASL 主报口径改为**服务端打点**——请求入队时刻 → 该片段首帧 PCM 写入响应流时刻；客户端端到端延迟作为辅助口径同步记录（两者差值即网络+排队开销，单独报告）。测量时固定单并发（排队为零），并发扫描（1/4/16）单独进系统指标附表。本地引擎与 Triton 两条链路的 FASL 不直接互比，各自内部跨行比较。

### 4.6 CER（%，↓）

整条拼接音频 → Paraformer（中文主报）/ Whisper-large-v3（英文与交叉验证，进附录）→ 与原文本对齐计 CER；文本正则化（数字/英文/标点）用同一套规则全行统一。长文本崩溃模式（复读/幻觉/提前终止）天然反映在 CER 上，另在 test-long 上单独报失败率（任一崩溃模式计失败）。

### 4.7 指标有效性验收（W2 出口）

在自然边界参考集上：F0/能量超额口径均值应 ≈ 0；对人工构造的劣化样本（强制变调 +3st、拼接零停顿、换说话人尾段）各指标应单调响应。通过后冻结。

---

## 5. 评测集

| 集合 | 规模 | 构成 | 用途 |
|---|---|---|---|
| **test-prosody**（表 2 主集） | 300 条 × 6–12 片段，中文为主 | 刻意含高跳变风险边界：疑问→陈述、感叹→平叙、长短句交替、数字/中英夹杂、情绪递进文本；3 预置音色 × 5 固定 instruct + 2 定制音色；每条标注每个边界的 punct_class | 表 2 全部六列 |
| test-long | 50 条 × 10 分钟级 | 有声书/播报长文 | 表 1 长文稳定性、崩溃率、FASL-vs-k 曲线 |
| test-general | seed-tts-eval（中/英） | 官方 | 表 1 常规能力门禁（官方口径：bf16、max_new_tokens=2048、默认采样参数） |
| test-switch | 100 条，含中途换 instruct | 情绪切换 | 表 4 |
| 自然边界参考集 | 20 说话人真实长录音 | 词级对齐 + 标点分桶 | d̂(c)、P50_natural、指标验收 |
| 注册集 | 每音色 60s held-out | 干净语音 | SIM Δ 参照 |

训练/评测源按节目、书目、说话人三重隔离；test-prosody 文本由 LLM 生成 + 人工审校，避免与任何训练语料重叠。

**v2.1 补充**：先行构建 **test-prosody-mini（50 条）** 快速版，供 E0 指标验收后立即复测 E1 各系统与迭代调试使用（E1 预跑仅单条 16s / 3 句样本，仅具方向性意义，样本量不足以支撑任何进表结论）；正式 300 条版本随迭代补齐，mini 集为其严格子集以保证数字可延续。

---

## 6. 实验流程（表格填充顺序）

| 步 | 内容 | 产出 | 状态（2026-07-07） |
|---|---|---|---|
| **E0（v2.1 新增，阻塞项）** | 统一 eval 包：实现 §4 全部口径（VAD 门控 + 浊音帧过滤 + 超额口径），三条链路（本地 / Triton / ttstest）全部收敛到同一包；过 §4.7 验收后冻结；建 test-prosody-mini（50 条）与自然边界参考集 | 冻结版 `eval/` + 参考分布 | **未开始，最高优先级**——E1 预跑暴露口径分叉与测量伪影（见 §9），冻结前任何数字不得进表 |
| E1 | 无状态行 + 整段离线 topline（纯官方 API / Triton 服务） | 表 2 无状态行、表 1 topline 列、SIMΔ 分母基准 | **预跑完成 ✅**（方向性结论成立，见 §9）；待 E0 后在 mini 集复测出正式数字 |
| **E1.5（v2.1 新增）** | 解剖 Triton 现有 stateful clause stream：backend 类型、携带状态内容、sequence batching 使用情况、codec 是否分块解码；产出一页机制说明 | 表 2 "现有 stateful" 行定义确认；C2 可实现档位结论；"仅声学尾"行定义是否收窄 | 未开始，与 E0 并行，**阻塞 E2/E3 排期** |
| E2 | 仅声学尾（C1，范围视 E1.5 结论裁剪）：Triton 分支实现 | 表 2 对应行 | 未开始 |
| E3 | 仅 KV/token 尾（C2）：Triton backend 内 sink+滑窗+session 路由（§2.1） | 表 2 对应行 + "现有 stateful"行同批复测；隐藏对照 C1+C2 | 未开始（团队建议的优先项，同意） |
| E4 | + 暂停恢复（C1+C2+C3） | 表 2 对应行 | 未开始 |
| E5 | C4 训练（LoRA→全参）后全开 | 表 2 末行、表 1 主行 | 未开始 |
| E6 | 表 3 扫描：T_tail ∈ {0.5, 1, 2}s × W ∈ {2, 5, 10, 20}s（完整配置上） | 表 3 | 未开始 |
| **E9（v2.2 新增）** | 表 4 并发压测：`steadystream_stress_client.py` + Poisson 调度 + `eval/{fasl_vad,jitter,stutter,stress_metrics}.py` + `make_table4.py`；6 档 × 3 seeds × {PAD 基线, stateful, 完整} | 表 4 + 4b | **进行中**（分支 `exp/steadystream-table4-20260707`） |
| E7 | 表 5：instruct 注入 {sink 驻留 / 每句重注 / 驻留+每 5 句重述} × test-switch 上 {换后清近窗 / 不清} | 表 5 | 未开始 |
| E8 | 主观：边界 ABX（拼接点前后各 1.5s，"是否同一人同一状态连续说"，每系统 100 边界）+ CMOS vs topline（20 人×50 条）+ 风格保持 MOS | 表 1 主观列 | 未开始 |

预期表 2 形态（用于结果 sanity check）：六列自上而下单调改善，其中——F0/能量跳变主要被 C1+C2 压低；停顿列在第 4 行才显著下降（C3 生效）；SIM Δ 在 CustomVoice 下各行差距较小（speaker embedding 锚定音色），若第 1 行 SIMΔ 已很小则如实报告并把论证重心放在韵律三列；FASL 第 2 行（re-prefill）可能高于第 3 行（KV 增量），完整行因 lookahead 略增待机但 FASL 本身不受影响；CER 在 test-prosody 上各行接近、差异主要在 test-long（进表 1）。若实测违背单调性，按行回查组件实现。

---

## 7. 里程碑与验收

| 周 | 里程碑 | 出口标准 | 状态（2026-07-07） |
|---|---|---|---|
| W1–2（07-07 起） | 环境 + E1 + **E0 统一 eval 包** + E1.5 解剖 | §4.7 验收通过，指标冻结；无状态 vs topline 差异显著；E1.5 机制说明产出、C2 档位确定 | E1 预跑 ✅（分支 `codex/steadystream-e1-20260707`）；E0/E1.5 进行中为 W1–2 剩余主任务 |
| W3–4 | Triton 分支实现 C2 → C1 → C3（顺序按团队建议：C2 优先）+ E2–E4 | 表 2 前 5 行（含现有 stateful 行）填齐；FASL-vs-k 恒定性验证；C2 在 test-long 复现"无界退化 / 有界稳定"对照；本地引擎与 Triton 输出对拍通过 | 未开始 |
| W3–5 | 数据管线（含 pause_ms/punct_class 字段）+ 50–100h 主线数据 | token 边界对齐 ±2 帧比例 >95%；停顿标注抽检准确率 >90% | 未开始 |
| W5–6 | C4 LoRA → E5 | 表 2 末行：F0 超额跳变较无状态行降 ≥60%、停顿偏差 ≤ 自然 P75、CER 不劣于无状态行、seed-tts-eval 不掉点 | 未开始 |
| W6–7 | 全参复现 + E6/E7 | 表 3/表 4 填齐；test-long 崩溃率 <2% | 未开始 |
| W8 | E8 主观 | 边界 ABX"同一人同一状态"判定率较无状态 +≥20pp；CMOS vs topline ≥ −0.2 | 未开始 |
| W9–10 | 报告（表 1–4 + 分析） | 评审通过 | 未开始 |

## 8. 风险（增量项，v1.1 全表仍适用）

| 风险 | 应对 |
|---|---|
| C1 与 C2 同开时声学尾 token 与 KV 中历史重复（双重条件化） | 严格互斥实现：C2 开启时 tail token 不再 re-prefill（历史已在 KV 中），C1 仅贡献解码器状态；表 2 行定义按 §1.2 执行 |
| C3 校正器裁剪静音引入新的边界伪影 | 校正只在 VAD 判定的纯静音区操作 + 10ms 淡入淡出；4.1/4.2 在校正后音频上复测把关 |
| 超额口径依赖自然参考分布的质量 | 参考集 ≥20 说话人、按标点分桶后每桶 ≥300 边界；分布连同代码一起冻结、随报告发布 |
| FASL 受实现细节噪声影响（编译/缓存） | 固定测量协议（§4.5），全行同机同环境同日跑，预热后计时 |
| SIM Δ 在 CustomVoice 下区分度不足 | 备用列：句间跳变 min-cos（§4.4 辅助口径）替换主列，报告中说明 |
| **评测脚本多套并存导致口径分叉（v2.1，已发生）** | E0 统一 eval 包为独立目录/仓库，本地 / Triton / ttstest 三条链路全部 import 同一包；冻结后任何指标改动走版本号 + 全量复测 |
| **Triton LM backend 为 TRT-LLM 类引擎，KV 细粒度操纵受限（v2.1）** | E1.5 前置确认；若受限则 C2 降档为"session 级 prefix 复用"并在报告中明确档位；完整 C2 结论由本地引擎在闲置卡上补证 |
| **5090-Host 常驻 engine.server 占 29.9GB 显存，挤占实验空间（v2.1，已发生）** | 服务侧实验全部走 gRPC 复用现有服务；需本地直载的对拍/训练任务改到其他卡执行；session cache 显存预算独立核算（§2.1） |
| **现有 stateful 隐式携带未知状态，污染组件归因（v2.1）** | E1.5 解剖为 E2/E3 的硬前置；解剖前不产出任何"仅声学尾/仅 KV"行的数字 |

---

## 9. 进度日志

### 2026-07-07（E1 预跑，分支 `codex/steadystream-e1-20260707`，未改源码）

**已完成**：
1. **E1 基线对照（本地 API）**：无状态逐片段拼接 vs 整段离线。旧口径下无状态边界能量跳变均值 29.2 dB（max 58.4）、F0 跳变 23.1 st；离线对应 6.0 dB / 4.6 st。结果目录 `Qwen3-TTS/logs/steadystream_e1_20260707/`。
2. **Triton 三组对照**：无状态 20.3 dB / 22.5 st；**stateful clause stream 13.5 dB / 11.3 st**；离线为参照。RTF 均 ~0.16。结果目录 `.../steadystream_e1_stream_compare_20260707/`。
3. **Triton 长文本 smoke**（ttstest 链路）：F0 delta mean 0.98 st（句级口径，与边界口径不同名同用，见结论 3）、CER 4.86%，3 句中 1 句 CER>5%。

**结论与决策**：
1. ✅ 核心假设获支持：无状态硬拼接边界不连续显著，方向性排序 无状态 < 现有 stateful < 离线 成立；
2. ⚠ **指标数字全部作废，不进表**：23.1 st ≈ 两个八度、max 58.4 dB 为物理不可能量级，判定为测量伪影（分析窗落入静音/清音段、拼接 click 污染基频估计、无 VAD 门控）；且三套脚本口径互不一致（smoke 的 0.98 st 为句级均值差而非边界 200ms 窗跳变）。→ 立项 E0（统一 eval 包 + §4.7 验收），为当前最高优先级阻塞项；
3. ✅ 离线合成的"边界代理值"（4.6 st / 6.0 dB）与超额口径设计互证：自然边界本就有数个 semitone 的合理起伏，进一步确认指标必须减自然基线；
4. ⚠ 发现 Triton 已有 stateful clause stream：表 2 新增"现有 stateful"行（§0.1），立项 E1.5 解剖其机制，实施主路径调整为 Triton 落地（§2.1）；
5. ⚠ 环境约束记录：5090-Host 常驻 engine.server 占 29.9GB 显存，需本地直载的实验改道（§8 风险表）；
6. 分支策略确认：Triton 侧改动在 `Qwen3-TTS-Triton` 单独开本地分支；eval 包独立成库被各链路引用。

**下一步（W1–2 剩余）**：E0（冻结 eval 包 + test-prosody-mini + 自然边界参考集）‖ E1.5（Triton stateful 解剖 + backend 类型确认）→ 通过后在 mini 集复测 E1 各系统出首批正式数字。

---

## 附录 A：训练样本 JSONL（v2.0，新增停顿字段）

```json
{
  "sample_id": "narrator01_book0123_chunk0042",
  "speaker_name": "narrator_01", "language": "Chinese",
  "instruct": "用平静温和的叙述语气说",
  "segments": [
    {"text": "其实我真的有发现，", "codes": [[...]], "lookahead": "我是一个",
     "pause_ms": 220, "punct_class": "comma"},
    {"text": "我是一个特别善于观察别人情绪的人。", "codes": [[...]], "lookahead": "",
     "pause_ms": 460, "punct_class": "period"}
  ],
  "instruct_switch": null,
  "meta": {"source": "audiobook", "dnsmos": 3.4, "align_conf": 0.97}
}
```

`codes` 含边界真实静音帧（≤600ms）；collate 组装 `[前缀] ⊕ Σ_k(<t>text_k[|lookahead]<eot> ⊕ codes_k ⊕ <bnd>)`，loss 仅施于 codes。

## 附录 B：与官方仓库改动清单

| 位置 | 改动 |
|---|---|
| `qwen_tts/core/models/modeling_qwen3_tts.py` / `qwen_tts/inference/qwen3_tts_model.py` | 只读；导出 CustomVoice 前缀模板对拍 |
| 新建 `steadystream/session.py` | C1/C2/C3 开关化推理引擎（§2） |
| 新建 `steadystream/pause_corrector.py` | C3 推理端校正器 |
| `finetuning/prepare_data.py` | 小改：整段抽码 + 时间戳切分 + pause_ms/punct_class 落盘 |
| `finetuning/` dataset/collate/processor | 重写：前缀 + 多段序列 + `<bnd>` + 增强（C4） |
| `finetuning/sft_12hz.py` | 小改：loss mask 开关，保留 speaker_name |
| 新建 `eval/` | `boundary_f0_energy.py`、`pause_dev.py`、`sim_delta.py`、`fasl.py`、`cer.py`、`natural_stats.py`（含表格自动汇总脚本 `make_table2.py`） |

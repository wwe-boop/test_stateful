# SteadyStream 表 2 进度汇报

日期：2026-07-08

范围：只覆盖表 2，当前评测集为 `test-prosody-mini`，规模为 3 seeds x 50 samples。表 3/表 4 不纳入本文档。

## 1. 一句话结论

表 2 当前已完成前 5 行的可运行实现、三种子音频生成、指标后处理和 CER 合并；新增的“完整可运行 SteadyStream 组合行”也已经完成 150/150 条音频生成、ASR/CER 合并和新版交付表生成。

2026-07-09 最新进展：C2 的 RoPE `position_offset`、terminal tail trim、`token_counts` 默认 reset、三档 `kv_terminal_drop_mode`、bounded token-history re-prefill 诊断均已落地并推送；随后已重编并临时部署 Phase B profile512 诊断引擎，验证完整 token-history re-prefill 不再受 `max_input_len=128` 卡住。用户复核后发现旧 profile512 `prosody_mini_002` smoke 存在实验设置污染：8 个设计子句被 gateway/driver 合并成 2 个 engine segment，7 个边界指标主要来自 `proxy_proportional` 假边界；同时 `eos_only` 的漏尾混入了 512 slot 容量截断。因此旧 `29.04%/36.42%` 只能作为污染诊断记录，不能作为 C2/C4 结论。

2026-07-09 修正实验设置后：已新增评测专用 `force_text_chunk_boundary=true`，在 `input_mode=token` 下强制每个上游 `TextChunk` 作为独立 engine segment，并让表 2 runner 默认拒绝 proxy 边界。单样本 `prosody_mini_002` 重跑已恢复 `exact_boundary_count=7/7`；CER 为 `stateful_stream=5.17%`、`acoustic_tail_only=6.03%`、`kv_tail_only=243.97%`、`tail_kv_pause_recovery=255.17%`、`full_steadystream=260.34%`。新结论是：分段设置修正后，C1/stateful 基线有效，KV/Full SteadyStream 的主要失败形态变为严重拖长和历史复读，而不是旧报告里的“29% 漏尾”口径。

需要特别说明：计划中真正的“完整 SteadyStream”定义为 C1+C2+C3+C4，其中 C4 是 continuation 后训练 checkpoint。当前没有找到真实 C4 continuation 训练 checkpoint，因此我没有把基座引擎或 smoke adapter 冒充成最终 C4 行；当前补测行标注为“C1+C2+C3，C4 未训练”。

## 2. 表 2 逐行状态

| 行 | 变体 | 当前状态 | 已有 CER | 汇报判断 |
|---:|---|---|---:|---|
| 1 | 无状态 | 已完成，3 seeds x 50 | 11.13% +/- 0.09% | 基线成立，可进当前交付表。 |
| 2 | 现有 stateful Triton | 已完成，3 seeds x 50 | 10.86% +/- 0.12% | 比无状态 CER 略好，边界 F0/能量明显改善。 |
| 3 | 仅声学尾 C1 prototype | 已完成，3 seeds x 50 | 11.40% +/- 0.85% | CER 接近基线，是目前最安全的组件原型。 |
| 4 | 仅 KV/token 尾 C2 prototype | P0 位置修复 + terminal drop + profile512 完整 token-history 诊断已落地，门禁仍失败 | 修复前 27.01% +/- 0.55%；最新 smoke 最好 29.04%；profile512 完整 token-history `eos_only=29.04%`、`silence=36.42%` | RoPE/早停/过裁/profile 容量都已量化；容量放开后仍漏尾/乱码，C2 仍不能进正式表。 |
| 5 | 尾 + KV + 暂停恢复 C1+C2+C3 prototype | 依赖修复后 C2 重跑；C3 口径仍待解耦 | 修复前 28.24% +/- 0.23% | 暂停指标被修好，但旧结果继承了 C2 的 CER 问题；不能作为最终 C3 结论。 |
| 6 | 完整 SteadyStream C1+C2+C3+C4 | C2 门禁 + C4 训练 checkpoint 均未完成 | - | 不能造数；正确顺序是 C2 token-history 形态达标，再训练/部署真实 C4。 |
| 6a | 当前可运行全开补测 C1+C2+C3，C4 未训练 | 音频 150/150 完成，ASR/CER 完成，delivery 已生成 | 24.36% +/- 0.47% | 用来交付“现有引擎能跑到什么程度”，不冒充训练后 C4。 |

## 3. 最新表 2 指标

产物位置：`workspace/table2_runs_impl_20260708/table2_current_full.md`

| 变体 | F0 raw↓ | 能量 raw↓ | 停顿↓ | SIM Δ↓ | FASL↓ | CER↓ |
|---|---:|---:|---:|---:|---:|---:|
| 无状态 | 6.88 +/- 0.14 st | 5.42 +/- 0.20 dB | 127.69 +/- 0.67 ms | 0.16 +/- 0.01 | 22.35 +/- 0.13 ms | 11.13% +/- 0.09% |
| 现有 stateful Triton | 3.39 +/- 0.13 st | 3.54 +/- 0.16 dB | 200.00 +/- 8.48 ms | 0.10 +/- 0.00 | 29.19 +/- 0.10 ms | 10.86% +/- 0.12% |
| C1 prototype | 3.57 +/- 0.42 st | 3.35 +/- 0.25 dB | 200.54 +/- 15.04 ms | 0.10 +/- 0.01 | 27.03 +/- 0.24 ms | 11.40% +/- 0.85% |
| C2 prototype | 3.07 +/- 0.12 st | 3.12 +/- 0.12 dB | 205.10 +/- 2.19 ms | 0.13 +/- 0.01 | 27.78 +/- 0.31 ms | 27.01% +/- 0.55% |
| C1+C2+C3 prototype | 4.00 +/- 0.93 st | 4.50 +/- 0.34 dB | 11.53 +/- 3.31 ms | 0.13 +/- 0.01 | 28.09 +/- 0.13 ms | 28.24% +/- 0.23% |
| 当前可运行全开 C1+C2+C3，C4 未训练 | 4.14 +/- 0.63 st | 4.80 +/- 0.14 dB | 9.80 +/- 3.33 ms | 0.14 +/- 0.00 | 28.38 +/- 1.65 ms | 24.36% +/- 0.47% |

注：上表 C2、C1+C2+C3、6a 为 P0 RoPE 位置修复前结果。修复后必须重新生成音频、ASR 和 CER，不能直接复用这些数字。

## 4. 当前主要问题

| 问题 | 现象 | 原因判断 | 处理策略 |
|---|---|---|---|
| C2/C3 CER 变差 | C2 到 27.01%，C1+C2+C3 到 28.24%；修复后 3 样本 smoke 仍在 29%-55% CER | 已定位五层问题：RoPE position 错位会破坏 compact KV；`pad_phase` 会一次丢掉 73%-95% 尾部帧造成严重复读；`eos_only/silence` 仍约 31% CER；profile128 bounded token-history 只能保留 37-42 帧 code tail；profile512 放入完整历史 codes 后仍漏尾/乱码 | `position_offset`、terminal drop mode、裁剪比例 metrics、token-history carry、profile512 诊断均已落地；下一步不是继续调 tail，而是对齐 C4 collate/reference 排布并判断是否需要 continuation 训练诊断。 |
| C3 只能修暂停 | 停顿从约 200 ms 降到约 10 ms，但 CER 没恢复 | pause recovery 是后处理/边界修正，不能补回已经缺失的文本 | 继续保留 C3 作为独立模块，但不把它宣传成解决语义完整性的手段。 |
| C3 停顿口径风险 | C3 停顿列非常好看 | 当前 pause recovery 目标表和指标期望来自同一硬编码表，存在循环论证风险 | 最终表需要把目标 pause 统计和指标测量解耦，并加 crossfade，避免只是在优化指标本身。 |
| 最终 C4 行未完成 | 没有 C4 checkpoint | `steadystream_plan_v2.md` 要的是多片段 continuation SFT，不是普通单句 speaker finetune；父目录训练代码可作为基础，但还要改 dataset/collate/loss mask | 已补 C4 manifest 校验、codes attach、batch dry-run、smoke train 工具；等真实数据或释放训练资源后推进。 |
| 当前引擎部署限制 | 0.6B smoke adapter 不能直接接入线上 custom-1.7b TensorRT fused engine | 线上是固定 TensorRT plan，不是可热插 LoRA 的 PyTorch runtime | smoke 只证明训练管线可走，不作为线上表 2 结果。 |
| 指标口径仍需备注 | F0/能量目前是 raw boundary jump，不是计划里的 natural-excess 终版口径 | 自然边界参考分布尚未完整冻结 | 当前交付表标注为 mini/prototype 口径；正式论文表需要 E0 冻结口径后重跑。 |

## 5. 我已完成的修改

| 类别 | 修改内容 | 结果 |
|---|---|---|
| 表 2运行开关 | 增加 `SessionConfig.experimental`，贯通 proto、gRPC 转换和 engine session config | 可以用统一 runner 切换 C1/C2/C3 原型。 |
| 引擎原型 | 在 `engine/backend/engine_loop.py` 增加 acoustic tail、bounded KV tail、pause recovery 相关实验路径 | 前 5 行可测，不再只停留在计划文档。 |
| 评测 runner | 扩展 `workspace/table2_timed_runner.py` 和相关脚本，支持 resume、三种子、变体标识和暂停恢复输出 | 已完成 750 行基础评测和 150 行全开补测。 |
| 后处理 | 增加/扩展 Table 2 postprocess、ASR manifest、CER merge、delivery generator | 已能生成当前表、诊断文档和交付 markdown。 |
| C4 准备 | 增加 continuation manifest validator、API synthetic smoke dataset、prepare_data codes attach、continuation batch dry-run、0.6B smoke train 脚本 | 已证明 C4 数据/训练 plumbing 能跑，但还不是最终 C4 模型。 |
| 完整行补测 | 新增 `scripts/python/run_table2_full_steadystream.py` | 150/150 条音频、ASR/CER 合并和 delivery 生成均已完成。 |
| C2 P0 修复 | 新增 `SlotKVState.position_offset`；`executor` 的 fused `position_ids` 改为 `position_offset + past_len`；SteadyStream carry 新增 `talker_logical_past_len`/`talker_position_offset`；段末 carry 支持 `pad_phase/eos_only/silence` 三档 terminal drop；`kv_inherit_token_counts` 默认改为 reset | 同时修复裁剪 KV tail 的 RoPE 逻辑位置合约，并把 over-trim 风险量化为 `dropped_tail_tokens/source_frames/ratio`。 |
| C2 token-history 诊断 | 新增 `kv_reprefill_token_history=true`：保存每步 `full_codec`，carry 上一段 text token ids + codes，按 TRT profile 构造 `prefix + history text + history codes tail` bounded re-prefill，当前段仍走 request/trailing streaming | 机制已通过单测并提交；profile128 3 样本 smoke 为 55.22% CER，profile512 完整 history 后最佳仍为 29.04% CER，证明容量不是唯一瓶颈。 |
| C2 回归测试 | 更新 `tests/unit/test_engine_loop_pipeline.py`，新增紧凑 KV tail position offset、terminal tail drop modes、drop mode config、默认 token_counts reset、显式继承兼容测试 | 目标测试已通过：32 passed, 1 warning。 |
| C4 真实数据 | 选定 WenetSpeech4TTS Premium 作为 C4 主数据源，完成 Premium_0 pilot 下载、MD5 校验、20 样本 manifest+codes 构建和 batch dry-run | 可用于 C4 扩容训练前置，但训练启动应等 C2 门禁通过。 |
| 版本管理 | 已配置 GitHub remote `github-test-stateful` 和专用 deploy key | 当前代码和本文档已持续推送到 GitHub 目标分支。 |

## 6. 当前交付与下一步

| 优先级 | 动作 | 当前结果 |
|---:|---|---|
| P0 | 为 6a 全开补测行生成 ASR manifest，复用已有 750 条 CER，只新增 150 条全开音频 ASR | 已完成，产物为 `table2_cer_full.json`。 |
| P0 | 合并 full-row CER 并重新生成交付表 | 已完成，产物为 `table2_current_full.md` 和 `table2_current_delivery_full_20260708.md`。 |
| P0 | 拉回本地 outputs，并提交/推送新增汇报与必要代码变更 | 已完成，本地目录为 `outputs/table2_current_delivery_full/`。 |
| P0 | 用 P0 C2 修复重跑 `kv_tail_only` 和 `tail_kv_pause_recovery` | 已完成 3 样本门禁 smoke：trim-tail 默认 384 为 65.57%/63.51% CER；tail sweep 最佳 `kv64` 为 30.19% CER；embedding replay re-prefill 为 50.69% CER；drop-mode 最佳 `silence` 为 31.05% CER；profile128 bounded token-history 为 55.22% CER；profile512 完整 token-history 最佳 `eos_only=29.04%`。均未达标。 |
| P0 | 如果修复后 C2 仍不达标，继续补 sink/window/token-history re-prefill 或 position-offset KV 细节 | 已新增 `kv_reprefill_history=true`、`kv_terminal_drop_mode`、`kv_reprefill_token_history=true` 并提交 `0179be1`、`b0512b2`、`102d688`、`5e02b87`；另完成 profile512 诊断引擎构建/部署。结论：over-trim 是 `pad_phase` 恶化主因，profile 容量也是 profile128 的主因之一，但二者都不是 C2 不达标的唯一主因；完整 text+codes 排布在未训练基座上仍不可靠。 |
| P1 | C2 门禁通过后，扩大 WenetSpeech4TTS Premium manifest，启动 continuation SFT | 真实 C4 checkpoint。 |
| P1 | 训练路径复用父目录官方 finetuning，但要改成多片段 continuation collate、history drop、lookahead 和 loss mask | LoRA 或全参 C4 checkpoint。 |

## 7. 当前命令状态

当前分支：`exp/steadystream-table4-20260707`

当前运行目录：`/home/zehan/workspace/Qwen3-TTS-Triton`

完整行音频输出：`workspace/table2_runs_impl_20260708`

完整交付表：`workspace/table2_runs_impl_20260708/table2_current_delivery_full_20260708.md`

完整行日志：`workspace/logs/table2_full_steadystream_20260708.log`

最新 C2 drop-mode smoke 输出：`workspace/table2_c2_dropmode_sweep_20260709`

本地已拉回：`outputs/table2_c2_dropmode_sweep_20260709/`

最新 C2 bounded token-history smoke 输出：`workspace/table2_c2_token_history_smoke_20260709`

本地已拉回：`outputs/table2_c2_token_history_smoke_20260709/`

最新 C2 profile512 完整 token-history smoke 输出：`workspace/table2_c2_token_history_profile512_smoke_20260709`

profile512 构建日志：`workspace/logs/build_fused_profile512_20260709.log`

诊断 runtime 当前状态：`workspace/model_repository/tts_orchestrator/1/runtime/model.plan` 已临时替换为 profile512，原 profile128 备份在 `workspace/backups/engine_profile_128_20260709/`，profile512 备份在 `workspace/backups/engine_profile_512_20260709/`。

截至本文档更新时，完整行音频生成、ASR/CER 合并和 delivery 生成均已完成；C2 门禁诊断已证明 profile512 放开容量后仍未过线，尚未批准 C4 正式训练。

## 8. 2026-07-09 C2 门禁复测结论

### 8.0 实验设置更正：token mode 必须逐 TextChunk 成段

用户复核 `prosody_mini_002` 后指出，旧 smoke 虽然指定了 token 输入，但 frontend 的 streaming driver 仍会按阈值合并短子句：8 个设计子句实际只形成 2 个 engine segment，事件流里多数边界是 `boundary_reason=flush_eos` 后的粗粒度提交，runner 再用 `proxy_proportional` 按文本比例补出了 7 个假边界。因此 F0/停顿和部分 CER 解释都被污染。

本轮已完成两个修正：

| 修正 | 文件 | 作用 |
|---|---|---|
| `force_text_chunk_boundary=true` | `engine/frontend/interface.py` | 评测专用开关；每个上游 `TextChunk` 走 `push_group_tokens()`，并发窗口满时进入 group queue，不再和后续子句合并。 |
| exact-boundary gate | `scripts/python/run_table2_full_steadystream.py` | 默认 `input_mode=token`、`group_policy=none`，并要求 `exact_boundary_count == len(segments)-1`；否则直接报错，不再产出 proxy 表 2 指标。 |

验证：`tests/unit/test_frontend_interface.py` 新增短 TextChunk 分段测试，目标单测共 `11 passed`；`prosody_mini_002` 单样本重跑中 `stateful_stream/acoustic_tail_only/kv_tail_only/tail_kv_pause_recovery/full_steadystream` 均为 `exact_boundary_count=7/7`。

修正后的单样本 CER/时长：

| 变体 | 时长 | CER | 观察 |
|---|---:|---:|---|
| `stateless_once` | 24.48s | 6.90% | 文本基本完整。 |
| `stateful_stream` | 24.64s | 5.17% | 文本基本完整，是当前修正 smoke 的强基线。 |
| `acoustic_tail_only` | 24.08s | 6.03% | 接近基线，说明声学尾不是 CER 暴涨主因。 |
| `kv_tail_only` | 77.68s | 243.97% | 大量历史复读，ASR 长度膨胀到 385 字符。 |
| `tail_kv_pause_recovery` | 89.13s | 255.17% | C3 只修停顿，不修 KV 复读。 |
| `full_steadystream` | 91.51s | 260.34% | 仍大量复读历史片段，当前不能作为完整 SteadyStream 行。 |

同一新口径下已补跑 `kv_tail_only` 的三档 `kv_terminal_drop_mode` 对照：

| drop mode | 时长 | CER | dropped tail | 判断 |
|---|---:|---:|---:|---|
| `eos_only` | 19.76s | 65.52% | 末段 1/22 帧 | 不再拖到 90s，但后半段局部乱码/漏读严重。 |
| `silence` | 20.72s | 65.52% | 末段 1/60 帧 | 与 `eos_only` 接近，说明只裁静音不能恢复语义。 |
| `pad_phase` | 92.24s | 258.62% | 末段 259/278 帧 | 明确 over-trim，严重诱发历史复读。 |

关键判断：旧 `29.04%/36.42%` 不是可靠的悲观结论；新口径下问题更清楚，`pad_phase` 过裁会把结果推向超长复读，但即使只丢 EOS 或尾部静音，C2 也停在 65.52% CER，仍远离 `stateful_stream=5.17%` 基线。因此当前卡点不是单纯边界裁剪，而是 KV 历史本身缺少可靠的当前文本语义约束，下一步应重跑 token-history/full-current 对照并收口 C4 collate/runtime 前缀一致性。

### 8.1 复测范围

本轮只验证 C2 是否能恢复文本完整性，不更新正式表 2 数字。所有结果均为 `seed=42`、`prosody_mini_001..003` 的 3 样本 smoke，用同一个 `workspace/table2_asr_batch.py` Paraformer/CER 口径。

| 产物 | 路径 | 说明 |
|---|---|---|
| terminal-tail trim 默认 384 | `workspace/table2_c2fix_p0_trim_tail_smoke_20260708/table2_cer.json` | 已消除 3 step 早停，但 384 tail 严重复读。 |
| tail 长度 sweep | `workspace/table2_c2_tail_sweep_20260708/table2_cer.json` | 补齐 `kv16/32/64/128` x 3 样本。 |
| embedding replay re-prefill | `workspace/table2_c2_reprefill_smoke_20260709/table2_cer.json` | 新增 `kv_reprefill_history=true` 诊断路径，已提交 `0179be1`。 |
| terminal drop mode sweep | `workspace/table2_c2_dropmode_sweep_20260709/table2_cer.json` | 新增 `kv_terminal_drop_mode=eos_only/silence/pad_phase` 三档对照，已提交 `b0512b2`。 |
| bounded token-history re-prefill | `workspace/table2_c2_token_history_smoke_20260709/table2_cer.json` | 新增 `kv_reprefill_token_history=true`，受 `max_input_len=128` 限制，已提交 `102d688`、`5e02b87`。 |
| profile512 完整 token-history re-prefill | `workspace/table2_c2_token_history_profile512_smoke_20260709/table2_cer.json` | 重编 Phase B `max_input_len=512/max_batch=1`，实际记录 `hist_codes=222-254/222-254`，验证完整上一段 codes 已放入。 |

### 8.2 结果表

| 变体 | CER 均值 | 关键现象 | 判断 |
|---|---:|---|---|
| stateless smoke 基线 | 3.34% | 3 样本文本基本完整 | smoke 口径下的可接受参考。 |
| stateful smoke 基线 | 3.63% | 3 样本文本基本完整 | smoke 口径下的可接受参考。 |
| acoustic tail only | 4.21% | 接近基线 | C1 仍是安全组件。 |
| trim-tail `kv_tail_only` 384 | 65.57% | 大量复读上一段 | 不通过。 |
| trim-tail `tail_kv_pause_recovery` 384 | 63.51% | C3 修停顿但不修语义 | 不通过。 |
| tail sweep `kv16` | 34.35% | 小 tail 多次 overflow/漏末段 | 不通过。 |
| tail sweep `kv32` | 30.47% | 部分样本 overflow，末段缺失 | 不通过。 |
| tail sweep `kv64` | 30.19% | sweep 最佳但仍远离基线 | 不通过。 |
| tail sweep `kv128` | 57.50% | 不 overflow 但复读历史 | 不通过。 |
| embedding replay re-prefill | 50.69% | replay 触发且不 overflow，但复读历史 | 不通过。 |
| drop-mode `eos_only` | 31.62% | 只丢 EOS，保留全部真实历史和静音 | 明显优于 `pad_phase`，但仍不达标。 |
| drop-mode `silence` | 31.05% | 丢尾部连续静音+EOS，上限 24 帧 | 三档最佳，证明 over-trim 是 `pad_phase` 主因之一；但仍不达标。 |
| drop-mode `pad_phase` | 58.48% | 丢弃比例 73%-95%，音频变长到 36-45s | 明确过裁并诱发复读，不应作为定型策略。 |
| bounded token-history `silence` | 55.22% | `prefill_len=128`，上一段 text token 62/64/67，codes tail 42/40/37 帧 | 仍复读，不通过；说明当前受限 profile 不是可用 C2。 |
| profile512 token-history `eos_only` | 29.04% | `prefill_len` 放宽到 512，上一段 codes 完整放入；最佳但仍漏尾/乱码 | 不通过；容量不是唯一瓶颈。 |
| profile512 token-history `silence` | 36.42% | `hist_codes=222-235/222-235`，第 1 条第二段 `overflow=True` | 不通过；比 `eos_only` 差，定型不宜全裁尾部静音。 |
| profile512 full-current token-history `eos_only` | 43.51% | 真实 C4-style 诊断：`prefix + history text + history codes + current text + codec BOS` 全量 prefill 后再解当前 codes | 不通过；保留全部尾部静音时更容易拖长并复读。 |
| profile512 full-current token-history `silence` | 28.46% | 同上，但段末裁掉连续 pad-silence + EOS，上限 24 帧 | 不通过；是当前 full-current 最佳，但仍远离 3%-4% smoke baseline。 |

### 8.2.1 full-current 诊断新增结果

本轮新增 `kv_reprefill_token_history_full_current=true`，用于验证计划和评审要求的 C2/C4-style token 级 re-prefill。旧 profile512 token-history 形态是：

`prefix + history_text + history_codes + boundary_EOS` 进入 prefill，当前段文本仍通过 request prefill/trailing 走流式解码。

新 full-current 形态是：

`prefix + history_text + history_codes + boundary_EOS + current_text + current_codec_BOS` 一次性进入 prefill，然后只让模型续写当前段 audio codes。

服务端日志已确认完整历史和当前文本进入同一个 prefill：例如第 1/2/3 条 `silence` 分别记录 `hist_text=62/64/67`、`hist_codes=221/221, 238/238, 268/268`、`current_text=12/32/32`，`mode=full_current`。

| 变体 | prosody_001 | prosody_002 | prosody_003 | 均值 |
|---|---:|---:|---:|---:|
| full-current `eos_only` | 44.32% | 48.28% | 37.93% | 43.51% |
| full-current `silence` | 21.59% | 30.17% | 33.62% | 28.46% |

逐样本 ASR 观察：

| 样本 | `silence` 主要错误 | `eos_only` 主要错误 |
|---|---|---|
| prosody_001 | 前半段对齐，末尾把“整颗咖啡豆静泡十四小时”重复，漏掉“听起来很费时间，但香气层次反而更干净”。 | 中后段开始复读“有点甜/新批次/烘焙程度”，音频时长 28.96s，明显拖长。 |
| prosody_002 | 前半段对齐，后段“探索一号/海斗一号/下潜任务”退化成串词。 | 尾部出现大量无意义重复和数字/音节串，CER 最高。 |
| prosody_003 | 前半段对齐，后段从“酸梅汤”附近回跳到开头并串词。 | 后段复读“绕着西湖走了整整三圈”，随后进入乱码。 |

本地已拉回音频和 ASR 结果：`outputs/table2_c2_token_history_fullcurrent_profile512_smoke_20260709/`。建议听同一样本的两档 wav：`silence` 通常更短、更少拖尾；`eos_only` 更容易保留过长尾部上下文后诱发复读。需要说明的是，我无法像人耳一样主观试听音频，只能基于已拉回 wav、时长、ASR 文本和服务端日志做可复核判断；本地 wav 已可直接播放复听。

### 8.3 失败模式

1. `position_offset` + terminal tail trim 的确修掉了早停：第二段不再只有 3 个 audio steps。
2. tail sweep 证明问题不是简单 tail 长度。小 tail 容易 overflow 或漏后半句，大 tail 不 overflow 但复读上一段。
3. embedding replay re-prefill 真实触发：3 条样本的 `prefix_len=21`，`replay_len=62/64/67`，第二段均 `overflow=False`；但 ASR 仍显示上一段内容被重复插入。
4. drop-mode sweep 证明 `pad_phase` 是过裁：两个 carry 点通常丢掉 171-283 帧，占该段 72.7%-94.6%；`eos_only/silence` 多数只丢 1 帧，CER 从 58.48% 降到约 31%。
5. 但 `eos_only/silence` 仍没有回到 baseline +/-1pp，说明即使不过裁，当前 KV 尾巴也不是计划要求的“有文字语义约束的历史”。本质仍是模型拿到一段 codec/embedding 历史，却缺少上一段真实 text token 的可解释条件。
6. bounded token-history 已经补入上一段 text token，但受当前 TRT `max_input_len=128` 限制，只能保留 37-42 帧 codes tail；ASR 显示它仍会重复上一段中段内容，CER 55.22%。
7. profile512 诊断已解除容量限制：`eos_only` 三条实际为 `hist_codes=243/243、226/226、254/254`，`silence` 三条实际为 `234/234、235/235、222/222`，说明完整上一段 codes 已进入 re-prefill。
8. 解除容量限制后仍未恢复：`eos_only=29.04%`、`silence=36.42%`；失败形态从 profile128 的明显复读转为第二段漏尾、提前终止或局部乱码。样例：`eos_only` 第 2 条尾部出现“科大雾残...探索探索”，`silence` 第 1 条第二段 `overflow=True` 且只识别到“冷萃吗”附近。
9. full-current 诊断进一步把当前文本也放入 prefill，排布已更接近 C4：`prefix + text1 + codes1 + text2 + codec_BOS -> decode codes2`。但最佳仍只有 28.46% CER，说明未训练基座对这种跨段 text+codes 续写分布仍不稳定。
10. 因此当前实现已经达到“prefix + 历史 text + 历史 codes + 当前 text”的容量要求，但未达到“可用 C2 续写”的行为要求；下一步要对齐 C4 collate/reference 排布，确认训练格式和推理格式是否逐 token 一致。

### 8.3.1 当前对比方法的第一性原理

表 2 C2 要验证的不是“音频能不能拼起来”，而是“模型在续写当前段 audio codes 时，能否同时保留上一段声学状态和当前段文本语义”。所以对照必须逐层拆开：

| 对照 | 控制变量 | 它回答的问题 |
|---|---|---|
| `stateless_once` / `stateful_stream` | 不启用实验 KV/token carry | 基座和普通 stateful 在同一 3 样本上最低能到什么 CER，这是门禁参考线。 |
| `acoustic_tail_only` | 只传声学尾，不传 KV/token 历史 | 如果它接近 baseline，说明音色/声学尾不是 CER 暴涨主因。 |
| `pad_phase/eos_only/silence` | 只改变段末丢多少 codec 帧 | 判断是不是 terminal tail over-trim 导致历史被裁坏。结果 `pad_phase` 最差，说明过裁确实有害。 |
| bounded token-history profile128 | 加入上一段 text + codes，但受 128 profile 限制 | 判断“没有上一段文本 token”是不是主因，同时暴露 profile 容量是否够。结果只保留 37-42 帧 codes，容量不够。 |
| profile512 token-history | 放宽到 512，让上一段完整 codes 进入 prefill | 判断容量放开后是否恢复。结果仍 29%-36%，说明容量不是唯一问题。 |
| profile512 full-current token-history | 再把当前完整 text 放进同一个 prefill | 判断真正 C4-style 排布能否让未训练基座直接续写。结果最佳 28.46%，说明还需要 C4 collate 对拍/continuation 训练，而不是继续盲调 tail。 |

从第一性原理看，CER 变差来自两个可分离层面。第一层是工程形态错误：RoPE 位置、terminal over-trim、profile 容量都会把历史 KV 变成错误上下文；这些已经逐项修复或量化。第二层是分布不匹配：即使 token 序列形态已经接近 `text1+codes1+text2->codes2`，未经过 continuation SFT 的基座仍可能把历史 codes 当成要复读的目标，或者在当前 codes 解码中丢失后半段语义。因此 C2 仍未过 baseline +/-1pp 门禁，不能批准正式 C4 训练或把当前行写成完整 SteadyStream。

### 8.4 当前卡点和决策

当前卡点已经从“TRT profile 放不下完整历史”推进为“完整 text+codes re-prefill 仍不能让未训练基座稳定续写”。旧 KV carry、position-offset KV、terminal trim、tail sweep、embedding replay、drop-mode sweep、profile128 bounded token-history 和 profile512 完整 token-history 都不能让 CER 回到基线。下一步应先做 C4 collate/reference 排布对拍，确保训练样本、推理 re-prefill 和 loss mask 是同一种 token 序列；在这个对拍完成前，C2 应在表 2 中标注为未通过，不应启动正式 C4 训练。

对 C4 的决策保持不变：C4 continuation SFT 要模拟“正确的 C2 推理形态”。在 C2 仍会复读/漏读的情况下启动 C4，会把推理侧实现 bug 混进训练目标，风险高且不可解释。

## 8.5 2026-07-09 第 0 步：离线 HF 复现与排布对拍

根据 `steadystream_table2_next_steps_20260709.md`，本轮开始执行“先离线 HF 复现 + 训推排布对拍，再进入 C4 LoRA 诊断”的收口路径。新增两个只服务于 Table 2/C4 的诊断脚本：

| 脚本 | 作用 | 当前结果 |
|---|---|---|
| `scripts/python/check_c4_runtime_layout_parity.py` | 对拍 C4 collate 的 `prefix + text1 + codes1 + text2 + codec_BOS` 槽位顺序，并和 runtime full-current 日志 prefix 长度比较 | Wenet pilot 前 5 条 `overall_status=warn`：continuation 顺序自洽，无槽位错位；但 C4 collate prefix 长度为 8，runtime full-current `prefix_len=21`，训推前缀仍不一致。 |
| `scripts/python/run_c4_hf_continuation_generate.py` | 直接用本地 PyTorch Qwen3-TTS 基座生成 continuation：喂入第 1 段 text+codes 和第 2 段 text+codec_BOS，只让模型生成第 2 段 codes | 0.6B Base 单样本生成成功但严重失败：目标 40 帧，生成 95 帧仍无 EOS；ASR 为“嗯有没有据据据据据据据据据据有没有证据”，CER=225%。 |

同时复跑已有 `run_c4_forward_smoke.py`，确认官方 PyTorch 模型可以吃进同一个 C4 continuation batch：

| 项目 | 结果 |
|---|---|
| manifest | `workspace/c4_wenet_premium0_pilot_20260708/c4_wenet_manifest_with_codes.jsonl` |
| model | `workspace/hf_models/Qwen3-TTS-12Hz-0.6B-Base` |
| device | `cuda:1` |
| 输入 | 1 条 Wenet pilot，前 2 段 |
| batch shape | `[1, 194, 2]`，实际 sequence length 100 |
| codec frames / loss positions | 72 / 74 |
| combined loss | 15.897184 |
| 显存 | reserved 约 2.35GB |

离线生成产物：`workspace/c4_wenet_premium0_pilot_20260708/hf_06b_continuation_generate_limit1_seg2/`；本地已拉回到 `outputs/c4_wenet_premium0_pilot_20260708_hf_06b_continuation_generate_limit1_seg2/`。生成音频 `hf_continuation.wav` 时长 7.6s，参考文本只有“笑得有些阴森研默，”8 个归一化字符，ASR 出现重复“据”字串。复核后将该单样本 HF 生成降级为旁证：当次脚本使用贪心解码、无单段/官方 API 对照，不能单独证明“基座无法消费续写排布”；真正不受采样污染的硬证据以后续 E10 teacher-forcing NLL 为准。

当前第 0 步结论：

1. C4 continuation batch 的内部顺序是对的：`current_codec_bos` 均与 `text1 + codes1 + text2` 推导位置一致。
2. 训推仍有一个必须收口的差异：训练 collate 的 prefix 是官方 finetune 8 槽结构，runtime full-current 的 cacheable prefix 是 21 槽结构。进入 LoRA 诊断前，应统一为同一个前缀构造函数或明确证明两者等价。
3. 离线 HF 生成当次表现为拖长、复读/乱码，但因使用贪心解码且缺少对照组，只能作为问题现象记录；不能作为 C4 必要性的独立证据。
4. 下一步优先级不是继续调 tail，而是先用 teacher-forcing NLL 建立无采样污染证据，再把 C4 collate 和 runtime full-current 共用前缀构造，最后做最小 LoRA/训练诊断，看 continuation CER 是否相对 E9 的 42%/65% 显著回落。

---

## 9. 2026-07-09 E9 容量预算修复与 full-current 复测

`24d181e` 新增 token-history 生成预算：在构造 `prefix + history_text + history_codes + current_text + codec_BOS` 时，先按 `当前文本 token 数 × 4.0 frames/token × 1.6` 预留当前段 decode 帧，再让历史 codes 使用剩余窗口；同时把 `generation_budget_frames`、`max_seq_len`、`trimmed_for_generation_budget` 写入 prefill metrics。目标单测已通过：`40 passed`。

在 E8 分段修复 + E9 容量预算后，重跑 `prosody_mini_001..003` 的 full-current token-history 3 样本 smoke：

| 变体 | CER 均值 | 时长现象 | overflow | 判断 |
|---|---:|---|---:|---|
| `full_current_silence` | 41.95% | 25.28s / 63.68s / 30.64s | 0/3 | 比 raw-KV 65%/259% 更可解释，但仍远离 3%-6% 基线。 |
| `full_current_eos_only` | 69.19% | 41.92s / 64.16s / 49.44s | 0/3 | 保留完整尾部更容易拖长和复读。 |

逐样本观察：`prosody_001` silence 仍复读开头一句但整体可读，CER 25.00%；`prosody_003` silence 较好，CER 16.38%；`prosody_002` silence 仍严重漏读/重复，CER 84.48%。所有样本 exact boundary 均有效，且本轮没有 overflow，因此旧“512 slot 截断”污染已经从这组实验中排除。

需要注意：本次 3 样本里 `trimmed_for_generation_budget` 均为 0，说明 E8 逐子句分段后单段历史通常没有挤满 512；E9 的价值是防止长历史/长子句再次污染实验，而不是解释本轮所有失败。当前失败仍主要是模型行为问题：即使上一段 text+codes 和当前 text 都进入 full-current prefill，未训练基座仍会复读、串词或漏读，C2 门禁继续失败。

下一步顺序不变但更明确：先做 teacher-forcing NLL 对比，判断真值 codes 在 continuation prefill 下的 NLL 是否显著劣于单段 prefill；再按官方采样配置重跑 HF 诊断。若 NLL 也差，C4 LoRA 的必要性才有干净证据；若 NLL 不差，问题更可能在采样/解码策略而非训练需求。

---

## 10. 2026-07-09 Teacher-Forcing NLL Pilot

按 E7 review §2.3-5，新增 `scripts/python/run_c4_teacher_forcing_nll.py`，在官方 PyTorch Qwen3-TTS 上比较同一目标段真值 codec_0 在两种条件下的逐帧 NLL：

| 条件 | 输入排布 | 统计目标 |
|---|---|---|
| `single_segment` | `prefix + current_text + current_codec_BOS + current_codes` | 当前段 codec_0 真值帧 + boundary EOS |
| `continuation_prefill` | `prefix + history_text + history_codes + boundary_EOS + current_text + current_codec_BOS + current_codes` | 同一个当前段 codec_0 真值帧 + boundary EOS |

为避免 sampling/ASR 干扰，本实验不生成音频，只对 teacher-forced logits 做 cross entropy。single 与 continuation 共享同一模型、同一 speaker embedding、同一目标段 codes。

当前先做 Wenet pilot smoke，尚未达到正式门槛 `N>=20` 且目标文本 `>=20` 字，不能替代最终 NLL 结论；但两组 pilot 信号一致：

| pilot | 样本 | single NLL | continuation NLL | ΔNLL | worse count |
|---|---:|---:|---:|---:|---:|
| target segment 1, min 12 chars | 6 | 1.3479 | 1.9922 | +0.6443 | 6/6 |
| target segment 2, min 12 chars | 5 | 1.1111 | 1.8511 | +0.7400 | 5/5 |

初步判断：在不采样、不解码的 teacher-forcing 条件下，continuation prefill 已经让真值 codec_0 概率显著变差；这支持“当前 continuation 排布超出未训练基座分布”的方向性判断，也解释了 E9 full-current 仍复读/串词。但由于 pilot 文本偏短，下一步需要扩充 manifest，按 review 要求做 `N>=20`、目标文本 `>=20` 字的正式 NLL 分布，再把它作为是否启动 C4 LoRA 的硬证据。

产物：

| 文件 | 说明 |
|---|---|
| `workspace/c4_wenet_premium0_pilot_20260708/teacher_forcing_nll_pilot_seg2_min12_limit6.json` | target segment 1 pilot，6 条。 |
| `workspace/c4_wenet_premium0_pilot_20260708/teacher_forcing_nll_pilot_seg3_min12_limit5.json` | target segment 2 pilot，5 条。 |

---

## 11. 2026-07-09 E10 正式 Teacher-Forcing NLL 门禁

按最新 review 要求，将 teacher-forcing NLL 从 pilot 扩到正式门槛：`N=20`、`target_segment_index=2`、目标文本 `>=20` 字。数据仍来自 WenetSpeech4TTS Premium_0，同一切分规则，窗口扩大到 1000 后筛选前 20 条合格样本；每条保留前 3 段，目标段为第 3 段，因此 continuation 条件包含两段真实历史 text+codes。

新增辅助脚本 `scripts/python/expand_c4_manifest_for_prepare.py`，只做 sample-level C4 manifest 到 segment-level `prepare_data.py` 输入的格式展开，并把筛选口径写入 manifest meta。随后复用官方 `../Qwen3-TTS/finetuning/prepare_data.py` 提取 60 段 audio codes，再 attach 回 20 条 continuation manifest。

| 项目 | 结果 |
|---|---|
| 源 manifest | `workspace/c4_wenet_premium0_nll20_20260709/c4_wenet_manifest_1000.jsonl` |
| 正式 manifest | `workspace/c4_wenet_premium0_nll20_20260709/c4_wenet_manifest_idx2_min20_limit20_with_codes.jsonl` |
| NLL summary | `workspace/c4_wenet_premium0_nll20_20260709/teacher_forcing_nll_idx2_min20_limit20.json` |
| 样本数 | 20 |
| target segment | index 2，第 3 段 |
| min target chars | 20 |
| single NLL mean | 1.180001 |
| continuation NLL mean | 1.882413 |
| ΔNLL mean | +0.702412 |
| continuation worse count | 19/20 |
| ratio mean | 1.600726 |

判读：这组结果已经满足 review 的 `N>=20` 和目标文本 `>=20` 字门槛。因为它是 teacher-forcing，对同一目标段真值 codec_0 计算 NLL，不经过采样、声码器和 ASR，所以排除了“CER 归一化/ASR 听错/采样偶然性”作为主要解释。continuation prefill 使真值 codes 的平均 NLL 上升约 60%，且 19/20 更差，说明未训练 0.6B base 在 `history_text + history_codes + current_text` 的 C4 续写排布上确实处于分布外。

这也解释了 E8/E9 后 full-current 仍复读、串词、漏读：工程侧的分段和容量污染已被修掉，但模型本身没有学会“带历史音频 token 的下一段 codec 续写”。因此 C2 门禁仍不应直接宣布通过；下一步可以启动最小 C4 LoRA 诊断，但训练目标必须使用与 runtime full-current 一致的 prefix/continuation 排布，并在训练前继续保留 E0/ASR 口径冻结问题作为最终表 2 风险项。

---

## 12. 2026-07-09 E11 C4 Continuation 训练可学习性 Smoke

在 E10 正式 NLL 证明 base 不会稳定消费 continuation 排布后，新增 `scripts/python/train_c4_continuation_smoke.py` 做最小 C4 teachability 诊断。该脚本不是正式训练 recipe，不保存 checkpoint；它复用已验证的 C4 batch 排布，在同一 20 条 Wenet continuation 样本上做少量全参 step，并比较训练前/训练后 target segment teacher-forcing NLL。

关键实现口径：历史段 `text+codes` 只作为上下文进入 attention；默认 `loss_scope=target`，只在当前目标段 codec frames + boundary EOS 上打 `codec_0_labels` 和 residual-code `codec_mask`。这比把历史段也纳入 loss 更符合 C2/C4 续写第一性原理：推理时历史已经发生，训练目标应是“给定历史，预测下一段”。

| smoke | loss scope | lr | epochs | eval N | before target NLL | after target NLL | Δ after-before | 判断 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 全段 loss | all | 2e-5 | 1 | 5 | 1.825440 | 4.280673 | +2.455233 | 高学习率下爆炸；不能据此否定 all-loss。 |
| target-only | target | 2e-5 | 1 | 5 | 1.825440 | 4.229811 | +2.404371 | 学习率过大，破坏 base 分布。 |
| target-only | target | 1e-6 | 1 | 5 | 1.825440 | 1.817586 | -0.007854 | 稳但基本不动。 |
| target-only | target | 5e-6 | 1 | 20 | 1.882413 | 1.669300 | -0.213113 | 当前最佳 smoke：C4 目标可优化。 |
| target-only | target | 5e-6 | 3 | 5 | 1.825440 | 1.737562 | -0.087878 | 多 epoch 没继续改善，tiny 全参有漂移/过拟合风险。 |

判读：C4 continuation 目标不是不可学，`target-only + 5e-6 + 1 epoch` 已在全 20 条正式 NLL 样本上把 target NLL 拉低约 0.21。但这还不能作为表 2 C4 交付数字，因为它没有保存/评估可生成 checkpoint，也没有跑 CER/音频；同时全参小数据训练对学习率非常敏感，高 LR 会快速破坏原模型分布。下一步应补 LoRA/冻结策略或使用官方训练框架改造后的 C4 collate，在更大 continuation 数据上训练，再回到 E8/E9 的 corrected token-mode runner 测 C2/C4 CER。


---

## 13. 2026-07-09 E12 Held-out C4 Teachability 复核

根据最新 review，对 E11 做两项修正：第一，E11 的 `-0.213` 是同 20 条训练集上的 NLL 改善，只能说明可记忆，不能说明泛化；第二，`all-loss` 只在 `2e-5` 高学习率下测过，不能得出“全段 loss 会伤害目标段”的结论。因此将 `train_c4_continuation_smoke.py` 增加 `--eval-offset`，用同一正式 manifest 做 `前 15 条训练 / 后 5 条 held-out 评估`，并在 `5e-6` 下补齐 target-only 与 all-loss 对照。

| held-out smoke | train N | eval offset/N | loss scope | lr | epochs | before held-out NLL | after held-out NLL | Δ after-before | 判断 |
|---|---:|---:|---|---:|---:|---:|---:|---:|---|
| C4 smoke | 15 | 15 / 5 | target-only | 5e-6 | 1 | 1.489866 | 1.287174 | -0.202692 | held-out 也改善，说明不只是训练集记忆。 |
| C4 smoke | 15 | 15 / 5 | all-loss | 5e-6 | 1 | 1.489866 | 1.317911 | -0.171955 | all-loss 同样正向，略弱于 target-only；不能排除。 |

逐样本看，5 条 held-out 中 4 条 NLL 下降、1 条轻微上升；target-only 与 all-loss 都主要改善了原始 NLL 最高的样本。当前结论修正为：C4 continuation 目标具备初步 held-out teachability；`5e-6` 是比 `1e-6`/`2e-5` 更合理的全参 smoke 学习率；但正式 C4 训练不应过早锁死 target-only。下一步应按计划做 LoRA/冻结策略、放量 Wenet continuation 数据，并把 all-loss + history drop 与 target-only 作为训练消融，而不是把 E11 的高 LR 爆炸归因给 loss scope。

同时，§8.5 的单样本 HF 生成结论已降级：那次贪心生成只能说明诊断现象，不能单独作为“基座不会 continuation”的证据；当前 C4 必要性的硬依据是 E10 正式 teacher-forcing NLL（N=20, ΔNLL +0.702, 19/20 更差）和 E9 within-engine corrected smoke。

---

## 14. 2026-07-09 E13 Held-out Single-NLL Gap 闭合率

根据最新 review §9.2，补报同 5 条 held-out 样本的 single-segment NLL，并把 C4 smoke 进度改写为 `continuation NLL - single NLL` 的 gap 闭合率。该统计直接复用 E10 正式 NLL JSON 与 E12 held-out smoke JSON，无需重新跑模型。

| held-out 口径 | single NLL | continuation before | continuation after | 原始 gap | 训后 gap | gap 闭合率 |
|---|---:|---:|---:|---:|---:|---:|
| target-only, lr=5e-6, 15 train / 5 held-out | 1.017207 | 1.489866 | 1.287174 | 0.472659 | 0.269967 | 42.88% |
| all-loss, lr=5e-6, 15 train / 5 held-out | 1.017207 | 1.489866 | 1.317911 | 0.472659 | 0.300704 | 36.38% |

判读：held-out 单段基线不是 E10 全 20 条的 1.180，而是更低的 1.017；因此 E12 的 `1.287` 还没有“接近单段基线”，但已经在 held-out 上闭合约 43% 的 continuation gap。all-loss 也闭合约 36%，仍应保留为正式 C4 训练消融项。后续 headline 指标统一改为 gap 闭合率，目标是让 held-out continuation NLL 接近同样本 single NLL，而不是只看绝对 NLL 是否下降。

下一步执行口径同步更新：正式 20 条 `workspace/c4_wenet_premium0_nll20_20260709/c4_wenet_manifest_idx2_min20_limit20_with_codes.jsonl` 冻结为常设 NLL eval，不再进入训练；训练数据应从 `c4_wenet_manifest_1000.jsonl` 的其余窗口另抽，放量后用 LoRA/冻结策略训练，并继续报告 held-out gap 闭合率。

---

## 15. 2026-07-09 E14 LoRA 放量首轮 NLL Gap

按 review §9.3/§10，正式 20 条 eval 已冻结不入训；从 `c4_wenet_manifest_1000.jsonl` 另抽 200 条训练样本（`target_segment_index=2`、目标段 ≥12 字、排除正式 eval sample_id），用官方 tokenizer 提取 600 段 codes 后得到：

| 文件 | 说明 |
|---|---|
| `workspace/c4_wenet_premium0_nll20_20260709/c4_wenet_train_idx2_min12_limit200_with_codes.jsonl` | 200 条 C4 LoRA train，和正式 20 eval overlap=0。 |
| `scripts/python/train_c4_lora_gap.py` | PEFT LoRA 训练 + 冻结 eval single/continuation NLL gap 统计脚本。 |

LoRA 目标模块先用 `q_proj,v_proj`，评估固定为正式 20 条 NLL eval。核心门禁不只看 gap 闭合率，还同时看 single NLL 是否退化；否则会出现“single 被打坏导致 gap 假闭合”。

| LoRA smoke | train N | r | lr | loss scope | single before→after | continuation before→after | gap after | gap 闭合率 | 判断 |
|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| train20 r8 | 20 | 8 | 5e-5 | target | 1.180→1.200 | 1.882→1.697 | 0.497 | 29.25% | 路径可跑，single 基本不退。 |
| train200 r32 | 200 | 32 | 5e-6 | target | 1.180→1.783 | 1.882→1.978 | 0.195 | 72.18% | gap 大幅闭合但 single 明显退化，不可直接作为候选。 |
| train200 r8 | 200 | 8 | 5e-5 | target | 1.180→4.691 | 1.882→4.689 | -0.002 | 100.27% | 假闭合：整体 NLL 崩坏。 |
| train200 r8 | 200 | 8 | 5e-6 | target | 1.180→1.204 | 1.882→1.695 | 0.491 | 30.12% | 稳但改善有限。 |
| train200 r8 | 200 | 8 | 5e-6 | all-loss | 1.180→1.220 | 1.882→1.609 | 0.389 | 44.63% | 当前最佳健康档：single 小退化，gap 闭合更好。 |

判读：LoRA 路径已经跑通，且 200 条训练在冻结正式 eval 上能稳定改善 continuation gap。当前最可用配置是 `r=8, lr=5e-6, all-loss, q_proj/v_proj`，比 target-only 更好，支持 review 里“all-loss + history/drop 不能排除”的判断。`r32` 或较大学习率会造成 single NLL 退化，说明下一阶段门禁必须同时约束：`gap 闭合率上升` + `single NLL 不明显退化`。下一步建议在当前健康档上加大数据量并实现 history drop/lookahead，同时加入 10% single/replay 防遗忘，再考虑 PyTorch 离线生成 CER。

---

## 16. 2026-07-09 E15 LoRA 数据放量与早停曲线

按 E14 review，固定健康基线 `r=8, lr=5e-6, all-loss, q_proj/v_proj`，不再继续调 rank/LR；先做数据量和早停曲线。将 Wenet Premium_0 manifest 扩到 3000 windows，从中筛选 `target_segment_index=2`、目标段 ≥12 字、排除正式 20 eval 的 1000 条训练样本，并提取 3000 段 audio codes：

| 文件 | 说明 |
|---|---|
| `workspace/c4_wenet_premium0_nll20_20260709/c4_wenet_manifest_3000.jsonl` | 3000 window pool。 |
| `workspace/c4_wenet_premium0_nll20_20260709/c4_wenet_train_idx2_min12_limit1000_with_codes.jsonl` | 1000 条 C4 train，和正式 eval overlap=0。 |

新增 `train_c4_lora_gap.py --max-steps` 支持早停曲线；同时加 `--single-replay-ratio` 做 10% single replay 防遗忘试验。结果如下：

| 配置 | steps | replay | single before→after | continuation before→after | gap after | gap 闭合率 | 判定 |
|---|---:|---:|---:|---:|---:|---:|---|
| 200 train baseline | 200 | 0 | 1.180→1.220 | 1.882→1.609 | 0.389 | 44.63% | E14 健康基线。 |
| 1000 pool early stop | 200 | 0 | 1.180→1.213 | 1.882→1.594 | 0.381 | 45.70% | 严格 single 预算内，略优于 E14。 |
| 1000 pool early stop | 250 | 0 | 1.180→1.255 | 1.882→1.563 | 0.308 | 56.12% | 更高闭合，但 single +0.075，略超 ≤0.05 预算。 |
| 1000 pool early stop | 250 | 10% | 1.180→1.251 | 1.882→1.589 | 0.338 | 51.89% | replay10 对 single 仅小幅帮助，换来 closure 下降。 |
| 1000 pool early stop | 350 | 0 | 1.180→1.410 | 1.882→1.605 | 0.195 | 72.20% | closure 高但 single 退化明显，不健康。 |
| 1000 pool early stop | 350 | 10% | 1.180→1.403 | 1.882→1.596 | 0.192 | 72.60% | replay10 未解决 350-step single 退化。 |
| 1000 pool full epoch | 1000 | 0 | 1.180→4.007 | 1.882→4.039 | 0.032 | 95.37% | 假闭合，整体 NLL 崩坏。 |

判读：数据放量有效，但训练步数是主要风险。`1000 pool + 200 steps` 是当前严格健康档，single 只退化 +0.033 且 gap 闭合 45.70%；`250 steps` 可作为激进候选，closure 到 56.12%，但 single 预算略超。10% single replay 在当前随机替换实现下帮助有限，说明防遗忘可能需要更强的 replay 比例、显式 single batch 调度或学习率/warmup/decay，而不是简单随机 10%。下一步优先不是继续拉长训练，而是用 `step200/250` 两个 adapter 做 PyTorch 离线生成 CER 小样本校准，看 NLL closure 对 CER 是否有实际收益；若 CER 有回落，再加 history drop/lookahead 与更稳的 replay 调度。

---

## 17. 2026-07-09 E16 C4 LoRA 生成级 Smoke 与音频回听包

按 E15 结论，不再只看 teacher-forcing NLL gap，而是把两个候选 LoRA adapter 放到 PyTorch 生成路径里做小样本 CER 校准。新增脚本 `scripts/python/run_c4_lora_continuation_generate.py`，输入仍是冻结的 20 条正式 eval manifest；本次只取前 3 条，排布为 `history text + history codes + current text + codec_BOS`，并使用官方采样参数 `top_k=50, top_p=1.0, temperature=0.9, repetition_penalty=1.05`，同时打开 `--subtalker-dosample`。为了判断 LoRA 是否真实改善，还增加同条件 `c4_base` 对照。

| 生成 smoke | adapter | NLL 侧状态 | 生成帧 / 目标帧 | 音频时长 | CER 均值 | 逐条失败形态 |
|---|---|---|---|---|---:|---|
| `c4_base` | 无 | E10 continuation OOD | 95/64, 102/71, 13/55 | 7.60s, 8.16s, 1.04s | 87.73% | 前两条长输出但历史串入/胡言，第三条短成“我走过去”。 |
| `c4_lora_step200` | `lora_train1000_r8_all_lr5e-6_step200` | single +0.033, gap closure 45.70% | 66/64, 71/71, 10/55 | 5.28s, 5.68s, 0.80s | 74.70% | 前两条长度达标但句首混入历史文本，第三条塌缩。 |
| `c4_lora_step250` | `lora_train1000_r8_all_lr5e-6_step250` | single +0.075, gap closure 56.12% | 39/64, 21/71, 10/55 | 3.12s, 1.68s, 0.80s | 69.55% | 第一条 CER 较好但明显短，第二/三条早停式塌缩。 |

逐条 ASR 关键信息：

| 样本 | reference | base hypothesis | step200 hypothesis | step250 hypothesis |
|---|---|---|---|---|
| 000 | 我点头看他平时做什么都好像一副玩世不恭的样子 | 他摇摇头等最后一起拿出来乌乌点头看他平时做什么都好像一副玩世不恭的样 | 他摇摇头等最后一起拿出来我我点头看他平时做什么都好像一副玩世不恭的样子 | 他摇摇头看他平时做什么都好像一副玩世不恭的样子 |
| 001 | 也不能将我们分开摊开的手掌上是两枚同心结 | 然后他回头看着我我是555 | 然后他回头看着我是那么懂将我们分开摊开德手掌上是两枚同心结 | 要算是轮回往生 |
| 002 | 伸手从袖子里拿出带出来的最后一点射魂香 | 我走过去 | 我走过去 | 我走过去 |

音频与 ASR 结果已拉回本地：

| 本地目录 | 内容 |
|---|---|
| `/Users/liuzehan/Documents/Codex/2026-07-08/ssh-4090-host-home-zehan-workspace/outputs/generate_c4_base_eval3/` | base continuation 3 条 wav、codes、`asr_cer.json`。 |
| `/Users/liuzehan/Documents/Codex/2026-07-08/ssh-4090-host-home-zehan-workspace/outputs/generate_c4_lora_step200_eval3/` | step200 LoRA continuation 3 条 wav、codes、`asr_cer.json`。 |
| `/Users/liuzehan/Documents/Codex/2026-07-08/ssh-4090-host-home-zehan-workspace/outputs/generate_c4_lora_step250_eval3/` | step250 LoRA continuation 3 条 wav、codes、`asr_cer.json`。 |

判读：LoRA 不是无效，`step200/250` 都比 base continuation 的 87.73% CER 有所下降，说明 C4 continuation 排布确实可被训练拉回一部分；但它还不能交付为完整 SteadyStream/C4 行，因为最佳小样本 CER 仍在约 70%，明显差于 corrected C2 smoke 的 41.95%，更远离 3%-6% 单段基线。失败原因从音频/ASR 形态看分成两类：第一，历史文本/音频仍会被插到当前句前面，说明模型还没学会“历史只作声学/韵律条件，不参与语义续写”；第二，部分样本出现短输出或固定短句，说明生成停止机制仍受 text-channel EOS / codec 无效步影响，NLL gap closure 还没有稳定转化为自回归生成质量。

下一步门禁保持不变：当前结果不批准启动表 2 完整 C4 大跑。应先在小样本上解决 `NLL 好但 generation 差` 的桥接问题，优先做三项：1）加入 C4 训练时的显式 current-text 对齐/历史 loss mask，降低历史语义串入；2）生成脚本增加 text-channel EOS、codec 有效码率和特殊码分布日志，定位为什么 `eos_step=-1` 仍提前停；3）在同一 3 条上补 `single current text -> codes` 生成对照，确认不是采样/声码器基础路径问题，再决定是否继续放大 LoRA 数据。

---

## 18. 2026-07-09 E17 生成评估装置复核与单段对照

根据新增 review §12，先不继续放大训练，而是审计 E16 的生成评估装置。两项最低成本裁决如下：

1. E16 三个 run 的 `summary.json` 均确认 `do_sample=true`、`subtalker_dosample=true`、`top_k=50`、`top_p=1.0`、`temperature=0.9`、`repetition_penalty=1.05`，因此 E16 不是“静默贪心”作废。
2. 样本 002 的历史段 1 文本就是“我走过去，”；三条件输出“我走过去”是明确的历史串入，不是随机 ASR 幻觉。

同时升级 `scripts/python/run_c4_lora_continuation_generate.py`：

| 修改 | 目的 |
|---|---|
| `--do-sample/--subtalker-dosample` 默认改为官方 True，并支持 `--no-do-sample` 显式关闭 | 防止漏传 flag 退回贪心。 |
| 增加 `--generation-mode single/continuation` | 同一手搓 prefill 管线直接跑单段健康对照。 |
| EOS 改用 `result.sequences` 判定，记录 `sequence_eos_step/raw_generated_steps/stop_reason` | 修复 hidden_states 看不到 EOS 导致 `eos_step=-1` 的问题。 |
| `--max-new-tokens` 显式记录；本轮统一设为 256 | 避免旧版 `target_frames+32` 真值长度泄漏和删失污染。 |

E17 重跑同 3 条样本，base 单段作为装置健康金标准；continuation 三组使用同一 `max_new_tokens=256`：

| 生成模式 | adapter | CER 均值 | 帧数 / 目标帧 | stop reason | 判读 |
|---|---|---:|---|---|---|
| single | 无 | 1.75% | 61/64, 64/71, 57/55 | 全部 `sequence_eos` | 装置、采样、声码器、ASR 口径健康；单段无灾难性遗忘。 |
| continuation | 无 | 97.88% | 99/64, 255/71, 13/55 | eos, max cap, eos | 放开 cap 后 base 更差，第二条撞 256；continuation 排布明显 OOD。 |
| continuation | step200 | 74.70% | 66/64, 71/71, 10/55 | 全部 `sequence_eos` | 与 E16 一致；LoRA 能减少 base 的长复读，但历史串入和第三条早停仍在。 |
| continuation | step250 | 69.55% | 39/64, 21/71, 10/55 | 全部 `sequence_eos` | CER 略低但靠短输出/删除换插入，不能视为真实达标。 |

逐条关键 ASR：

| 样本 | single base | continuation base | step200 | step250 |
|---|---|---|---|---|
| 000 | 完整正确 | 历史“他摇摇头...”+ 当前句 | 历史前缀 + 当前句 | 历史前缀 + 当前句，较短 |
| 001 | 完整正确 | 跑到 255 帧 cap，明显胡言/复读 | 历史前缀 + 当前句 | 早停成无关短句 |
| 002 | “摄魂香/射魂香”1 字差 | 直接输出历史段“我走过去” | 同左 | 同左 |

本地回听包已同步：

| 本地目录 | 内容 |
|---|---|
| `/Users/liuzehan/Documents/Codex/2026-07-08/ssh-4090-host-home-zehan-workspace/outputs/generate_c4_base_single_eval3_e17/` | base 单段对照 wav/codes/ASR。 |
| `/Users/liuzehan/Documents/Codex/2026-07-08/ssh-4090-host-home-zehan-workspace/outputs/generate_c4_base_cont_eval3_e17/` | base continuation wav/codes/ASR。 |
| `/Users/liuzehan/Documents/Codex/2026-07-08/ssh-4090-host-home-zehan-workspace/outputs/generate_c4_lora_step200_cont_eval3_e17/` | step200 continuation wav/codes/ASR。 |
| `/Users/liuzehan/Documents/Codex/2026-07-08/ssh-4090-host-home-zehan-workspace/outputs/generate_c4_lora_step250_cont_eval3_e17/` | step250 continuation wav/codes/ASR。 |

判读更新：E16 的“LoRA 有方向性改善但不达标”仍成立，且现在由单段健康对照加固；但 step250 的较低 CER 不能当成更好路线，因为它主要通过短输出减少插入。当前最可信结论是：0.6B base 单段生成健康，continuation 排布会触发历史语义串入；LoRA step200/250 只部分缓解，尚未压过 corrected C2 的 41.95% 门槛，更不能进入完整表 2 C4 大跑。

项目决策同步：后续新实验按 review §12.3 切到 `1.7B 基座 + input_mode=token + force_text_chunk_boundary + exact-boundary gate`；0.6B E10-E17 保留为方法论记录。下一步不再扩大 0.6B 训练，先确认线上同源 custom-1.7B PyTorch 权重位置，然后在 1.7B 上重建 NLL 门禁和生成装置单段/continuation 对照。

---

## 19. 2026-07-09 E18 1.7B CustomVoice NLL 门禁与生成装置降级

按 review §12.3，后续新实验切到线上同源 `1.7B + input_mode=token`。远端确认 `workspace/models/Qwen3-TTS-12Hz-1.7B-CustomVoice/` 是完整 HF/PyTorch 模型（约 4.3GB，含 `model.safetensors` 和 speech tokenizer），`workspace/exported/custom-1.7b/` 是 24GB TRT 导出包。该 1.7B custom 模型没有 `speaker_encoder`，speaker 槽应按引擎逻辑从 `talker_config.spk_id` 取 codec embedding；本轮已为 NLL/生成脚本增加兼容：0.6B 继续走 x-vector，1.7B custom 走 `spk_id`，默认 speaker=`serena`。

### 19.1 1.7B teacher-forcing NLL

| 模型 | N | single NLL | continuation NLL | Δ cont-single | continuation worse | 判读 |
|---|---:|---:|---:|---:|---:|---|
| 0.6B Base（E10） | 20 | 1.180 | 1.882 | +0.702 | 19/20 | 续写排布强 OOD。 |
| 1.7B CustomVoice | 3 | 7.837 | 7.306 | -0.531 | 0/3 | 小样本 continuation 反而更低，不可外推。 |
| 1.7B CustomVoice | 20 | 7.436 | 7.809 | +0.372 | 11/20 | 仍有 continuation gap，但明显弱于 0.6B。 |

判读：1.7B custom 上，continuation 排布不是 0.6B 那种“一边倒失败”；正式 20 条只有 11/20 更差，平均 gap 约为 0.6B 的一半。绝对 NLL 很高，原因是当前 eval 的真值 codes 来自 Wenet 公开音色，而 1.7B custom 的 speaker 槽固定为 `serena`，所以绝对值不能和 0.6B 直接比；当前只把 ΔNLL 作为相对门禁。

### 19.2 1.7B PyTorch 生成装置未过单段健康门禁

同 E17 生成脚本、官方采样、`max_new_tokens=256`、speaker=`serena`，跑前 3 条单段/continuation：

| 生成模式 | CER 均值 | 帧数 / 目标帧 | stop reason | ASR 形态 | 判读 |
|---|---:|---|---|---|---|
| single | 100.00% | 255/64, 255/71, 175/55 | max cap, max cap, eos | 三条基本空识别 | 单段金标准失败，装置不健康。 |
| continuation | 98.33% | 255/64, 15/71, 20/55 | max cap, eos, eos | 空识别或无关短句 | 不能用于判断 C4。 |

这和 0.6B 的 E17 不同：0.6B 同一手搓生成装置的 single CER 是 1.75%，可以作为健康对照；1.7B custom single 直接 100%，说明问题不在 C4 continuation 本身，而是 1.7B custom 的 PyTorch 手搓 prefill 路径没有复现线上/官方 CustomVoice 生成形态。可能原因包括：custom 模型依赖 speaker/language/instruct 的官方 runtime 前缀；Wenet ref/codes 与 custom speaker 分布不匹配；或 8 槽 C4 collate 前缀不适合直接驱动 custom 生成。

因此 E18 生成级结论必须降级：1.7B 目前只完成 teacher-forcing NLL 门禁，不能用当前 PyTorch hand-prefill 生成 CER 评价 C4。下一步改走两条更可信路径：

1. **runtime/token-mode 路径**：用已部署的 custom-1.7B TRT engine，在 `input_mode=token + force_text_chunk_boundary + exact-boundary gate` 下做单段/continuation smoke，这是表 2 的真实测量口径。
2. **官方 CustomVoice PyTorch 前缀路径**：若必须离线生成，先复现官方 `speaker/language/instruct` 单段生成到 CER 3%-6%，再把 C4 continuation 接入；单段不过线不得评 C4。

当前门禁：不扩大 1.7B LoRA，不导出新 TRT；先让 1.7B 生成装置通过单段健康门禁。

---

## 20. 2026-07-09 E19 1.7B Runtime Token-mode 单样本 Smoke

为绕开 E18 中 PyTorch hand-prefill 对 1.7B custom 生成不健康的问题，改用已部署的 custom-1.7B TRT engine 跑真实表 2 runtime 口径：`input_mode=token`、`stream_group_policy=none`、默认启用 `force_text_chunk_boundary=true`，样本为 `seed=42/prosody_mini_001`。

命令产物：`workspace/table2_1p7b_runtime_token_smoke_e19_20260709/`。该 run 成功生成 7 个 wav，并完成 Paraformer ASR/CER。

| variant | audio sec | exact boundaries | CER | 形态 |
|---|---:|---:|---:|---|
| `stateless_once` | 21.20 | concat exact | 1.14% | 单段/offline 口径健康。 |
| `stateful_stream` | 23.28 | 7/7 | 3.41% | runtime token-mode 健康。 |
| `acoustic_tail_only` | 22.64 | 7/7 | 2.27% | C1 声学尾健康。 |
| `offline_full` | 24.24 | n/a | 2.27% | offline 对照健康。 |
| `kv_tail_only` | 78.32 | 7/7 | 259.09% | 严重历史复读，hyp 314 chars vs ref 88。 |
| `tail_kv_pause_recovery` | 97.50 | event partially non-exact in summary | 270.45% | 更严重拖长/复读。 |
| `full_steadystream` | 80.59 | event partially non-exact in summary | 244.32% | Full 当前仍退化为 KV 历史复读。 |

这组结果非常关键：1.7B custom 的真实 runtime 单段/流式是健康的（1%-3% CER），所以 E18 的 PyTorch hand-prefill single=100% 不是模型本身坏，而是离线生成装置没有复现 custom runtime 前缀/推理形态。另一方面，runtime token-mode 下 KV/Full 仍 244%-270% CER，和 E8/E9 的“历史复读”诊断一致，说明当前线上 C2/C4-style history 仍不能直接交付表 2 Full 行。

判读：后续 1.7B 生成级验证应以 runtime/token-mode 为准；PyTorch 离线生成只有在先复现 runtime 单段 1%-3% CER 后才可恢复使用。当前优先级变为：在 runtime 侧继续做 C2/C4 诊断（如 full-current / drop-mode / history text mask），而不是继续用 Wenet codes 训练 1.7B LoRA 或导出新 TRT。

本地拉回状态：`table2_cer.json` 和部分 wav 已开始同步到 `/Users/liuzehan/Documents/Codex/2026-07-08/ssh-4090-host-home-zehan-workspace/outputs/table2_1p7b_runtime_token_smoke_e19_20260709/`；但 SSH 文件流在拉取 wav 时两次断开，远端完整产物仍在上述 workspace 目录。若需要人工回听，下一步单独用分块/base64 或重新建立稳定传输通道拉取剩余 wav。

---

## 21. 2026-07-09 E20 1.7B Runtime Full-current C2 诊断

按最新 review 要求，本轮不再用 PyTorch hand-prefill 评价 1.7B，而是在真实 TRT runtime/token-mode 上补 C2 诊断。核心修复与实验口径如下：

| 项 | 内容 | 目的 |
|---|---|---|
| 引擎新增 flag | `kv_reprefill_token_history_drop_history_text=true` | 允许 token-history re-prefill 时只保留历史 codes，不带历史 text token。 |
| runner 新增 flag | `--include-c2-diagnostics` | 在表 2 runner 里追加 C2 诊断变体，避免手工脚本口径漂移。 |
| 新变体 1 | `full_current_silence` | history text + history codes + current text，尾部按 silence 裁剪。 |
| 新变体 2 | `full_current_eos_only` | history text + history codes + current text，只丢 EOS，保留真实尾部静音。 |
| 新变体 3 | `full_current_codes_only_silence` | history codes + current text，不带 history text，验证“历史文本串入”是否主因。 |

命令产物：`workspace/table2_1p7b_runtime_c2diag3_e20_20260709/`。本轮跑 3 条样本、seed=42、`input_mode=token`、`stream_group_policy=none`、默认强制 text chunk boundary。Paraformer ASR/CER 汇总如下：

| variant | CER 均值 | 逐样本 CER | 判读 |
|---|---:|---|---|
| `stateless_once` | 4.40% | 1.14%, 6.90%, 5.17% | 单段基线健康。 |
| `stateful_stream` | 4.01% | 3.41%, 5.17%, 3.45% | 普通流式健康。 |
| `acoustic_tail_only` | 4.21% | 2.27%, 6.03%, 4.31% | C1 声学尾健康。 |
| `offline_full` | 4.78% | 2.27%, 9.48%, 2.59% | offline 对照健康。 |
| `kv_tail_only` | 286.08% | 259.09%, 243.97%, 355.17% | KV-only 严重历史复读。 |
| `tail_kv_pause_recovery` | 277.80% | 270.45%, 255.17%, 307.76% | pause recovery 不能救 KV 复读。 |
| `full_steadystream` | 276.55% | 244.32%, 260.34%, 325.00% | 现状 Full 仍退化为历史复读。 |
| `full_current_silence` | 49.69% | 56.82%, 49.14%, 43.10% | current text re-prefill 显著降低复读，但仍远高于 4% 基线。 |
| `full_current_eos_only` | 56.82% | 45.45%, 85.34%, 39.66% | 保留尾部静音不稳定，样本 002 明显变差。 |
| `full_current_codes_only_silence` | 73.26% | 78.41%, 71.55%, 69.83% | 去掉 history text 更差，主要表现为删字/漏尾。 |

关键结论：E20 基本确认“当前文本必须进入 token-history re-prefill”，它把 244%-286% 的灾难复读压到 50% 左右，是目前最有效的 runtime 侧杠杆。但这还不能作为完整 SteadyStream 表 2 结果交付，因为健康 runtime 对照只有约 4%，而 full-current 仍差一个数量级。`codes_only` 变体反而更差，说明问题不是简单的“历史 text token 串入”；如果把 history text 拿掉，模型会更容易丢当前文本，证明 current/history 的联合排布和 history codes 长度才是下一步要拆的变量。

当前卡点更新：完整 SteadyStream 还卡在 C2 history conditioning，不是模型/声码器/ASR 基线坏。1.7B runtime 正常单段、普通流式、C1 acoustic tail 都稳定在 4%-5%；坏的是 KV/Full 的历史条件如何进入自回归上下文。下一步建议在同一 3 条样本上做 `full_current_silence/eos_only` 的 history code tail sweep（例如 16/32/64/128/384），判断 384 帧历史是否过长导致复读；如果短尾能回到基线 ±1pp，再扩 10 条/全量；如果短尾仍在 30%-50%，再进入 C4 训练或更强 alignment 设计。

---

## 22. 2026-07-09 E21 History code tail sweep

按 E20 下一步，新增 runner 参数 `--c2-diagnostic-kv-tail-tokens`，对 full-current C2 诊断扫 history code tail 长度。该参数只影响 `--include-c2-diagnostics` 追加的三类诊断变体，并把输出 key/wav 自动加后缀，例如 `full_current_silence_tail64`，便于同一目录 resume 多档。产物目录：`workspace/table2_1p7b_runtime_tail_sweep_e21_20260709/`；ASR/CER 文件：`table2_cer_tail_sweep.json`。

本轮 3 样本、seed=42、`input_mode=token`、`stream_group_policy=none`，扫 `kv_tail_tokens=16/32/64/128`，并与 E20 的 384 结果对照：

| 变体 | tail16 | tail32 | tail64 | tail128 | tail384/E20 |
|---|---:|---:|---:|---:|---:|
| `full_current_silence` | 55.54% | 79.56% | **43.93%** | 64.00% | 49.69% |
| `full_current_eos_only` | 93.40% | **44.91%** | 74.33% | 94.75% | 56.82% |
| `full_current_codes_only_silence` | 85.38% | **63.53%** | 77.21% | 111.29% | 73.26% |

逐样本看，最佳两档仍不稳定：

| 变体 | prosody_mini_001 | prosody_mini_002 | prosody_mini_003 | 均值 |
|---|---:|---:|---:|---:|
| `full_current_silence_tail64` | 35.23% | 59.48% | 37.07% | 43.93% |
| `full_current_eos_only_tail32` | 4.55% | 67.24% | 62.93% | 44.91% |

判读：history code tail 长度确实影响生成，短尾可把 E20 的 50%-57% 进一步压到约 44%-45%，而且运行时长明显缩短，说明 384 帧长历史会加重长复读/长生成。但它没有解决根因：最好的 `silence_tail64` 仍比 4%-5% 健康 streaming baseline 高约 10 倍；`eos_only_tail32` 在样本 001 可到 4.55%，但样本 002/003 仍 60%+，说明这不是可定型策略。`codes_only` 全部劣于 text+codes，继续支持 E20 结论：简单去掉 history text 不是解法。

当前门禁结论：C2 还没有确认回到 baseline ±1pp，不能启动 C4 训练/完整 Full 表 2 大跑。下一步优先级应从“裁剪多少历史 codes”转向“history/current 的排布和对齐”：要么做 token-level text+codes re-prefill 的整句滑窗诊断，保证当前 text 在 decode 前最后、最强；要么进入 C4 训练前先构造与 runtime 完全同构的 collate/teacher-forcing gate。E21 还暴露了一个工程小问题：同一目录扫多档时旧版 resume 只保留基础变体 metadata，早期 tail wav 和 ASR 不丢，但 `results.json` 会只保留最新 tail metadata；已修复为 resume 时保留所有已有 variant key。

---

## 23. 2026-07-09 E23 Baseline / C1 / C3 同文本对比

按最新要求，用同一批文本重跑 baseline、C1、C3 对照，避免把 C2 tail sweep 的额外诊断混入口径。运行目录：`workspace/table2_baseline_c1_c3_same_text_e23_20260709/`；样本为 `prosody_mini_001-003`、seed=42、`input_mode=token`、`stream_group_policy=none`、默认强制 text chunk boundary。C3 采用当前表 2 runner 中已有定义：`tail_kv_pause_recovery`，即 C1+C2+C3 prototype，不是纯 pause-only 单独开关。

| 行 | variant | CER 均值 | prosody_mini_001 | prosody_mini_002 | prosody_mini_003 | 判读 |
|---|---|---:|---:|---:|---:|---|
| baseline | `stateless_once` | 4.40% | 1.14% | 6.90% | 5.17% | 单段拼接健康。 |
| baseline | `stateful_stream` | 4.01% | 3.41% | 5.17% | 3.45% | 普通流式健康。 |
| baseline | `offline_full` | 4.78% | 2.27% | 9.48% | 2.59% | 离线整段健康。 |
| C1 | `acoustic_tail_only` | 4.21% | 2.27% | 6.03% | 4.31% | 声学尾本身不破坏语义。 |
| C3 prototype | `tail_kv_pause_recovery` | 277.80% | 270.45% | 255.17% | 307.76% | C3 停顿恢复无法修复 C2/KV 历史复读。 |

进一步检查：`stateful_stream`、`acoustic_tail_only`、`tail_kv_pause_recovery` 都拿到 exact boundaries（001/002 为 7/7，003 为 9/9），所以这次不是 proxy boundary 或分段合并导致的假结果。C3 的 pause deviation 在样本 001 上从 stateful 的约 196.8ms、C1 的约 290.6ms 被校正到 0ms，但 CER 同时爆炸，hyp_chars 变成 324/409/469，而 ref_chars 只有 88/116/116。结论是：C3 后处理确实能把“停顿列”做漂亮，但它不能恢复语义；当前 C3 行差的根因仍是 C2/KV 历史条件导致的长复读，而不是 baseline、C1、ASR 或同文本设置问题。

---

## 24. 2026-07-09 E24 Serving prefix 收口对照：language=auto + 空 instruct

按 E7 review §3.2，补一个最小 serving-prefix 消融：runner 新增 `--override-language` 和 `--override-instruct`，同一批 `prosody_mini_001-003`、seed=42、`input_mode=token`，只把数据集默认的 `language="Chinese"`、非空 instruct 覆盖为 `language="auto"`、`instruct=""`，使 serving 前缀从带 instruct/language tag 的长前缀收敛到更接近官方 8 槽核心。产物目录：`workspace/table2_prefix_auto_empty_e24_20260709/`。

| variant | E23 原始前缀 | E24 auto/空 instruct | 判读 |
|---|---:|---:|---|
| `stateless_once` | 4.40% | 5.36% | baseline 仍健康。 |
| `stateful_stream` | 4.01% | 4.21% | 普通流式基本不变。 |
| `offline_full` | 4.78% | 4.78% | 离线整段不变。 |
| `acoustic_tail_only` | 4.21% | 6.69% | C1 仍是健康量级，略有波动。 |
| `tail_kv_pause_recovery` | 277.80% | 280.49% | C3/KV 灾难复读没有改善。 |

E24 metadata 确认三条样本均为 `language=auto`、`instruct=""`，并且 exact boundaries 正常（001/002 为 7/7，003 为 9/9）。因此，21 槽 serving 前缀膨胀不是当前 runtime C3/KV 复读的主因；即使收敛到 auto/空 instruct，KV/history conditioning 仍会把输出拖长到 89-122 秒并造成 250%-320% CER。下一步不应继续押注前缀长度本身，而应做 content-level layout parity 与 token-level text+codes re-prefill/训练同构门禁。

---

## 25. 2026-07-09 E25 Content-level runtime/C4 prefix parity

按 E7 review §3.2，将 `scripts/python/check_c4_runtime_layout_parity.py` 从符号/长度检查升级为内容级 prefix token-pair 检查：脚本现在可用 `--runtime-language/--runtime-speaker/--runtime-instruct` 从 tokenizer/config 构造 runtime prefix 的双通道 token 槽，并与 C4 collate 的 `input_ids[:, :prefix_len, :]` 逐位比对；不一致时 exit 非 0，并在 summary 中记录前 8 个 mismatch。产物在 `workspace/c4_wenet_premium0_nll20_20260709/`。

| 检查 | summary | overall | 关键结果 |
|---|---|---|---|
| 旧符号级 prefix8 | `runtime_layout_parity_prefix8_e25_summary.json` | pass | C4 collate 内部顺序通过，prefix_len=8 与 auto/空 instruct 目标长度一致。 |
| 旧符号级 prefix21 | `runtime_layout_parity_prefix21_e25_summary.json` | warn | C4 prefix_len=8，不等于原始 serving 的 21 槽长前缀。 |
| 内容级 auto/空 instruct + Vivian | `runtime_layout_content_parity_auto_empty_e25_summary.json` | fail | 长度同为 8，但 speaker 槽不一致：runtime pos6=`[tts_pad, 3065]`，C4 pos6=`[tts_pad, 0]`。 |
| 内容级 Chinese + instruct + Vivian | `runtime_layout_content_parity_chinese_instruct_e25_summary.json` | fail | 长度 21 vs 8，且 instruct/language/speaker 多处内容 mismatch。 |

判读：E24 已经说明“前缀长度从 21 收到 8”不能单独解决 runtime 复读；E25 则进一步证明即便在 auto/空 instruct 条件下，C4 collate 与 runtime 仍没有内容级同构，最小差异集中在 speaker 槽。当前 C4 collate 的 8 槽核心使用 `[nothink, think_bos, think_eos, 0, codec_pad]`，而 custom runtime 会按 speaker 查表写入 Vivian=`3065`。这解释了为什么“8 槽长度一致”不能等价于“训推一致”。下一步若要让 C4 训练诊断真正服务 CustomVoice 表 2，必须先决定 speaker 槽策略：要么训练 collate 写入与 runtime 相同的 `spk_id`，要么在导出/运行时证明训练中的 `0` 槽与 custom speaker embedding 等价；否则 LoRA 训练和 runtime Full 仍有隐性分布差。

---

## 26. 2026-07-09 E26 Speaker 槽复核与 C4 训练诊断接口修复

复核 E25 后补充一个重要 nuance：C4 batch 的 token-level speaker 槽确实写 `0`，但现有 teacher-forcing NLL 与 LoRA generation 的 embedding 构造路径会显式把第 6 槽覆盖为 speaker embedding。对 1.7B CustomVoice，没有 `speaker_encoder` 时 `resolve_speaker_embedding()` 会从 `talker_config.spk_id` 查表，例如 `Vivian -> 3065`，再取 `model.talker.model.codec_embedding(3065)`；因此 E25 的 token-level mismatch 不一定等价于 embedding-level mismatch。真正的风险是：所有训练/评估脚本必须显式携带同一个 speaker，否则这个覆盖逻辑会漂移。

本轮发现并修复两个接口漂移：

| 脚本 | 问题 | 修复 |
|---|---|---|
| `scripts/python/train_c4_lora_gap.py` | `codec0_nll_for_segment()` 已要求 `speaker`，但 eval gap 调用未传；后续重跑会直接 TypeError。 | 增加 `--speaker`（默认 `serena`），before/after eval 均传入并写入 summary。 |
| `scripts/python/train_c4_continuation_smoke.py` | 同样在 `evaluate_rows()` 调用 NLL 时缺 `speaker`。 | 增加 `--speaker` 并传入 before/after eval，summary 记录 speaker。 |

验证：`py_compile` 已通过。下一步如果继续 C4 LoRA/NLL 训练诊断，必须在命令里显式传 `--speaker Vivian/Serena/Ethan`，并把 speaker 作为结果维度记录；否则无法证明训练诊断与表 2 CustomVoice runtime 同源。

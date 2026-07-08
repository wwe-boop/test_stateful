# SteadyStream 表 2 进度汇报

日期：2026-07-08

范围：只覆盖表 2，当前评测集为 `test-prosody-mini`，规模为 3 seeds x 50 samples。表 3/表 4 不纳入本文档。

## 1. 一句话结论

表 2 当前已完成前 5 行的可运行实现、三种子音频生成、指标后处理和 CER 合并；新增的“完整可运行 SteadyStream 组合行”也已经完成 150/150 条音频生成、ASR/CER 合并和新版交付表生成。

2026-07-08 最新进展：C2 的 CER 劣化已经拆成两层 P0 问题处理。第一层是裁剪 KV tail 后 RoPE 逻辑位置被重置：本轮已落地 `position_offset`，紧凑 KV 继续紧凑存放，但后续 decode 的 `position_ids` 按原始逻辑时间线继续递增。第二层是 smoke 发现未发生裁剪时仍早停：原因更像是段末 pad/stop phase 的 terminal tail 被带进下一段；已改为保存 SteadyStream carry 时根据 `pad_start_frame` 丢掉 terminal tail token。`kv_inherit_token_counts` 默认也已从继承改为 reset。旧 C2/C3 数字仍保留作“修复前原型”记录，不能当作修复后结果。

需要特别说明：计划中真正的“完整 SteadyStream”定义为 C1+C2+C3+C4，其中 C4 是 continuation 后训练 checkpoint。当前没有找到真实 C4 continuation 训练 checkpoint，因此我没有把基座引擎或 smoke adapter 冒充成最终 C4 行；当前补测行标注为“C1+C2+C3，C4 未训练”。

## 2. 表 2 逐行状态

| 行 | 变体 | 当前状态 | 已有 CER | 汇报判断 |
|---:|---|---|---:|---|
| 1 | 无状态 | 已完成，3 seeds x 50 | 11.13% +/- 0.09% | 基线成立，可进当前交付表。 |
| 2 | 现有 stateful Triton | 已完成，3 seeds x 50 | 10.86% +/- 0.12% | 比无状态 CER 略好，边界 F0/能量明显改善。 |
| 3 | 仅声学尾 C1 prototype | 已完成，3 seeds x 50 | 11.40% +/- 0.85% | CER 接近基线，是目前最安全的组件原型。 |
| 4 | 仅 KV/token 尾 C2 prototype | P0 位置修复 + terminal-token drop 已落地，待重跑 | 修复前 27.01% +/- 0.55% | 修复前边界指标好看但文本完整性失败；已定位到 RoPE position 错位和段末 EOS 进入下一段上下文两个风险。 |
| 5 | 尾 + KV + 暂停恢复 C1+C2+C3 prototype | 依赖修复后 C2 重跑；C3 口径仍待解耦 | 修复前 28.24% +/- 0.23% | 暂停指标被修好，但旧结果继承了 C2 的 CER 问题；不能作为最终 C3 结论。 |
| 6 | 完整 SteadyStream C1+C2+C3+C4 | C2 门禁 + C4 训练 checkpoint 均未完成 | - | 不能造数；正确顺序是 C2 修复重跑达标，再训练/部署真实 C4。 |
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
| C2/C3 CER 变差 | C2 到 27.01%，C1+C2+C3 到 28.24% | 已定位两层 P0 根因：裁剪后的 Talker KV tail 保留旧绝对位置 RoPE 相位但新 query position 重置；另外未裁剪 smoke 仍早停，说明上一段 pad/stop phase terminal tail 也不能直接作为下一段上下文 | 已新增 `position_offset` 修复逻辑位置连续性；保存 carry 时按 `pad_start_frame` 丢掉 terminal tail；token_counts 默认 reset；下一步重跑 C2 门禁。 |
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
| C2 P0 修复 | 新增 `SlotKVState.position_offset`；`executor` 的 fused `position_ids` 改为 `position_offset + past_len`；SteadyStream carry 新增 `talker_logical_past_len`/`talker_position_offset`；段末 carry 按 `pad_start_frame` 默认丢掉 terminal tail；`kv_inherit_token_counts` 默认改为 reset | 同时修复裁剪 KV tail 的 RoPE 逻辑位置合约，以及上一段 pad/stop phase 污染下一段上下文的问题。 |
| C2 回归测试 | 更新 `tests/unit/test_engine_loop_pipeline.py`，新增紧凑 KV tail position offset、terminal tail drop、默认 token_counts reset、显式继承兼容测试 | 目标测试已通过：30 passed, 1 warning。 |
| C4 真实数据 | 选定 WenetSpeech4TTS Premium 作为 C4 主数据源，完成 Premium_0 pilot 下载、MD5 校验、20 样本 manifest+codes 构建和 batch dry-run | 可用于 C4 扩容训练前置，但训练启动应等 C2 门禁通过。 |
| 版本管理 | 已配置 GitHub remote `github-test-stateful` 和专用 deploy key | 当前代码和本文档已持续推送到 GitHub 目标分支。 |

## 6. 当前交付与下一步

| 优先级 | 动作 | 当前结果 |
|---:|---|---|
| P0 | 为 6a 全开补测行生成 ASR manifest，复用已有 750 条 CER，只新增 150 条全开音频 ASR | 已完成，产物为 `table2_cer_full.json`。 |
| P0 | 合并 full-row CER 并重新生成交付表 | 已完成，产物为 `table2_current_full.md` 和 `table2_current_delivery_full_20260708.md`。 |
| P0 | 拉回本地 outputs，并提交/推送新增汇报与必要代码变更 | 已完成，本地目录为 `outputs/table2_current_delivery_full/`。 |
| P0 | 用 P0 C2 修复重跑 `kv_tail_only` 和 `tail_kv_pause_recovery` | 正在执行 smoke；通过标准是 CER 回到无状态/stateful 基线约 1 个百分点范围内，同时保持边界 F0/能量收益。 |
| P0 | 如果修复后 C2 仍不达标，继续补 sink/window/token-history re-prefill 或 position-offset KV 细节 | 不启动 C4 训练，避免把 C2 实现 bug 误归因到 C4。 |
| P1 | C2 门禁通过后，扩大 WenetSpeech4TTS Premium manifest，启动 continuation SFT | 真实 C4 checkpoint。 |
| P1 | 训练路径复用父目录官方 finetuning，但要改成多片段 continuation collate、history drop、lookahead 和 loss mask | LoRA 或全参 C4 checkpoint。 |

## 7. 当前命令状态

当前分支：`exp/steadystream-table4-20260707`

当前运行目录：`/home/zehan/workspace/Qwen3-TTS-Triton`

完整行音频输出：`workspace/table2_runs_impl_20260708`

完整交付表：`workspace/table2_runs_impl_20260708/table2_current_delivery_full_20260708.md`

完整行日志：`workspace/logs/table2_full_steadystream_20260708.log`

截至本文档更新时，完整行音频生成、ASR/CER 合并和 delivery 生成均已完成。

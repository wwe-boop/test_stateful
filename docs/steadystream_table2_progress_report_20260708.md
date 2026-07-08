# SteadyStream 表 2 进度汇报

日期：2026-07-08

范围：只覆盖表 2，当前评测集为 `test-prosody-mini`，规模为 3 seeds x 50 samples。表 3/表 4 不纳入本文档。

## 1. 一句话结论

表 2 当前已完成前 5 行的可运行实现、三种子音频生成、指标后处理和 CER 合并；新增的“完整可运行 SteadyStream 组合行”也已经完成 150/150 条音频生成和非 CER 指标后处理，下一步只差 ASR/CER 合并后即可生成新版交付表。

需要特别说明：计划中真正的“完整 SteadyStream”定义为 C1+C2+C3+C4，其中 C4 是 continuation 后训练 checkpoint。当前没有找到真实 C4 continuation 训练 checkpoint，因此我没有把基座引擎或 smoke adapter 冒充成最终 C4 行；当前补测行标注为“C1+C2+C3，C4 未训练”。

## 2. 表 2 逐行状态

| 行 | 变体 | 当前状态 | 已有 CER | 汇报判断 |
|---:|---|---|---:|---|
| 1 | 无状态 | 已完成，3 seeds x 50 | 11.13% +/- 0.09% | 基线成立，可进当前交付表。 |
| 2 | 现有 stateful Triton | 已完成，3 seeds x 50 | 10.86% +/- 0.12% | 比无状态 CER 略好，边界 F0/能量明显改善。 |
| 3 | 仅声学尾 C1 prototype | 已完成，3 seeds x 50 | 11.40% +/- 0.85% | CER 接近基线，是目前最安全的组件原型。 |
| 4 | 仅 KV/token 尾 C2 prototype | 已完成，3 seeds x 50 | 27.01% +/- 0.55% | 边界指标好看，但文本完整性失败，不能作为最终方案。 |
| 5 | 尾 + KV + 暂停恢复 C1+C2+C3 prototype | 已完成，3 seeds x 50 | 28.24% +/- 0.23% | 暂停指标被修好，但继承了 C2 的 CER 问题。 |
| 6 | 完整 SteadyStream C1+C2+C3+C4 | C4 未训练，最终行未完成 | - | 不能造数；需要真实 continuation checkpoint。 |
| 6a | 当前可运行全开补测 C1+C2+C3，C4 未训练 | 音频 150/150 完成，非 CER 指标完成，CER 待 ASR | 待补 | 用来交付“现有引擎能跑到什么程度”，不冒充训练后 C4。 |

## 3. 最新非 CER 指标

产物位置：`workspace/table2_runs_impl_20260708/table2_full_no_cer.md`

| 变体 | F0 raw↓ | 能量 raw↓ | 停顿↓ | SIM Δ↓ | FASL↓ | CER↓ |
|---|---:|---:|---:|---:|---:|---:|
| 无状态 | 6.88 +/- 0.14 st | 5.42 +/- 0.20 dB | 127.69 +/- 0.67 ms | 0.16 +/- 0.01 | 22.35 +/- 0.13 ms | 11.13% +/- 0.09% |
| 现有 stateful Triton | 3.39 +/- 0.13 st | 3.54 +/- 0.16 dB | 200.00 +/- 8.48 ms | 0.10 +/- 0.00 | 29.19 +/- 0.10 ms | 10.86% +/- 0.12% |
| C1 prototype | 3.57 +/- 0.42 st | 3.35 +/- 0.25 dB | 200.54 +/- 15.04 ms | 0.10 +/- 0.01 | 27.03 +/- 0.24 ms | 11.40% +/- 0.85% |
| C2 prototype | 3.07 +/- 0.12 st | 3.12 +/- 0.12 dB | 205.10 +/- 2.19 ms | 0.13 +/- 0.01 | 27.78 +/- 0.31 ms | 27.01% +/- 0.55% |
| C1+C2+C3 prototype | 4.00 +/- 0.93 st | 4.50 +/- 0.34 dB | 11.53 +/- 3.31 ms | 0.13 +/- 0.01 | 28.09 +/- 0.13 ms | 28.24% +/- 0.23% |
| 当前可运行全开 C1+C2+C3，C4 未训练 | 4.14 +/- 0.63 st | 4.80 +/- 0.14 dB | 9.80 +/- 3.33 ms | 0.14 +/- 0.00 | 28.38 +/- 1.65 ms | 待 ASR |

## 4. 当前主要问题

| 问题 | 现象 | 原因判断 | 处理策略 |
|---|---|---|---|
| C2/C3 CER 变差 | C2 到 27.01%，C1+C2+C3 到 28.24% | 裁剪后的 Talker KV tail 没有验证 sink、CustomVoice 前缀、position id、EOS 合约；音频里出现提前 EOS/漏读末句，不是单纯 ASR 统计问题 | 当前只作为诊断原型；最终需要真实 C4 continuation 训练或更严格的 KV 合约实现。 |
| C3 只能修暂停 | 停顿从约 200 ms 降到约 10 ms，但 CER 没恢复 | pause recovery 是后处理/边界修正，不能补回已经缺失的文本 | 继续保留 C3 作为独立模块，但不把它宣传成解决语义完整性的手段。 |
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
| 完整行补测 | 新增 `scripts/python/run_table2_full_steadystream.py` | 150/150 条音频已完成，非 CER 指标已完成。 |
| 版本管理 | 已配置 GitHub remote `github-test-stateful` 和专用 deploy key | 当前代码提交已推送，最新已推提交为 `0ceed1b`；本文档会作为后续进度提交。 |

## 6. 下一步交付路径

| 优先级 | 动作 | 预期产物 |
|---:|---|---|
| P0 | 为 6a 全开补测行生成 ASR manifest，复用已有 750 条 CER，只新增 150 条全开音频 ASR | `table2_cer_full.json` |
| P0 | 合并 full-row CER 并重新生成交付表 | `table2_current_full.md` 和 `table2_current_delivery_full_20260708.md` |
| P0 | 拉回本地 outputs，并提交/推送新增汇报与必要代码变更 | GitHub 分支同步，用户可直接查看。 |
| P1 | 若要把第 6 行从“C4 未训练”变成真正完整 SteadyStream，需要提供或训练 continuation checkpoint | 真实 C1+C2+C3+C4 表 2 末行。 |
| P1 | 训练路径复用父目录官方 finetuning，但要改成多片段 continuation collate 和 loss mask | LoRA 或全参 C4 checkpoint。 |

## 7. 当前命令状态

当前分支：`exp/steadystream-table4-20260707`

当前运行目录：`/home/zehan/workspace/Qwen3-TTS-Triton`

完整行音频输出：`workspace/table2_runs_impl_20260708`

完整行日志：`workspace/logs/table2_full_steadystream_20260708.log`

截至本文档写入时，完整行音频生成已经完成 150/150，后台进程正常退出。

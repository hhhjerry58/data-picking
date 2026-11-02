# 数学文档对目标 SFT loss 的直接训练效用

配置：[document_utility.yaml](../configs/document_utility.yaml)。本轮直接检验四种文档来源的训练效果，
不再以来源识别 AUC 替代训练效用：从同一个原始 reference 模型出发，分别继续训练，再测同一套
留出 MATH 问答的回答 NLL。没有在文档训练后追加 target SFT。

## 已完成结果（2026-10-25 UTC）

12 个 run 均完成 64 步和最终评分，完整分析见
[已执行 notebook](../experiments-analysis/2026-10-25_161254Z_document-utility-qwen25-1p5b-t1048576-s64/analysis.ipynb)。
Reference 的平均回答 NLL 为 **0.743498**。下表为三个种子的平均值；相对下降比例先逐题计算，
再对问题和种子平均，因此不等于平均 loss 下降量除以平均 reference loss。负变化表示恶化。

| 文档训练来源 | 最终目标 NLL | loss 下降量 | 相对下降比例 |
| --- | --- | --- | --- |
| General FineWeb | 0.715040 | +0.028459 | +3.441% |
| OpenWebMath | 0.709711 | +0.033788 | +3.951% |
| OpenStax | 0.891925 | −0.148427 | −27.372% |
| FineMath-4+ | 0.721796 | +0.021703 | +2.206% |

以下是同种子、同问题的数学来源相对 general 的改善差；正值有利于数学来源。
区间同时重采样 seed 和目标问题组，只有三个 seed，属于近似的不确定性描述。
仅重采样问题组的条件区间及所有逐种子结果也保存在 notebook 中。

| 来源 | 下降量优势及 hierarchical 95% CI | 相对下降优势及 hierarchical 95% CI（百分点） |
| --- | --- | --- |
| OpenWebMath | +0.005329 [−0.000590, +0.014682] | +0.509 [−0.359, +1.883] |
| OpenStax | −0.176885 [−0.191908, −0.163486] | −30.813 [−33.736, −28.095] |
| FineMath-4+ | −0.006756 [−0.011913, −0.002111] | −1.235 [−2.002, −0.511] |

OpenWebMath 平均略优，但两项优势区间均跨零；下降量优势主要来自 seed 44，
相对下降比例在 seed 42/43 反而略低于 general。FineMath 三个 seed 都能降低目标 loss，
但两项改善均小于 general。OpenStax 三个 seed 均让目标 loss 上升，且明显不如 general。
因此，本轮不支持“数学文档必然比 general 更能降低 MATH SFT loss”；FineMath/OpenStax 的
结果与此前观察相容，但不足以验证逐样本 loss-change 选数方法，OpenWebMath 也仍有不确定性。

已核对全部训练日志的步数、有效 token 预算、有限梯度，以及最终 optimizer/scheduler/cursor。
Reference 与旧评分的最大逐题 NLL 差为 4.77e-6，低于预设容差 2e-5；原有六份 notebook 未改动。
本轮相关测试 20 项通过，分析 notebook 的 6 个代码单元全部执行完成。

## 在训练前固定的对照

- General：原 9,000 篇 FineWeb 背景，标签未人工核验，可能含数学。
- OpenWebMath：原 1,000 篇注入文档。
- OpenStax：已准备的 117 个教材模块。
- FineMath-4+：已准备的固定 1,000 篇网页。

四组均从 `Qwen/Qwen2.5-1.5B-Instruct` revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306` 初始化。每组 seed 42/43/44，共 12 个 run。
每个 run 恰好训练 **1,048,576 个有效预测 token**，64 步，每步 16,384 个；micro batch 为 2，
按窗口片段动态累积，不能把它误称为固定的每步样本数。学习率 2e-5、warmup 7 步、线性衰减，
AdamW、weight decay 0.01、梯度裁剪 1.0，全部一致。

已有文档窗口按 seed 不放回打乱，消耗固定 token 预算。步骤边界落在窗口中间时，下一片段保留
原窗口的全部前缀作为上下文，但前缀已消费 token 的 loss mask 为零；不会跨文档拼接。
每个训练目标 token 最多出现一次。原始正文不修改；最后一个窗口按总预算只消费前一部分，
未消费部分不参与本次训练。每个种子同时改变所用窗口子集和顺序，数据配方和 ID 均落盘。

采用 token NLL 总和除以该优化步的有效 token 总数，避免可变长度样本被隐式等权。
长文可能贡献更多训练 token；这是固定 token 预算下的来源对照，不是等篇数对照。
模型为 FP32 master 参数/optimizer，bf16 autocast；SDPA math 覆盖 forward、backward 和重计算，
保留非有限梯度检查。

## 状态保存与结果评估

每 32 步保存完整 checkpoint（model、optimizer、scheduler、RNG、global step、data orders、
batch/cursor、累计有效 token），启用自动恢复；模型、训练配置、数据或数值实现不符时拒绝恢复。
General seed 42 已在 step 32 实际退出后恢复，验证状态接续。固定最终 step 64 评估，
step 32 仅用于恢复，不看评估 loss 选模型。

评估目标为原 512 条留出 MATH validation，七个科目，排除在所有文档训练之外。
沿用完全相同的聊天模板、token IDs、response mask，full FP32 / eval / SDPA math，
逐样本回答平均 NLL。重新测 reference 并与旧值核对。所有数学源和 general 在相同问题上配对比较。

两项变化指标并列：

- `delta_loss = L_ref − L_document_trained`
- `relative_gain = delta_loss / max(L_ref, 1e-6)`

每种数学来源相对 general 的对照为同种子下 `gain_math − gain_general`，正值表示数学源更有利。
同时报告最终 NLL 和绝对变化，不能把“损失增加较少”表述为“损失下降”。
逐个展示三个 seed、均值、范围，避免仅报告最优 seed。

2,000 次配对 bootstrap：一套仅重采样目标问题组，条件于这三个训练模型；另一套同时重采样
seed 和问题组，并在来源间共享重采样索引。只有三个 seed，后者也是近似的不确定性描述。
没有预先指定有实际意义的等价界限，因此“未检出优势”不能当作证明相等。

## 对此前结论的解释

如果数学来源比 general 更能降低目标 loss，说明此前的来源 loss 变化在这个训练设置下未能
正确预测训练效用。如果没有优势，则与此前观察相容，但仍只提供来源层面的证据。
本轮没有比较同一候选池内按分数挑选的数据与随机选择的数据，因而不能证明逐样本排序准确。
效果也只针对这些来源、模型、训练预算与优化设置。

## 复现入口

```bash
source scripts/env.sh
.venv/bin/python scripts/experiment.py prepare-utility
.venv/bin/python scripts/experiment.py train-utility --arm general --seed 42 --check-only
# 显式选择空闲 GPU；blob 配置仅用于同一持久位置的检查点传输，不输出凭据。
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/experiment.py train-utility \
  --arm general --seed 42 --checkpoint-blob-config <existing-blob-config> --direct-checkpoint-io
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/experiment.py score-utility --arm general --seed 42
.venv/bin/python scripts/experiment.py analyze-utility
.venv/bin/python scripts/experiment.py export-utility \
  --run-dir /mnt/tianzhu_hot/data-picking/runs/document-utility-qwen25-1p5b-t1048576-s64
```

各子 run 在同一根目录下追加 `-<arm>-seed<seed>`。配方位于挂载盘 `data/processed/`，
完整产物位于 `runs/`。训练前计划、预检、输入哈希、源码快照和旧 notebook 哈希保留。

## 检查点 I/O 修订

首批 step-32 保存遇到 FUSE EIO，检查发现其底层缓存盘仅余约 0.01 GiB。
改用相同 blob 目的路径的分块上传，服务端逐块验证 Content-MD5，提交后核对完整块顺序、大小及
稳定 ETag，并计算完整本地 SHA-256，最后写 manifest 和完成标记。首批恢复检查点及 general 的
首次检查点另外做全文件各段 SHA-256 回读；小文件测试也验证错误 MD5 被服务端拒绝。
后续检查点避免重复下载完整 18.5 GB 文件；数值 trainer、配置和数据指纹不变。
完整 RAM 状态只有通过 optimizer/scheduler/RNG/data cursor 检查才恢复；没有完整状态的 general
按原种子从 reference 重跑，并保留初次日志以核对重跑结果。未完成检查点不作为有效状态使用。
General 前 32 步重跑的 loss 最大差为 2.98e-7，梯度范数最大差为零；步数、token、片段数和
学习率一致。FineMath seed 42 的最终 RAM 状态也已完成持久上传与评分，没有重新训练。

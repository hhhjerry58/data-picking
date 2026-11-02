# 数学 SFT 的 QA 交叉复测与网页 checkpoint 诊断

配置：[probe_diagnostics.yaml](../configs/probe_diagnostics.yaml)。这是用户在观察到
step-64 网页反向结果后授权的两个诊断，不属于此前冻结的确认性评估，也不用于重新选择 checkpoint。
只评估已有模型，没有 optimizer update，没有新增梯度测量。

[已执行的诊断 notebook](../experiments-analysis/2026-10-23_165639Z_math-web-probe-diagnostics-qwen25-1p5b-seed42/analysis.ipynb)
包含两个检查的逐项结果、固定长度控制、配对比较和 5 张图。7 个代码单元格全部执行，
10 个结果/配置快照哈希已验证；相关 18 项测试通过，旧四个 notebook 均保持原样。

## 两个检查

1. 用多科目 MATH 4,096 条训练得到的 step-64 模型，重新评估原 Algebra A2 与 UltraChat，
   各 512 条。复用原 `data/evaluation.jsonl` 的完整 input IDs、response mask 和样本 ID，
   不重抽样、不添加内容筛选。重新评分 reference，并与旧分析的每条 reference NLL 核对。
2. 在原 10,000 篇网页、11,688 个窗口上补测同一训练 run 的 step-16/32/48。
   reference 和 step-64 复用已校验结果；所有模型共享窗口边界、token IDs、mask。
   每篇仍按 NLL 总和除以评分 token 数聚合；背景来源未标注，不能当作确认的非数学负例。

两项变化指标 `delta_loss`、`relative_gain` 并列报告，方向固定为越大越像目标类；保留四个
原始 loss 方向基线。QA 与网页分别以 A2 和注入的 OpenWebMath 来源为正类。
长度控制沿用两个旧分析的固定 ID：QA 每域 126 条，网页 729 / 6,561 条，未重新匹配。

## 数值与统计

- 完整 FP32 forward / CE，eval 模式，SDPA math；QA 评分回答平均 NLL，网页评分全文窗口。
- QA 重复 forward、padding 检查和新旧 reference 交叉复核容差均为 `2e-5`。
  网页沿用已记录的整篇 token 加权 NLL 校验，容差 `2e-5`，同时保留窗口级差值。
- 检查新训练集与固定 QA 池的规范化问题重合。新训练数据准备时已按精确/近重复问题组
  保留旧 A2/B，诊断不改变训练集或评估集。
- 2,000 次按类分层、按问题组/重复文档组 bootstrap。旧 QA 与新 QA 使用相同重采样规则，
  另外报告同批样本上的配对 AUC 变化区间；两个变化指标均计算。
- QA 的旧模型区间在诊断 seed 下重算，点估计与旧 notebook 一致，旧区间和 notebook 保留原样。
- 网页各步沿用原网页统计 seed、top-k 规则；正式 step-64 指标直接复用且与逐文档重新聚合值逐项核对。
  配对报告 step-64 相对中间 checkpoint 的 AUC 差，未使用它选择 checkpoint。
- 区间不包含训练 seed、网页 crawl 或所有网站相关性。最早 checkpoint 是 step 16，
  不能据此推断 step 1–15 的信号。两个检查无法单独区分格式、预测目标、题型和网页来源差异的因果贡献。

## 复现

```bash
source scripts/env.sh
# GPU 仅为示例，先确认空闲。
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/experiment.py score-probe-qa
# 每个中间 checkpoint 的 shard 0 / 1 均运行；路径来自 diagnostic_plan.json。
CUDA_VISIBLE_DEVICES=2 .venv/bin/python scripts/experiment.py score-web \
  --model sft --sets web --shard-index 0 --document-sanity --checkpoint <saved-checkpoint>
.venv/bin/python scripts/experiment.py analyze-probe
.venv/bin/python scripts/experiment.py export-probe \
  --run-dir /mnt/tianzhu_hot/data-picking/runs/math-web-probe-diagnostics-qwen25-1p5b-seed42
```

`diagnostic_plan.json` 在新增评分前冻结配置、数据哈希、checkpoint SHA 和匹配 ID 文件。
QA 完成而网页尚未完成时，可用 `analyze-probe --qa-only` 查看第一个检查。
新增 QA 分数、汇总、日志和图位于独立 diagnostic run；网页中间评分存入原训练 run 的
`scores_web/step-000016|000032|000048/web/`，不改动原 reference / step-64 评分和正式分析。
Notebook 写入新的 UTC 时间目录，包含无需挂载即可查看的已执行输出和校验快照；旧 notebook 逐个核对哈希。

## QA 检查的观测结果

新 step-64 模型在固定旧 A2/UltraChat 上，下降量 AUC 为 **0.991497**
（95% CI 0.985424–0.996464），相对下降比例 AUC 为 **0.996025**（0.992752–0.998448）。
原长度匹配子集上分别为 **0.982048**（0.963462–0.995717）和 **0.992315**（0.985261–0.997671）。

A2 平均 NLL 为 **0.809258 → 0.623749**，99.22% 样本改善；UltraChat 为
**1.099355 → 1.159328**，6.05% 改善。因此新训练设置保留了原 QA 区分信号。
原始池中，两项 AUC 相对旧模型分别增加 0.005825、0.009167；配对区间分别为
[0.002132, 0.010391]、[0.004349, 0.015118]。这不代表独立训练 seed 上的显著性。

重新评分 reference 与旧结果的最大逐样本 NLL 差为 **1.0371e-5**，低于预设 `2e-5`；
额外核对中，若完全复用旧 reference 分数，新模型的两项原始池 AUC 均不变。
固定 QA 池与新训练集的规范化问题重合为 0。

## 网页 checkpoint 诊断的观测结果

| Step | 下降量 AUC | 相对下降比例 AUC | 长度匹配：下降量 | 长度匹配：相对比例 |
| --- | --- | --- | --- | --- |
| 16 | 0.2866 | 0.2320 | 0.2788 | 0.2321 |
| 32 | 0.2849 | 0.2039 | 0.2731 | 0.2022 |
| 48 | 0.2919 | 0.2088 | 0.2814 | 0.2089 |
| 64 | 0.3046 | 0.2137 | 0.2924 | 0.2123 |

四个 checkpoint 上，两项指标在原始池和固定长度匹配池的 95% 区间均低于 0.5。
最早观测的 step 16 已出现反向区分：数学来源文档平均 NLL 为 **1.705336 → 1.728197**，
背景为 **2.733867 → 2.738543**。数学来源文档在该时点已经恶化更多。

QA 区分信号保留与早期网页反向信号可以同时成立。这支持关注问答训练向网页全文预测的迁移限制，
削弱“只有后期训练过度才造成网页失效”的解释；尚未把格式、预测目标、内容构成和网页来源差异
各自的因果贡献拆开。四步轨迹也不支持把两项指标概括成同一种单调变化：相对下降比例在 step 32
低于 step 16，随后略回升，但四个时点都未恢复到预设正方向。

# OpenStax 与 FineMath-4+ 文档来源复验

配置：[document_quality.yaml](../configs/document_quality.yaml)。用户授权分别测量两种来源，
复用已训练的 Qwen2.5-1.5B-Instruct reference / MATH 4,096 条 step-64 模型，以及原 9,000 篇
FineWeb 背景。没有训练、checkpoint 选择或新增梯度测量。两个变化指标并列，不按结果翻转方向。

## 固定来源与样本

- **OpenStax**：`openstax/osbooks-college-algebra-bundle`，revision
  `463991614337632b0e02cbb6c76223cbb0d423d3`，College Algebra 2e / Precalculus 2e 的目录。
  取命名章节内的完整模块，排除目录结构中的独立前后置材料；同一 module ID 只计一次。
  原有 121 个模块，评分前发现 4 个模块含空的源 MathML，按基本完整性规则隔离，最终 117 个。
  不按 loss、主题、最短长度或图像数量筛选模块。
- **FineMath-4+**：`HuggingFaceTB/finemath`，revision
  `e92b25a616738fe95dc186b64dfb19f9c8525594`。seed 20261024 从 64 个分片固定抽出 22/9/39 三个，
  SHA-256、字节数写入配置。仅取与背景相同的 `CC-MAIN-2023-06`，共 6,173 条；
  4 条 target 重合被隔离，从剩余 6,169 条中均匀随机抽 1,000 条。
  不叠加主题、代码、语言、最短长度或质量分数筛选。官方子集的整数质量分数为 4/5，
  浮点分数可以小于 4；没有错误地再按浮点值重筛。
- 与背景按精确正文/URL、MinHash 候选及实际五词 Jaccard ≥0.8 检查重合，未发现需隔离的新来源行。
  target 检查涵盖新训练/validation 和原 A2/B，使用固定的规范化文本包含和 shingle containment 规则；
  它不保证排除所有语义改写或所有短题重合。

正文和 tokenization 在 `/mnt/tianzhu_hot/data-picking/data/processed/math-document-quality-v1-seed20261024`。
新来源共 1,117 篇、2,017 个窗口、2,767,528 个评分 tokens。背景和旧结果保持原样。

## 提取、审查与已知限制

CNXML 保留完整正文、例题、练习和根级 glossary；MathML 按结构转换为 LaTeX，分数、上下标、
根式、矩阵和左右附标有专项测试。未知数学标签报错，不能静默剥离；37k 量级公式逐节点检查，
保留样本中共 36,534 个 MathML 公式。源文档空公式的排除属于评分前规则修订，记录在
`preparation_plan_v2.json`，其中保留原始计划。每篇无截断，窗口最多 2,048 tokens。

OpenStax 的 2,625 个图像仅保留文本描述，图形题仍可能依赖视觉信息；教材源文件也有个别不完善的
学习目标文字。121 个模块并非 121 个完全独立主题，也不是 CC 网页样本。

每来源固定随机导出 50 篇。人工查看全部 50 篇 FineMath 的首/中/尾摘录、前 12 个固定随机
OpenStax 模块摘录，并对所有教材执行结构检查；不是逐篇数学正确性证明。FineMath 摘录中仍发现
推广、目录、代码/物理、公式缺失和错误解答。例如一个解答把 `(x+1)^2` 展开成 `2x+2`，另一个页面
仍来自 `istopdeath.com` 并缺失大量公式。审查记录在 `quality_audit_review.json`，没有据此手动替换样本。

## 评分、长度与统计

完整 FP32 forward/CE，eval、SDPA math；两模型使用相同 token IDs、mask 和窗口边界。
每篇除首 token 外评分一次，累加各窗口 NLL 总和后除以 token 总数，随后计算两项变化分数。
原背景直接复用经过哈希核对的相同模型评分。整篇重复/padding 检查容差 `2e-5`，保留窗口级差值。

分别比较 117:9,000 与 1,000:9,000；报告实际来源比例、ROC-AUC、AP、top-k 来源 precision/recall
及相对随机富集。不同来源比例的 AP/precision 不能直接横向比较。原始 loss 基线四个方向全部保留。

长度控制在评分前固定为 2 的幂次 token 桶，每桶 1:9。OpenStax 保留 25 小节及 225 背景，其中
17 个正例是无 MathML 的短章节导言，只有 8 个较长教学小节；因此此控制的覆盖范围很有限。
FineMath 保留 720 / 6,480 条。报告保留率及桶内剩余长度 AUC，不宣称分桶彻底消除了长度因素。
固定 ID 保存在 `length_control_plan.json`，分析时重新核对。

2,000 次按来源分层的 bootstrap：OpenStax 按章节聚类，并合并重复连接的章节，共 14 组；
FineMath 和背景按重复文档组。区间不包括训练 seed 和全部站点相关性。
这是两个来源复验，不能单独识别质量、格式、题型或序列化形式的因果贡献，也没有验证选数训练收益。

## 运行入口

```bash
source scripts/env.sh
.venv/bin/python scripts/experiment.py prepare-quality
# 评分前完成固定样本审查并冻结 evaluation_plan.json。
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/experiment.py score-quality --model reference
CUDA_VISIBLE_DEVICES=2 .venv/bin/python scripts/experiment.py score-quality --model sft
.venv/bin/python scripts/experiment.py analyze-quality
.venv/bin/python scripts/experiment.py export-quality \
  --run-dir /mnt/tianzhu_hot/data-picking/runs/math-document-quality-qwen25-1p5b-step64-seed20261024
```

持久结果目录与上面相同；小型结果快照与已执行 notebook 单独导出到仓库的 UTC 时间目录。
源语料、完整审阅文本、窗口、模型、日志及图保留在挂载盘。旧 notebook 保留原样。

## 观测结果

[已执行 notebook](../experiments-analysis/2026-10-24_102053Z_math-document-quality-qwen25-1p5b-step64-seed20261024/analysis.ipynb)
包含两组独立对比、原始 loss 基线、固定长度控制、来源比例及质量审查说明。

| 来源 / 评估池 | 下降量 AUC（95% CI） | 相对下降比例 AUC（95% CI） |
| --- | --- | --- |
| OpenStax，117 / 9,000 | 0.1099 [0.0901, 0.1326] | 0.0691 [0.0472, 0.0950] |
| OpenStax 长度匹配，25 / 225 | 0.3666 [0.2476, 0.5264] | 0.3324 [0.2136, 0.4932] |
| FineMath-4+，1,000 / 9,000 | 0.1685 [0.1539, 0.1834] | 0.0818 [0.0702, 0.0942] |
| FineMath 长度匹配，720 / 6,480 | 0.1507 [0.1333, 0.1693] | 0.0783 [0.0640, 0.0932] |

OpenStax 平均 NLL 为 **1.164409 → 1.200566**，仅 3/117 篇改善；FineMath 为
**1.340716 → 1.395827**，21/1,000 篇改善。相同背景为 **2.733867 → 2.746040**。
两个新来源都没有恢复预设的正向信号，数学来源文档平均恶化更多。
OpenStax 的匹配子集很小且由导言主导，其中下降量区间覆盖 0.5，不能把完整池的强反向信号
不加限制地推广到长度控制后的所有教材。

按分数取原始池前 10%，两项变化分数在 OpenStax 都只找回 1 个正例（随机期望约 11.7）；
FineMath 分别找回 13 / 16 篇（随机期望 100）。负 reference loss 的来源 AUC 分别为
0.9560、0.9534，说明这两组来源本身容易区分，但所测的 SFT 更新没有按预设方向降低它们的文档 loss。

这次来源替换未支持“换成更干净数学文档即可恢复正向指标”的假设。OpenStax 的保真公式与实际
教学内容仍出现负变化，使“只有 OpenWebMath 噪声造成失效”的解释更不充分；这并非质量因素的
独立因果实验。FineMath 审查也证实了其残余噪声。未翻转指标、未重选 checkpoint 或手动替换样本。

验证：20 项相关测试通过；7 个 notebook 代码单元格全部执行；11 个快照哈希校验通过；
5 张图经检查。原始背景分数、输入文件和旧五个 notebook 均保持不变。

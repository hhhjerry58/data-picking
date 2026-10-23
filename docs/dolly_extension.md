# Dolly C 评估与 A2/Dolly loss notebook

这是 Algebra/Number Theory 实验的评估扩展。固定沿用原 ref、step 32 SFT 模型和 A1/A2/B 划分，
不执行新的训练。扩展结果独立保存，原始 A/B notebook 保留。

## 数据规则

C 来自 `databricks/databricks-dolly-15k` 的 `general_qa`，revision 固定为
`bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a`。按用户确认的规则：

- 完整模板序列最多 2,048 tokens，与 A/B 一致；超长整条排除，不截断答案。
- 排除空问题、空答案、没有有效 response token 等无法正常评分的样本。
- 不设最短答案长度，不按主题、语言、代码或回答风格过滤，不人工挑选或替换。
- 重复只记录；同题不同答保留。从所有有效候选行均匀随机抽 512 条，seed 为 20261022。
- 当前固定 revision 的 general_qa context 全为空，沿用 A/B 模板，无需新增拼接模板。

这里的 C 表示「Dolly 通用问答中满足本实验长度与有效性要求的随机样本」，其中可以包含数学、
代码和短答案。它不是经过内容筛查的非数学子集。

问题采用 Unicode NFKC、转小写和空白合并后按完全相同分组。报告完整类别、有效候选和选中 C 的
重复组数、同题不同答组数，并以问题组为单位计算分层 bootstrap 区间。
同时记录与 A1/A2/B 的规范化问题重合，主随机样本保持不变；若有重合，另列排除重合 C 的敏感性结果。
该检查不能保证发现语义改写。原 A1/A2/B 的分组和去重结果保持不变。

每条 general_qa 的有效性、长度和选中状态写入 manifest；保存数据 revision、原始文件 SHA256、
原 A1/A2/B split hashes、模板 hash、tokenizer revision、抽样与 anchor IDs。
使用独立的 `dolly-lengthonly` run，旧的内容筛选版本不进入本次统计。

本次实际准备结果：2,191 条 general_qa 中仅 1 条超长被排除；2,190 条候选中抽取 512 条。
选中样本包含 504 个规范化问题组，其中 8 组各有两条不同回答，全部保留。
未发现与 A1/A2/B 规范化后相同的问题。完整结果见
[本次 notebook](../experiments-analysis/2026-10-22_093305Z_math-algebra-numbertheory-dolly-lengthonly-qwen25-1p5b-seed42/analysis.ipynb)。

## 梯度定义

A2/B/C 各 512 条，在两个模型上分别执行单样本 forward/backward，使用 float32 参数和完整 float32
计算、SDPA math 后端、eval 模式和回答 token 平均 NLL。梯度不裁剪，不更新参数。

每条样本记录：

- 全局梯度 L2 norm、RMS、单个标量参数的最大绝对梯度。
- 每个命名参数张量的 norm/RMS/最大绝对梯度，以及 norm 最大的参数张量名称。
- 每层聚合 norm/RMS/最大绝对梯度；绑定的 embedding/lm-head 权重只计一次。
- 与同一模型上固定 128 条 A1 平均梯度的 cosine；两个模型分别构建自己的参照，anchor seed 为 2027。
- 同一次 forward 的 mean/sum NLL，并与原 A2/B 分数核对。

保存标量、逐层和逐参数张量统计，不保存逐样本完整梯度向量。按块写入并校验哈希，支持恢复已完成块。

## 分析与呈现

当前 review notebook 展示 A2/C 的 loss，并在末尾追加同一批样本的梯度分析，A2 为正类。loss 下降量 `loss_ref - loss_sft`
和相对下降比例 `(loss_ref - loss_sft) / max(loss_ref, 1e-6)` 并列考察，不设主次。
按原 A/B notebook 的布局展示实验设置、训练曲线、五联 loss 图、ROC、top-k、长度控制、配对比较和结论。
五联图包含 ref NLL、SFT NLL、下降量、相对下降比例和逐样本散点图。
置信区间继续采用按域分层、按问题组抽样的 2,000 次 bootstrap。
原始 A2/C 的估计与区间沿用已有结果，仅重新计算两域长度控制。

只在 A2/C 合并池上计算 prompt/response 长度联合四分位桶，每域取同样数量，报告匹配结果和保留率。
长度匹配后的区间仍按问题组重采样。原始 loss 基线与长度基线同时报告。
Dolly 与 MATH 的来源、题型、文风和回答质量混杂；可分性不构成选数训练收益的证明。

当前 notebook 不展示 B 对比；末尾展示全局/逐层梯度、最大单参数梯度、A1 cosine、长度对照及观察。
已完成的完整测量继续保存在挂载目录；原 A2/B notebook 的梯度附录复用其中经过模型与样本一致性核对的 A2/B 部分。
用于这版展示的两域统计保存在 `metrics/a2_dolly_loss/`，不改动原始三域结果。
更新已有 notebook 时保持目录时间，并在挂载的 `analysis_history/` 备份修改前的版本。
追加或更新梯度部分使用 `scripts/append_gradient_analysis.py`，具体入口见实验分析索引。

结果存入 `experiments-analysis/YYYY-MM-DD_HHMMSSZ_<run_id>/`。已执行 notebook 和小结果快照在仓库，
完整数据、日志和梯度明细在 `/mnt/tianzhu_hot/data-picking`。

## 运行入口

```bash
source scripts/env.sh
uv --no-cache pip install --python .venv/bin/python -r requirements-generalization.txt
.venv/bin/python scripts/prepare_dolly.py
```

先检查空闲 GPU，再分别测量两个模型。模型大文件可使用 SHA256 校验通过的临时内存副本；
如已有该副本，复用对应 `runtime_weights.json` 即可。挂载读取缓慢且尚无副本时，可使用现有
blobfuse 配置读取同一存储中的原文件：

```bash
.venv/bin/python scripts/stage_evaluation_weights.py --blob-config /path/to/existing/blobfuse2.yaml
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/measure_gradients.py --model reference
CUDA_VISIBLE_DEVICES=3 .venv/bin/python scripts/measure_gradients.py --model sft
.venv/bin/python scripts/analyze_generalization.py
.venv/bin/python scripts/export_generalization_notebook.py \
  --run-dir /mnt/tianzhu_hot/data-picking/runs/math-algebra-numbertheory-dolly-lengthonly-qwen25-1p5b-seed42
```

单样本 FP32 autograd/math 与原批量推理的 NLL 校验容差为 `1e-4`。此前在相同 tokenization hash 的
A2 样本上观察到约 `2.38e-5` 的差异，因此不再使用原先的 `2e-5` 容差。
实际最大误差随结果报告；不调整分数或梯度定义来消除差异。

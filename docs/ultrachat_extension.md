# UltraChat 外部数据源复验

本轮沿用原 `Qwen/Qwen2.5-1.5B-Instruct` reference 与 Algebra A1-SFT 的最终 step 32 模型，
固定原 A2=512。只做评分和逐样本反向传播，不执行 optimizer step、不重新训练或选择 checkpoint。
配置见 `configs/ultrachat_extension.yaml`。

## 在评分前固定的数据定义

- C：`HuggingFaceH4/ultrachat_200k` 的 `test_sft`，revision
  `8049631c405ae6576f93f445c6b8166f76f5505a`。完整 parquet 的 SHA-256 写入配置，并在下载后验证。
- 每段对话只取首轮完整 user/assistant 问答，使用与 Dolly、A2 相同的聊天模板和 response mask。
  后续轮次不拼接，也不替补无效或超长的首轮。这是单轮问答视图，不是对完整多轮对话的测量。
- 仅排除完整模板序列超过 2,048 tokens、空问题、空答案、缺少有效首轮或无法正常评分的样本。
  不截断答案、不设最短长度，不按数学、代码、语言、主题、图像引用或回答风格过滤。
- 在整个有效候选池中按行均匀随机抽 512 条，seed 为 20261022。重复问题和不同回答都保留。
  以规范化问题为 bootstrap 单位；与原 A1/A2/B 的规范化问题重合单独记录，不默默替换主样本。
- 保存原始行号、prompt_id、原对话轮数、revision、数据和 tokenization hashes、完整候选 manifest 和抽样 ID。

UltraChat 是合成对话；H4 的 200k 版经过上游子集选择、大小写修正和部分拒答措辞清理。
本实验不叠加内容过滤，结果仍只代表该指定版本的首轮问答。数据许可 MIT；原始作者为 Ding et al. (2023)。

本次实际准备：23,110 条首轮中，22 条超长、2 条空回答被排除；有效候选 23,086 条。
选中 512 条对应 512 个规范化问题组，未发现与原固定数据的规范化问题重合。

## 测量与统计

只测 A2 与 UltraChat C，各 512 条；两模型都使用完整 float32、eval 模式和 SDPA math，
逐样本计算回答平均 NLL。同一次反向传播记录未裁剪的全局 norm/RMS/最大绝对梯度、逐层/逐参数统计，
以及与同模型固定 128 条 A1 平均梯度的 cosine；A1 抽样 seed 2027 与 Dolly 实验一致。
绑定参数只计一次，完成的数据块带哈希并支持恢复。

比较为 A2/C，A2 为正类。loss 下降量 `loss_ref - loss_sft` 和相对下降比例
`(loss_ref - loss_sft) / max(loss_ref, 1e-6)` 并列考察，不设主次，同时保留原始 loss 基线。
按域分层、按问题组进行 2,000 次 bootstrap。长度控制只用 A2/C 的 prompt/response 联合四分位分桶，
每桶两域取相同数量，报告保留率及桶内剩余差异，不按模型分数挑样本。

Notebook 沿用 Dolly 的实验设置、训练曲线、五联 loss 图、ROC、top-k、长度控制、配对比较与验证结构，
不展示梯度数据。相对下降比例图沿用固定显示范围并列出范围外样本；全部样本参与统计。
完整梯度保存在挂载目录；原始 Dolly 和 A/B notebook 保持不变。

换一个通用数据来源可检验结果对来源的敏感性，但一次 C 抽样和一个训练 seed 不能估计所有随机性。
bootstrap 区间不包含训练随机性，域识别效果也不直接证明进一步选数训练的收益。

## 运行

```bash
source scripts/env.sh
.venv/bin/python scripts/prepare_ultrachat.py
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/measure_gradients.py --config configs/ultrachat_extension.yaml --model reference
CUDA_VISIBLE_DEVICES=3 .venv/bin/python scripts/measure_gradients.py --config configs/ultrachat_extension.yaml --model sft
.venv/bin/python scripts/analyze_generalization.py --config configs/ultrachat_extension.yaml
.venv/bin/python scripts/export_generalization_notebook.py \
  --run-dir /mnt/tianzhu_hot/data-picking/runs/math-algebra-ultrachat-lengthonly-qwen25-1p5b-seed42
```

运行前检查空闲 GPU。沿用经过 SHA-256 校验的模型临时内存副本与 `runtime_weights.json`，
所有持久数据、日志和完整梯度保存在 `/mnt/tianzhu_hot/data-picking`。
Notebook 与小快照按结果 UTC 时间存入仓库 `experiments-analysis/`。

## 本次观测结果

[已执行 notebook](../experiments-analysis/2026-10-22_113752Z_math-algebra-ultrachat-lengthonly-qwen25-1p5b-seed42/analysis.ipynb)
只展示 A2/UltraChat 的 loss 对比。下降量 AUC 为 0.9857（95% CI 0.9784–0.9921），
相对下降比例 AUC 为 0.9869（95% CI 0.9795–0.9929）。
长度匹配每域保留 126 条，下降量 AUC 为 0.9727（95% CI 0.9520–0.9895），
相对下降比例 AUC 为 0.9722（95% CI 0.9513–0.9889）。
A2 平均 NLL 为 0.8093→0.6374，UltraChat 为 1.0994→1.1654。
高可分性同时包含 A2 改善和这批通用问答损失恶化，不能直接推出实际选数训练收益。

两个模型各完成 1,024 条样本的测量，完整记录在该 run 的 `gradients/` 和 `metrics/`。
与 Dolly run 相同 A2 的复核中，平均 NLL 最大差为 4.77e-7，低于已有 1e-4 容差；
全局 norm、最大绝对梯度和 A1 cosine 统计一致，未据此调整数据或分数。

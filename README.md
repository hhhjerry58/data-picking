# data-picking

研究如何从候选数据中选择最有助于降低目标 domain loss 的样本。

第一步是做一个 domain 区分预实验：将 A 分成 A1/A2，仅在 A1 上做 SFT，
然后比较 reference model 与 SFT model 在 A2、B 上的逐样本 loss，检验
loss 下降量和相对下降比例能否识别 A2。

- [预实验实现方案（已确认）](docs/pilot_design.md)
- [A/B 数据源建议](docs/data_sources.md)
- [Dolly C 与训练前后梯度分析](docs/dolly_extension.md)
- [UltraChat 外部数据源复验](docs/ultrachat_extension.md)
- [项目约定](AGENTS.md)
- [实验分析索引（已执行的 notebooks）](experiments-analysis/README.md)

完整训练产物、模型和数据缓存存放在 `/mnt/tianzhu_hot/data-picking`。
用于 review 的 notebook 和小体积结果快照保存在本仓库的 `experiments-analysis/`，
按 UTC 结果时间命名子目录；代码和配置也保留在本仓库。

首轮已在本地 GPU 1（H100）完成。loss 下降量与相对下降比例并列考察，不设主次。
两项 AUC 分别为 **0.6849**（95% CI 0.6534–0.7160）和 **0.7007**（95% CI 0.6688–0.7314）；
长度匹配后分别为 **0.6524** 和 **0.6770**。
当前保存的下降量与原始 loss 的配对对照尚未确认额外收益。
这支持同域识别信号，尚不证明实际选数训练收益。

- [本次 notebook：指标、图表、对照和结论](experiments-analysis/2026-10-21_172347Z_math-algebra-numbertheory-qwen25-1p5b-seed42/analysis.ipynb)
- [原始 Markdown 报告](/mnt/tianzhu_hot/data-picking/runs/math-algebra-numbertheory-qwen25-1p5b-seed42/report.md)
- [完整指标与置信区间](/mnt/tianzhu_hot/data-picking/runs/math-algebra-numbertheory-qwen25-1p5b-seed42/metrics/metrics.json)
- [最终 checkpoint 位置](/mnt/tianzhu_hot/data-picking/runs/math-algebra-numbertheory-qwen25-1p5b-seed42/training_complete.json)
- [测试和真实续训验证](/mnt/tianzhu_hot/data-picking/runs/math-algebra-numbertheory-qwen25-1p5b-seed42/validation_summary.json)

Dolly C 扩展已完成：仅按完整序列长度（≤2,048 tokens）和基本有效性筛选，
从 2,190 条有效 `general_qa` 中固定随机抽 512 条，保留重复问题的不同回答。
新的 [A2/Dolly loss 与梯度 notebook](experiments-analysis/2026-10-22_093305Z_math-algebra-numbertheory-dolly-lengthonly-qwen25-1p5b-seed42/analysis.ipynb)
按首轮 notebook 的结构展示两域 loss 分布、ROC、top-k 和仅基于 A2/C 的长度对照。
A2/C 的 loss 下降量 AUC 为 **0.6162**，相对下降比例 AUC 为 **0.8563**；
原始 loss 与 prompt 长度本身已有很强的区分信号，解释时需一起比较。

UltraChat 复验也已完成：[A2/UltraChat loss 与梯度 notebook](experiments-analysis/2026-10-22_113752Z_math-algebra-ultrachat-lengthonly-qwen25-1p5b-seed42/analysis.ipynb)。
固定抽取 `ultrachat_200k/test_sft` 的 512 条有效首轮问答，loss 下降量和相对下降比例 AUC 分别为 **0.9857**、**0.9869**，
两域长度匹配后分别为 **0.9727**、**0.9722**。本次 UltraChat 平均 loss 上升，A2 平均 loss 下降；
与 Dolly 的差异说明结果依赖通用数据来源，单次复验不能估计全部随机性。
三个 notebook 的末尾均追加了全局/逐层梯度、最大单参数梯度、A1 方向相似度、长度对照及观察；原有 loss 分析保持不变。

已实现数据准备、完整状态保存/恢复、SFT、逐样本打分、bootstrap 评估和作图。
首轮配置为 MATH Algebra A1=1,024、A2=512，Number Theory B=512，
`Qwen/Qwen2.5-1.5B-Instruct` 全参数 SFT 1 epoch。

本地环境已配置 `.venv`，复用现有 CUDA PyTorch，其他依赖见 `requirements.txt`。
在仓库根目录选择空闲 GPU 后运行或恢复整个流程：

```bash
source scripts/env.sh
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/run_pilot.py --config configs/pilot.yaml
```

该命令复用完整的下载、准备数据和评分产物，训练从最近一个完整且配置匹配的
checkpoint 自动恢复。训练配置或训练代码改变时会拒绝续接旧 run，需使用新 run ID。
官方 test 不参与这轮实验；最终 checkpoint 是预先固定的主结果。
半程 checkpoint 同样完整保存，如需趋势诊断可加 `--score-intermediate` 单列报告。

测试与真实续训检查：

```bash
source scripts/env.sh
mkdir -p /mnt/tianzhu_hot/data-picking/tests
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest -q \
  -o cache_dir=/mnt/tianzhu_hot/data-picking/cache/pytest \
  --basetemp=/mnt/tianzhu_hot/data-picking/tests/core-check tests/
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/train_sft.py \
  --subset-size 64 --run-suffix=-resume-check-new --stop-after-steps 1
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/train_sft.py \
  --subset-size 64 --run-suffix=-resume-check-new
```

完整训练产物保留在指定挂载，便于 review 的分析快照放在仓库。
因 blobfuse2 不支持部分 POSIX 临时文件操作，仅短暂的运行 scratch 使用
`/dev/shm/data-picking-*` 内存盘。

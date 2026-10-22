# data-picking

研究如何从候选数据中选择最有助于降低目标 domain loss 的样本。

第一步是做一个 domain 区分预实验：将 A 分成 A1/A2，仅在 A1 上做 SFT，
然后比较 reference model 与 SFT model 在 A2、B 上的逐样本 loss，检验
loss 下降量和相对下降比例能否识别 A2。

- [预实验实现方案（已确认）](docs/pilot_design.md)
- [A/B 数据源建议](docs/data_sources.md)
- [项目约定](AGENTS.md)

所有数据缓存、checkpoint、训练日志、逐样本分数和实验结果统一存放在
`/mnt/tianzhu_hot/data-picking`，代码和配置保留在本仓库。

首轮已在本地 GPU 1（H100）完成。主指标 loss 下降量的 AUC 为 **0.6849**
                                                                                                          ,,,,,,,,,, （95% CI 0.6534–0.7160），相对下降比例的 AUC 为 **0.7007**。
长度匹配后主指标 AUC 为 **0.6524**，仍高于随机；但相对原始 loss 的额外
收益在该对照中尚不确定，也尚未确认优于只看 SFT 后的 loss。
这支持同域识别信号，尚不证明实际选数训练收益。

- [完整结果报告](/mnt/tianzhu_hot/data-picking/runs/math-algebra-numbertheory-qwen25-1p5b-seed42/report.md)
- [完整指标与置信区间](/mnt/tianzhu_hot/data-picking/runs/math-algebra-numbertheory-qwen25-1p5b-seed42/metrics/metrics.json)
- [最终 checkpoint 位置](/mnt/tianzhu_hot/data-picking/runs/math-algebra-numbertheory-qwen25-1p5b-seed42/training_complete.json)
- [测试和真实续训验证](/mnt/tianzhu_hot/data-picking/runs/math-algebra-numbertheory-qwen25-1p5b-seed42/validation_summary.json)

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

所有需保留的产物都在指定挂载中；因 blobfuse2 不支持部分 POSIX 临时文件操作，
仅短暂的运行 scratch 使用 `/dev/shm/data-picking-*` 内存盘。

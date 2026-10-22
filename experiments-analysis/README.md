# 实验分析

每个实验用一个独立子目录保存已执行的 `analysis.ipynb` 和小体积结果快照。
直接打开 notebook 即可查看表格、图和结论；重跑分析只依赖快照和 CPU，不需要模型、GPU 或原始挂载。

目录格式为 `YYYY-MM-DD_HHMMSSZ_<run_id>/`：使用 UTC 时间，按目录名字典序即可从旧到新排列。
时间代表该版原始分析结果的生成时间，后续编辑 notebook 不改变目录时间。
首轮 A/B run 没有独立的分析时间字段，因此首次导出取 `metrics/metrics.json` 的修改时间，
作为分析完成时间的近似。Dolly 扩展使用指标文件中的 `completed_at_utc`。
每个目录的时间及依据固定记录在 `manifest.json`，不是 notebook 导出时间。

<!-- experiments-index:start -->

| 结果时间（UTC） | 实验 notebook | 模型 | 下降量 AUC | 相对下降比例 AUC |
| --- | --- | --- | --- | --- |
| 2026-10-21 17:23:47 | [algebra / number_theory · seed 42](2026-10-21_172347Z_math-algebra-numbertheory-qwen25-1p5b-seed42/analysis.ipynb) | Qwen2.5-1.5B-Instruct | 0.6849 | 0.7007 |
| 2026-10-22 09:33:05 | [algebra / Dolly C · loss](2026-10-22_093305Z_math-algebra-numbertheory-dolly-lengthonly-qwen25-1p5b-seed42/analysis.ipynb) | Qwen2.5-1.5B-Instruct | 0.6162 | 0.8563 |
| 2026-10-22 11:37:52 | [algebra / UltraChat C · loss](2026-10-22_113752Z_math-algebra-ultrachat-lengthonly-qwen25-1p5b-seed42/analysis.ipynb) | Qwen2.5-1.5B-Instruct | 0.9857 | 0.9869 |

<!-- experiments-index:end -->

每个实验目录的内容：

```text
YYYY-MM-DD_HHMMSSZ_<run_id>/
  analysis.ipynb       # 已执行的设置、指标、图表、对照、结论和验证记录
  manifest.json        # 来源、时间依据、快照哈希、notebook 执行状态
  snapshot/
    metrics.json
    per_example.parquet
    length_matched_ids.json
    effective_config.json
    training_complete.json
    validation_summary.json
    train.jsonl
```

Dolly 和 UltraChat notebook 只展示各自的 A2/C loss 对比，包含 C 抽样 manifest、重复统计和两域长度匹配快照；通过
`scripts/export_generalization_notebook.py --run-dir <run_dir>` 导出，具体配置见
[Dolly 扩展说明](../docs/dolly_extension.md)。
UltraChat 的首轮问答定义和来源说明见 [UltraChat 复验方案](../docs/ultrachat_extension.md)。

模型、optimizer、原始数据、token 缓存和完整运行日志保留在
`/mnt/tianzhu_hot/data-picking/runs/<run_id>/` 及其他对应产物目录，原始路径在 manifest 和 notebook 中注明。
快照是便于 review 的固定副本，不随挂载上的文件变化而自动改变。

导出后续实验（在仓库根目录执行）：

```bash
source scripts/env.sh
uv --no-cache pip install --python .venv/bin/python -r requirements-notebook.txt
.venv/bin/python scripts/export_analysis_notebook.py \
  --run-dir "/mnt/tianzhu_hot/data-picking/runs/<run_id>"
```

导出命令会复制小体积结果、检查哈希、执行所有 notebook 单元格并更新上面的索引，
不重新训练或评分，也不重新运行 bootstrap；notebook 重算并核对点估计，区间来自原始结果快照。
每个源文件限制为 10 MiB，防止意外把大规模数据放进仓库。
已有目录默认拒绝覆盖；`--overwrite` 会明确替换 notebook 和导出快照，使用前需保留人工 review 笔记。
索引标记之间由脚本维护，其他说明和人工备注可写在标记之外。

第一次在其他 Python 环境重跑 notebook 时，安装 `requirements-notebook.txt` 中的分析依赖，
并将 kernel 工作目录设为当前实验子目录或仓库根目录。

# 仓库结构与实验操作

所有命令从仓库根目录执行。训练、数据和完整测量产物写入 `/mnt/tianzhu_hot/data-picking`；
可 review 的 notebook 和小快照保存在 `experiments-analysis/`。

## 目录分工

```text
configs/                    # 已完成实验的固定配置，保持原路径和内容
src/data_picking/           # 数据、训练、评分、统计与 checkpoint 实现
scripts/
  env.sh                   # 缓存、挂载产物与运行环境
  experiment.py            # 统一命令入口
  workflows/run_pilot.py   # A1/A2/B 全流程编排
  notebooks/               # loss notebook 导出、梯度附录和共享展示工具
  maintenance/             # 下载、权重暂存、数值诊断
requirements/              # 分层依赖；共享版本只维护一份
tests/                     # 数值、数据规则、续训、统计与 CLI 检查
docs/                      # 实验设计、来源说明和本操作指南
experiments-analysis/      # 按 UTC 时间排序的已执行 notebook 和快照
```

核心模块按用途划分如下。模块路径稳定，便于核对已保存的训练/测量代码指纹。

| 用途 | 模块 |
| --- | --- |
| 配置、文件校验、随机状态 | `common.py` |
| MATH / Dolly / UltraChat 数据 | `data.py` / `generalization_data.py` / `ultrachat_data.py` |
| Math 文档检索数据与污染检查 | `web_retrieval.py`、`document_utils.py` |
| 共同模板和逐样本回答 NLL | `tokenization.py`、`loss.py` |
| SFT 与完整状态恢复 | `training.py`、`checkpoint.py` |
| loss 评分与梯度测量 | `scoring.py`、`gradient_patterns.py` |
| 两域/扩展统计 | `analysis.py`、`generalization_analysis.py` |
| notebook 用的小型分析快照 | `loss_review.py`、`gradient_review.py` |

## 环境

```bash
source scripts/env.sh
# 实验与 notebook 导出；复用本机已验证的 CUDA PyTorch。
uv --no-cache pip install --python .venv/bin/python -r requirements.txt
# 开发与测试额外包含 pytest。
uv --no-cache pip install --python .venv/bin/python -r requirements/dev.txt
.venv/bin/python scripts/experiment.py --help
```

在其他机器上只重跑已导出的 notebook，可仅安装 `requirements/notebook.txt`。
`requirements/training.txt` 和 `requirements/notebook.txt` 共用 `requirements/analysis.txt`。

## 命令入口

| 命令 | 用途 |
| --- | --- |
| `download` | 下载 pilot 固定的模型和 MATH 原始数据 |
| `prepare --config ...` | 按配置准备 MATH、Dolly、UltraChat 或文档检索数据包 |
| `train --config ...` | A1 SFT，自动恢复完整 checkpoint；`--check-only` 仅预检实际数据和训练计划 |
| `score --config ...` | pilot 逐样本评分；不传 checkpoint 时使用 ref |
| `analyze --config ...` | 根据配置选择 pilot 或扩展统计 |
| `gradients --config ... --model reference\|sft` | 同时记录逐样本 loss 和完整梯度统计 |
| `score-web --model reference\|sft --document-sanity` | 固定网页/target validation 评分；中间模型使用 `--checkpoint` |
| `score-probe-qa` | 新数学 probe 在原 A2/UltraChat 上的交叉评分 |
| `analyze-probe [--qa-only]` | 固定 QA 与网页 checkpoint 轨迹诊断；不选择 checkpoint |
| `export-probe --run-dir ...` | 将两个诊断写入新的已执行 notebook |
| `prepare-quality` / `score-quality --model reference\|sft` | 固定 OpenStax / FineMath 来源准备及现有模型评分 |
| `analyze-quality` / `export-quality --run-dir ...` | 两种来源分别与原背景比较并导出 notebook |
| `run-pilot --config ...` | 下载、准备、ref 评分、SFT、SFT 评分和统计全流程 |
| `export --run-dir ...` | 根据结果类型选择 loss notebook 模板 |
| `append-gradients --notebook-dir ... --measurement-run ...` | 保留已有 loss 单元格，追加梯度附录 |
| `stage-weights --config ... --blob-config ...` | 校验并暂存模型到 RAM，缓解 blobfuse 大文件读取 |
| `check-gradients --config ...` | 诊断数值后端；不执行 optimizer step |

每个命令都支持 `--help`。需要 GPU 的命令先用 `nvidia-smi` 检查空闲设备，再显式设置
`CUDA_VISIBLE_DEVICES`；不要使用别人正在占用的 GPU。

## Pilot 训练与恢复

```bash
source scripts/env.sh
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/experiment.py run-pilot --config configs/pilot.yaml
```

单独执行某阶段：

```bash
.venv/bin/python scripts/experiment.py download --config configs/pilot.yaml
.venv/bin/python scripts/experiment.py prepare --config configs/pilot.yaml
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/experiment.py train --config configs/pilot.yaml
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/experiment.py score --config configs/pilot.yaml --checkpoint final
.venv/bin/python scripts/experiment.py analyze --config configs/pilot.yaml
```

新训练实验复制配置并更换 `run_id`，确认 `save_optimizer: true`、`auto_resume: true` 和持久产物路径。
恢复时继续核对模型、数据、训练配置和实现指纹，不能删除校验来复用不匹配的 checkpoint。
配置或训练方法有变时使用新 run；已完成实验的配置保持固定。

## Dolly / UltraChat 评估扩展

以 UltraChat 为例，Dolly 替换为 `configs/dolly_extension.yaml`：

```bash
source scripts/env.sh
.venv/bin/python scripts/experiment.py prepare --config configs/ultrachat_extension.yaml
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/experiment.py gradients --config configs/ultrachat_extension.yaml --model reference
CUDA_VISIBLE_DEVICES=3 .venv/bin/python scripts/experiment.py gradients --config configs/ultrachat_extension.yaml --model sft
.venv/bin/python scripts/experiment.py analyze --config configs/ultrachat_extension.yaml
```

先完成数据准备再启动测量，两个模型均完整结束后再统计。挂载慢时，可沿用经过校验的
`runtime_weights.json`，或先运行 `stage-weights`。这些是评估扩展，不执行新的 SFT。

文档检索的 4,096 条目标集使用 `prepare --config configs/web_retrieval_data_4096.yaml`。它只准备 MATH target、
数学文档/CC 混合池及通用 SFT 对照池，定义和产物格式见 [数据说明](web_retrieval_data.md)。
对应 SFT 配置为 `configs/web_math_sft.yaml`，先用 `train --config ... --check-only` 核对 64 步计划。

## Notebook 与验证

```bash
.venv/bin/python scripts/experiment.py export --run-dir /mnt/tianzhu_hot/data-picking/runs/<run_id>
.venv/bin/python scripts/experiment.py append-gradients \
  --notebook-dir experiments-analysis/<directory> \
  --measurement-run /mnt/tianzhu_hot/data-picking/runs/<measurement_run_id>
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest --basetemp=/dev/shm/data-picking-tests -q
```

导出先生成 loss 部分，梯度另行追加。已有 notebook 默认拒绝覆盖；不要在更新附录时重建整个
notebook。`--replace-appendix` 只替换标记的附录，保留其前面的单元格；已有人工附录笔记应先保存。
目录时间、原始 scores、历史 manifest 中的代码路径与哈希是来源记录，不随仓库整理改写。

pytest 的缓存路径已在 `pyproject.toml` 固定到挂载；测试的临时 checkpoint 使用上述 RAM scratch。

## 旧入口对应关系

旧的单阶段包装脚本已移除，功能全部由统一入口提供：

| 原文件名 | 新命令（接在 `scripts/experiment.py` 后） |
| --- | --- |
| `prepare_data.py` / `prepare_dolly.py` / `prepare_ultrachat.py` | `prepare --config <对应配置>` |
| `train_sft.py` | `train` |
| `score.py` | `score` |
| `analyze.py` / `analyze_generalization.py` | `analyze --config <对应配置>` |
| `measure_gradients.py` | `gradients --config <对应配置> --model ...` |
| `run_pilot.py` / `download_assets.py` | `run-pilot` / `download` |
| `export_analysis_notebook.py` / `export_generalization_notebook.py` | `export --run-dir ...` |
| `append_gradient_analysis.py` | `append-gradients` |
| `stage_evaluation_weights.py` / `check_gradients.py` | `stage-weights` / `check-gradients` |

旧的 `requirements-generalization.txt` 改用根目录 `requirements.txt`；
`requirements-notebook.txt` 改用 `requirements/notebook.txt`。

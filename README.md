# data-picking

研究如何从候选数据中选择最有助于降低目标 domain loss 的样本。
实验从同域识别扩展到文档来源识别和直接训练效用对照：比较 ref/SFT 模型的逐样本 loss 和梯度，
并检验不同文档来源的训练是否降低目标 SFT loss。loss 下降量与相对下降比例并列考察，不设主次。
域区分效果本身不证明后续选数训练的实际收益。

## 已完成实验

模型为 `Qwen/Qwen2.5-1.5B-Instruct`。前三项沿用 Algebra A1=1,024、step 32；
文档检索另用七类 MATH 4,096 条训练 64 步。Notebook 保存执行输出与结果快照，可在 CPU 上重跑；
原四项实验包含梯度附录，后续交叉诊断复用已有 checkpoint 测量 loss。

| 对比 | 下降量 AUC | 相对下降比例 AUC | Notebook |
| --- | --- | --- | --- |
| A2 / Number Theory | 0.6849 | 0.7007 | [A2/B](experiments-analysis/2026-10-21_172347Z_math-algebra-numbertheory-qwen25-1p5b-seed42/analysis.ipynb) |
| A2 / Dolly | 0.6162 | 0.8563 | [Dolly](experiments-analysis/2026-10-22_093305Z_math-algebra-numbertheory-dolly-lengthonly-qwen25-1p5b-seed42/analysis.ipynb) |
| A2 / UltraChat | 0.9857 | 0.9869 | [UltraChat](experiments-analysis/2026-10-22_113752Z_math-algebra-ultrachat-lengthonly-qwen25-1p5b-seed42/analysis.ipynb) |
| OpenWebMath 注入文档 / CC 背景 | 0.3046 | 0.2137 | [文档检索](experiments-analysis/2026-10-23_095832Z_math-web-probe-qwen25-1p5b-n4096-b64-s64-seed42/analysis.ipynb) |
| 新 MATH step-64 模型：原 A2 / UltraChat | 0.9915 | 0.9960 | [QA 交叉复测与网页 checkpoint 诊断](experiments-analysis/2026-10-23_165639Z_math-web-probe-diagnostics-qwen25-1p5b-seed42/analysis.ipynb) |
| OpenStax 教材 / CC 背景（117 / 9,000） | 0.1099 | 0.0691 | [文档来源复验](experiments-analysis/2026-10-24_102053Z_math-document-quality-qwen25-1p5b-step64-seed20261024/analysis.ipynb) |
| FineMath-4+ / CC 背景（1,000 / 9,000） | 0.1685 | 0.0818 | [文档来源复验](experiments-analysis/2026-10-24_102053Z_math-document-quality-qwen25-1p5b-step64-seed20261024/analysis.ipynb) |

上表为原始池点估计。置信区间、长度控制、原始 loss 基线和梯度诊断见各 notebook。
UltraChat 平均 loss 上升，A2 下降；与 Dolly 的差异说明需要考察通用数据来源的影响。
文档检索中，MATH 验证 NLL 降到 0.5940，但两项变化分数都未按预设方向富集注入文档；
长度匹配 AUC 分别为 0.2924、0.2123。未翻转分数，标签仍只表示来源，不能当作语义数学标签或选数训练收益。
后续诊断中，新模型保留了原 A2/UltraChat 区分信号；网页在最早观测的 step 16 已反向
（两项 AUC 0.2866、0.2320），step 32/48 也反向。完整轨迹与固定长度控制见诊断 notebook。
换用 OpenStax / FineMath-4+ 后也未恢复预设正方向。OpenStax 的长度控制仅保留 25 个小节，
其中 17 个是短章节导言，其代表性有限；FineMath 固定抽样仍有推广、公式缺失等噪声。

直接训练效用实验另从相同 reference 出发，每种文档来源训练三个 seed，每次 64 步、
1,048,576 个有效预测 token，再测同一套 512 条留出 MATH 回答。以下为三个 seed 的平均值，
reference NLL 为 0.743498；相对下降比例是逐题比例的平均，负值表示 loss 上升。

| 来源 | 最终目标 NLL | loss 下降量 | 相对下降比例 |
| --- | --- | --- | --- |
| General FineWeb | 0.715040 | +0.028459 | +3.441% |
| OpenWebMath | 0.709711 | +0.033788 | +3.951% |
| OpenStax | 0.891925 | −0.148427 | −27.372% |
| FineMath-4+ | 0.721796 | +0.021703 | +2.206% |

OpenWebMath 相对 general 的两项平均优势均为正，但同时重采样 seed 和问题组的 95% 区间均跨零；
FineMath 改善较小，OpenStax 使目标 loss 上升。这是固定预算下的来源对照，尚未验证逐样本选数。
逐种子结果、配对区间与检查记录见
[直接训练效用 notebook](experiments-analysis/2026-10-25_161254Z_document-utility-qwen25-1p5b-t1048576-s64/analysis.ipynb)。

## 使用入口

```bash
source scripts/env.sh
.venv/bin/python scripts/experiment.py --help
```

所有操作由 `scripts/experiment.py` 提供，包括数据准备、训练/恢复、评分、统计、notebook 导出与梯度附录。
完整命令、环境安装、GPU 选择、目录说明和旧入口对应关系见 [操作指南](docs/workflow.md)。

```text
configs/                  已完成实验的固定配置
src/data_picking/         核心数据、训练、测量与统计实现
scripts/                  统一入口，以及 workflows / notebooks / maintenance 工具
requirements/             共享分析、训练、notebook 与测试依赖
experiments-analysis/     按 UTC 时间排列的 notebook 和小型结果快照
```

所有模型、数据、checkpoint、日志和完整测量保存在 `/mnt/tianzhu_hot/data-picking`；
仓库只保留代码、配置、文档及用户指定的 notebook 快照。

## 文档

- [实验分析索引与 notebook 更新约定](experiments-analysis/README.md)
- [预实验设计](docs/pilot_design.md) · [最初的数据源选择](docs/data_sources.md)
- [Dolly 实验](docs/dolly_extension.md) · [UltraChat 复验](docs/ultrachat_extension.md)
- [Math documents / CC 网页数据准备](docs/web_retrieval_data.md)
- [Math SFT → 网页检索验证方案](docs/web_retrieval_experiment.md)
- [QA 交叉复测与网页 checkpoint 诊断](docs/probe_diagnostics.md)
- [OpenStax / FineMath-4+ 文档来源复验](docs/document_quality.md)
- [数学文档对目标 SFT loss 的直接训练效用](docs/document_utility.md)
- [仓库结构与操作指南](docs/workflow.md) · [项目约定](AGENTS.md)

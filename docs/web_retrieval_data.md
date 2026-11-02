# Math documents / Common Crawl 数据准备

本轮准备数据与训练配置，不执行模型训练或评分，也不报告检索效果。
当前 4,096 条训练目标集使用 [web_retrieval_data_4096.yaml](../configs/web_retrieval_data_4096.yaml)。
原 [web_retrieval_data.yaml](../configs/web_retrieval_data.yaml) 和 1,024 条版本保留为历史数据。
统一入口：

```bash
source scripts/env.sh
CUDA_VISIBLE_DEVICES='' .venv/bin/python scripts/experiment.py prepare --config configs/web_retrieval_data_4096.yaml
```

当前产物位于 `/mnt/tianzhu_hot/data-picking/data/processed/math-web-retrieval-v2-4096-seed20261023/`。
原模型、旧实验配置和三个 notebook 保持不变。

## 来源与抽样范围

| 部分 | 数据源 | 范围和用途 |
| --- | --- | --- |
| math SFT target | `EleutherAI/hendrycks_math`，revision `21a5633873b6a120296cce3e2df9d5550074f4a3` | 七个题型的官方 train；4,096 条 target train、512 条 validation |
| 注入的数学相关文档 | `open-web-math/open-web-math`，revision `fde8ef8de2300f5e778f56261843dab89f230815` | 按 seed 抽取三个固定分片，取 `CC-MAIN-2023-06` 内的文档，选 1,000 篇 |
| 未标注背景网页 | `HuggingFaceFW/fineweb`，revision `9bb295ddab0e05d785b879661af7260fed5140fc` | 同一 CC 抓取批次的一个固定随机分片，选 9,000 篇 |
| 通用 SFT 对照池 | `HuggingFaceH4/ultrachat_200k`，revision `8049631c405ae6576f93f445c6b8166f76f5505a` | 一个固定 `train_sft` 分片的首轮问答，准备 4,096 条供后续等 token 预算对照 |

每个源文件的路径、大小和 SHA-256 均固定在配置中；下载后验证。源数据和 dataset card 保留在挂载。
网页的抽样总体是指定分片与抓取批次，**不是整个 Common Crawl 的均匀样本**。
已选分片内的有效文档按固定随机顺序选取，来源字段和抽样优先序都可复核。

OpenWebMath 与 FineWeb 都来自 Common Crawl，但上游正文提取、质量过滤不同。
本地只统一换行和首尾空白，不能消除上游来源差异。
OpenWebMath 还包含物理、计算机等数学相关内容；正例标记表示“来自注入的 OpenWebMath 集合”，
不代表人工核验过的数学标签。背景集可能包含数学内容，不把它当作已确认的非数学负例。
两种网页语料的数据集许可为 ODC-By 1.0，原网页自身许可仍适用；MATH 和 UltraChat 数据集声明 MIT。

## Target 划分与污染检查

MATH 只使用官方 train。沿用文本模型的基本有效性、完整聊天模板不超过 2,048 tokens 的要求，
排除明显依赖图像/图示的题目；保留 problem/solution、题型、难度、源行号和 tokenization hash。
先按问题精确/近重复分组，每组取一个代表，再按题型和难度分层抽取 train/validation。

原 Algebra/Number Theory 实验的 A2/B 及检测到的重复问题组整组保留为旧评估数据，
不进入本次新 target train/validation。新训练与验证的 group ID 也互不相交。

4,096 条版本保留原 1,024 条训练样本，从其余可用题目按题型/难度补充 3,072 条。
原 512 条 validation 的内容、顺序及哈希不变。网页和通用 SFT 池按扩大后的 target 重新检查重合。

网页和通用 SFT 对照池检查新 target train/validation 以及原 A2/B 中的题目、答案。
对至少 12 个规范化 token 的问题、至少 40 个 token 的答案，用稀有 5-token 锚点检索，
再检查规范化原文包含或至少 0.8 的 shingle 包含比例。疑似重合网页单独保存，
按预先固定的随机顺序顺延补足数量，不依据模型分数或人工主题判断替换。
短的通用答案不会用作污染证据；这不是对语义改写、官方 test 或模型预训练污染的完整保证。

网页内部按规范化正文、URL 和 MinHash 候选的精确 Jaccard 复核建立重复组；
保留重复网页及跨来源重合记录，后续统计可按文档组重采样。
近重复搜索采用 64 permutations、16×4 LSH bands、5-token shingle、阈值 0.8，是近似检查。

## 原始文档与评分窗口

网页不加聊天模板，不设整篇长度上限，不因数学、代码、语言或风格额外过滤。
正文完整分词后切成最多 2,048 tokens 的窗口；相邻窗口共享一个上下文 token。
每篇文档除第一个 token 外，其余 token **恰好评分一次**，尾部不丢弃、不把不同网页拼接。

以后先按各窗口的实际评分 token 数聚合整篇平均 NLL，再并列计算 loss 下降量和相对下降比例。
文档来源、URL、上游分数等只放在 provenance/labels 中，不拼进模型输入。

```text
math-web-retrieval-v2-4096-seed20261023/
  effective_data_config.json
  source_manifest.json
  summary.json
  COMPLETED.json                 # 所有文件关闭、校验后写入
  target/
    train.jsonl                  # 4,096 条；split=A1
    validation.jsonl             # 512 条；split=A2
    manifest.jsonl
    duplicate_edges.json
    summary.json
  web/
    documents.parquet            # 正文、文档 ID、group ID、token 数
    windows.parquet              # 固定 input_ids、回答/评分 mask、边界及 hash
    document_manifest.parquet    # URL、crawl、源文件/源行、上游 metadata
    labels.jsonl                 # 注入来源标记；semantic_math_label 未人工标注
    sampling_audit.jsonl
    quarantine.jsonl             # 疑似 target 重合文档及匹配证据
    duplicate_edges.json
  control_sft/
    pool.jsonl                   # 首轮完整问答的通用 SFT 候选池
    manifest.jsonl
    summary.json
  sft/                           # 供现有 trainer 读取的校验后副本
    A1.jsonl
    A2.jsonl
    manifest.jsonl
    summary.json
    COMPLETED.json
```

通用 SFT 池尚未决定实际训练子集和步数；下一步训练前需要固定与 math SFT 相同的训练 token 预算。
此配置是数据配置，CLI 会拒绝直接将它作为训练或评分配置使用。

## SFT 配置

训练配置是 [web_math_sft.yaml](../configs/web_math_sft.yaml)，与纯数据配置分开：

| 设置 | 值 |
| --- | --- |
| 训练样本 / 验证样本 | 4,096 / 512 |
| 有效 batch size | 64 |
| micro batch / 梯度累积 | 2 / 32 |
| epoch / optimizer steps | 1 / 64 |
| checkpoint steps | 16、32、48、64 |
| 学习率 / warmup | 2e-5 / 7 steps |
| 参数 / autocast / attention | float32 / bf16 / SDPA math |
| 保存 optimizer / 自动恢复 | true / true |

所有 checkpoint 保存模型、optimizer、scheduler、RNG、global step 和数据迭代状态。
新增配置使用独立 run ID，不续接旧 Algebra checkpoint。预检直接读取已准备数据，
核对哈希、样本数、总步数与保存频率，输出 `training_preflight.json`：

```bash
source scripts/env.sh
CUDA_VISIBLE_DEVICES='' .venv/bin/python scripts/experiment.py train --config configs/web_math_sft.yaml --check-only
# 启动训练时先检查空闲 GPU，再使用：
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/experiment.py train --config configs/web_math_sft.yaml
```

上述记录对应数据和配置准备阶段。后续训练及检索验证见 [验证方案](web_retrieval_experiment.md)。

4,096 条版本已准备并预检：包含原 1,024 条训练样本，训练回答共 830,862 tokens；
512 条 validation 与旧版本逐字节一致。重新进行 target 重合检查后，网页混合池也与旧版本一致。
38 项测试通过，数据哈希、训练集包含关系和验证隔离均已核对。

[当前数据校验记录](/mnt/tianzhu_hot/data-picking/data/validation/math-web-retrieval-v2-4096-seed20261023.json) ·
[实际训练计划](/mnt/tianzhu_hot/data-picking/runs/math-web-probe-qwen25-1p5b-n4096-b64-s64-seed42/training_preflight.json)

## 原 1,024 条版本的准备记录

- Target：1,024 条训练、512 条验证，覆盖全部七个 MATH 题型；训练回答共 204,713 tokens。
  原 A2/B 的 1,024 条题目及其检测到的重复组已保留，未进入新划分。
- 网页：1,000 篇注入文档、9,000 篇背景文档；抽样总体分别为固定分片中的 4,540 与 309,148 篇同批次网页。
- 共 11,688 个窗口、9,188,836 个可评分 token；最长文档 352,187 tokens，完整保留并分窗。
- 当前规则下未检测到入选网页之间的重复或与受保护 target 的重合；不表示语义重复已被完全排除。
- 通用 SFT 对照池：4,096 条；准备时排除 3 条超长和 3 条空回答。
- 34 项测试通过。独立校验核对了 18 个产物文件哈希、所有窗口的 mask/边界/覆盖、
  所有网页的原始 crawl 元数据，并重新分词核验 33 篇文档（含最长文档）。

[汇总 JSON](/mnt/tianzhu_hot/data-picking/data/processed/math-web-retrieval-v1-seed20261023/summary.json) ·
[独立校验记录](/mnt/tianzhu_hot/data-picking/data/validation/math-web-retrieval-v1-seed20261023.json)

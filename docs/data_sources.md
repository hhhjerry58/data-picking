# A/B 数据源建议

状态：用户已确认首轮采用 MATH Algebra/Number Theory，并授权本地 GPU 训练。下文保留数据源选择依据；正式准备后成功保留 A1=1,024、A2=B=512，A2/B 的 level 分布完全一致。完整过滤统计和 manifest 保存在产物根目录的 `data/processed/` 中。

## 首选：MATH 中的 Algebra 与 Number Theory

来源：[EleutherAI/hendrycks_math](https://huggingface.co/datasets/EleutherAI/hendrycks_math)，[数据卡](https://huggingface.co/datasets/EleutherAI/hendrycks_math/blob/main/README.md)。本次查到的 revision 为 `21a5633873b6a120296cce3e2df9d5550074f4a3`，正式准备数据时固定这个 revision。

| 用途 | 子集与 split | 官方原始条数 | 建议抽取量 |
| --- | --- | --- | --- |
| A1/A2 的来源 A | `algebra / train` | 1,744 | A1=1,024，A2=512 |
| B | `number_theory / train` | 869 | 512 |
| 后续独立 A_test 候选 | `algebra / test` | 1,187 | 首轮不训练、不打分、不用来调整方案 |

推荐这一组的原因：同一数据集、同一种题目与完整解答格式，并且两边都有 `problem`、`solution`、`level`、`type`。可以直接把 `problem` 映射为 prompt、`solution` 映射为 response，用完整解答计算 loss，无需生成答案或额外补充解释。

这测的是**数学子领域**的区分能力。代数与数论之间存在技能迁移，分离可能比“数学 vs 历史”更弱；有效说明同来源的细粒度信号值得研究，无效也不能据此否定所有更宽领域的效果。

准备规则：

- 只使用官方 train 划分构造 A1/A2/B。保留官方 test 作为后续候选评估数据，使用前仍检查与训练数据及候选池的重复。官方 test 并不保证没有出现在初始模型预训练中。
- 所有领域采用同样的文本过滤规则。首轮剔除 prompt 或 solution 中含 `[asy]`、图片标签或必须依赖外部图片的样本，不把绘图代码当作额外的领域标记；记录剔除数量。
- 先去重、过滤空回答和超过 2,048 tokens 的样本，再分组划分；若不足建议条数，缩小规模，不混入官方 test 凑数。
- 对 A1/A2 按 `level` 尽量保持相同难度组成；在 A2/B 中优先使用共同难度层、按相同配额抽样。`level` 是粗粒度标注，仍需报告实际分布和原始 loss 基线。
- 两域统一模板，不将 `type` 或 `level` 注入模型输入；另外做方案中规定的 prompt/response 长度匹配分析。
- 数据集没有直接提供题族 ID，应由规范化题目 hash 和近重复分组生成实验 ID/group ID，并保留原始子集及行号用于追溯。
- 按 1,024 条 A1、有效 batch 32、1 epoch 计算，共 32 个 optimizer steps。因此保存间隔改为 16 steps，以保留半程与最终状态；其余训练约定沿用已确认方案。

为什么暂时不选 Geometry：按数据服务返回的 train 前 100 条检查，Geometry 有 65 条在题目或解答中包含 `[asy]`，Algebra 有 5 条，Number Theory 为 0 条。Geometry 会额外引入图形表示及绘图代码差异。这是按返回顺序的小样本检查，不是总体比例估计，也不代表 Algebra/Number Theory 全部没有图形问题。

本次各取前 100 条时，Algebra 和 Number Theory 的平均解答字符数分别为 361 和 457；这也说明同来源仍需要长度控制。尚未完成 tokenizer 长度过滤和全部数据去重，不能保证最终一定保留建议数量。

## 较宽领域的复验：Stack Exchange Physics 与 History

来源：[HuggingFaceH4/stack-exchange-preferences](https://huggingface.co/datasets/HuggingFaceH4/stack-exchange-preferences)。本次查到 revision 为 `c7bda74048748f55749cd663c3d8d1025a841fd9`。

- A：`data/physics.stackexchange.com/` 下的问答。
- B：`data/history.stackexchange.com/` 下的问答。
- 两个路径已在仓库文件列表中确认存在；不使用对应的 `*.meta.stackexchange.com`，后者主要讨论网站管理。
- 数据有 `qid`、`question`、`answers`、`date`、`metadata`；每条回答含 `answer_id`、`text`、`pm_score`、`selected` 等字段。
- 每个问题只保留一条参考回答：固定选最高 `pm_score`，同分时优先被接受的回答，再按 `answer_id` 确定性打破同分。若希望过滤低分答案，两域使用同一阈值。
- 对问题和答案执行相同的 HTML 清洗，保留正文、段落、数学内容；剔除依赖缺失图片的条目。来源链接、作者和站点字段保存在 metadata 中，不作为模型的领域提示。
- 使用 `site + qid` 作为问题组 ID，避免不同子站相同数字 qid 相互冲突；同题的不同回答不能跨 A1/A2。
- 只下载对应子站的分片，不下载整个约 22 GB 的集合。子站完成清洗后的数量尚未核验，原方案的 2,048/1,024/1,024 仅作为目标量。

这组更接近“物理 vs 历史”的大领域场景，且都来自同一个问答平台。代价是清洗更多、答案质量不完全一致，并且**领域与子站来源仍然绑定**，无法单凭这个实验排除社群写作风格的影响。需要保留来源和署名信息，并按数据集许可处理后续对外发布。

建议顺序：先用 MATH Algebra/Number Theory 完成首轮，再用 Stack Exchange Physics/History 检验能否推广到更宽的领域差异。两组实验分开报告，不根据观察到的 AUC 只保留其中表现较好的组合。

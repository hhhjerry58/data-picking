# 用 SFT 前后的 loss 变化区分目标域：预实验实现方案

状态：首轮本地训练、双模型打分和评估已完成，结果入口见 [README](../README.md)。实验产物根目录为 `/mnt/tianzhu_hot/data-picking`。首轮采用 MATH Algebra/Number Theory，见 [数据源说明](data_sources.md)；生效参数以 `configs/pilot.yaml` 及 run 内 `effective_config.json` 为准。

## 1. 要回答的问题

给定初始模型 θ₀，以及来自不同 domain 的数据 A、B：

1. 将 A 划分为训练集 A1 和未见过的评估集 A2。
2. 只在 A1 上做 SFT，得到 θ_A；θ₀ 保持不变，作为 reference model。
3. 对 A2、B 的每条样本，分别计算 θ₀ 和 θ_A 的 loss。
4. 检验 loss 的下降量、相对下降比例能否将 A2 排在 B 前面。

核心假设是：A1 上学到的能力可以泛化到 A2，因此 A2 的 loss 下降应当比 B 更明显。这需要实验验证，不能预设一定成立。

一个直观依据是：若先把训练简化成很小的一步 SGD，`θ_A ≈ θ₀ - η g_A1`，那么 `L₀(x) - L_A(x) ≈ η ∇L₀(x) · g_A1`。loss 下降量近似反映样本梯度与 A1 平均梯度的对齐程度；实际使用多步 AdamW 后，这只是理解信号的局部近似。

这个实验测的是“对 A 域适应的响应是否可用于识别同域数据”。它不能直接证明“得分越高的样本，拿来继续训练就越能降低 A 域 loss”：同域数据也可能过于简单、重复，或没有额外训练价值。

## 2. 首轮实验配置

| 项目 | 建议默认值 | 说明 |
| --- | --- | --- |
| Reference model | `Qwen/Qwen2.5-1.5B-Instruct` | 小模型便于先验证流程；固定模型/tokenizer revision |
| A1 | 1,024 条 | MATH Algebra，唯一的 SFT 训练数据 |
| A2 | 512 条 | MATH Algebra 评估，正类 |
| B | 512 条 | MATH Number Theory 评估，负类 |
| A/B 来源 | `EleutherAI/hendrycks_math` 的官方 train | 固定 revision，官方 test 不使用 |
| 划分 seed | 2026 | 首轮固定一次划分并落盘 |
| 训练 seed | 42 | 有信号后可在固定划分上补 43、44 |
| 训练方式 | 全参数 SFT，1 epoch | 不混入 B；不使用 RL、蒸馏或额外数据 |
| 优化器 | AdamW，lr `2e-5`，weight decay `0.01` | betas `(0.9, 0.999)`，eps `1e-8`，梯度裁剪 `1.0` |
| 学习率调度 | warmup 10%，之后 linear decay | 首轮固定，不在 A2/B 上调参 |
| 有效 batch size | 32 条 | 单卡 micro batch 2，梯度累积 16；显存不足可等效调整 |
| 数值精度 | 训练使用 fp32 参数/optimizer + bf16 autocast；打分使用完整 fp32 | token CE 和累计 loss 使用 float32 |
| 最大序列长度 | 2,048 tokens | prompt + response；首轮丢弃超长样本，不静默截断 |
| checkpoint | 每 16 个 optimizer steps，另存最终状态 | 默认规模共 32 steps，保留半程和最终模型 |
| 主报告 checkpoint | 最终的 1 epoch 模型 | 半程仅用于趋势诊断，不按 A2/B 结果选最优模型 |
| 恢复训练 | 保存 optimizer；自动 resume | 还需 scheduler、RNG、step 和数据迭代位置 |

数据不足时减少规模并记录实际数量，不通过复制或有放回抽样凑数。每个数据源经过过滤后的数量决定最终规模；配置在训练前固定。

A/B 的一个起点是数学问答和非数学问答，但应尽量匹配语言、回答形式和长度。若两边分别来自完全不同的数据集，首轮结论只能称为“这两个数据集可区分”；需要同来源或额外来源复验，才能更有把握地解释为 domain 信号。只含分类标签/选项字母的数据也能跑，但 loss 会受很少几个 token 支配，不作为首选。

## 3. 数据准备和防泄漏

首轮统一成单轮、带参考回答的 JSONL。示例仅展示字段，不是实际数据：

```json
{"id":"a-0001","domain":"A","group_id":"problem-0001","prompt":"问题文本","response":"参考回答","source":"dataset-name","source_revision":"pinned-revision"}
```

- `domain` 只供划分和分析使用，不进入模型文本。训练文件只包含 A1。
- 多轮数据、纯文本语料先不混入首轮；以后可分别定义多轮 assistant loss 或全文 loss。
- 对 A/B 使用同一 system prompt（默认无额外 system 指令）、同一 chat template。不额外加入数据源、领域名称或只出现在单侧的格式提示。
- 先统一字段、过滤空回答和超长样本，再分组去重和划分。记录各域原始数、过滤数、去重数和最终数。
- 精确去重同时检查规范化的 `prompt` 和 `prompt + response`。同一个问题的不同答案、改写、同文档切片应归到同一 `group_id`，不能跨 A1/A2。
- 没有原始 group ID 时，对规范化问题做字符 5-gram 相似度检查，例如 Jaccard ≥ 0.8 的样本建立近重复候选组；阈值在看模型分数前固定，并抽查候选。这个启发式不能保证排除所有语义重复。
- 检查 A/B 之间的重复或标签冲突；无法确认归属的重复组从主实验移除。对分组后的 A 做确定性划分，组边界优先于精确条数。
- 保存 manifest：样本 ID、group ID、split、来源/revision、文本 hash、token 数、过滤原因、随机种子。
- A2/B 不参与超参数、训练步数或 checkpoint 的选择；首轮不学习分类器，也不在评估集上搜索分类阈值。

## 4. 逐样本 loss 的定义

对样本 x = (prompt, response)，使用 teacher forcing，只计算参考回答正文的 token NLL，单位为 nats/token：

```text
Lθ(x) = - (1 / N(x)) * Σ[t ∈ response_mask] log pθ(token_t | tokens_<t)
```

其中 N(x) 是被计分的回答 token 数。主配置的训练和打分都使用这份 response mask：排除 system/user 文本、padding、角色标记和结束控制 token，保留回答正文的全部 token，包括推理过程和最终答案。不对生成答案计分，也不计算 prompt 的平均 loss。

实现时共享同一个预处理函数，并缓存 tokenization，保证两个模型输入的 token IDs、attention mask、response mask 完全一致。固定 tokenizer revision 和 chat template，不能分别使用两个 checkpoint 里可能不同的 tokenizer 设置。

需要特别注意：

1. 使用 causal shift：`logits[:, :-1]` 预测 `input_ids[:, 1:]`，mask 也同步移位。
2. `cross_entropy(..., reduction="none")` 得到 token loss；每条样本先求 mask 内总和，再除以自己的 N。不能直接把 `model.loss` 的 batch 标量当作逐样本分数。
3. 用经验证的 assistant token mask 或模板文本区间与 tokenizer offsets 建立 mask；验证首个回答 token、末尾控制 token、padding 和边界 token 的处理。不能只用“prompt 单独 tokenize 后的长度”盲切完整序列。
4. 两模型都使用 `eval()` 和 inference/no-grad 模式，关闭 dropout；打分不做 packing，按相同设置分批。回答 mask 为空时直接报错或在准备阶段剔除，不能除以零。
5. 训练目标同样按“每条回答先做 token 平均、再对有效 batch 做样本平均”计算，避免长回答仅因长度而获得更大权重；梯度累积与最后不足 batch 的归一化也必须一致。
6. 同时存 `sum_nll`、`mean_nll`、`prompt_tokens`、`response_tokens`，便于检查长度效应。样本 loss 使用 mean NLL。

## 5. 比较哪些指标

记 `L0 = Lθ₀(x)`，`LA = Lθ_A(x)`；统一为“分数越大，越倾向 A2”。
loss 下降量与相对下降比例并列考察，不设主次。

| 名称 | 定义 | 用途 |
| --- | --- | --- |
| `delta_loss` | `L0 - LA` | loss 下降量；正值表示 SFT 后 loss 下降 |
| `relative_gain` | `(L0 - LA) / max(L0, 1e-6)` | 相对下降比例；考虑原始 loss 尺度，但低 L0 时会不稳定 |
| `neg_ref_loss` | `-L0` | 基线：只用原始模型就能否识别 A 域 |
| `neg_sft_loss` | `-LA` | 基线：是否只看训练后的 loss 就够了 |

原始 loss 的方向还会受领域难度影响，因此附录也预先报告 `+L0`、`+LA` 两个反向基线，分别展示，不在看完 AUC 后翻转或替换主基线。若反向原始 loss 已能强区分，同样说明原始模型已有区分信息，不能只因 `-L0` 表现差就宣称 delta 提供了独有信息。

用户提出的 loss 比值 `LA / L0` 越小越倾向 A2；`relative_gain = 1 - LA/L0` 是它的同序改写（忽略数值保护），因此无需将二者当作独立结果。记录触发分母保护的样本数，不为了提高 AUC 而过滤这些样本。

`PPL_ref / PPL_sft = exp(delta_loss)` 与 `delta_loss` 排序相同，也不算一个新指标。对相同的参考回答，`N(x) * delta_loss` 才是完整回答的条件 log-likelihood ratio；使用平均 loss 是为了减少长度主导，且 θ₀ 并不是专门的 B 域模型，不能把这个分数直接解释成 A/B 的后验概率。

## 6. 如何判断有没有信号

固定 `label(A2)=1`、`label(B)=0`。首轮平衡的 A2/B 混合池中，每个指标都报告：

- **ROC-AUC**：排序区分能力，随机排序期望为 0.5。保持预定方向，即使 AUC < 0.5 也不事后翻转。
- **Average precision（AP）**：明确使用 average precision 定义，避免与另一种 PR 曲线积分混淆。
- **Precision@top 10% / top 20%**：按分数降序选出后，A2 所占比例；随机期望为 0.5。边界同分用预先固定的随机次序打破，不能用域标签或带域前缀的 ID 排序。
- A2/B 的 `L0`、`LA`、`delta_loss` 均值、中位数、分位数及正 delta 比例。

对 A2/B 分层 bootstrap 2,000 次，分别给两项指标的 AUC 和 top-k precision 的 95% 区间。有重复题族/文档组时按 group 重采样，同一组不能拆成独立样本。对于与原始 loss 基线的 AUC 差，在同一 bootstrap 样本上计算差值，使用配对区间，并明确标注所比较的指标。

首轮结果按下面方式解释：

- 对 `delta_loss` 和 `relative_gain` 分别判断：AUC 区间下界高于 0.5 表示该指标有同域区分信号。
- `AUC(delta_loss) - AUC(neg_ref_loss)` 的配对区间下界高于 0：说明相比按 ref loss 从低到高排序，有额外区分收益；也同时展示与 `neg_sft_loss` 及预定反向基线的差异，避免遗漏领域难度本身提供的信号。
- 能区分但不优于原始 loss：不能宣称 SFT 前后比较带来了增益。
- 若两项指标结论不同，分别报告其证据和局限，不因预设优先级忽略其中一项，也不在看到结果后翻转方向。
- Bootstrap 区间只描述给定模型和划分上的样本不确定性；不能代替多个训练 seed 或独立数据划分的复现。

平衡数据上的 precision/AP 仅适用于本实验的 50% A 占比；真实候选池中 A 的占比不同，需要重新估计，不能直接沿用。

## 7. 必要的对照和诊断

**实现正确性。** 在任何训练前，给同一个 θ₀ 打两遍分，delta 应在事先约定的数值容差内接近 0。当所有样本的分数都相等时，AUC 实现应返回 0.5；若有微小浮点残差，不能把它的排序解释为 domain 信号。使用小样本手工核对 shift、mask、逐样本 reduction，检查改变 batch size、右侧 padding 不会改变有效 loss。

**是否真的学到了 A。** 额外对固定 128 条 A1 训练样本记录训练前后 loss，只用作诊断，不放入 A2/B 的 ROC。A1 下降但 A2 不下降，说明可能主要是记忆；A2 明显下降才支持泛化。若区分主要来自 B loss 上升，需要报告域外退化，不能说成 A 域获得了同等程度的提升。

**长度/格式控制。** 画 prompt/response token 数的分布，报告只用 `+长度` 和 `-长度` 两个预定分数的 AUC；方向分别展示，不挑较大者冒充预设结果。再做一份不使用模型分数的长度匹配分析：在两域混合数据上分别按 prompt、response 长度建立四分位区间，在联合分桶中按固定 seed 从两域各取相同数量。只保留有两域覆盖的桶，报告保留比例，并重新计算所有指标。若样本大幅减少，明确说明匹配结果适用范围较窄。

**来源/模板控制。** 统一外部模板不能消除答案内部写作风格、参考答案生成模型和数据源差异。尽可能选同来源、同任务形式的数据；后续换一个 A/B 数据源组合复验。长度匹配后仍有效，也不能单凭这一项排除所有混杂因素。

**Checkpoint 变化。** 半程和最终 checkpoint 的分布可以同时展示，但首轮主结果始终取最终 checkpoint。根据本轮结果调整训练强度的下一轮实验应标明探索性质；若要做确认性结论，需另留未使用的评估数据。

建议输出四张图：delta 的 A2/B 分布、四种分数的 ROC、每条样本 `(L0, LA)` 散点图及 `y=x` 参考线、长度匹配前后的指标对比。

## 8. 拟实现的代码结构和接口

已实现的主要文件：

```text
configs/pilot.yaml                 # 数据、模型、训练、打分、评估配置
src/data_picking/data.py           # 规范化、分组去重、划分与 manifest
src/data_picking/tokenization.py   # 训练和评估共享的模板与 response mask
src/data_picking/loss.py           # 逐样本 masked causal NLL
src/data_picking/checkpoint.py     # 完整 checkpoint 校验、保存与恢复
scripts/prepare_data.py            # 输出 A1/A2/B 和缓存 tokenization
scripts/train_sft.py               # 只读取 A1，执行 SFT
scripts/score.py                   # 对固定 split 输出每条样本的 NLL
scripts/analyze.py                 # 合并、计算指标、bootstrap 和画图
tests/                            # mask/loss、数据隔离、真实续训等关键检查
```

使用 PyTorch + Transformers 实现单卡 trainer。固定 revision 直接下载 Parquet，数据处理和分析使用 pyarrow、NumPy、scikit-learn、Matplotlib；依赖版本记录在 requirements 和生效配置中。若后续换 LoRA 或更大模型，应作为显式实验变体记录，不能和全参数结果混称。

拟定的执行顺序：

```text
prepare_data(config)
  -> splits + manifest + token_cache
score(reference, A2 + B + A1_diagnostic)
  -> ref_scores.parquet
train_sft(reference, A1, config, auto_resume=True)
  -> resumable checkpoints + effective_config
score(final_checkpoint, same_token_cache)
  -> sft_scores.parquet
analyze(ref_scores, sft_scores, manifest)
  -> per_example.parquet + metrics.json + figures + report.md
```

两模型可串行加载，避免同时驻留显存。评分合并按唯一 `id` 做一一关联，校验文本/tokenization hash、response token 数和覆盖率；缺失、重复、非有限值不能悄悄丢弃。

逐样本结果至少包含：

```text
id, group_id, domain, split, source, text_hash, tokenization_hash,
prompt_tokens, response_tokens, ref_sum_nll, sft_sum_nll,
ref_mean_nll, sft_mean_nll, delta_loss, relative_gain,
model_revision, checkpoint_step, run_id
```

`effective_config` 记录解析后的参数、模型/tokenizer revision、模板 hash、数据 manifest hash、代码 commit/dirty 状态、依赖版本、随机种子和恢复位置。报告中给出实际使用的数据量和训练 token 数。

## 9. Checkpoint 与自动恢复要求

用户指定本项目的训练及数据产物统一存入 `/mnt/tianzhu_hot/data-picking`。目录已经完成当前挂载与小文件写入/读取检查，挂载类型为 blobfuse2；这不替代运行时的完整 checkpoint 持久化和恢复检查。

```text
/mnt/tianzhu_hot/data-picking/
  cache/huggingface/          # 数据集、模型、tokenizer 的下载缓存
  cache/                     # 其他显式配置的库缓存
  data/raw/                  # 原始数据与来源/revision 记录
  data/processed/            # A1/A2/B、manifest、tokenization 缓存
  data/source_audits/         # 数据源可用性和 schema 检查结果
  runs/<run_id>/
    effective_config.json
    checkpoints/             # 包含 optimizer 等状态的完整 checkpoint
    logs/
    scores/                  # 逐样本分数
    metrics/
    figures/
    report.md
```

实现时显式设置 HF 等库的缓存目录及 trainer 输出目录，确保下载缓存、处理数据、训练日志和结果也遵守该根目录。代码、配置和设计文档留在 Git 仓库。

按用户后续确定的分析组织方式，用于 review 的已执行 notebook 和小体积结果快照另存于
仓库 `experiments-analysis/YYYY-MM-DD_HHMMSSZ_<run_id>/`。目录按原始结果时间（UTC）排序，
每个分析保留来源路径和快照哈希；模型、optimizer 与原始训练数据仍保留在挂载中。
入口和导出方法见 [实验分析索引](../experiments-analysis/README.md)。

实际挂载不支持部分 POSIX 临时文件操作；仅这些短暂的运行/依赖安装 scratch 文件使用 `/dev/shm/data-picking-*` 内存盘，不用来保存实验产物。bf16 打分的 A1 sanity 检查发现最大 batch/padding 误差约 0.00468 nats/token，因此在正式训练和域区分分析前将打分改为完整 float32；重新检查后该误差约为 0.00000268。这个调整依据实现数值误差，未使用 A2/B 的区分结果。

训练前的 64 条 A1 检查还发现，当前本地 PyTorch/CUDA 环境的自动 attention 后端会在一个 padded batch 上产生非有限 bf16 梯度，关闭 activation checkpointing 后仍复现。固定 SDPA math 后端后，同一个完整 batch 的梯度检查通过。因此正式训练在 forward、backward 和重计算期间都固定 math 后端，继续保留非有限梯度即停止的检查；超参数、数据和训练步数没有按 A2/B 的结果调整。

所有新训练实验默认显式开启 `save_optimizer: true`、`auto_resume: true`。新 trainer 的这些字段必须落实为实际行为，不能仅写在配置里：

- 在 optimizer step 完成后保存模型、optimizer、scheduler、RNG、global step、epoch 和数据迭代状态；恢复时继续原来的 LR 和数据顺序。
- 每个 checkpoint 使用独立目录，所有文件关闭并完成大小/哈希校验后，最后写入 manifest 和完成标记；自动恢复再次验证完整性，只读取最近一个完整、配置匹配的 checkpoint。blobfuse2 上不依赖目录原子 rename 的假设。
- 未写完的临时 checkpoint 不覆盖上一个完整状态；记录其存在并尝试恢复最近的完整状态。已有训练记录却没有可用 checkpoint，或正式 checkpoint 损坏、配置不兼容时，明确报错，不静默从头开始。训练已经达到目标 steps 时跳过重复训练。
- 使用经核验可写、重启后保留的存储路径。临时目录不能作为唯一副本；远程作业需要检查持久挂载或同步后的完整性。
- 启动前打印并保存最终生效配置，确认 optimizer 保存、自动恢复和持久路径。实现验收时做一次“训练数步 → 退出 → 自动继续”的小规模检查，验证 optimizer step、scheduler、数据位置及 RNG 恢复，不能只验证模型权重能加载。

如果实际实现改为复用 `/home/v-guanhechen/rl/scripts/train-sopd-sing/train.sh`，必须显式传：

```text
--save_optimizer True
--auto_resume True
--autosync_dir <已核验的持久目录>
```

已查看的该脚本版本要求 `autosync_dir` 位于 `/mnt/tianzhu/` 的子目录，与用户为本项目指定的 `/mnt/tianzhu_hot/data-picking` 不一致。当前方案采用独立 trainer 直接使用用户指定路径；如果以后复用旧脚本，需要先让其路径校验和同步实现支持该挂载，不能直接传入当前不支持的目录，也不能静默改存到旧路径。此处占位符不是可启动的完整命令。

## 10. 实施顺序与验收

1. **数据与 loss 流程检查**：用极少量样本验证 schema、分组隔离、模板/mask、逐样本 NLL，以及“同模型 delta≈0”。
2. **训练与续训检查**：在独立测试输出目录用几十条 A1 样本跑通训练、完整保存、自动恢复和打分；测试 checkpoint 不混入正式实验。
3. **首轮正式预实验**：冻结配置，仅用 A1 训练；完成 A2/B 四种指标、区间、原始 loss 分布和长度匹配对照。
4. **结果交付**：保存数据 manifest、生效配置、可恢复 checkpoint、逐样本分数、汇总指标和图。无明显区分信号也应完整记录，不能只交付最佳图。
5. **信号复验**：按需要补训练 seeds、独立划分或同来源数据。首轮评价本身不以达到某个 AUC 为“流程完成”的条件。

## 11. 如何进一步验证“选出来的数据确实更有用”

如果这个预实验有信号，下一步应做实际训练效用比较：另留完全未参与打分、调参和数据选择的 A_test，用一个候选池 C 比较按 delta、relative gain、ref loss、随机方法选择的数据，以及可用时的已知 A 域样本基线。

所有选择方法从同一个初始模型重新训练，使用同一个目标 token 预算、优化器和训练 token 进度下的学习率调度，并报告实际 tokens/steps；不要仅匹配条数却让长文本组多训练很多 token。候选池、A1 和 A_test 需要按问题/文档组隔离。

最终比较各模型在 A_test 上的平均回答 NLL 及相对初始模型的下降；不同随机选择和训练 seed 用于估计稳定性。这一步才能直接回答“这种选法是否更能降低目标 domain 的 loss”。

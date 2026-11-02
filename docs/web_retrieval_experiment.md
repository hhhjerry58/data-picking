# Math SFT → 网页文档检索验证

训练配置：[web_math_sft.yaml](../configs/web_math_sft.yaml)。评估规则在查看本轮检索结果前固定于
[web_retrieval_eval.yaml](../configs/web_retrieval_eval.yaml) 及 run 的 `evaluation_plan.json`。
数据定义和去重/重合检查见 [数据说明](web_retrieval_data.md)。

## 固定设置

- 4,096 条多题型 MATH target，仅训练 A1；512 条 validation 和网页均不参与优化。
- 有效 batch 64，micro batch 2，梯度累积 32，1 epoch / 64 optimizer steps。
- 每 16 步保存完整 checkpoint。第 64 步为固定最终模型；16/32/48 只用于留出 NLL 趋势诊断。
- float32 master 参数和 optimizer 状态、bf16 autocast、SDPA math；有限梯度检查保持启用。
- checkpoint 保存 model、optimizer、scheduler、RNG、global step 与数据顺序，自动恢复并拒绝不匹配的训练。

## 文档评分与统计

ref 和最终 SFT 模型都对固定的 11,688 个窗口执行 full-float32、eval、SDPA-math forward。
两模型共享 input IDs、attention/评分 mask 和窗口边界；按 2 个固定 shard、128 窗口一块落盘并校验哈希。
每篇文档先聚合 NLL 总和与实际评分 token 数，再计算文档平均 NLL。
对每模型、每评分集检查同输入重复 forward 和 batch/padding 变化，容差固定为 2e-5。
网页检查以整篇 token 加权平均 NLL 为单位，并保留窗口级差值；QA 按回答平均 NLL 检查。
首次校验曾把只有 1 个评分 token 的尾窗口当作完整样本，得到 2.10e-5 的窗口差值，
其对整篇 2,048 个评分 token 的贡献约 1e-8。完整文档复核的最大差为约 1.7e-6。
修正记录在 `numerical_validation_amendment.json`：阈值、评分计算、输入、模型和最终 checkpoint 规则均未改动。

下降量 `L_ref - L_sft` 与相对下降比例 `(L_ref - L_sft) / max(L_ref, 1e-6)` 并列报告。
四个原始 loss 方向基线同时保存，不按结果翻转指标。
报告来源 ROC-AUC、AP、前 1%/5%/10% 的注入集 recall、来源 precision 和相对随机的富集倍数。
两项变化指标与原始 loss 基线的 AUC 差使用同一批 bootstrap 样本计算配对区间。

长度控制只用原始文档 token 数四分位分桶，桶内随机抽样并维持 1:9 来源比例；同时报告保留率和剩余长度信号。
2,000 次按来源分层、按重复文档组 bootstrap；它不覆盖训练 seed、抓取批次或全部站点相关性。

OpenWebMath 标签仅表示注入来源，背景网页未人工标注且可能包含数学内容。
因此来源 precision 不是数学内容纯度。上游提取方式不同仍是解释限制。

## 梯度附录

[web_retrieval_gradients.yaml](../configs/web_retrieval_gradients.yaml) 固定每来源随机 64 篇，seed 20261030，
抽样不依赖分数或文档长度。每篇所有窗口都参与：累积 `sum(window NLL) / document scored tokens` 的梯度后，
计算全局/逐层/逐参数统计；不将窗口 norm 平均来替代文档梯度。
采用完整 FP32、eval、SDPA math，不裁剪、不更新参数，绑定权重只计一次。
两个模型分别使用同一批 128 条 A1（seed 2027）建立自己的平均梯度参照。
与完整评分的文档平均 NLL 复核容差沿用 1e-4。

## 执行入口

```bash
source scripts/env.sh
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/experiment.py train --config configs/web_math_sft.yaml \
  --checkpoint-blob-config /path/to/existing/blobfuse2.yaml
CUDA_VISIBLE_DEVICES=2 .venv/bin/python scripts/experiment.py score-web --model reference --shard-index 0 --document-sanity
CUDA_VISIBLE_DEVICES=3 .venv/bin/python scripts/experiment.py score-web --model reference --shard-index 1 --document-sanity
```

SFT 评估使用经过 SHA256 校验的 RAM checkpoint 副本，元数据文件为
`runtime_weights_step-000064.json`（中间 checkpoint 对应 16/32/48）。可通过 `stage-weights`
的 `--checkpoint`、`--metadata-name` 和既有 `--blob-config` 参数生成；不记录签名 URL 或凭据。
接着运行相同的两个 `score-web` 命令，将 `--model` 换为 `sft`。
中间 checkpoint 的评分加 `--sets validation --checkpoint <目录>`，不用于模型选择。

本机 blobfuse 在 checkpoint 校验时触发过缓慢的整文件回读。可选的
`--checkpoint-blob-config` 仅针对当前 run 的 `checkpoints/*/state.pt`：持久文件仍按原路径写入，
同时把同一字节流镜像到临时 RAM；成功关闭持久文件后核对 blob 的 ETag/大小并计算完整 SHA。
恢复时也验证已有 checkpoint SHA。其他文件的读写和训练数值代码不变。
本轮在完整的第 48 步 checkpoint 上切换 I/O 方式并实际恢复了 optimizer、scheduler、RNG 和数据顺序；
原日志保存在 `logs/train-before-checkpoint-io.jsonl`，切换记录为 `checkpoint_io_transition.json`。
缓存损坏拒绝、镜像字节一致性和训练状态恢复均有测试覆盖。

```bash
CUDA_VISIBLE_DEVICES=2 .venv/bin/python scripts/experiment.py gradients-web --model reference
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/experiment.py gradients-web --model sft
.venv/bin/python scripts/experiment.py analyze-web
.venv/bin/python scripts/experiment.py export-web \
  --run-dir /mnt/tianzhu_hot/data-picking/runs/math-web-probe-qwen25-1p5b-n4096-b64-s64-seed42
```

本轮验证 math SFT 信号能否识别注入文档；未训练通用 SFT 对照，也未进行按不同分数选文档后的等 token 再训练。
后两项是进一步检验目标特异性和实际训练收益的实验，不将本轮来源检索结果当作其证明。

## 已观测结果

[已执行 notebook](../experiments-analysis/2026-10-23_095832Z_math-web-probe-qwen25-1p5b-n4096-b64-s64-seed42/analysis.ipynb)
包含完整训练、检索、长度对照、数值检查及固定子集梯度附录。

| 指标 | 原始池 AUC（95% CI） | 长度匹配 AUC（95% CI） | 前 1,000 篇找回注入文档 |
| --- | --- | --- | --- |
| loss 下降量 | 0.3046 [0.2856, 0.3226] | 0.2924 [0.2700, 0.3142] | 39 / 1,000 |
| 相对下降比例 | 0.2137 [0.1962, 0.2312] | 0.2123 [0.1918, 0.2343] | 45 / 1,000 |
| ref loss，低值优先 | 0.8872 [0.8745, 0.8999] | 0.8796 [0.8639, 0.8951] | 见 notebook |

随机选取 1,000 篇时，预计找回约 100 篇注入文档。两项变化分数在固定的“越大越优先”方向上低于随机，
没有事后翻转。长度匹配保留 729 篇注入文档与 6,561 篇背景，仍为 1:9。

留出 MATH 的平均回答 NLL 从 0.7435 降到 0.5940，97.66% 样本改善。
原始网页中，OpenWebMath 平均 NLL 为 1.7053→1.7377，背景为 2.7339→2.7460。
因此 target 问答改善并未转化成注入文档的正向 loss-change 检索信号。
格式差异、上游来源差异、训练路径等均需进一步对照才能归因；背景中的实际数学内容尚未人工核验。

第 16/32/48/64 步完整 checkpoint 均已保存，实际恢复检查发生在第 48 步。
评分共覆盖 11,688 个窗口、9,188,836 个 token；梯度诊断为固定 128 篇文档、两个模型。

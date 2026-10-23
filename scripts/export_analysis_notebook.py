"""Export a completed pilot to a dated, executed, self-contained review notebook.

This command only copies small analysis artifacts and runs CPU analysis cells.
It never loads model weights, optimizer state, or the original training dataset.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import textwrap

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager


SNAPSHOTS = {
    "metrics.json": "metrics/metrics.json",
    "per_example.parquet": "metrics/per_example.parquet",
    "length_matched_ids.json": "metrics/length_matched_ids.json",
    "effective_config.json": "effective_config.json",
    "training_complete.json": "training_complete.json",
    "validation_summary.json": "validation_summary.json",
    "train.jsonl": "logs/train.jsonl",
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def loss_metric_overview(scores):
    parts = []
    for key, label in [("delta_loss", "loss 下降量"), ("relative_gain", "相对下降比例")]:
        value = scores[key]
        low, high = value["roc_auc_ci95"]
        parts.append(f"{label} ROC-AUC **{value['roc_auc']:.4f}**（95% CI {low:.4f}–{high:.4f}）")
    return "两项指标并列考察，不设主次：" + "；".join(parts) + "。"


def metric_conclusion_prefix():
    return textwrap.dedent(r"""
    texts = []
    for key in ['delta_loss', 'relative_gain']:
        value = metrics['main']['scores'][key]
        lo, hi = value['roc_auc_ci95']
        signal = ('在固定方向上有区分信号' if lo > .5 else
                  '区分方向与预设假设相反，没有事后翻转分数' if hi < .5 else
                  '区间覆盖 0.5，尚无稳定的区分证据')
        texts.append(f"{score_labels[key]}：AUC {value['roc_auc']:.4f}，95% CI {interval(value['roc_auc_ci95'])}；{signal}。")
    """).strip()


def update_index(root):
    readme = root / "README.md"
    if not readme.exists():
        readme.write_text("# 实验分析\n\n<!-- experiments-index:start -->\n<!-- experiments-index:end -->\n")
    start, end = "<!-- experiments-index:start -->", "<!-- experiments-index:end -->"
    content = readme.read_text()
    if start not in content or end not in content:
        raise ValueError(f"Preserving README without index markers: {readme}")
    lines = [start, "", "| 结果时间（UTC） | 实验 notebook | 模型 | 下降量 AUC | 相对下降比例 AUC |",
             "| --- | --- | --- | --- | --- |"]
    for folder in sorted(root.iterdir()):
        if not (folder / "manifest.json").is_file():
            continue
        manifest = read_json(folder / "manifest.json")
        if manifest.get("execution", {}).get("status") != "completed":
            continue
        cfg = read_json(folder / "snapshot/effective_config.json")["config"]
        metrics = read_json(folder / "snapshot/metrics.json")
        title = f"{cfg['data']['a_subset']} / {cfg['data']['b_subset']} · seed {cfg['training']['seed']}"
        timestamp = datetime.fromisoformat(manifest["result_time_utc"]).strftime("%Y-%m-%d %H:%M:%S")
        model = cfg["model"]["name"].split("/")[-1]
        if metrics.get("analysis_type") in {"dolly_loss_comparison", "ultrachat_loss_comparison"}:
            title = f"{cfg['data']['a_subset']} / {metrics.get('c_display_name', 'Dolly')} C · loss"
            scores = metrics["main"]["scores"]
        elif metrics.get("analysis_type") == "dolly_gradient_extension":
            title = f"{cfg['data']['a_subset']} / {cfg['data']['b_subset']} / Dolly C · loss + gradients"
            comparison = metrics["primary_comparison"]
            scores = metrics["comparisons"][comparison]["raw"]["loss_scores"]
        else:
            scores = metrics["main"]["scores"]
        if manifest.get("gradient_appendix"):
            title = title.replace(" · loss", "") + " · loss + grad"
        lines.append(f"| {timestamp} | [{title}]({folder.name}/analysis.ipynb) | {model} | "
                     f"{scores['delta_loss']['roc_auc']:.4f} | {scores['relative_gain']['roc_auc']:.4f} |")
    lines += ["", end]
    prefix, rest = content.split(start, 1)
    _, suffix = rest.split(end, 1)
    readme.write_text(prefix + "\n".join(lines) + suffix)


def make_notebook(manifest, cfg, metrics):
    cells = []

    def md(source):
        cells.append(nbformat.v4.new_markdown_cell(textwrap.dedent(source).strip()))

    def code(source):
        cells.append(nbformat.v4.new_code_cell(textwrap.dedent(source).strip()))

    model = cfg["config"]["model"]["name"]
    a = cfg["config"]["data"]["a_subset"].replace("_", " ").title()
    b = cfg["config"]["data"]["b_subset"].replace("_", " ").title()
    overview = loss_metric_overview(metrics["main"]["scores"])
    md(f"""
    # {a} / {b}：SFT 前后 loss 变化的预实验

    **Run：** `{manifest['run_id']}`  
    **结果时间（UTC）：** {manifest['result_time_utc']}  
    **模型：** `{model}`；主报告 checkpoint：step {metrics['checkpoint_step']}。

    本 notebook 将实验设置、结果、对照和局限放在一起。{overview}
    指标是否优于基线以及长度控制后的表现，见后面的配对比较。

    已保存执行输出，直接打开即可查看表格和图。重跑只读取同目录的 `snapshot/`，
    不需要 GPU、checkpoint 或原始数据挂载，也不会重新训练模型。
    来源路径、时间依据和快照 SHA-256 见 `manifest.json`。
    """)
    md("""
    ## 1. 问题与评价约定

    只在 A1 上做 SFT；A2 是未参与这次 SFT 的 A 域样本，B 是另一个域。
    两个模型使用完全相同的 token IDs 和回答 mask，只计算回答正文的平均 token NLL（nats/token）。
    A2 为正类，所有排序方向预先固定；A1 诊断样本不参与 A2/B 的分类评价。

    下降量和相对下降比例并列考察，不设主次。

    - loss 下降量：`delta_loss = L_ref - L_sft`。
    - 相对下降比例：`relative_gain = (L_ref - L_sft) / max(L_ref, 1e-6)`。
    - 基线：`-L_ref`、`-L_sft`，即 loss 低的样本优先；反向基线在附录单列。
    - 95% 区间来自原始分析的 2,000 次按 domain 分层、按 group 重采样的 bootstrap。
      本 notebook 重算 AUC、AP 和 top-k 点估计并与快照交叉核对，直接展示原始区间。

    这是**同域识别**预实验，不直接检验被选数据用于后续训练时的实际收益。
    """)
    code("""
    import hashlib
    import json
    import math
    import os
    import sys
    import tempfile
    from pathlib import Path

    if 'MPLCONFIGDIR' not in os.environ:
        scratch = Path('/dev/shm/data-picking-runtime') if Path('/dev/shm').is_dir() else Path(tempfile.gettempdir()) / 'data-picking-runtime'
        os.environ['MPLCONFIGDIR'] = str(scratch / 'matplotlib')
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from IPython.display import display, Markdown
    from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

    %matplotlib inline
    plt.rcParams.update({"figure.dpi": 120, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    pd.set_option("display.max_colwidth", 100)
    """)
    # Resolve relative to the notebook working directory or a repository root.
    code(f"""
    candidates = [Path.cwd(), Path.cwd() / 'experiments-analysis' / {manifest['directory']!r}]
    HERE = next((p for p in candidates if (p / 'manifest.json').is_file() and (p / 'snapshot').is_dir()), None)
    if HERE is None:
        raise FileNotFoundError('请将 notebook 工作目录设为本实验目录，或仓库根目录。')
    SNAPSHOT = HERE / 'snapshot'
    manifest = json.loads((HERE / 'manifest.json').read_text())
    for name, info in manifest['files'].items():
        content = (SNAPSHOT / name).read_bytes()
        assert len(content) == info['bytes'], name
        assert hashlib.sha256(content).hexdigest() == info['sha256'], name

    def load_json(name):
        return json.loads((SNAPSHOT / name).read_text())

    metrics = load_json('metrics.json')
    effective = load_json('effective_config.json')
    completed = load_json('training_complete.json')
    validation = load_json('validation_summary.json')
    df = pd.read_parquet(SNAPSHOT / 'per_example.parquet')
    training = pd.read_json(SNAPSHOT / 'train.jsonl', lines=True)
    evaluation = df.loc[df['split'].isin(['A2', 'B'])].reset_index(drop=True)
    matched_ids = load_json('length_matched_ids.json')
    assert df['id'].is_unique
    assert len(set(matched_ids)) == len(matched_ids)
    assert set(matched_ids) <= set(evaluation['id'])
    matched = evaluation.set_index('id').loc[matched_ids].reset_index()
    assert len(evaluation) == metrics['main']['examples']
    assert set(evaluation['split']) == {{'A2', 'B'}}
    assert completed['global_step'] == metrics['checkpoint_step']
    assert metrics['primary_checkpoint'] is True
    print(f"快照校验通过；全部逐样本记录 {{len(df)}} 条，正式评估 {{len(evaluation)}} 条，长度匹配 {{len(matched)}} 条。")
    print('Python:', sys.executable)
    """)
    md("## 2. 实验设置与数据划分")
    code("""
    cfg = effective['config']
    settings = [
        ('模型', cfg['model']['name']),
        ('模型 revision', cfg['model']['revision']),
        ('数据源', cfg['data']['name']),
        ('数据 revision', cfg['data']['revision']),
        ('A / B', f"{cfg['data']['a_subset']} / {cfg['data']['b_subset']}"),
        ('划分 seed / 训练 seed', f"{cfg['data']['split_seed']} / {cfg['training']['seed']}"),
        ('训练规模', f"{effective['actual_train_size']} examples; {completed['seen_response_tokens']:,} response tokens"),
        ('训练长度', f"{cfg['training']['epochs']} epoch; {completed['global_step']} optimizer steps"),
        ('有效 batch / 学习率', f"{cfg['training']['effective_batch_size']} / {cfg['training']['learning_rate']}"),
        ('训练精度 / attention', f"{cfg['training']['parameter_dtype']} parameters + {cfg['training']['autocast_dtype']} autocast / {cfg['training']['attention_backend']}"),
        ('打分精度', cfg['scoring']['forward_dtype']),
        ('主报告 checkpoint', completed['global_step']),
    ]
    display(pd.DataFrame(settings, columns=['设置', '值']))
    splits = effective['data_summary']['splits']
    display(pd.DataFrame([
        {'split': name, '条数': value['examples'], '回答 tokens': value['response_tokens'], **value['level_counts']}
        for name, value in splits.items()
    ]).set_index('split').fillna(0))
    assert splits['A2']['level_counts'] == splits['B']['level_counts']
    display(pd.DataFrame(effective['data_summary']['filter_counts']).fillna(0).astype(int).rename_axis('过滤 / 选择状态'))
    """)
    md("""
    A2/B 的难度等级组成完全匹配。按规范化问题及字符 5-gram 相似度分组后，每组保留一个代表，
    跨域重复组排除；图形表示、空内容和超长样本按同一规则过滤。
    `eligible` 表示过滤后可用但未抽入本轮的样本。官方 test 没有用于这轮训练或打分。
    这些措施不能保证排除所有语义重复，也未核验初始模型预训练是否包含 MATH。
    """)
    md("## 3. 训练过程与两域 loss 变化")
    code("""
    assert training['step'].tolist() == list(range(1, completed['global_step'] + 1))
    assert np.isfinite(training[['loss', 'grad_norm']].to_numpy()).all()
    assert int(training['response_tokens'].sum()) == completed['seen_response_tokens']
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    axes[0].plot(training['step'], training['loss'], marker='.', color='#1674B8')
    axes[0].set(xlabel='Optimizer step', ylabel='Mean response NLL', title='Training batches (A1 only)')
    axes[1].plot(training['step'], training['lr'], color='#DF7931')
    axes[1].set(xlabel='Optimizer step', ylabel='Learning rate', title='Fixed warmup / linear decay')
    fig.tight_layout()
    plt.show()
    plt.close(fig)
    print(f"训练 step 记录的耗时合计 {training['seconds'].sum():.1f} 秒；不含模型加载、checkpoint I/O 和评分。")
    """)
    md("每一步的 batch 内容不同，因此训练曲线不等于固定评估集上的学习曲线。下面比较相同样本在两个模型下的 loss；A1 只展示固定的 128 条训练诊断样本。")
    code("""
    loss_table = pd.DataFrame([
        {'split': name, '条数': len(group), 'ref NLL': group['ref_mean_nll'].mean(),
         'SFT NLL': group['sft_mean_nll'].mean(), '平均下降量': group['delta_loss'].mean(),
         'loss 下降样本比例': (group['delta_loss'] > 0).mean()}
        for name, group in df.groupby('split', sort=True)
    ]).set_index('split')
    display(loss_table.round(4))
    colors = {'A': '#1674B8', 'B': '#DF7931'}
    domain_names = {'A': 'A2: ' + cfg['data']['a_subset'].replace('_', ' ').title(),
                    'B': 'B: ' + cfg['data']['b_subset'].replace('_', ' ').title()}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    bins = np.histogram_bin_edges(evaluation['delta_loss'], bins=40)
    for domain, group in evaluation.groupby('domain'):
        axes[0].hist(group['delta_loss'], bins=bins, density=True, alpha=0.5,
                     label=domain_names[domain], color=colors[domain])
        axes[1].scatter(group['ref_mean_nll'], group['sft_mean_nll'], s=10, alpha=0.4,
                        label=domain_names[domain], color=colors[domain])
    axes[0].axvline(0, color='black', linewidth=0.8)
    axes[0].set(xlabel='Reference NLL - SFT NLL', ylabel='Density', title='Loss change on held-out examples')
    bound = evaluation[['ref_mean_nll', 'sft_mean_nll']].to_numpy().max() * 1.02
    axes[1].plot([0, bound], [0, bound], 'k--', linewidth=0.8)
    axes[1].set(xlabel='Reference NLL', ylabel='SFT NLL', title='Paired per-example losses')
    for ax in axes:
        ax.legend(fontsize=8)
    fig.tight_layout()
    plt.show()
    plt.close(fig)
    """)
    md("## 4. 能否把 A2 排在 B 前面？")
    code("""
    score_keys = ['delta_loss', 'relative_gain', 'neg_ref_loss', 'neg_sft_loss']
    score_labels = {'delta_loss': 'loss 下降量', 'relative_gain': '相对下降比例',
                    'neg_ref_loss': 'ref loss，低值优先', 'neg_sft_loss': 'SFT loss，低值优先'}

    def point_metrics(frame, key, ties):
        y = (frame['domain'] == 'A').to_numpy(dtype=int)
        scores = frame[key].to_numpy()
        order = np.lexsort((ties, -scores))
        return {'roc_auc': roc_auc_score(y, scores), 'average_precision': average_precision_score(y, scores),
                'precision_top_10pct': y[order[:math.ceil(len(y) * .1)]].mean(),
                'precision_top_20pct': y[order[:math.ceil(len(y) * .2)]].mean()}

    for frame, stored in [(evaluation, metrics['main']), (matched, metrics['length_matched'])]:
        ties = np.random.default_rng(cfg['analysis']['bootstrap_seed'] + 1).random(len(frame))
        for key in score_keys:
            for name, value in point_metrics(frame, key, ties).items():
                np.testing.assert_allclose(value, stored['scores'][key][name], rtol=0, atol=1e-12)

    def interval(values):
        return f'[{values[0]:.4f}, {values[1]:.4f}]'

    def metric_table(result, keys=score_keys):
        records = []
        for key in keys:
            row = result['scores'][key]
            records.append({'排序指标': score_labels.get(key, key), 'ROC-AUC': round(row['roc_auc'], 4),
                            'AUC 95% CI': interval(row['roc_auc_ci95']), 'AP': round(row['average_precision'], 4),
                            'Top 10% A2': f"{row['precision_top_10pct']:.1%}",
                            'Top 20% A2': f"{row['precision_top_20pct']:.1%}"})
        return pd.DataFrame(records).set_index('排序指标')

    display(metric_table(metrics['main']))
    print('原始池和长度匹配池的 AUC、AP、top-k 点估计均已重新计算，并与快照一致。')
    print(f"原始池 top 10% 取前 {math.ceil(len(evaluation) * .1)} 条；A2/B 各占一半，随机 A2 占比期望为 50%。")
    """)
    code("""
    y = (evaluation['domain'] == 'A').to_numpy(dtype=int)
    fig, ax = plt.subplots(figsize=(6.8, 5))
    for key in score_keys:
        fpr, tpr, _ = roc_curve(y, evaluation[key])
        ax.plot(fpr, tpr, label=f"{key} (AUC {metrics['main']['scores'][key]['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8)
    ax.set(xlabel='False positive rate', ylabel='True positive rate', title='A2 vs B; fixed score directions')
    ax.legend(fontsize=9)
    fig.tight_layout()
    plt.show()
    plt.close(fig)
    """)
    md("""
    ## 5. 长度控制及相对基线的增益

    原始分析按混合池的 prompt/response 长度四分位建立联合分桶，每桶从两域各取相同数量，
    未使用模型分数挑样本。下面读取保存的匹配 ID，展示同一份控制样本。
    匹配改变了评估人群，且分桶不能保证完全消除长度或其他混杂因素。
    """)
    code("""
    print(f"匹配保留 {len(matched)}/{len(evaluation)} 条（{len(matched)/len(evaluation):.1%}）。")
    display(pd.DataFrame({'原始池': evaluation.groupby('domain').size(),
                          '长度匹配': matched.groupby('domain').size()}))
    display(metric_table(metrics['length_matched']))
    length_rows = []
    for pool, result in [('原始池', metrics['main']), ('长度匹配', metrics['length_matched'])]:
        for column, values in result['length_baselines'].items():
            length_rows.append({'评估集': pool, '长度特征': column, '+长度 AUC': values['positive_auc'], '-长度 AUC': values['negative_auc']})
    display(pd.DataFrame(length_rows).round(4))
    """)
    code("""
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for shift, pool, result in [(-0.15, 'Original', metrics['main']), (.15, 'Length matched', metrics['length_matched'])]:
        positions = np.arange(len(score_keys)) + shift
        values = [result['scores'][k]['roc_auc'] for k in score_keys]
        ax.scatter(positions, values, label=pool, zorder=3)
        for position, key in zip(positions, score_keys):
            low, high = result['scores'][key]['roc_auc_ci95']
            ax.vlines(position, low, high, linewidth=1)
    ax.axhline(.5, linestyle='--', color='black', linewidth=.8)
    ax.set(xticks=np.arange(len(score_keys)), xticklabels=score_keys, ylabel='ROC-AUC', ylim=(0, 1))
    ax.legend()
    fig.tight_layout()
    plt.show()
    plt.close(fig)

    comparisons = []
    for pool, result in [('原始池', metrics['main']), ('长度匹配', metrics['length_matched'])]:
        for baseline in ['neg_ref_loss', 'neg_sft_loss']:
            value = result['paired_delta_auc_minus_baseline'][baseline]
            low, high = value['ci95']
            comparisons.append({'评估集': pool, '比较': 'delta_loss − ' + baseline,
                                'AUC 差': round(value['difference'], 4), '配对 95% CI': interval(value['ci95']),
                                '区间覆盖 0': low <= 0 <= high})
    display(pd.DataFrame(comparisons))
    """)
    md("## 6. 结论与边界")
    code(metric_conclusion_prefix() + "\n" + textwrap.dedent(r"""
    a2, b = metrics['loss_summary']['A2'], metrics['loss_summary']['B']
    texts.append(f"A2 平均 NLL：{a2['ref_mean_nll']['mean']:.4f} → {a2['sft_mean_nll']['mean']:.4f}；"
                 f"B：{b['ref_mean_nll']['mean']:.4f} → {b['sft_mean_nll']['mean']:.4f}。")
    if a2['delta_loss']['mean'] > 0 and b['delta_loss']['mean'] > 0:
        texts.append('两个域的平均 loss 都下降，区分效果没有依赖 B 的平均 loss 恶化。')
    for key in ['delta_loss', 'relative_gain']:
        v = metrics['length_matched']['scores'][key]
        texts.append(f"长度匹配后，{score_labels[key]} AUC 为 {v['roc_auc']:.4f}，95% CI {interval(v['roc_auc_ci95'])}。")
    for label, result, baseline in [('loss 下降量在原始池中与 SFT loss 比较', metrics['main'], 'neg_sft_loss'),
                                     ('loss 下降量在长度匹配后与 ref loss 比较', metrics['length_matched'], 'neg_ref_loss')]:
        value = result['paired_delta_auc_minus_baseline'][baseline]
        lo, hi = value['ci95']
        if lo <= 0 <= hi:
            texts.append(f"{label}：配对区间覆盖 0，额外区分收益尚不确定。")
        else:
            texts.append(f"{label}：AUC 差 {value['difference']:.4f}，95% CI {interval(value['ci95'])}。")
    best = max(score_keys, key=lambda k: metrics['main']['scores'][k]['roc_auc'])
    texts.append(f"本轮 {score_labels[best]} 的 AUC 点估计最高；这不自动证明它显著优于其他指标。")
    display(Markdown('\n\n'.join(texts)))
    """).strip())
    md("""
    本轮只有一个模型、一个训练 seed 和一次数据划分；bootstrap 区间不包含训练随机性。
    平衡评估池上的 AP/top-k 占比也不能直接外推到其他 A/B 比例的候选池。

    后续可做（本轮未运行）：固定方案补独立 seed/划分，再在未参与选择的 A_test 上比较
    按不同分数和随机方法选数后的实际训练收益，匹配训练 token 预算。
    """)
    md("## 7. 验证记录与原始产物")
    code("""
    display(pd.DataFrame([
        ('保存 optimizer / 自动恢复', f"{effective['save_optimizer']} / {effective['auto_resume']}"),
        ('真实续训检查', validation['real_resume']['passed']),
        ('核心 / 分析测试', f"{validation['unit_tests']['core_passed']} / {validation['unit_tests']['analysis_passed']}"),
        ('同模型重复评分最大差', validation['scoring_sanity']['identical_model_max_delta']),
        ('batch / padding 最大差', validation['scoring_sanity']['batch_padding_max_delta']),
        ('相对下降分母保护触发数', metrics['relative_gain_denominator_clamps']),
    ], columns=['检查', '结果']))
    print('原始 run：', manifest['source_run_dir'])
    print('最终 checkpoint：', completed['checkpoint'])
    print('训练 fingerprint：', effective['training_fingerprint'])
    print('时间来源：', manifest['timestamp_basis'])
    print('快照文件 SHA-256：')
    display(pd.DataFrame([{'文件': name, 'bytes': info['bytes'], 'SHA-256': info['sha256']}
                           for name, info in manifest['files'].items()]).set_index('文件'))
    """)
    md("""
    训练在当前本地 PyTorch/CUDA 环境中固定使用 SDPA math 后端，避免自动后端在检查中出现的非有限梯度。
    评分使用完整 float32，因为 bf16 在不同 batch/padding 下曾出现约 0.0047 nats/token 的差异。
    这些数值修正发生在正式训练和域区分分析之前，没有依据 A2/B 的效果调整超参数。

    模型权重、optimizer、数据下载和 token 缓存保留在原始挂载。
    本目录只包含用于 review 的 notebook 和小体积结果快照。
    """)
    md("## 附录：预先规定的反向原始 loss 基线")
    code("""
    display(metric_table(metrics['main'], keys=['pos_ref_loss', 'pos_sft_loss']))
    """)
    nb = nbformat.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python", "version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"}
    nb.metadata["data_picking"] = {"run_id": manifest["run_id"], "result_time_utc": manifest["result_time_utc"], "snapshot_manifest": "manifest.json"}
    return nb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parents[1] / "experiments-analysis")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace an existing exported notebook and snapshots")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    cfg = read_json(run / "effective_config.json")
    metrics = read_json(run / "metrics/metrics.json")
    run_id = cfg["config"]["run_id"]
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
        raise ValueError("Run ID must be a safe directory name")
    if not metrics["primary_checkpoint"]:
        raise ValueError("Export the final, primary checkpoint result as the main notebook")
    result_time = datetime.fromtimestamp((run / "metrics/metrics.json").stat().st_mtime, timezone.utc)
    folder_name = result_time.strftime("%Y-%m-%d_%H%M%SZ_") + run_id
    folder = args.output_root.resolve() / folder_name
    if folder.exists() and not args.overwrite:
        raise FileExistsError(f"Preserving existing analysis: {folder}; --overwrite explicitly replaces exported files")
    snapshot = folder / "snapshot"
    snapshot.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, relative in SNAPSHOTS.items():
        source = run / relative
        if source.stat().st_size > 10 * 1024 * 1024:
            raise ValueError(f"Snapshot exceeds 10 MiB review limit: {source}; curate a smaller analysis export")
        shutil.copy2(source, snapshot / name)
        files[name] = {"source_relative_path": relative, "bytes": source.stat().st_size, "sha256": sha256(snapshot / name)}
        if files[name]["sha256"] != sha256(source):
            raise RuntimeError(f"Snapshot copy mismatch: {source}")
    repo = Path(__file__).resolve().parents[1]
    manifest = {
        "run_id": run_id, "directory": folder_name, "source_run_dir": str(run),
        "result_time_utc": result_time.isoformat(),
        "timestamp_basis": "metrics/metrics.json filesystem modification time captured at first export; proxy for original analysis completion, not export time",
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "exporter": {"script": "scripts/export_analysis_notebook.py", "sha256": sha256(__file__),
                     "repository_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
                     "working_tree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip())},
        "files": files, "execution": {"status": "pending", "gpu_required": False},
    }
    write_json(folder / "manifest.json", manifest)
    notebook = make_notebook(manifest, cfg, metrics)
    notebook_path = folder / "analysis.ipynb"
    nbformat.write(notebook, notebook_path)
    scratch = Path(os.environ.get("TMPDIR", "/dev/shm/data-picking-runtime"))
    for key, subdir in [("JUPYTER_RUNTIME_DIR", "jupyter"), ("IPYTHONDIR", "ipython")]:
        value = scratch / subdir
        value.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(key, str(value))
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv[0] = sys.executable
    client = NotebookClient(notebook, km=manager, timeout=300, resources={"metadata": {"path": str(folder)}})
    print(f"Executing CPU analysis notebook: {notebook_path}", flush=True)
    try:
        client.execute()
    finally:
        try:
            nbformat.write(notebook, notebook_path)
        finally:
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
    nbformat.validate(notebook)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert all(cell.execution_count is not None for cell in code_cells)
    assert not any(out.output_type == "error" for cell in code_cells for out in cell.outputs)
    manifest["execution"] = {"status": "completed", "gpu_required": False, "python": sys.executable,
                             "code_cells": len(code_cells), "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                             "notebook_sha256": sha256(notebook_path)}
    write_json(folder / "manifest.json", manifest)
    update_index(args.output_root.resolve())
    print(f"Export complete: {notebook_path} ({notebook_path.stat().st_size:,} bytes)", flush=True)


if __name__ == "__main__":
    main()

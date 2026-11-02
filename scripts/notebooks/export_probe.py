"""Export the QA cross-check and web checkpoint trajectory into a new notebook."""
import argparse
import base64
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import sys
import textwrap

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient

from _shared import read_json, sha256, update_index, write_json


def build_notebook(manifest):
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"display_name": "Python (data-picking)", "language": "python", "name": "python3"}
    nb.metadata["data_picking"] = {"analysis_type": "probe_transfer_diagnostics", "run_id": manifest["run_id"]}
    def md(s): nb.cells.append(nbformat.v4.new_markdown_cell(textwrap.dedent(s).strip()))
    def code(s): nb.cells.append(nbformat.v4.new_code_cell(textwrap.dedent(s).strip()))
    md("""
    # 数学 SFT → QA / 网页：两个交叉检查

    本分析回答两个问题：新的 4,096 条 MATH / step-64 模型能否继续区分原来的 A2 与 UltraChat；
    同一轮训练的 step-16/32/48 是否已经出现网页反向信号。

    这是观察到原 step-64 网页结果后的诊断。只评估已有 checkpoint，没有训练或新增梯度测量，
    不以这些评估集选择 checkpoint，保留原 step-64 正式结果。
    **loss 下降量与相对下降比例并列考察，不设主次，不翻转分数。**
    """)
    code("""
    from pathlib import Path
    import hashlib, json
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from IPython.display import display, Markdown
    from sklearn.metrics import roc_auc_score, roc_curve
    root = Path.cwd(); snap = root / 'snapshot'
    manifest = json.loads((root / 'manifest.json').read_text())
    for name, info in manifest['files'].items():
        assert hashlib.sha256((snap / name).read_bytes()).hexdigest() == info['sha256'], name
    metrics = json.loads((snap / 'metrics.json').read_text())
    qa = pd.read_parquet(snap / 'qa_per_example.parquet')
    web = pd.read_parquet(snap / 'web_per_document.parquet')
    qa_ids = json.loads((snap / 'qa_length_matched_ids.json').read_text())
    web_ids = json.loads((snap / 'web_length_matched_ids.json').read_text())
    keys = ['delta_loss', 'relative_gain']
    baselines = ['neg_ref_loss', 'neg_sft_loss']
    probes = ['legacy_algebra1024_step32', 'math4096_step64']
    probe_names = {probes[0]: '原 Algebra 1024 / step 32', probes[1]: '新 MATH 4096 / step 64'}
    fresh = qa.loc[qa.probe == probes[1]].copy()
    colors = {'A': '#2379B5', 'C': '#DD7C32', 'math': '#2379B5', 'background': '#DD7C32'}
    def interval(x): return f'[{x[0]:.4f}, {x[1]:.4f}]'
    print('已验证快照 SHA-256：', len(manifest['files']))
    """)
    md("""
    ## 1. 固定设置与比较边界

    QA 使用原来的 A2=512、UltraChat=512，逐条复用 token IDs 和 response mask；仅计算回答平均 NLL。
    旧模型是 Algebra 1,024 条 / batch 32 / step 32，新模型是多科目 MATH 4,096 条 / batch 64 / step 64。
    两者训练集和训练预算不同，因此比较检验原 QA 信号能否保留，不把变化归因于单一超参数。

    网页使用原来的 1,000 篇 OpenWebMath 与 9,000 篇 FineWeb，共 11,688 个窗口。
    每篇先累加窗口 NLL 与评分 token 数，再相除。所有 checkpoint 使用相同文档、窗口和掩码。
    网页没有聊天模板。背景来源未标注，可能含数学；来源检索结果不能直接当作语义分类或选数训练收益。

    原长度匹配 ID 完全保留：QA 每域 126 条，网页 729 / 6,561 条。
    95% 区间来自 2,000 次按类分层、按问题/重复文档组 bootstrap，不包含训练随机性。
    """)
    code("""
    val = pd.DataFrame(metrics['target_validation'])
    display(val[['checkpoint_step', 'mean_nll', 'fraction_improved']].round(5))
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(val.checkpoint_step, val.mean_nll, 'o-')
    ax.set(xlabel='Optimizer step', ylabel='Held-out answer NLL', title='MATH validation throughout the same training run', xticks=val.checkpoint_step)
    fig.tight_layout(); plt.show(); plt.close(fig)
    """)
    md('## 2. 新模型对原 A2 / UltraChat 的 loss 变化')
    code("""
    display(pd.DataFrame([{'probe': probe_names[p], 'domain': d, **v}
        for p, r in metrics['qa']['probes'].items() for d, v in r['summary'].items()]).round(5))
    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    columns = ['ref_mean_nll', 'sft_mean_nll', 'delta_loss', 'relative_gain']
    shared = np.histogram_bin_edges(fresh[['ref_mean_nll', 'sft_mean_nll']].to_numpy().ravel(), bins=35)
    for ax, col in zip(axes, columns):
        bins = shared if col.endswith('mean_nll') else np.histogram_bin_edges(fresh[col], bins=40)
        for domain, group in fresh.groupby('domain'):
            ax.hist(group[col], bins=bins, density=True, alpha=.5, color=colors[domain], label='A2' if domain == 'A' else 'UltraChat')
        ax.set(xlabel=col, ylabel='Density', title=col)
        if col in keys: ax.axvline(0, color='black', linewidth=.8)
    for domain, group in fresh.groupby('domain'):
        axes[4].scatter(group.ref_mean_nll, group.sft_mean_nll, s=8, alpha=.4, color=colors[domain], label='A2' if domain == 'A' else 'UltraChat')
    upper = fresh[['ref_mean_nll','sft_mean_nll']].to_numpy().max() * 1.02
    axes[4].plot([0, upper], [0, upper], 'k--', linewidth=.8)
    axes[4].set(xlabel='Reference NLL', ylabel='New SFT NLL', title='Paired QA losses')
    for ax in axes: ax.legend(fontsize=8)
    fig.tight_layout(); plt.show(); plt.close(fig)
    print('全部样本参与图示和统计，未按分数或内容删除样本。')
    """)
    md('## 3. QA 区分度是否保留：旧模型与新模型、长度控制和原始 loss 基线')
    code("""
    rows = []
    for probe in probes:
        for pool in ['main', 'length_matched']:
            for key, value in metrics['qa']['probes'][probe][pool]['scores'].items():
                rows.append({'probe': probe_names[probe], 'pool': pool, 'score': key,
                    'AUC': value['roc_auc'], '95% CI': interval(value['roc_auc_ci95']), 'AP': value['average_precision']})
    display(pd.DataFrame(rows).round(4))
    display(pd.DataFrame([{'pool': pool, 'score': key, '新模型 AUC − 旧模型 AUC': value['auc_difference_after_minus_before'],
        'paired 95% CI': interval(value['ci95'])} for pool, r in metrics['qa']['paired_new_minus_old'].items() for key, value in r.items()]).round(4))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for probe, style in zip(probes, ['--', '-']):
        frame = qa.loc[qa.probe == probe]; y = (frame.domain == 'A').astype(int)
        for key in keys:
            x, t, _ = roc_curve(y, frame[key])
            axes[0].plot(x, t, linestyle=style, label=f'{probe}: {key}')
    axes[0].plot([0,1],[0,1],'k:',linewidth=.8)
    axes[0].set(xlabel='UltraChat false-positive rate', ylabel='A2 recall', title='Same QA pool, two SFT probes')
    for offset, pool in [(-.12, 'main'), (.12, 'length_matched')]:
        scores = metrics['qa']['probes'][probes[1]][pool]['scores']
        for i, key in enumerate(keys + baselines):
            axes[1].vlines(i+offset, *scores[key]['roc_auc_ci95'])
        axes[1].scatter(np.arange(4)+offset, [scores[k]['roc_auc'] for k in keys+baselines], label=pool)
    axes[1].axhline(.5, linestyle=':', color='black')
    axes[1].set(xticks=np.arange(4), xticklabels=keys+baselines, ylabel='ROC-AUC', ylim=(0,1.02), title='New probe: length control and baselines')
    axes[1].tick_params(axis='x', labelrotation=15)
    for ax in axes: ax.legend(fontsize=7)
    fig.tight_layout(); plt.show(); plt.close(fig)
    """)
    md('## 4. 同一轮训练的网页信号：step 16 / 32 / 48 / 64')
    code("""
    records = []
    for step, result in metrics['web'].items():
        for pool in ['main', 'length_matched']:
            for key, value in result[pool]['scores'].items():
                records.append({'step': int(step), 'pool': pool, 'score': key, 'AUC': value['roc_auc'],
                    '95% CI': interval(value['roc_auc_ci95']), 'AP': value['average_precision'],
                    'top-10% math-source retrieved': value['top']['0.1']['retrieved_math_source'],
                    'top-10% source precision': value['top']['0.1']['source_precision']})
    table = pd.DataFrame(records)
    display(table.loc[table.score.isin(keys)].round(4))
    display(table.loc[(table.pool == 'main') & table.score.isin(baselines)].round(4))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    steps = sorted(web.checkpoint_step.unique())
    for ax, pool in zip(axes, ['main', 'length_matched']):
        for key in keys + baselines:
            values = [metrics['web'][str(s)][pool]['scores'][key] for s in steps]
            line, = ax.plot(steps, [v['roc_auc'] for v in values], 'o-', label=key)
            ax.fill_between(steps, [v['roc_auc_ci95'][0] for v in values], [v['roc_auc_ci95'][1] for v in values], color=line.get_color(), alpha=.12)
        ax.axhline(.5, color='black', linestyle=':')
        ax.set(xlabel='Optimizer step', ylabel='Source ROC-AUC', title=pool, xticks=steps, ylim=(0,1))
        ax.legend(fontsize=8)
    fig.tight_layout(); plt.show(); plt.close(fig)
    print('未画 step-0 的差值 AUC：两个模型相同时全部差值为零，AUC 按定义为 0.5。')
    print('原始池 top 10% 选择 1,000 篇，随机期望找回 100 篇注入文档；长度匹配池选择 729 篇，随机期望 72.9 篇。')
    """)
    md('## 5. 哪类网页的 loss 在变化？')
    code("""
    summaries = pd.DataFrame([{'step': int(step), 'source': source, 'fraction_improved': v['positive_delta_fraction'],
        **{key: v[key]['mean'] for key in ['ref_mean_nll','sft_mean_nll','delta_loss','relative_gain']}}
        for step, result in metrics['web'].items() for source, v in result['document_summary'].items()])
    display(summaries.round(6))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for source, group in summaries.groupby('source'):
        group = group.sort_values('step')
        axes[0].plot([0]+group.step.tolist(), [group.ref_mean_nll.iloc[0]]+group.sft_mean_nll.tolist(), 'o-', label=source, color=colors[source])
        for ax, key in zip(axes[1:], keys):
            ax.plot([0]+group.step.tolist(), [0]+group[key].tolist(), 'o-', label=source, color=colors[source])
    for ax, title in zip(axes, ['Document mean NLL', 'Mean loss decrease', 'Mean relative decrease']):
        ax.set(xlabel='Optimizer step', title=title, xticks=[0,16,32,48,64]); ax.legend()
    for ax in axes[1:]: ax.axhline(0, color='black', linestyle=':')
    fig.tight_layout(); plt.show(); plt.close(fig)
    display(pd.DataFrame([{'earlier_step': int(step), 'score': key,
        'AUC64 − AUCearlier': v['auc_difference_after_minus_before'], 'paired 95% CI': interval(v['ci95'])}
        for step, r in metrics['web_paired_auc_final_minus_intermediate'].items() for key,v in r.items()]).round(4))
    """)
    md('## 6. 结果复核与解释边界')
    code("""
    for probe in probes:
        frame = qa.loc[qa.probe == probe]
        assert len(frame) == 1024 and frame.id.is_unique
        for pool, subset in [('main', frame), ('length_matched', frame.loc[frame.id.isin(qa_ids)])]:
            for key in keys+baselines:
                np.testing.assert_allclose(roc_auc_score((subset.domain == 'A').astype(int), subset[key]),
                    metrics['qa']['probes'][probe][pool]['scores'][key]['roc_auc'], atol=1e-12, rtol=0)
    for step, frame in web.groupby('checkpoint_step'):
        assert len(frame) == 10000 and frame.id.is_unique and frame.injected_math_source.sum() == 1000
        assert frame.scored_tokens.sum() == 9188836 and frame.windows.sum() == 11688
        for pool, subset in [('main', frame), ('length_matched', frame.loc[frame.id.isin(web_ids)])]:
            for key in keys+baselines:
                np.testing.assert_allclose(roc_auc_score(subset.injected_math_source.astype(int), subset[key]),
                    metrics['web'][str(step)][pool]['scores'][key]['roc_auc'], atol=1e-12, rtol=0)
    print('QA reference 与旧结果最大 NLL 差：', metrics['qa']['reference_max_nll_difference'])
    print('与新训练集的规范化问题重合：', metrics['qa']['overlap']['normalized_question_overlap_ids'])
    print('旧 notebook 保持原样：', metrics['original_notebooks_unchanged'])
    for key in keys:
        q = metrics['qa']['probes'][probes[1]]['main']['scores'][key]
        w = [metrics['web'][str(s)]['main']['scores'][key]['roc_auc'] for s in [16,32,48,64]]
        display(Markdown(f"**{key}**：新模型 QA AUC **{q['roc_auc']:.4f}**（95% CI {interval(q['roc_auc_ci95'])}）；网页 step 16/32/48/64 AUC 依次为 **" + ' / '.join(f'{v:.4f}' for v in w) + '**。'))
    qa_preserved = all(metrics['qa']['probes'][probes[1]]['main']['scores'][k]['roc_auc_ci95'][0] > .5 for k in keys)
    early_inverted = all(metrics['web']['16']['main']['scores'][k]['roc_auc_ci95'][1] < .5 for k in keys)
    if qa_preserved and early_inverted:
        display(Markdown('**本轮观测：新模型保留了原 QA 区分信号，而网页反向信号在 step 16 已出现。** '
            '这支持 QA 与网页任务之间的迁移限制，并削弱“只有后期训练过度才造成网页失效”的解释。'))
    """)
    md("""
    最早观测是 step 16，不能推断 step 1–15 的情况。
    这两个检查仍无法单独区分聊天格式、答案/全文预测目标、题型及网页来源差异的因果贡献。
    step 64 的正式结果保持不变，没有从诊断中重新挑选最优 checkpoint。
    """)
    return nb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(); run = args.run_dir.resolve()
    metrics = read_json(run/"metrics/metrics.json")
    plan = read_json(run/"diagnostic_plan.json")
    date = datetime.fromisoformat(metrics["completed_at_utc"]).astimezone(timezone.utc)
    repo = Path(__file__).resolve().parents[2]
    folder = repo/"experiments-analysis"/(date.strftime("%Y-%m-%d_%H%M%SZ_")+metrics["run_id"])
    if folder.exists(): raise FileExistsError("Preserving existing analysis notebook")
    snap = folder/"snapshot"; snap.mkdir(parents=True)
    sources = {name: run/"metrics"/name for name in ["metrics.json", "qa_metrics.json", "qa_per_example.parquet",
        "web_per_document.parquet", "qa_length_matched_ids.json", "web_length_matched_ids.json"]}
    sources.update({name: run/name for name in ["diagnostic_plan.json", "original_notebook_hashes.json", "qa_overlap.json"]})
    sources["effective_config.json"] = Path(plan["identity"]["web_run"])/"effective_config.json"
    files = {}
    for name, source in sources.items():
        if source.stat().st_size > 10*1024**2: raise ValueError(f"Snapshot too large: {name}")
        shutil.copy2(source, snap/name)
        files[name] = {"source_path": str(source), "bytes": source.stat().st_size, "sha256": sha256(snap/name)}
        assert files[name]["sha256"] == sha256(source)
    manifest = {"run_id": metrics["run_id"], "directory": folder.name, "analysis_type": metrics["analysis_type"],
        "source_run_dir": str(run), "result_time_utc": date.isoformat(), "timestamp_basis": "metrics/metrics.json completed_at_utc",
        "files": files, "exporter": {"script": str(Path(__file__).relative_to(repo)), "sha256": sha256(__file__)},
        "execution": {"status": "pending"}}
    write_json(folder/"manifest.json", manifest)
    nb = build_notebook(manifest); path = folder/"analysis.ipynb"; nbformat.write(nb, path)
    scratch = Path("/dev/shm/data-picking-runtime")
    for key, sub in [("JUPYTER_RUNTIME_DIR", "jupyter"), ("IPYTHONDIR", "ipython")]:
        (scratch/sub).mkdir(parents=True, exist_ok=True); os.environ.setdefault(key, str(scratch/sub))
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    manager = KernelManager(kernel_name="python3"); manager.kernel_spec.argv[0] = sys.executable
    client = NotebookClient(nb, km=manager, timeout=300, resources={"metadata": {"path": str(folder)}})
    try: client.execute()
    finally:
        nbformat.write(nb, path)
        if manager.has_kernel: manager.shutdown_kernel(now=True)
    nbformat.validate(nb)
    cells = [c for c in nb.cells if c.cell_type == "code"]
    assert all(c.execution_count is not None for c in cells)
    assert not any(o.output_type == "error" for c in cells for o in c.outputs)
    manifest["execution"] = {"status": "completed", "code_cells": len(cells), "notebook_sha256": sha256(path), "gpu_required": False}
    write_json(folder/"manifest.json", manifest)
    figures = run/"figures"; figures.mkdir(exist_ok=True)
    count = 0
    for cell in cells:
        for output in cell.outputs:
            if "image/png" in output.get("data", {}):
                count += 1; (figures/f"figure-{count:02d}.png").write_bytes(base64.b64decode(output.data["image/png"]))
    update_index(repo/"experiments-analysis")
    print("PROBE_NOTEBOOK_COMPLETE", path, "figures", count, flush=True)


if __name__ == "__main__": main()

"""Export the completed OpenStax / FineMath source comparisons."""
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
    nb=nbformat.v4.new_notebook()
    nb.metadata['kernelspec']={'display_name':'Python (data-picking)','language':'python','name':'python3'}
    nb.metadata['data_picking']={'analysis_type':'document_quality_sources','run_id':manifest['run_id']}
    def md(s):nb.cells.append(nbformat.v4.new_markdown_cell(textwrap.dedent(s).strip()))
    def code(s):nb.cells.append(nbformat.v4.new_code_cell(textwrap.dedent(s).strip()))
    md('''
    # OpenStax / FineMath-4+：数学文档来源复验

    分别将 OpenStax 教材和 FineMath-4+ 与相同的 9,000 篇未标注 FineWeb 背景比较。
    使用原 reference 与 MATH 4,096 条 / step-64 SFT 模型，不训练、不选择 checkpoint。
    **loss 下降量与相对下降比例并列报告，方向固定为越大越像数学来源，不设主次、不翻转分数。**

    两次比较的正例分别是教材来源和 FineMath 来源；背景可能含数学。来源检索不等同于语义分类或选数训练收益。
    ''')
    code('''
    from pathlib import Path
    import json, hashlib
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
    from IPython.display import display, Markdown
    root=Path.cwd(); snap=root/'snapshot'
    manifest=json.loads((root/'manifest.json').read_text())
    for name, info in manifest['files'].items():
        assert hashlib.sha256((snap/name).read_bytes()).hexdigest()==info['sha256'], name
    metrics=json.loads((snap/'metrics.json').read_text())
    data=pd.read_parquet(snap/'per_document.parquet')
    metadata=pd.read_parquet(snap/'new_source_metadata.parquet')
    keys=['delta_loss','relative_gain']; baselines=['neg_ref_loss','neg_sft_loss']
    names={'openstax':'OpenStax','finemath':'FineMath-4+'}
    matched_ids={c:json.loads((snap/f'{c}_length_matched_ids.json').read_text()) for c in names}
    colors={'math':'#2379B5','background':'#DD7C32'}
    def ci(x):return f'[{x[0]:.4f}, {x[1]:.4f}]'
    print('校验快照数量：',len(manifest['files']))
    display(pd.DataFrame(metrics['dataset_summary']['sources']).T)
    ''')
    md('''
    ## 数据与评分定义

    - OpenStax：固定 Git revision 的 College Algebra 2e / Precalculus 2e，目录内 121 个独立模块，
      共享 module ID 只计一次。完整保留正文、例题、练习和词汇表，MathML 结构转换为 LaTeX。
      在查看 loss 前隔离 4 个包含空源公式的模块，最终保留 117 个小节、36,534 个数学公式节点。
      图像仅保留文字描述；图形题仍可能缺少视觉信息，不能宣称每篇均适合纯文本理解。
    - FineMath-4+：固定版本、seed 选出的 3 个分片，仅取 CC-MAIN-2023-06，与背景抓取批次一致。
      6,173 条候选中隔离 4 条 target 重合，固定随机抽 1,000 条；没有本地主题、代码或质量分数筛选。
      该抽样代表这三个分片中的合格池，不是全部 FineMath 的均匀样本。
    - 所有文档不套聊天模板、不截断正文；窗口上限 2,048 tokens，每篇除首 token 外均评分一次。
      文档 NLL 为窗口 NLL 总和除以评分 token 数，两模型共享 tokens、mask 与窗口边界。
      完整 FP32、eval、SDPA math；按整篇 NLL 校验重复/padding 误差，容差 2e-5。
    - 原始比较为 117:9,000 和 1,000:9,000。AP、precision 的随机基线因正例比例不同而不同。
      区间来自 2,000 次按来源分层的 bootstrap：OpenStax 按教材章节聚类（14 组），
      FineMath 与背景按重复文档组。区间不包含训练 seed 或全部站点相关性。
    ''')
    md('''
    ## 评分前的质量审查

    每来源固定随机导出 50 篇。人工查看 FineMath 50 篇的首/中/尾摘录、OpenStax 前 12 个随机模块摘录，
    并自动核对全部教材公式转换；未逐篇验证所有数学结论。审查没有改变固定样本。

    FineMath 摘录中同时出现完整数学解释、物理/代码内容、目录页、推广文字、公式缺失及错误解答。
    因此 FineMath-4+ 的分类器高分不能当作人工核验的纯数学/高质量标签。
    OpenStax 也包含章节导言、图表与少量不完善的源说明文字。
    ''')
    code('''
    rows=[]
    for corpus,result in metrics['comparisons'].items():
        for source,s in result['document_summary'].items():
            rows.append({'comparison':names[corpus],'source':source,'documents':s['documents'],
                'fraction improved':s['positive_delta_fraction'],
                **{k:s[k]['mean'] for k in ['ref_mean_nll','sft_mean_nll','delta_loss','relative_gain','token_count']}})
    display(pd.DataFrame(rows).round(6))
    print('准备阶段排除/保留统计：',metrics['dataset_summary']['status_counts'])
    print('参考模型数值检查：',metrics['sanity']['reference'])
    print('SFT 模型数值检查：',metrics['sanity']['sft'])
    ''')
    for corpus in ['openstax','finemath']:
        md(f'## {names_for_title(corpus)}：loss 分布与配对变化')
        code(f"corpus={corpus!r}\n"+textwrap.dedent('''
        frame=data.loc[data.comparison==corpus].copy()
        fig,axes=plt.subplots(1,5,figsize=(22,4))
        nll_bins=np.histogram_bin_edges(frame[['ref_mean_nll','sft_mean_nll']].to_numpy().ravel(),bins=40)
        for ax,key in zip(axes,['ref_mean_nll','sft_mean_nll','delta_loss','relative_gain']):
            bins=nll_bins if key.endswith('mean_nll') else np.histogram_bin_edges(frame[key],bins=50)
            for source,g in frame.groupby('source'):
                ax.hist(g[key],bins=bins,density=True,alpha=.5,color=colors[source],label=names[corpus] if source=='math' else 'Background')
            ax.set(xlabel=key,ylabel='Density',title=key)
            if key in keys:ax.axvline(0,color='black',linewidth=.8)
        ylim=max(axes[0].get_ylim()[1],axes[1].get_ylim()[1])
        for ax in axes[:2]:ax.set_ylim(0,ylim)
        for source,g in frame.groupby('source'):
            axes[4].scatter(g.ref_mean_nll,g.sft_mean_nll,s=6,alpha=.3,color=colors[source],label=names[corpus] if source=='math' else 'Background')
        upper=frame[['ref_mean_nll','sft_mean_nll']].to_numpy().max()*1.02
        axes[4].plot([0,upper],[0,upper],'k--',linewidth=.8)
        axes[4].set(xlabel='Reference NLL',ylabel='SFT NLL',title='Paired document losses')
        for ax in axes:ax.legend(fontsize=8)
        fig.tight_layout();plt.show();plt.close(fig)
        ''').strip())
    md('## 来源区分、原始 loss 基线与 top-k')
    code('''
    rows=[];top=[]
    for corpus,result in metrics['comparisons'].items():
        for pool in ['main','length_matched']:
            m=result[pool]
            if m is None:continue
            for key,v in m['scores'].items():
                rows.append({'source':names[corpus],'pool':pool,'score':key,'AUC':v['roc_auc'],
                    '95% CI':ci(v['roc_auc_ci95']),'AP':v['average_precision'],'random AP':m['source_prevalence']})
                if key in keys+baselines:
                    t=v['top']['0.1'];top.append({'source':names[corpus],'pool':pool,'score':key,'selected docs':t['k'],
                        'retrieved source docs':t['retrieved_math_source'],'source precision':t['source_precision'],
                        'source recall':t['source_recall'],'enrichment over random':t['source_enrichment']})
    display(pd.DataFrame(rows).round(4));display(pd.DataFrame(top).round(4))
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,corpus in zip(axes,names):
        f=data.loc[data.comparison==corpus];y=f.injected_math_source.astype(int)
        for key in keys+baselines:
            x,t,_=roc_curve(y,f[key]);ax.plot(x,t,label=f'{key} ({roc_auc_score(y,f[key]):.3f})')
        ax.plot([0,1],[0,1],'k:',linewidth=.8)
        ax.set(xlabel='Background-source retrieval rate',ylabel='Positive-source recall',title=names[corpus]);ax.legend(fontsize=8)
    fig.tight_layout();plt.show();plt.close(fig)
    ''')
    md('''
    ## 长度控制及其覆盖范围

    在评分前按固定的 2 的幂次 token 长度桶抽样，每桶保持 1:9。没有截断长教材或根据 loss 挑选样本。
    OpenStax 仅保留 25/117 个模块（21.37%），其中较短的章节导言占比较高；这个子集不能代表全部长教学小节。
    FineMath 保留 720/1,000 篇（72%）。分桶仍可能留下桶内长度差异，下表同时报告长度本身的 AUC。
    ''')
    code('''
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    length_rows=[]
    for ax,corpus in zip(axes,names):
        result=metrics['comparisons'][corpus]
        print(names[corpus],result['length_matching'])
        for offset,pool in [(-.12,'main'),(.12,'length_matched')]:
            m=result[pool]
            if m is None:continue
            length_rows.append({'source':names[corpus],'pool':pool,**m['length_baselines']})
            values=m['scores'];x=np.arange(4)+offset
            ax.scatter(x,[values[k]['roc_auc'] for k in keys+baselines],label=pool)
            for xi,k in zip(x,keys+baselines):ax.vlines(xi,*values[k]['roc_auc_ci95'])
        ax.axhline(.5,color='black',linestyle=':')
        ax.set(xticks=np.arange(4),xticklabels=keys+baselines,ylabel='Source ROC-AUC',ylim=(0,1.02),title=names[corpus]);ax.tick_params(axis='x',labelrotation=15);ax.legend()
    display(pd.DataFrame(length_rows).round(4))
    fig.tight_layout();plt.show();plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,3.8))
    for ax,corpus in zip(axes,names):
        f=data.loc[data.comparison==corpus]
        for source,g in f.groupby('source'):
            ax.hist(np.log2(g.token_count),bins=np.arange(5,17,.5),density=True,alpha=.5,color=colors[source],label=names[corpus] if source=='math' else 'Background')
        ax.set(xlabel='log2(document tokens)',ylabel='Density',title=names[corpus]);ax.legend()
    fig.tight_layout();plt.show();plt.close(fig)
    ''')
    md('## 结果复核与解读')
    code('''
    for corpus,result in metrics['comparisons'].items():
        f=data.loc[data.comparison==corpus].sort_values('id')
        assert f.id.is_unique and (f.source=='background').sum()==9000
        for pool,part in [('main',f),('length_matched',f.loc[f.id.isin(matched_ids[corpus])])]:
            m=result[pool]
            if m is None:continue
            y=part.injected_math_source.astype(int)
            for key,v in m['scores'].items():
                np.testing.assert_allclose(roc_auc_score(y,part[key]),v['roc_auc'],atol=1e-12,rtol=0)
                np.testing.assert_allclose(average_precision_score(y,part[key]),v['average_precision'],atol=1e-12,rtol=0)
        for key in keys:
            v=result['main']['scores'][key];lo,hi=v['roc_auc_ci95']
            interpretation='预设方向上有区分信号' if lo>.5 else '区分方向与预设假设相反' if hi<.5 else '区间覆盖 0.5'
            display(Markdown(f"**{names[corpus]} / {key}**：AUC **{v['roc_auc']:.4f}**（95% CI {ci(v['roc_auc_ci95'])}），{interpretation}。"))
    old=metrics['original_web_reference']['scores']
    print('原 OpenWebMath 对照（1,000/9,000）：', {k:old[k]['roc_auc'] for k in keys})
    print('所有 snapshot 的哈希、两组 raw/matched AUC 与 AP 已核对。')
    ''')
    md('''
    本实验同时改变了数学文档的来源、内容构成、长度和抽取/序列化形式。
    来源区分改善不能单独归因于质量；没有改善也不能推出教材对数学训练无用。
    尤其要区分“来源 AUC”和“正例自身 loss 是否下降”：相对背景更容易找回，并不要求正例绝对 loss 下降。
    本轮没有训练所选文档，因此没有直接验证进一步选数训练的收益。
    ''')
    return nb


def names_for_title(corpus):
    return {'openstax':'OpenStax','finemath':'FineMath-4+'}[corpus]


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',type=Path,required=True)
    run=parser.parse_args().run_dir.resolve();metrics=read_json(run/'metrics/metrics.json');plan=read_json(run/'evaluation_plan.json')
    cfg=plan['identity']['config'];bundle=Path(cfg['artifact_root'])/'data/processed'/cfg['bundle_id']
    date=datetime.fromisoformat(metrics['completed_at_utc']).astimezone(timezone.utc)
    repo=Path(__file__).resolve().parents[2];folder=repo/'experiments-analysis'/(date.strftime('%Y-%m-%d_%H%M%SZ_')+metrics['run_id'])
    if folder.exists():raise FileExistsError('Preserving existing analysis notebook')
    snap=folder/'snapshot';snap.mkdir(parents=True)
    sources={name:run/'metrics'/name for name in ['metrics.json','per_document.parquet','new_source_metadata.parquet',
        'openstax_length_matched_ids.json','finemath_length_matched_ids.json']}
    sources.update({name:run/name for name in ['evaluation_plan.json','preparation_plan_v2.json','quality_audit_review.json','length_control_plan.json']})
    sources['dataset_summary.json']=bundle/'summary.json'
    base_run=Path(plan['identity']['checkpoint']).parent.parent;sources['effective_config.json']=base_run/'effective_config.json'
    files={}
    for name,p in sources.items():
        if p.stat().st_size>10*1024**2:raise ValueError(f'Snapshot too large: {name}')
        shutil.copy2(p,snap/name);files[name]={'source_path':str(p),'bytes':p.stat().st_size,'sha256':sha256(snap/name)}
        assert files[name]['sha256']==sha256(p)
    manifest={'run_id':metrics['run_id'],'directory':folder.name,'analysis_type':metrics['analysis_type'],
        'source_run_dir':str(run),'result_time_utc':date.isoformat(),'timestamp_basis':'metrics/metrics.json completed_at_utc',
        'files':files,'exporter':{'script':str(Path(__file__).relative_to(repo)),'sha256':sha256(__file__)},'execution':{'status':'pending'}}
    write_json(folder/'manifest.json',manifest);nb=build_notebook(manifest);path=folder/'analysis.ipynb';nbformat.write(nb,path)
    scratch=Path('/dev/shm/data-picking-runtime')
    for key,sub in [('JUPYTER_RUNTIME_DIR','jupyter'),('IPYTHONDIR','ipython')]:
        (scratch/sub).mkdir(parents=True,exist_ok=True);os.environ.setdefault(key,str(scratch/sub))
    os.environ['CUDA_VISIBLE_DEVICES']=''
    manager=KernelManager(kernel_name='python3');manager.kernel_spec.argv[0]=sys.executable
    client=NotebookClient(nb,km=manager,timeout=300,resources={'metadata':{'path':str(folder)}})
    try:client.execute()
    finally:
        nbformat.write(nb,path)
        if manager.has_kernel:manager.shutdown_kernel(now=True)
    nbformat.validate(nb);cells=[c for c in nb.cells if c.cell_type=='code']
    assert all(c.execution_count is not None for c in cells)
    assert not any(o.output_type=='error' for c in cells for o in c.outputs)
    manifest['execution']={'status':'completed','code_cells':len(cells),'notebook_sha256':sha256(path),'gpu_required':False}
    write_json(folder/'manifest.json',manifest)
    figures=run/'figures';figures.mkdir(exist_ok=True);count=0
    for cell in cells:
        for out in cell.outputs:
            if 'image/png' in out.get('data',{}):
                count+=1;(figures/f'figure-{count:02d}.png').write_bytes(base64.b64decode(out.data['image/png']))
    update_index(repo/'experiments-analysis');print('QUALITY_NOTEBOOK_COMPLETE',path,'figures',count,flush=True)


if __name__=='__main__':main()

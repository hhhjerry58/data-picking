"""Export direct source-level document-training utility with all seeds."""
import argparse
import base64
from datetime import datetime,timezone
import os
from pathlib import Path
import shutil
import sys
import textwrap

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient
from _shared import read_json,sha256,write_json,update_index


def build(manifest):
    nb=nbformat.v4.new_notebook()
    nb.metadata['kernelspec']={'display_name':'Python (data-picking)','language':'python','name':'python3'}
    nb.metadata['data_picking']={'analysis_type':'document_training_utility','run_id':manifest['run_id']}
    def md(s):nb.cells.append(nbformat.v4.new_markdown_cell(textwrap.dedent(s).strip()))
    def code(s):nb.cells.append(nbformat.v4.new_code_cell(textwrap.dedent(s).strip()))
    md('''
    # 数学文档是否比 general 更能降低目标 SFT loss？

    从相同 reference 模型出发，分别用 General FineWeb、OpenWebMath、OpenStax 和 FineMath-4+
    做文档继续训练，随后在同一套留出 MATH 问答上计算回答 NLL。没有再训练 target SFT。
    **loss 下降量与相对下降比例并列考察，不设主次。** 本轮指标是实际目标 loss 的变化，使用来源识别 AUC 的旧结果不等于训练效用。
    ''')
    code('''
    from pathlib import Path
    import json,hashlib
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from IPython.display import display,Markdown
    root=Path.cwd();snap=root/'snapshot';manifest=json.loads((root/'manifest.json').read_text())
    for name,info in manifest['files'].items():assert hashlib.sha256((snap/name).read_bytes()).hexdigest()==info['sha256'],name
    metrics=json.loads((snap/'metrics.json').read_text())
    data=pd.read_parquet(snap/'per_example.parquet');training=pd.read_parquet(snap/'training.parquet')
    names={'general':'General FineWeb','openwebmath':'OpenWebMath','openstax':'OpenStax','finemath':'FineMath-4+'}
    arms=list(names);seeds=metrics['seeds'];keys=['delta_loss','relative_gain'];ref=metrics['reference_mean_nll']
    colors={'general':'#777777','openwebmath':'#2379B5','openstax':'#31935F','finemath':'#D8792E'}
    def ci(x):return f'[{x[0]:.5f}, {x[1]:.5f}]'
    print('Reference held-out mean answer NLL:',ref)
    print('目标样本 / 问题组：',metrics['target_examples'],metrics['target_groups'])
    ''')
    md('''
    ## 固定训练预算与实验单位

    四种来源 × seed 42/43/44，共 12 个 run。各 64 步，每步 16,384 个有效预测 token，
    合计 1,048,576 个；学习率 2e-5、warmup 7 步、线性衰减，AdamW 设置一致。
    不放回打乱已有文档窗口，分步边界保留窗口前缀作为上下文并 mask 已消费目标；无跨文档拼接。
    不同 seed 同时改变窗口抽样和顺序。按 token 配平，样本数和输入上下文量可以不同。

    FP32 master/optimizer、bf16 autocast、SDPA math；保持有限梯度检查。固定 step 64 评估，
    step 32 只用于完整状态保存/恢复。General seed 42 实际从 step 32 恢复。
    目标为原 512 条留出 MATH，完整 FP32 评分、同 tokens/response mask，未用于选模型。
    General 是原未标注背景，不能等同于人工确认的非数学语料。
    ''')
    code('''
    recipes=pd.DataFrame(metrics['data_preparation']['runs'])
    display(recipes[['arm','seed','documents','fragments','supervised_tokens','input_tokens','steps']])
    assert recipes.supervised_tokens.nunique()==1 and recipes.supervised_tokens.iloc[0]==1048576
    display(pd.DataFrame([{'arm':a,'seed':s,'observed steps':len(g),'tokens':g.response_tokens.sum(),
        'finite gradients':np.isfinite(g.grad_norm).all()} for (a,s),g in training.groupby(['arm','seed'])]))
    fig,ax=plt.subplots(figsize=(10,4))
    for arm in arms:
        g=training.loc[training.arm==arm].groupby('step').loss.mean()
        ax.plot(g.index,g,label=names[arm],color=colors[arm])
    ax.set(xlabel='Optimizer step',ylabel='Training token NLL',title='Training stability (different source losses are not utility comparisons)');ax.legend()
    fig.tight_layout();plt.show();plt.close(fig)
    ''')
    md('## 留出 MATH 回答 loss：全部种子与绝对改善')
    code('''
    per_seed=[];summary=[]
    for arm,result in metrics['arms'].items():
        for i,seed in enumerate(seeds):
            per_seed.append({'source':names[arm],'seed':seed,'target NLL':result['per_seed_target_nll'][i],
                **{k:result['scores'][k]['per_seed'][i] for k in keys}})
        summary.append({'source':names[arm],'mean target NLL':result['mean_target_nll'],
            **{k:result['scores'][k]['mean'] for k in keys}})
    display(pd.DataFrame(per_seed).round(6));display(pd.DataFrame(summary).round(6))
    fig,ax=plt.subplots(figsize=(9,4))
    for i,arm in enumerate(arms):
        r=metrics['arms'][arm];ax.scatter([i-.12,i,i+.12],r['per_seed_target_nll'],color=colors[arm])
        ax.plot([i-.2,i+.2],[r['mean_target_nll']]*2,color=colors[arm],linewidth=3)
    ax.axhline(ref,color='black',linestyle='--',label='No document training')
    ax.set(xticks=range(4),xticklabels=[names[a] for a in arms],ylabel='Held-out mean answer NLL',title='Three seeds per source; lower is better');ax.legend()
    fig.tight_layout();plt.show();plt.close(fig)
    ''')
    md('''
    ## 数学来源相对 general 的配对效用

    对同一个 seed、同一个问题计算 `gain_math − gain_general`，正值表示数学源更有利。
    下降量差也等于 `L_general − L_math`。绝对 gain 为负表示损失增加，不能把“增加较少”当作“下降”。

    表中给出两套 95% 区间：conditional 仅重采样问题组，条件于三个训练模型；hierarchical 同时重采样
    seed 与问题组，并在来源之间共享索引。只有三个 seed，训练随机性的总体不确定性仍只能粗略估计。
    ''')
    code('''
    rows=[]
    for arm in arms[1:]:
        for key in keys:
            v=metrics['arms'][arm]['contrast_vs_general'][key]
            rows.append({'source':names[arm],'metric':key,'mean advantage over general':v['mean'],
                **{f'seed {s}':v['per_seed'][i] for i,s in enumerate(seeds)},
                'conditional 95% CI':ci(v['conditional_ci95']),'hierarchical 95% CI':ci(v['hierarchical_ci95'])})
    display(pd.DataFrame(rows).round(6))
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,key in zip(axes,keys):
        for i,arm in enumerate(arms[1:]):
            v=metrics['arms'][arm]['contrast_vs_general'][key]
            ax.hlines(i,*v['hierarchical_ci95'],color=colors[arm],linewidth=2)
            ax.scatter(v['per_seed'],[i-.06,i,i+.06],color=colors[arm],alpha=.5,s=20)
            ax.scatter([v['mean']],[i],color=colors[arm],s=55,marker='D')
        ax.axvline(0,color='black',linestyle=':')
        ax.set(yticks=range(3),yticklabels=[names[a] for a in arms[1:]],xlabel='Math improvement minus general improvement',title=key)
    fig.tight_layout();plt.show();plt.close(fig)
    ''')
    md('## 同一批目标问题上的配对结果')
    code('''
    avg=data.groupby(['arm','id']).trained_mean_nll.mean().unstack('arm')
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for ax,arm in zip(axes,arms[1:]):
        ax.scatter(avg.general,avg[arm],s=8,alpha=.4,color=colors[arm]);bound=avg[['general',arm]].to_numpy().max()*1.02
        ax.plot([0,bound],[0,bound],'k--',linewidth=.8)
        ax.set(xlabel='General-trained answer NLL',ylabel='Math-trained answer NLL',title=names[arm])
    fig.tight_layout();plt.show();plt.close(fig)
    print('每个点为同一道留出题在三个 seed 上的平均 NLL；对角线下方表示数学来源更有利。')
    ''')
    md('## 校验与解释边界')
    code('''
    assert len(data)==12*512
    for arm,result in metrics['arms'].items():
        f=data.loc[data.arm==arm]
        for i,seed in enumerate(seeds):
            g=f.loc[f.seed==seed];assert g.id.is_unique and len(g)==512
            np.testing.assert_allclose(g.trained_mean_nll.mean(),result['per_seed_target_nll'][i],atol=1e-12,rtol=0)
            for key in keys:np.testing.assert_allclose(g[key].mean(),result['scores'][key]['per_seed'][i],atol=1e-12,rtol=0)
    print('Reference 交叉复核最大 NLL 差：',metrics['reference_crosscheck_max_nll_difference'])
    print('真实 resume 验证：',metrics['actual_resume'])
    for arm in arms[1:]:
        for key in keys:
            v=metrics['arms'][arm]['contrast_vs_general'][key];lo,hi=v['hierarchical_ci95']
            meaning='数学来源更有利' if lo>0 else '数学来源更不利' if hi<0 else '区间覆盖零，尚无明确方向'
            display(Markdown(f"**{names[arm]} / {key}**：相对 general 的改善差 **{v['mean']:.6f}**，hierarchical 95% CI {ci(v['hierarchical_ci95'])}；{meaning}。"))
    ''')
    md('''
    这些结论只对应本轮 reference 模型、语料池、训练预算与优化设置。
    “没有检出数学来源优势”不能证明两者等价；即使结果与先前 loss 变化的方向一致，也只是来源层面的证据。
    本轮没有在同一候选池内比较按分数挑选和随机挑选，因此没有证明逐样本数据排序的准确性。
    检查点 I/O 的修订不改变数值训练代码；完整状态、有效 token 数和原始日志均保留。
    ''')
    return nb


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True)
    run=p.parse_args().run_dir.resolve();metrics=read_json(run/'metrics/metrics.json');plan=read_json(run/'experiment_plan.json')
    date=datetime.fromisoformat(metrics['completed_at_utc']).astimezone(timezone.utc)
    repo=Path(__file__).resolve().parents[2];folder=repo/'experiments-analysis'/(date.strftime('%Y-%m-%d_%H%M%SZ_')+metrics['run_id'])
    if folder.exists():raise FileExistsError('Preserving existing analysis notebook')
    snap=folder/'snapshot';snap.mkdir(parents=True)
    write_json(run/'analysis_effective_config.json',{'config':plan['config']})
    sources={name:run/'metrics'/name for name in ['metrics.json','per_example.parquet','training.parquet']}
    sources.update({name:run/name for name in ['experiment_plan.json','data_preparation.json','checkpoint_io_transition.json','checkpoint_io_v2.json','checkpoint_io_v3.json','checkpoint_transport_checks.json']})
    sources['effective_config.json']=run/'analysis_effective_config.json'
    files={}
    for name,source in sources.items():
        if source.stat().st_size>10*1024**2:raise ValueError('Snapshot too large')
        shutil.copy2(source,snap/name);files[name]={'source_path':str(source),'bytes':source.stat().st_size,'sha256':sha256(snap/name)}
        assert files[name]['sha256']==sha256(source)
    manifest={'run_id':metrics['run_id'],'directory':folder.name,'analysis_type':metrics['analysis_type'],
        'source_run_dir':str(run),'result_time_utc':date.isoformat(),'timestamp_basis':'metrics/metrics.json completed_at_utc',
        'files':files,'exporter':{'script':str(Path(__file__).relative_to(repo)),'sha256':sha256(__file__)},'execution':{'status':'pending'}}
    write_json(folder/'manifest.json',manifest);nb=build(manifest);path=folder/'analysis.ipynb';nbformat.write(nb,path)
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
    for c in cells:
        for out in c.outputs:
            if 'image/png' in out.get('data',{}):
                count+=1;(figures/f'figure-{count:02d}.png').write_bytes(base64.b64decode(out.data['image/png']))
    update_index(repo/'experiments-analysis');print('UTILITY_NOTEBOOK_COMPLETE',path,'figures',count,flush=True)


if __name__=='__main__':main()

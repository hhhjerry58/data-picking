"""Export the completed math-document retrieval verification and gradient appendix."""
import argparse
import base64
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import sys
import textwrap

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

from data_picking.web_analysis import prepare_gradient_snapshots
from _shared import read_json, write_json, sha256, update_index, loss_metric_overview


def build_notebook(manifest, metrics):
    cells=[]
    def md(s):cells.append(nbformat.v4.new_markdown_cell(textwrap.dedent(s).strip()))
    def code(s):cells.append(nbformat.v4.new_code_cell(textwrap.dedent(s).strip()))
    md(f"""
    # Math documents / CC：math SFT 后的文档检索验证

    **Run：** `{manifest['run_id']}`  
    **结果时间（UTC）：** {manifest['result_time_utc']}

    在 4,096 条多题型 MATH 问答上训练 64 步，比较原始 ref 与预先固定的 **step 64** 模型。
    候选池包含 1,000 篇 OpenWebMath 注入文档和 9,000 篇未标注 CC 背景网页。
    {loss_metric_overview(metrics['main']['scores'])}

    本页验证问答任务的学习信号能否迁移到文档检索；没有进行选中文档后的再次训练。
    所有表格和图已执行保存，重跑只需要本目录快照和 CPU。
    """)
    md("""
    ## 1. 数据与评价约定

    两种来源都限定在 `CC-MAIN-2023-06`，在固定源分片内随机抽样。
    OpenWebMath 和 FineWeb 的上游提取/质量筛选不同；这一来源差异仍然存在。
    **标签表示是否来自注入的 OpenWebMath 集合，不是人工核验的数学内容标签。**
    CC 背景可能含数学内容，因此以下 precision/recall/enrichment 都是注入来源的检索指标。

    网页使用原始正文，不加聊天模板。每个 token（每篇第一个除外）评分一次，窗口最多 2,048 tokens。
    整篇 NLL = 所有窗口 NLL 总和 / 实际评分 token 总数；再计算下降量与相对下降比例。
    分数越大越倾向注入来源，方向固定。raw-loss 基线同时展示，未按观察到的 AUC 翻转分数。
    置信区间为按来源分层、按重复文档组重采样的 2,000 次 bootstrap。
    """)
    code(r"""
    import hashlib,json,math,os
    import tempfile
    from pathlib import Path
    if 'MPLCONFIGDIR' not in os.environ:
        scratch=Path('/dev/shm/data-picking-runtime') if Path('/dev/shm').is_dir() else Path(tempfile.gettempdir())/'data-picking-runtime'
        os.environ['MPLCONFIGDIR']=str(scratch/'matplotlib')
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from IPython.display import display,Markdown
    from sklearn.metrics import roc_auc_score,average_precision_score,roc_curve
    %matplotlib inline
    plt.rcParams.update({'figure.dpi':120,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    pd.set_option('display.max_colwidth',80)
    """)
    code('DIRECTORY = '+repr(manifest['directory'])+'\n'+textwrap.dedent(r"""
    HERE=next((p for p in [Path.cwd(),Path.cwd()/'experiments-analysis'/DIRECTORY] if (p/'snapshot').is_dir() and (p/'manifest.json').is_file()),None)
    if HERE is None:raise FileNotFoundError('请使用当前实验目录或仓库根目录。')
    SNAP=HERE/'snapshot';manifest=json.loads((HERE/'manifest.json').read_text())
    for name,info in manifest['files'].items():
        data=(SNAP/name).read_bytes()
        assert len(data)==info['bytes'] and hashlib.sha256(data).hexdigest()==info['sha256'],name
    def load(name):return json.loads((SNAP/name).read_text())
    metrics=load('metrics.json');effective=load('effective_config.json');dataset=load('dataset_summary.json')
    plan=load('evaluation_plan.json');completed=load('training_complete.json')
    df=pd.read_parquet(SNAP/'per_document.parquet')
    windows=pd.read_parquet(SNAP/'per_window.parquet')
    target=pd.read_parquet(SNAP/'target_validation.parquet')
    training=pd.read_json(SNAP/'train.jsonl',lines=True)
    matched=df.set_index('id').loc[load('length_matched_ids.json')].reset_index()
    assert df['id'].is_unique and df.groupby('source').size().to_dict()=={'background':9000,'math':1000}
    assert len(windows)==metrics['windows'] and int(df['scored_tokens'].sum())==metrics['scored_tokens']
    totals=windows.groupby('document_id')[['ref_sum_nll','sft_sum_nll','scored_tokens']].sum()
    aligned=df.set_index('id').loc[totals.index]
    for prefix in ['ref','sft']:
        np.testing.assert_allclose(totals[f'{prefix}_sum_nll']/totals['scored_tokens'],aligned[f'{prefix}_mean_nll'],rtol=0,atol=1e-12)
    assert completed['global_step']==64 and training['step'].tolist()==list(range(1,65))
    assert training['response_tokens'].sum()==completed['seen_response_tokens']==dataset['target']['splits']['train']['response_tokens']
    keys=['delta_loss','relative_gain','neg_ref_loss','neg_sft_loss']
    names={'math':'Injected OpenWebMath','background':'Unlabeled CC background'}
    colors={'math':'#1674B8','background':'#DF7931'}
    labels={'delta_loss':'loss 下降量','relative_gain':'相对下降比例','neg_ref_loss':'ref loss，低值优先','neg_sft_loss':'SFT loss，低值优先'}
    def interval(v):return f'[{v[0]:.4f}, {v[1]:.4f}]'
    print(f'快照及文档 NLL 聚合核对通过：{len(df):,} 篇网页，{len(windows):,} 个窗口，{metrics["scored_tokens"]:,} 个评分 token。')
    """))
    md('## 2. 固定训练设置与留出验证集')
    code(r"""
    cfg=effective['config']
    display(pd.DataFrame([
        ('模型',cfg['model']['name']),('模型 revision',cfg['model']['revision']),
        ('训练 / 验证样本',f"{effective['actual_train_size']} / {dataset['target']['splits']['validation']['examples']}"),
        ('有效 batch / micro batch / 累积',f"{cfg['training']['effective_batch_size']} / {cfg['training']['micro_batch_size']} / {effective['gradient_accumulation_steps']}"),
        ('训练步数 / 学习率',f"{completed['global_step']} / {cfg['training']['learning_rate']}"),
        ('checkpoint', '16, 32, 48, 64；最终结果固定为 64'),
        ('训练 / 评分精度','float32 master + bf16 autocast / full float32；SDPA math'),
        ('optimizer 保存 / 自动恢复',f"{effective['save_optimizer']} / {effective['auto_resume']}"),
    ],columns=['设置','值']))
    display(pd.DataFrame(dataset['target']['splits']['train']['subjects'],index=['train']).T.join(
        pd.Series(dataset['target']['splits']['validation']['subjects'],name='validation')))
    display(pd.DataFrame(load('checkpoint_state_checks.json')).T)
    checks=[]
    for model in ['reference_web','sft_web']:
        for value in metrics['score_sanity'][model]:
            checks.append({'model':model,'shard':value['shard'],'unit':value.get('unit','window_mean_nll'),
                           'repeat max':value['repeat_max_nll_difference'],'padding max':value['batch_padding_max_nll_difference'],
                           'window padding max':value.get('window_batch_padding_max_nll_difference'), 'tolerance':value['tolerance']})
    display(pd.DataFrame(checks))
    fig,axes=plt.subplots(1,2,figsize=(11,3.8))
    axes[0].plot(training['step'],training['loss'],marker='.',color=colors['math'])
    axes[0].set(xlabel='Optimizer step',ylabel='Mean response NLL',title='Training batches (MATH only)')
    trend=pd.DataFrame(metrics['target_validation'])
    axes[1].plot(trend['checkpoint_step'],trend['mean_nll'],marker='o',color=colors['background'])
    axes[1].set(xlabel='Checkpoint step (0 = reference)',ylabel='Mean response NLL',title='Fixed held-out MATH validation')
    fig.tight_layout();plt.show();plt.close(fig)
    display(trend.round(5))
    """)
    md('每个训练 batch 不同；中间 checkpoint 的留出 loss 仅作趋势诊断，未用于选 checkpoint。这里测的是回答 NLL，没有测解题准确率。网页数值检查按完整文档聚合，窗口差同时保留：一次 1-token 尾窗口差为 2.10e-5，对整篇均值的贡献约 1e-8。阈值仍为 2e-5，原始评分计算与数值没有改写，详见 numerical_validation_amendment.json。')
    md('## 3. 两种网页来源的 loss 变化')
    code(r"""
    table=df.groupby('source').agg(documents=('id','size'),ref_NLL=('ref_mean_nll','mean'),sft_NLL=('sft_mean_nll','mean'),
                                   mean_drop=('delta_loss','mean'),median_relative_drop=('relative_gain','median'),mean_tokens=('token_count','mean'))
    table['fraction_improved']=df.groupby('source')['delta_loss'].apply(lambda x:(x>0).mean())
    display(table.round(5))
    fig=plt.figure(figsize=(12,12));grid=fig.add_gridspec(3,2)
    axes=[fig.add_subplot(grid[i,j]) for i in range(2) for j in range(2)]+[fig.add_subplot(grid[2,0])]
    nll_bins=np.histogram_bin_edges(df[['ref_mean_nll','sft_mean_nll']].to_numpy().ravel(),bins=40)
    relative_range=(-.25,.85);relative_bins=np.linspace(*relative_range,41)
    specs=[('ref_mean_nll','1. Reference NLL'),('sft_mean_nll','2. SFT NLL (step 64)'),
           ('delta_loss','3. Reference NLL - SFT NLL'),('relative_gain','4. Relative loss reduction')]
    for ax,(key,title) in zip(axes,specs):
        bins=relative_bins if key=='relative_gain' else nll_bins if key.endswith('mean_nll') else np.histogram_bin_edges(df[key],bins=40)
        for source,g in df.groupby('source'):
            if key=='relative_gain':
                ax.hist(g[key],bins=bins,weights=np.full(len(g),1/(len(g)*(bins[1]-bins[0]))),alpha=.5,color=colors[source],label=names[source])
            else:ax.hist(g[key],bins=bins,density=True,alpha=.5,color=colors[source],label=names[source])
        ax.set(xlabel=key,ylabel='Density',title=title)
        if key in ['delta_loss','relative_gain']:ax.axvline(0,color='black',linewidth=.8)
    ymax=max(ax.get_ylim()[1] for ax in axes[:2])
    for ax in axes[:2]:ax.set(xlim=(nll_bins[0],nll_bins[-1]),ylim=(0,ymax))
    for source,g in df.groupby('source'):axes[4].scatter(g.ref_mean_nll,g.sft_mean_nll,s=5,alpha=.25,color=colors[source],label=names[source])
    bound=df[['ref_mean_nll','sft_mean_nll']].to_numpy().max()*1.02
    axes[4].plot([0,bound],[0,bound],'k--',linewidth=.8)
    axes[4].set(xlabel='Reference NLL',ylabel='SFT NLL',title='5. Paired document mean losses',xlim=(0,bound),ylim=(0,bound))
    for ax in axes:ax.legend(fontsize=7)
    fig.tight_layout();plt.show();plt.close(fig)
    outside=df.loc[~df.relative_gain.between(*relative_range)]
    print('相对下降比例图显示 [-0.25, 0.85]；范围外样本仍参与所有统计：',outside.groupby('source').size().to_dict())
    """)
    md('## 4. 能否找回注入的数学相关文档？')
    code(r"""
    for frame,result in [(df,metrics['main']),(matched,metrics['length_matched'])]:
        y=frame.injected_math_source.to_numpy(dtype=int)
        ties=np.random.default_rng(result['tie_seed']).random(len(frame))
        for key,value in result['scores'].items():
            np.testing.assert_allclose(roc_auc_score(y,frame[key]),value['roc_auc'],rtol=0,atol=1e-12)
            np.testing.assert_allclose(average_precision_score(y,frame[key]),value['average_precision'],rtol=0,atol=1e-12)
            order=np.lexsort((ties,-frame[key].to_numpy()))
            for f,top in value['top'].items():assert int(y[order[:top['k']]].sum())==top['retrieved_math_source']
    def metric_table(result,columns=keys):
        return pd.DataFrame([{'指标':labels.get(k,k),'AUC':round(result['scores'][k]['roc_auc'],4),
            'AUC 95% CI':interval(result['scores'][k]['roc_auc_ci95']),'AP':round(result['scores'][k]['average_precision'],4)} for k in columns])
    display(metric_table(metrics['main']))
    rows=[]
    for key in keys:
        for f,v in metrics['main']['scores'][key]['top'].items():
            rows.append({'指标':labels[key],'选取比例':f'{float(f):.0%}','文档数':v['k'],'找回注入文档':v['retrieved_math_source'],
                         '来源 precision':v['source_precision'],'注入集 recall':v['source_recall'],
                         'recall 95% CI':interval(v['source_recall_ci95']),'相对随机富集倍数':v['source_enrichment']})
    display(pd.DataFrame(rows).round(4))
    fig,axes=plt.subplots(1,2,figsize=(12,4.4))
    y=df.injected_math_source.to_numpy(dtype=int)
    ties=np.random.default_rng(metrics['main']['tie_seed']).random(len(df))
    for key in keys:
        fpr,tpr,_=roc_curve(y,df[key]);axes[0].plot(fpr,tpr,label=f'{key} ({roc_auc_score(y,df[key]):.3f})')
        order=np.lexsort((ties,-df[key].to_numpy()))
        axes[1].plot(np.arange(1,len(df)+1)/len(df),np.cumsum(y[order])/y.sum(),label=key)
    for ax in axes:ax.plot([0,1],[0,1],'k--',linewidth=.8);ax.legend(fontsize=8)
    axes[0].set(xlabel='Background-source retrieval rate',ylabel='Injected-source recall',title='Source ROC')
    axes[1].set(xlabel='Fraction of candidate documents selected',ylabel='Injected-source recall',xlim=(0,.2),title='Retrieval under selection budgets')
    fig.tight_layout();plt.show();plt.close(fig)
    print('原始池随机 precision 期望为 10%；这里的 precision 衡量注入来源，不能当作真实数学内容纯度。')
    """)
    md('## 5. 长度控制与相对原始 loss 基线的差异')
    code(r"""
    print('长度匹配：',metrics['length_matching'])
    display(metric_table(metrics['length_matched']))
    display(pd.DataFrame([{'pool':name,**m['length_baselines']} for name,m in [('raw',metrics['main']),('matched',metrics['length_matched'])]]))
    fig,ax=plt.subplots(figsize=(9,4))
    for shift,name,result in [(-.15,'Original',metrics['main']),(.15,'Length matched',metrics['length_matched'])]:
        x=np.arange(len(keys))+shift;ax.scatter(x,[result['scores'][k]['roc_auc'] for k in keys],label=name)
        for xi,k in zip(x,keys):ax.vlines(xi,*result['scores'][k]['roc_auc_ci95'])
    ax.axhline(.5,color='black',linestyle='--',linewidth=.8)
    ax.set(xticks=np.arange(len(keys)),xticklabels=keys,ylabel='ROC-AUC',ylim=(0,1));ax.legend()
    fig.tight_layout();plt.show();plt.close(fig)
    records=[]
    for pool,result in [('raw',metrics['main']),('matched',metrics['length_matched'])]:
        for key in ['delta_loss','relative_gain']:
            for base in ['neg_ref_loss','neg_sft_loss']:
                v=result['paired_auc_differences'][key][base]
                records.append({'pool':pool,'比较':key+' − '+base,'AUC 差':v['difference'],'配对 95% CI':interval(v['ci95'])})
    display(pd.DataFrame(records).round(4))
    """)
    md('长度匹配按混合池的 token 数四分位分桶，桶内随机抽取并维持 1:9 来源比例。它改变了评估人群，不能保证消除桶内长度、网站风格或上游清洗差异。')
    md('## 6. 观察与局限')
    code(r"""
    lines=[]
    for key in ['delta_loss','relative_gain']:
        raw=metrics['main']['scores'][key];control=metrics['length_matched']['scores'][key]
        v=raw['top']['0.1']
        lines.append(f"{labels[key]}：原始 AUC {raw['roc_auc']:.4f}（{interval(raw['roc_auc_ci95'])}），"
                     f"长度匹配 AUC {control['roc_auc']:.4f}；前 {v['k']} 篇找回 {v['retrieved_math_source']} 篇注入文档（recall {v['source_recall']:.1%}）。")
        lo,hi=raw['roc_auc_ci95']
        if hi<.5:
            lines.append(f"{labels[key]}在预设方向上的 AUC 低于 0.5：较高分数更偏向背景来源，没有事后翻转分数。")
        elif lo>.5:
            lines.append(f"{labels[key]}在预设方向上有注入来源的区分信号；仍需结合原始 loss 基线判断额外信息。")
        else:
            lines.append(f"{labels[key]}的区间覆盖 0.5，本轮尚无稳定的正向区分证据。")
    for source,g in metrics['document_summary'].items():
        lines.append(f"{names[source]}：平均文档 NLL {g['ref_mean_nll']['mean']:.4f} → {g['sft_mean_nll']['mean']:.4f}。")
    display(Markdown('\n\n'.join(lines)))
    print('相对下降分母保护触发数：',metrics['relative_gain_denominator_clamps'])
    display(metric_table(metrics['main'],columns=['pos_ref_loss','pos_sft_loss']))
    """)
    md("""
    这是指定分片、指定抓取批次、一个训练 seed 上的来源检索结果。Bootstrap 不包含训练随机性，
    也没有完整覆盖同站点文风的相关性。上游提取差异、网页主题与来源标记之间的混杂仍需后续检验。
    尚未人工标注背景网页的数学相关性，未训练通用 SFT 对照，也未做等 token 预算的选数后训练对照。
    因此本轮不能直接证明选中的文档带来更大的数学训练收益。
    """)
    md('## 附录：固定随机子集的整篇文档梯度')
    code(r"""
    gradient=load('gradient_summary.json')
    g=pd.read_parquet(SNAP/'gradient_per_example.parquet')
    layers=pd.read_parquet(SNAP/'gradient_per_layer.parquet')
    assert g.groupby(['model','source']).size().to_dict()=={('reference','background'):64,('reference','math'):64,('sft','background'):64,('sft','math'):64}
    display(g.groupby(['model','source']).agg(documents=('id','size'),tokens_median=('token_count','median'),
        norm_median=('grad_l2','median'),max_abs_median=('grad_max_abs','median'),cos_to_A1_median=('cos_to_a1','median')).round(5))
    display(pd.DataFrame(gradient['anchors']).loc[['norm','mean_nll','parameter_count']])
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    order=[('reference','math'),('sft','math'),('reference','background'),('sft','background')]
    for ax,key in zip(axes,['grad_l2','grad_max_abs','cos_to_a1']):
        values=[g.loc[(g.model==m)&(g.source==s),key] for m,s in order]
        ax.boxplot(values,tick_labels=['ref/math','sft/math','ref/bg','sft/bg'],flierprops={'markersize':2})
        ax.set(title=key);ax.tick_params(axis='x',rotation=35)
        if key!='cos_to_a1':ax.set_yscale('log')
    fig.tight_layout();plt.show();plt.close(fig)
    layer_order=['embedding']+sorted(k for k in layers.layer.unique() if k.startswith('layer_'))
    layer_order+=sorted(k for k in layers.layer.unique() if k not in layer_order)
    fig,axes=plt.subplots(1,2,figsize=(12,4),sharey=True)
    for ax,model in zip(axes,['reference','sft']):
        for source in ['math','background']:
            values=layers.loc[(layers.model==model)&(layers.source==source)].groupby('layer').grad_rms.median().reindex(layer_order)
            ax.plot(range(len(values)),values,label=names[source],color=colors[source])
        ax.set(yscale='log',title=model,xlabel='Embedding / transformer blocks / final norm',ylabel='Median gradient RMS');ax.legend(fontsize=8)
    fig.tight_layout();plt.show();plt.close(fig)
    print('文档梯度与完整评分的 NLL 最大差：',gradient['max_nll_crosscheck_error'])
    """)
    md("""
    梯度样本在查看检索结果前按 seed 固定，每来源 64 篇，全部窗口参与计算。
    `g(d)` 是整篇 token 加权平均 NLL 的梯度，先累积各窗口梯度再计算 norm；不是窗口 norm 的平均。
    完整 FP32、eval、SDPA math，不裁剪、不执行 optimizer update，绑定参数只计一次。
    两模型分别用同一批 128 条 A1 构造各自平均梯度，cosine 参照会随模型改变。
    这里的子集诊断不替代 10,000 篇候选池的检索统计；原始逐参数明细保存在运行目录。
    """)
    nb=nbformat.v4.new_notebook(cells=cells)
    nb.metadata['kernelspec']={'display_name':'Python 3','language':'python','name':'python3'}
    nb.metadata['language_info']={'name':'python','version':f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}'}
    nb.metadata['data_picking']={'analysis_type':'web_document_retrieval','run_id':manifest['run_id']}
    return nb


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args();run=args.run_dir.resolve()
    metrics=read_json(run/'metrics_web/metrics.json')
    prepare_gradient_snapshots(run)
    date=datetime.fromisoformat(metrics['completed_at_utc']).astimezone(timezone.utc)
    repo=Path(__file__).resolve().parents[2]
    name=date.strftime('%Y-%m-%d_%H%M%SZ_')+metrics['run_id']
    folder=repo/'experiments-analysis'/name
    if folder.exists():raise FileExistsError('Preserving existing notebook; edit or archive it explicitly')
    snap=folder/'snapshot';snap.mkdir(parents=True)
    bundle=Path(read_json(run/'evaluation_plan.json')['config']['bundle'])
    sources={name:run/'metrics_web'/name for name in ['metrics.json','per_document.parquet','per_window.parquet',
        'target_validation.parquet','length_matched_ids.json','gradient_per_example.parquet','gradient_per_layer.parquet',
        'gradient_parameter_summary.parquet','gradient_summary.json']}
    sources.update({'effective_config.json':run/'effective_config.json','training_complete.json':run/'training_complete.json',
        'train.jsonl':run/'logs/train.jsonl','evaluation_plan.json':run/'evaluation_plan.json','dataset_summary.json':bundle/'summary.json'})
    sources['numerical_validation_amendment.json']=run/'numerical_validation_amendment.json'
    sources['checkpoint_io_transition.json']=run/'checkpoint_io_transition.json'
    sources['resume_verification.json']=run/'resume_verification.json'
    checks={}
    for step in [16,32,48,64]:
        checks[str(step)]=read_json(run/'scores_web'/f'step-{step:06d}'/'validation'/'shard-00-of-02'/'checkpoint_state_validation.json')
    checkpath=run/'metrics_web/checkpoint_state_checks.json';write_json(checkpath,checks);sources['checkpoint_state_checks.json']=checkpath
    files={}
    for name,source in sources.items():
        if source.stat().st_size>10*1024**2:raise ValueError(f'Snapshot too large: {name}')
        shutil.copy2(source,snap/name)
        files[name]={'source_path':str(source),'bytes':source.stat().st_size,'sha256':sha256(snap/name)}
        assert files[name]['sha256']==sha256(source)
    manifest={'run_id':metrics['run_id'],'directory':folder.name,'source_run_dir':str(run),'analysis_type':'web_document_retrieval',
        'result_time_utc':date.isoformat(),'timestamp_basis':'metrics_web/metrics.json completed_at_utc','files':files,
        'gradient_appendix':{'sample':'fixed random 64 documents per source','whole_document_gradients':True},
        'exporter':{'script':str(Path(__file__).relative_to(repo)),'sha256':sha256(__file__)},'execution':{'status':'pending'}}
    write_json(folder/'manifest.json',manifest)
    nb=build_notebook(manifest,metrics);path=folder/'analysis.ipynb';nbformat.write(nb,path)
    scratch=Path('/dev/shm/data-picking-runtime')
    for key,sub in [('JUPYTER_RUNTIME_DIR','jupyter'),('IPYTHONDIR','ipython')]:
        (scratch/sub).mkdir(exist_ok=True,parents=True);os.environ.setdefault(key,str(scratch/sub))
    os.environ['CUDA_VISIBLE_DEVICES']=''
    manager=KernelManager(kernel_name='python3');manager.kernel_spec.argv[0]=sys.executable
    client=NotebookClient(nb,km=manager,timeout=300,resources={'metadata':{'path':str(folder)}})
    print('EXECUTING_WEB_NOTEBOOK',path,flush=True)
    try:client.execute()
    finally:
        nbformat.write(nb,path)
        if manager.has_kernel:manager.shutdown_kernel(now=True)
    nbformat.validate(nb)
    code_cells=[c for c in nb.cells if c.cell_type=='code']
    assert all(c.execution_count is not None for c in code_cells)
    assert not any(o.output_type=='error' for c in code_cells for o in c.outputs)
    manifest['execution']={'status':'completed','code_cells':len(code_cells),'notebook_sha256':sha256(path),'gpu_required':False}
    write_json(folder/'manifest.json',manifest)
    figures=run/'figures/web';figures.mkdir(parents=True,exist_ok=True)
    count=0
    for c in code_cells:
        for output in c.outputs:
            if 'image/png' in output.get('data',{}):
                count+=1;(figures/f'figure-{count:02d}.png').write_bytes(base64.b64decode(output.data['image/png']))
    update_index(repo/'experiments-analysis')
    print('WEB_NOTEBOOK_COMPLETE',path,'figures',count,flush=True)


if __name__=='__main__':main()

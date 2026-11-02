"""Append executed, self-contained gradient diagnostics to an existing review notebook."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import sys
import textwrap

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

from data_picking.common import file_hash, read_json, write_json
from data_picking.gradient_review import prepare_gradient_review


APPENDIX_TAG = 'data_picking_gradient_appendix'


def appendix_cells(directory, summary):
    cells = []
    def md(text): cells.append(nbformat.v4.new_markdown_cell(textwrap.dedent(text).strip(), metadata={APPENDIX_TAG:True}))
    def code(text): cells.append(nbformat.v4.new_code_cell(textwrap.dedent(text).strip(), metadata={APPENDIX_TAG:True}))
    md(r"""
    ## 8. 训练前后的逐样本梯度分析

    下文是新增的探索性诊断，沿用上文同一批两域样本、reference 和 step 32 SFT 模型。
    对每条样本的**回答正文平均 token NLL**计算 `g(x) = ∇θ L(x)`；完整 float32、eval 模式、SDPA math，
    梯度不裁剪，没有 optimizer step，绑定参数只计一次。数据已经测量并保存，本节重跑仅需 CPU。

    - 全局 norm：`||g||₂`；RMS 为 `||g||₂ / √参数量`。两模型参数量相同，二者的排序信息相同。
    - 最大绝对梯度：`max |gᵢ|`，对应单个标量参数；逐参数张量统计另列。
    - 层间分布：每层 RMS 和每层所占的梯度平方范数比例。
    - 方向相似度：样本梯度与**同一模型上固定 128 条 A1 平均梯度**的 cosine。

    norm 的前后差是两个 norm 的差，不能解释成两个梯度向量之差的 norm。
    loss 下降量与相对下降比例仍按上文并列考察；本节不会用梯度诊断替换它们。
    """)
    code('GRADIENT_DIRECTORY = '+repr(directory)+'\n'+textwrap.dedent(r"""
    import hashlib
    import json
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from IPython.display import display, Markdown
    from sklearn.metrics import roc_auc_score
    %matplotlib inline
    GROOT = next((p for p in [Path.cwd(),Path.cwd()/'experiments-analysis'/GRADIENT_DIRECTORY]
                  if (p/'snapshot/gradient_metrics.json').is_file()),None)
    if GROOT is None: raise FileNotFoundError('请使用当前实验目录或仓库根目录作为工作目录。')
    GMANIFEST = json.loads((GROOT/'manifest.json').read_text())
    for name in ['gradient_metrics.json','gradient_per_example.parquet','gradient_per_layer.parquet',
                 'gradient_parameter_summary.parquet','gradient_provenance.json']:
        raw = (GROOT/'snapshot'/name).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==GMANIFEST['files'][name]['sha256'],name
    gm = json.loads((GROOT/'snapshot/gradient_metrics.json').read_text())
    gd = pd.read_parquet(GROOT/'snapshot/gradient_per_example.parquet')
    gl = pd.read_parquet(GROOT/'snapshot/gradient_per_layer.parquet')
    gp = pd.read_parquet(GROOT/'snapshot/gradient_parameter_summary.parquet')
    gmatched = gd.set_index('id').loc[gm['length_matched_ids']].reset_index()
    gdomains = gm['domains'];gnames = gm['domain_names']
    gcolors = {'A':'#1674B8',gdomains[1]:'#DF7931'}
    gshort = {d:('A2' if d=='A' else gnames[d].split(': ',1)[1]) for d in gdomains}
    assert gd.id.is_unique and gd.groupby('domain').size().to_dict()=={d:512 for d in gdomains}
    assert set(gl.id)==set(gd.id) and set(gp.domain)==set(gdomains)
    assert gm['checkpoint_step']==32
    assert max(gm['nll_difference_from_loss_notebook'].values())<=gm['nll_crosscheck_tolerance']
    gfigures = []
    print('梯度数据来源：',gm['measurement_run'])
    print('全参数数量（绑定权重只计一次）：',f"{gm['parameter_count']:,}")
    print('与上文 loss 最大数值复核差：',gm['nll_difference_from_loss_notebook'])
    print('这些差异来自原有单样本/批量 FP32 计算路径；本节未改写上文的 loss 数据。')
    """))
    md('### 8.1 全局梯度大小、最大值及前后变化')
    code(r"""
    grow = []
    for domain,group in gd.groupby('domain'):
        for prefix in ['ref','sft']:
            norm = group[f'{prefix}_grad_l2']
            grow.append({'数据':gnames[domain],'模型':prefix,'样本数':len(group),
                         'norm 均值':norm.mean(),'norm 中位数':norm.median(),
                         'norm Q25':norm.quantile(.25),'norm Q75':norm.quantile(.75),
                         'RMS 中位数':group[f'{prefix}_grad_rms'].median(),
                         'max|g| 中位数':group[f'{prefix}_grad_max_abs'].median()})
    # Plain DataFrame display works without pandas' optional Jinja2/Styler dependency.
    gsummary = pd.DataFrame(grow)
    gsummary_display = gsummary.copy()
    for column,pattern in {'norm 均值':'{:.4f}','norm 中位数':'{:.4f}',
                          'norm Q25':'{:.4f}','norm Q75':'{:.4f}',
                          'RMS 中位数':'{:.3e}','max|g| 中位数':'{:.4f}'}.items():
        gsummary_display[column] = gsummary_display[column].map(pattern.format)
    display(gsummary_display)
    # Each metric gets three overlaid two-domain distributions, in the same order.
    for key,title,xlabel in [('grad_l2','Global gradient L2','Gradient L2 norm'),
                             ('grad_max_abs','Maximum absolute scalar gradient','Maximum absolute gradient')]:
        value_bins = np.linspace(0,gd[[f'ref_{key}',f'sft_{key}']].to_numpy().max(),61)
        paired_difference = gd[f'ref_{key}']-gd[f'sft_{key}']
        difference_bins = np.histogram_bin_edges(paired_difference,bins=60)
        fig,axes = plt.subplots(1,3,figsize=(15,4.2))
        for ax,stage in zip(axes,['ref','sft','difference']):
            bins = difference_bins if stage=='difference' else value_bins
            for domain,group in gd.groupby('domain'):
                values = (group[f'ref_{key}']-group[f'sft_{key}'] if stage=='difference'
                          else group[f'{stage}_{key}'])
                median = values.median()
                assert np.histogram(values,bins=bins)[0].sum()==len(values)
                ax.hist(values,bins=bins,density=True,alpha=.5,color=gcolors[domain],
                        label=f'{gnames[domain]} (median={median:.2f})')
                ax.axvline(median,color=gcolors[domain],linestyle='--',linewidth=1)
            ax.set(xlim=(bins[0],bins[-1]),ylabel='Density')
            if stage=='difference':
                ax.axvline(0,color='black',linestyle=':',linewidth=1)
                difference_label = 'Reference L2 - SFT L2' if key=='grad_l2' else 'Reference max|g| - SFT max|g|'
                ax.set(title='Reference - SFT (paired)',xlabel=difference_label)
            else:
                ax.set(title='Reference' if stage=='ref' else 'SFT',xlabel=xlabel)
            ax.legend(fontsize=8)
        value_ymax = max(ax.get_ylim()[1] for ax in axes[:2])
        for ax in axes[:2]: ax.set_ylim(0,value_ymax)
        fig.suptitle(title,fontsize=12)
        fig.tight_layout();plt.show();plt.close(fig)
    gchanges=[];gnotes=[]
    for key,label in [('grad_l2','global gradient L2'),('grad_max_abs','maximum absolute gradient')]:
        for domain,group in gd.groupby('domain'):
            before,after = group[f'ref_{key}'],group[f'sft_{key}']
            difference = before-after
            fraction = (difference>0).mean()
            gchanges.append({'指标':label,'数据':gnames[domain],'ref 中位数':before.median(),
                             'SFT 中位数':after.median(),'配对差值中位数':difference.median(),'差值 > 0 比例':fraction})
            gnotes.append(f"{gnames[domain]} 的 {label}：ref/SFT 中位数为 {before.median():.4f}/{after.median():.4f}；"
                          f"逐样本 ref−SFT 差值的中位数为 {difference.median():.4f}，{fraction:.1%} 的样本差值为正。")
        target_group,other_group = gd.loc[gd.domain=='A'],gd.loc[gd.domain==gdomains[1]]
        ref_gap = target_group[f'ref_{key}'].median()-other_group[f'ref_{key}'].median()
        sft_gap = target_group[f'sft_{key}'].median()-other_group[f'sft_{key}'].median()
        gnotes.append(f"两域 {label} 的中位数差（A2−{gshort[gdomains[1]]}）从 ref 的 {ref_gap:.4f} 变为 SFT 的 {sft_gap:.4f}；这是中位数的位置差，分布形状见图。")
    display(pd.DataFrame(gchanges).round(4))
    display(Markdown('**观测：**\n\n'+'\n\n'.join(gnotes)))
    """)
    md("""
    每项指标依次展示 ref、SFT、逐样本 ref−SFT 的两域叠加分布，共六张分布图。
    直方图使用全部样本、线性坐标和每域归一化的密度；同一指标的 ref/SFT 共用分桶、横轴范围和纵轴尺度，
    两域在差值图中也共用分桶。彩色虚线表示各域中位数，差值图的黑色点线表示零。
    正差值表示 SFT 后该指标下降；负差值表示上升。
    本节展示原始评估池，长度匹配对照见 8.4。

    差值先按同一条样本计算，再画分布：global gradient 使用 `||g_ref||₂ − ||g_sft||₂`，
    maximum 使用 `max|g_ref| − max|g_sft|`，不是先相减梯度向量再求 norm/maximum。
    norm、最大单参数梯度和 loss 描述不同的量，它们不必同步变化。
    """)
    md('### 8.2 哪些层和参数承担了梯度？')
    code(r"""
    gnorms = pd.concat([gd[['id','ref_grad_l2']].rename(columns={'ref_grad_l2':'global_norm'}).assign(model='reference'),
                       gd[['id','sft_grad_l2']].rename(columns={'sft_grad_l2':'global_norm'}).assign(model='sft')],ignore_index=True)
    gle = gl.merge(gnorms,on=['id','model'],validate='many_to_one')
    gle['energy_share'] = gle.grad_l2**2/gle.global_norm**2
    np.testing.assert_allclose(gle.groupby(['id','model']).energy_share.sum(),1,rtol=1e-6,atol=1e-6)
    gorder = ['embedding']+sorted(k for k in gl.layer.unique() if k.startswith('layer_'))
    gorder += sorted(k for k in gl.layer.unique() if k not in gorder)
    fig,axes = plt.subplots(1,2,figsize=(13,4.2),sharey=True)
    for ax,model in zip(axes,['reference','sft']):
        for domain in gdomains:
            group = gl.loc[(gl.model==model)&(gl.domain==domain)].groupby('layer').grad_rms
            ax.plot(range(len(gorder)),group.median().reindex(gorder),label=gnames[domain],color=gcolors[domain])
            ax.fill_between(range(len(gorder)),group.quantile(.25).reindex(gorder),group.quantile(.75).reindex(gorder),color=gcolors[domain],alpha=.12)
        ax.set(yscale='log',xlabel='Layer group (embedding, blocks, final norm)',ylabel='Per-layer gradient RMS',title=model)
        ax.set_xticks(range(0,len(gorder),5));ax.legend(fontsize=8)
    fig.tight_layout();plt.show();plt.close(fig)
    gshares = gle.groupby(['model','domain','layer']).energy_share.mean().unstack('layer').reindex(columns=gorder)
    fig,ax = plt.subplots(figsize=(13,3.8))
    heat = ax.imshow(gshares.to_numpy(),aspect='auto',cmap='viridis')
    ax.set_yticks(range(len(gshares)));ax.set_yticklabels([f'{m}/{gshort[d]}' for m,d in gshares.index])
    ax.set_xticks(range(len(gorder)));ax.set_xticklabels(gorder,rotation=90,fontsize=7)
    fig.colorbar(heat,ax=ax,label='Mean share of squared gradient norm')
    fig.tight_layout();plt.show();plt.close(fig)
    top=[]
    for domain,group in gd.groupby('domain'):
        for prefix,model in [('ref','reference'),('sft','sft')]:
            for parameter,count in group[f'{prefix}_grad_max_abs_parameter'].value_counts().head(3).items():
                top.append({'domain':domain,'model':model,'parameter':parameter,'max_abs_winner_count':int(count),
                            'max_abs_winner_fraction':count/len(group)})
    gtop = pd.DataFrame(top).merge(gp,on=['domain','model','parameter'],validate='many_to_one')
    display(gtop[['domain','model','parameter','numel','max_abs_winner_count','max_abs_winner_fraction',
                  'grad_l2_median','grad_rms_median','grad_max_abs_median']].round(6))
    layer_notes=[]
    for domain in gdomains:
        ref_share,sft_share = gshares.loc[('reference',domain)],gshares.loc[('sft',domain)]
        layer_notes.append(f"{gnames[domain]}：平方 norm 占比最大的组为 ref 的 `{ref_share.idxmax()}`（{ref_share.max():.1%}）"
                           f"和 SFT 的 `{sft_share.idxmax()}`（{sft_share.max():.1%}）。")
    display(Markdown('**观测：**\n\n'+'\n\n'.join(layer_notes)))
    """)
    md("""
    曲线阴影为样本间 Q25–Q75。RMS 按层参数量归一化；热图先按每条样本的全局平方 norm 归一化，
    再跨样本取均值。二者分别反映每个参数的尺度和层的总贡献，不能互相替代。
    `embedding` 包含绑定的输入 embedding / 输出 lm-head 合并梯度，权重只计一次。

    表中列出每组最常出现全局最大单参数梯度的三个参数张量，次数和占比以该域的全部 512 条为分母。
    `grad_max_abs_median` 是这个张量逐样本最大绝对梯度的中位数。
    完整逐参数聚合表保存在 `gradient_parameter_summary.parquet`，逐样本逐参数明细仍在原始测量目录。
    """)
    md('### 8.3 梯度方向与 A1 平均梯度的相似度')
    code(r"""
    display(pd.DataFrame([{'模型':k,'A1 条数':len(v['ids']),'A1 平均梯度 norm':v['norm'],
                          'A1 平均 NLL':v['mean_nll'],'embedding/head 绑定':v['tied_input_output_embeddings']}
                         for k,v in gm['anchor_summary'].items()]).round(6))
    cosine_rows=[]
    for domain,group in gd.groupby('domain'):
        for prefix in ['ref','sft']:
            values = group[f'{prefix}_cos_to_a1']
            cosine_rows.append({'数据':gnames[domain],'模型':prefix,'cosine 均值':values.mean(),
                                'cosine 中位数':values.median(),'Q25':values.quantile(.25),'Q75':values.quantile(.75),
                                'cosine > 0 比例':(values>0).mean()})
    display(pd.DataFrame(cosine_rows).round(5))
    # Both distributions use the same mean A1 gradient at the reference model.
    ref_cosine_bins = np.histogram_bin_edges(gd['ref_cos_to_a1'],bins=40)
    fig,ax = plt.subplots(figsize=(8,4.6))
    ref_cosine_notes=[]
    for domain,group in gd.groupby('domain'):
        values = group['ref_cos_to_a1']
        assert np.histogram(values,bins=ref_cosine_bins)[0].sum()==len(values)
        ax.hist(values,bins=ref_cosine_bins,density=True,alpha=.5,color=gcolors[domain],
                label=f'{gnames[domain]} (median={values.median():.3f})')
        ax.axvline(values.median(),color=gcolors[domain],linestyle='--',linewidth=1)
        ref_cosine_notes.append(f"{gnames[domain]}：cosine 均值 {values.mean():.4f}，中位数 {values.median():.4f}，"
                                f"{(values>0).mean():.1%} 的样本与该 A1 平均梯度正向对齐。")
    ax.axvline(0,color='black',linestyle=':',linewidth=1)
    ax.set(xlim=(ref_cosine_bins[0],ref_cosine_bins[-1]),xlabel='Cosine to reference-model A1 mean gradient',
           ylabel='Density',title='Reference model: gradient cosine distributions')
    ax.legend(fontsize=9);fig.tight_layout();plt.show();plt.close(fig)
    display(Markdown('**Ref 模型上的观测：**\n\n'+'\n\n'.join(ref_cosine_notes)))
    fig,axes = plt.subplots(1,2,figsize=(11,4))
    for ax,prefix in zip(axes,['ref','sft']):
        for domain,group in gd.groupby('domain'):
            values = np.sort(group[f'{prefix}_cos_to_a1'])
            ax.plot(values,np.arange(1,len(values)+1)/len(values),label=gnames[domain],color=gcolors[domain])
        ax.axvline(0,color='black',linestyle='--',linewidth=.8)
        ax.set(title=prefix,xlabel='Cosine to same-model A1 mean gradient',ylabel='Empirical CDF')
        ax.legend(fontsize=8)
    fig.tight_layout();plt.show();plt.close(fig)
    """)
    md(r"""
    新增直方图比较 `cos(g_ref(A2 样本), mean g_ref(A1))` 和
    `cos(g_ref(对照域样本), mean g_ref(A1))`。先对固定 128 条 A1 的梯度求均值，再计算每条样本的 cosine，
    因而两组使用同一个 ref 参照向量。两域使用共同分桶，密度分别归一化；彩色虚线为中位数，黑色点线为零。

    下方 CDF 的两模型使用同一批 128 条 A1，但平均梯度分别在各自参数位置计算，所以参照向量会改变。
    这里的两个 cosine 不是 `cos(g_ref(x), g_sft(x))`，也不是相对同一个固定向量的前后夹角变化。

    在单步、足够小的普通梯度下降近似下，正向对齐意味着沿样本的负梯度更新可能降低该 A1 子集的平均 loss。
    实际 AdamW 更新还受预条件、动量和有限步长影响；而 A1 是训练数据，因此这个参照也不能替代未见目标域上的训练收益验证。
    """)
    md('### 8.4 与 loss、长度的关系及区分信息')
    code(r"""
    display(pd.DataFrame(gm['norm_correlations']).rename(columns={'nll':'rho(norm,NLL)',
        'prompt_tokens':'rho(norm,prompt length)','response_tokens':'rho(norm,response length)'}).round(4))
    display(pd.DataFrame(gm['norm_drop_loss_correlations']).rename(columns={
        'delta_loss':'rho(norm drop,loss decrease)','relative_gain':'rho(norm drop,relative decrease)'}).round(4))
    print('以上均为域内 Spearman 相关；loss 下降量与相对下降比例分别报告。')
    print('长度匹配沿用上文样本：',gm['length_matched_counts'])
    gkeys = ['ref_grad_l2','sft_grad_l2','grad_l2_delta','ref_grad_max_abs','sft_grad_max_abs',
             'grad_max_abs_delta','ref_cos_to_a1','sft_cos_to_a1']
    glabels = ['ref L2','SFT L2','L2 drop','ref max|g|','SFT max|g|','max|g| drop','ref cos(A1)','SFT cos(A1)']
    def ginterval(v): return f'[{v[0]:.4f}, {v[1]:.4f}]'
    auc_rows=[]
    for key in gkeys:
        row={'梯度特征':key}
        for pool,frame in [('raw',gd),('length_matched',gmatched)]:
            saved = gm['comparisons'][pool]['gradient_scores'][key]
            np.testing.assert_allclose(roc_auc_score(frame.domain=='A',frame[key]),saved['roc_auc'],rtol=0,atol=1e-12)
            row[f'{pool} AUC'] = round(saved['roc_auc'],4)
            row[f'{pool} 95% CI'] = ginterval(saved['roc_auc_ci95'])
        auc_rows.append(row)
    display(pd.DataFrame(auc_rows).set_index('梯度特征'))
    fig,ax = plt.subplots(figsize=(9,5))
    for shift,pool,label in [(-.12,'raw','Original'),(.12,'length_matched','Length matched')]:
        for i,key in enumerate(gkeys):
            value = gm['comparisons'][pool]['gradient_scores'][key]
            lo,hi = value['roc_auc_ci95'];point = value['roc_auc']
            ax.plot([lo,hi],[i+shift,i+shift],color=('C0' if pool=='raw' else 'C1'),linewidth=1)
            ax.scatter(point,i+shift,color=('C0' if pool=='raw' else 'C1'),label=label if i==0 else None,s=24)
    ax.axvline(.5,color='black',linestyle='--',linewidth=.8)
    ax.set(xlim=(0,1),yticks=range(len(gkeys)),yticklabels=glabels,xlabel='ROC-AUC (larger raw value predicts A2)')
    ax.invert_yaxis();ax.legend();fig.tight_layout();plt.show();plt.close(fig)
    """)
    md("""
    A2 固定为正类，所有梯度特征采用“数值越大越倾向 A2”，没有看到结果后翻转方向。
    AUC 小于 0.5 表示较大值更倾向另一组，并不等于没有区分信息。
    区间来自按域分层、按问题组抽样的 2,000 次 bootstrap；这些多指标结果属于探索性诊断。
    长度对照沿用本 notebook 上文的匹配 ID，并未重新挑选适合梯度的样本。
    """)
    md('### 8.5 本轮观察与解释边界')
    code(r"""
    observations=[]
    for domain,group in gd.groupby('domain'):
        loss_drop = group.delta_loss.mean()
        norm_drop = (1-group.sft_grad_l2/group.ref_grad_l2).median()
        observations.append(f"{gnames[domain]}：平均 loss 变化量（ref−SFT）为 {loss_drop:.4f}，"
                             f"配对 norm 相对下降的中位数为 {norm_drop:.1%}。")
        if loss_drop<0 and norm_drop>0:
            observations.append(f"{gnames[domain]} 的平均 loss 上升，但 norm 的配对中位变化仍为下降；梯度变小与 loss 改善并非等价。")
    for key,label in [('ref_grad_l2','训练前 norm'),('sft_grad_l2','训练后 norm'),
                      ('ref_cos_to_a1','训练前 A1 cosine'),('sft_cos_to_a1','训练后 A1 cosine')]:
        graw_result = gm['comparisons']['raw']['gradient_scores'][key]
        gmatched_result = gm['comparisons']['length_matched']['gradient_scores'][key]
        low,high = gmatched_result['roc_auc_ci95']
        direction = ('在该匹配子集中，较大值倾向 A2' if low>.5 else
                     f'在该匹配子集中，较大值倾向 {gnames[gdomains[1]]}' if high<.5 else
                     '该匹配子集的区间覆盖 0.5，尚无稳定的区分证据')
        observations.append(f"{label}：原始 AUC {graw_result['roc_auc']:.4f}；长度匹配 AUC {gmatched_result['roc_auc']:.4f} "
                             f"（95% CI {ginterval(gmatched_result['roc_auc_ci95'])}）；{direction}。")
        raw_low,raw_high = graw_result['roc_auc_ci95']
        if key=='ref_grad_l2' and (raw_low>.5 or raw_high<.5) and low<=.5<=high:
            observations.append('训练前 norm 在原始池中的区分信号未在长度对照中稳定保留。匹配也改变了评估人群，不能把差异全部归因于长度。')
    display(Markdown('**观测结果：**\n\n'+'\n\n'.join(observations)))
    """)
    md("""
    **解释边界：** norm 衡量局部梯度大小，cosine 描述相对指定 A1 参照的方向；单独任何一个都不能当作
    “继续训练这个样本的收益”。层间占比、回答长度、loss 尺度和来源/格式差异需要一起看。
    尤其要区分原始池与长度匹配子集的结论，匹配后的区间不包含训练 seed 的不确定性。
    要验证选数效用，仍需固定预算进行实际选数训练，并在未参与选择的目标域测试集上评估。
    """)
    return cells


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--notebook-dir',required=True,type=Path)
    parser.add_argument('--measurement-run',required=True,type=Path)
    parser.add_argument('--replace-appendix',action='store_true',help='Replace only this generated appendix; preserve every original cell')
    args=parser.parse_args()
    folder=args.notebook_dir.resolve();path=folder/'analysis.ipynb'
    original_notebook_sha=file_hash(path)
    notebook=nbformat.read(path,as_version=4)
    tagged=[i for i,c in enumerate(notebook.cells) if c.metadata.get(APPENDIX_TAG)]
    if tagged and not args.replace_appendix: raise FileExistsError('Gradient appendix already exists; preserve notes or explicitly replace it')
    if tagged and tagged != list(range(tagged[0],len(notebook.cells))):
        raise ValueError('Additional cells follow the appendix; refusing to discard them')
    prefix=notebook.cells[:tagged[0]] if tagged else notebook.cells[:]
    prefix_sources=[c.source for c in prefix]
    artifacts,summary=prepare_gradient_review(folder,args.measurement_run)
    if file_hash(path)!=original_notebook_sha:
        raise RuntimeError('Notebook changed during preparation; refusing to overwrite concurrent edits')
    manifest=read_json(folder/'manifest.json')
    original_files={name:info['sha256'] for name,info in manifest['files'].items() if not name.startswith('gradient_')}
    backup=Path(manifest['source_run_dir'])/'analysis_history'/datetime.now(timezone.utc).strftime('%Y-%m-%d_%H%M%S_%fZ_gradients')
    shutil.copytree(folder,backup)
    names=['gradient_per_example.parquet','gradient_per_layer.parquet','gradient_parameter_summary.parquet','gradient_metrics.json']
    sources={name:artifacts/name for name in names}
    sources['gradient_provenance.json']=artifacts/'COMPLETED.json'
    for name,source in sources.items():
        if source.stat().st_size>10*1024*1024: raise ValueError(f'Snapshot exceeds 10 MiB: {name}')
        target=folder/'snapshot'/name;shutil.copy2(source,target)
        manifest['files'][name]={'source_path':str(source),'bytes':target.stat().st_size,'sha256':file_hash(target)}
        assert file_hash(target)==file_hash(source)
    manifest['gradient_appendix']={'measurement_run':str(args.measurement_run.resolve()),'artifact_directory':str(artifacts),
        'domains':summary['domains'],'added_at_utc':datetime.now(timezone.utc).isoformat(),'backup_directory':str(backup),
        'first_cell':len(prefix),'script':'scripts/notebooks/append_gradients.py','script_sha256':file_hash(__file__),
        'preserved_loss_cells':len(prefix),'loss_snapshots_unchanged':True}
    manifest['execution']['status']='pending'
    write_json(folder/'manifest.json',manifest)
    notebook.cells=prefix+appendix_cells(folder.name,summary)
    nbformat.write(notebook,path)
    scratch=Path('/dev/shm/data-picking-runtime')
    for key,sub in [('JUPYTER_RUNTIME_DIR','jupyter'),('IPYTHONDIR','ipython')]:
        (scratch/sub).mkdir(parents=True,exist_ok=True);os.environ.setdefault(key,str(scratch/sub))
    os.environ['CUDA_VISIBLE_DEVICES']=''
    manager=KernelManager(kernel_name='python3');manager.kernel_spec.argv[0]=sys.executable
    client=NotebookClient(notebook,km=manager,timeout=300,resources={'metadata':{'path':str(folder)}})
    print('EXECUTING_GRADIENT_APPENDIX',folder.name,flush=True)
    try: client.execute()
    finally:
        nbformat.write(notebook,path)
        if manager.has_kernel: manager.shutdown_kernel(now=True)
    nbformat.validate(notebook)
    assert [c.source for c in notebook.cells[:len(prefix)]]==prefix_sources
    for name,sha in original_files.items(): assert file_hash(folder/'snapshot'/name)==sha
    code=[c for c in notebook.cells if c.cell_type=='code']
    assert all(c.execution_count is not None for c in code)
    assert not any(o.output_type=='error' for c in code for o in c.outputs)
    manifest['execution']={'status':'completed','code_cells':len(code),'notebook_sha256':file_hash(path),'gpu_required':False,
                           'completed_at_utc':datetime.now(timezone.utc).isoformat()}
    write_json(folder/'manifest.json',manifest)
    print('GRADIENT_APPENDIX_COMPLETE',path,'code cells',len(code),flush=True)


if __name__=='__main__':main()

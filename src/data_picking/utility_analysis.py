"""Paired target-loss comparisons of document sources, with explicit seed variation."""
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd

from .common import file_hash,read_json,write_json
from .utility_data import settings


def load_scores(experiment,label):
    folder=experiment/'scores'/label;complete=read_json(folder/'COMPLETED.json');parts=[]
    for path in sorted(folder.glob('chunk-*.complete.json')):
        r=read_json(path)
        if r['fingerprint']!=complete['fingerprint'] or file_hash(folder/r['file'])!=r['sha256']:raise RuntimeError('Invalid target scores')
        parts.append(pd.read_parquet(folder/r['file']))
    f=pd.concat(parts,ignore_index=True).sort_values('id').reset_index(drop=True)
    if not f.id.is_unique or len(f)!=complete['rows']:raise RuntimeError('Missing/duplicate target examples')
    return f


def summarize_arrays(ref, losses, groups, seeds, arms, draws, rng_seed):
    """losses[seed, arm, question]; shared seed/question draws keep contrasts paired."""
    if losses.shape!=(len(seeds),len(arms),len(ref)) or not np.isfinite(losses).all():raise ValueError('Invalid loss cube')
    general=arms.index('general')
    values={'delta_loss':ref[None,None,:]-losses}
    values['relative_gain']=values['delta_loss']/np.maximum(ref[None,None,:],1e-6)
    blocks=[np.where(np.asarray(groups)==g)[0] for g in sorted(set(groups))]
    rng=np.random.default_rng(rng_seed)
    sampled={k:{'conditional':[],'hierarchical':[]} for k in values}
    for _ in range(draws):
        ix=np.concatenate([blocks[i] for i in rng.integers(len(blocks),size=len(blocks))])
        ss=rng.integers(len(seeds),size=len(seeds))
        for k,v in values.items():
            sampled[k]['conditional'].append(v[:,:,ix].mean(axis=(0,2)))
            sampled[k]['hierarchical'].append(v[ss][:,:,ix].mean(axis=(0,2)))
    result={}
    for a,arm in enumerate(arms):
        record={'mean_target_nll':float(losses[:,a,:].mean()),'per_seed_target_nll':losses[:,a,:].mean(axis=1).tolist(),
                'scores':{},'contrast_vs_general':{}}
        for key,v in values.items():
            per_seed=v[:,a,:].mean(axis=1)
            score={'mean':float(per_seed.mean()),'per_seed':per_seed.tolist(),'seed_range':[float(per_seed.min()),float(per_seed.max())],
                   'seed_sample_sd':float(per_seed.std(ddof=1))}
            contrast=v[:,a,:].mean(axis=1)-v[:,general,:].mean(axis=1)
            compared={'mean':float(contrast.mean()),'per_seed':contrast.tolist(),'seed_range':[float(contrast.min()),float(contrast.max())]}
            for mode in ['conditional','hierarchical']:
                boot=np.asarray(sampled[key][mode]);score[f'{mode}_ci95']=np.quantile(boot[:,a],[.025,.975]).tolist()
                compared[f'{mode}_ci95']=np.quantile(boot[:,a]-boot[:,general],[.025,.975]).tolist()
            record['scores'][key]=score;record['contrast_vs_general'][key]=compared
        result[arm]=record
    return result


def analyze(path):
    cfg,experiment,_=settings(path);ref=load_scores(experiment,'reference')
    metadata=['id','group_id','subject','level','tokenization_hash','scored_tokens']
    seeds,arms=cfg['data']['seeds'],cfg['data']['arms'];cube=np.zeros((len(seeds),len(arms),len(ref)))
    all_rows=[];training=[];checks={};sanities={'reference':read_json(experiment/'scores/reference/sanity.json')}
    for si,seed in enumerate(seeds):
        for ai,arm in enumerate(arms):
            label=f'{arm}-seed{seed}';f=load_scores(experiment,label)
            pd.testing.assert_frame_equal(ref[metadata],f[metadata])
            cube[si,ai]=f.mean_nll
            child,_,run=settings(path,arm,seed);done=read_json(run/'training_complete.json')
            if done['global_step']!=cfg['training']['steps'] or done['seen_response_tokens']!=cfg['training']['total_supervised_tokens']:raise RuntimeError('Training budget mismatch')
            log=[__import__('json').loads(s) for s in (run/'logs/train.jsonl').read_text().splitlines()]
            if [r['step'] for r in log]!=list(range(1,cfg['training']['steps']+1)):raise RuntimeError('Missing/duplicate optimizer steps')
            if any(r['response_tokens']!=cfg['training']['tokens_per_step'] or not np.isfinite(r['grad_norm']) for r in log):raise RuntimeError('Invalid gradient/token audit')
            training.extend(dict(r,arm=arm,seed=seed) for r in log)
            checks[label]=read_json(experiment/'scores'/label/'checkpoint_state_validation.json')
            if checks[label]['optimizer_steps']!=[64] or checks[label]['scheduler_step']!=64:raise RuntimeError('Incomplete training state')
            sanities[label]=read_json(experiment/'scores'/label/'sanity.json')
            rows=f[metadata].copy();rows['arm']=arm;rows['seed']=seed;rows['ref_mean_nll']=ref.mean_nll;rows['trained_mean_nll']=f.mean_nll
            rows['delta_loss']=rows.ref_mean_nll-rows.trained_mean_nll;rows['relative_gain']=rows.delta_loss/np.maximum(rows.ref_mean_nll,1e-6)
            all_rows.append(rows)
    results=summarize_arrays(ref.mean_nll.to_numpy(),cube,ref.group_id.to_list(),seeds,arms,
                            cfg['analysis']['bootstrap_samples'],cfg['analysis']['bootstrap_seed'])
    baseline_run=Path(cfg['artifact_root'])/'runs/math-web-probe-qwen25-1p5b-n4096-b64-s64-seed42'
    previous=pd.read_parquet(baseline_run/'metrics_web/target_validation.parquet')
    previous=previous.loc[previous.checkpoint_step==0].sort_values('id').reset_index(drop=True)
    pd.testing.assert_frame_equal(ref[['id','tokenization_hash','scored_tokens']],previous[['id','tokenization_hash','scored_tokens']])
    error=float((ref.mean_nll-previous.mean_nll).abs().max())
    if error>cfg['scoring']['sanity_tolerance']:raise RuntimeError('Reference model no longer reproduces target losses')
    for p,sha in read_json(experiment/'original_notebook_hashes.json').items():
        if file_hash(p)!=sha:raise RuntimeError('A prior notebook changed')
    resume=read_json(Path(cfg['artifact_root'])/'runs'/f"{cfg['run_id']}-general-seed42"/'resume_verification.json')
    metrics={'analysis_type':'document_training_utility','run_id':cfg['run_id'],'reference_mean_nll':float(ref.mean_nll.mean()),
             'target_examples':len(ref),'target_groups':ref.group_id.nunique(),'arms':results,'seeds':seeds,
             'reported_loss_scores':cfg['analysis']['reported_loss_scores'],'reference_crosscheck_max_nll_difference':error,
             'training':cfg['training'],'data_preparation':read_json(experiment/'data_preparation.json'),
             'sanity':sanities,'checkpoint_state_checks':checks,'actual_resume':resume,
             'conditional_ci_scope':'Target-question-group bootstrap conditional on the three fixed trained models.',
             'hierarchical_ci_scope':'Resample three seeds and target question groups, paired across arms. Only three seeds; population uncertainty remains approximate.',
             'interpretation':cfg['interpretation'],'completed_at_utc':datetime.now(timezone.utc).isoformat()}
    out=experiment/'metrics';out.mkdir(exist_ok=True)
    pd.concat(all_rows,ignore_index=True).to_parquet(out/'per_example.parquet',index=False)
    pd.DataFrame(training).to_parquet(out/'training.parquet',index=False)
    write_json(out/'metrics.json',metrics)
    write_json(out/'COMPLETED.json',{'files':{p.name:file_hash(p) for p in out.iterdir() if p.name!='COMPLETED.json'}})
    print('UTILITY_RESULTS',__import__('json').dumps(results),flush=True)
    return metrics

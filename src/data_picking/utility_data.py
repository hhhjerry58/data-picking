"""Exact-token, source-controlled document training recipes."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import random

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from .common import digest, file_hash, read_json, read_jsonl, write_json


def settings(path, arm=None, seed=None):
    cfg=yaml.safe_load(Path(path).read_text())
    root=Path(cfg['artifact_root']);experiment=root/'runs'/cfg['run_id']
    if arm is not None:
        if arm not in cfg['data']['arms'] or seed not in cfg['data']['seeds']:raise ValueError('Unregistered arm/seed')
        cfg['run_id']+=f'-{arm}-seed{seed}'
        cfg['training']['seed']=seed
        cfg['arm']=arm
        cfg['data']['prepared_dir']=str(root/'data/processed'/experiment.name/f'{arm}-seed{seed}')
    return cfg,experiment,root/'runs'/cfg['run_id']


def token_batches(windows, seed, steps, tokens_per_step):
    """Consume every chosen target once; preserve original within-window context."""
    order=list(range(len(windows)));random.Random(seed).shuffle(order)
    if sum(w['scored_tokens'] for w in windows)<steps*tokens_per_step:raise ValueError('Insufficient unique training tokens')
    cursor=0;position=1;rows=[];batches=[]
    for step in range(steps):
        remaining=tokens_per_step;indices=[]
        while remaining:
            window=windows[order[cursor]];ids=list(window['input_ids'])
            if list(window['response_mask']) != [0]+[1]*(len(ids)-1):raise ValueError('Expected complete causal document window')
            take=min(remaining,len(ids)-position);end=position+take
            mask=[0]*position+[1]*take
            row={'id':f"{window['id']}:targets:{position}-{end}",'document_id':window['document_id'],
                 'source_window_id':window['id'],'step':step+1,'input_ids':ids[:end],'response_mask':mask,
                 'response_tokens':take,'source_token_start':window['token_start']+position,
                 'source_token_end':window['token_start']+end,'tokenization_hash':digest({'input_ids':ids[:end],'response_mask':mask})}
            indices.append(len(rows));rows.append(row);remaining-=take;position=end
            if position==len(ids):cursor+=1;position=1
        batches.append(indices)
    return rows,batches


def prepare(path):
    cfg,experiment,_=settings(path)
    plan=read_json(experiment/'experiment_plan.json')
    if plan['config']!=cfg:raise RuntimeError('Frozen utility plan changed')
    web=Path(cfg['data']['web_bundle']);quality=Path(cfg['data']['quality_bundle'])
    for bundle in [web,quality]:
        complete=read_json(bundle/'COMPLETED.json')
        for name in (['web/windows.parquet','web/labels.jsonl',cfg['data']['target_file']] if bundle==web else ['windows.parquet','document_manifest.parquet']):
            if file_hash(bundle/name)!=complete['files'][name]:raise RuntimeError('Source bundle changed')
    labels={r['id']:('openwebmath' if r['injected_math_source'] else 'general') for r in read_jsonl(web/'web/labels.jsonl')}
    pools={arm:[] for arm in cfg['data']['arms']}
    for r in pq.read_table(web/'web/windows.parquet').to_pylist():pools[labels[r['document_id']]].append(r)
    labels={r['id']:r['corpus'] for r in pq.read_table(quality/'document_manifest.parquet').to_pylist()}
    for r in pq.read_table(quality/'windows.parquet').to_pylist():pools[labels[r['document_id']]].append(r)
    records=[]
    for arm in cfg['data']['arms']:
        pool=sorted(pools[arm],key=lambda r:r['id'])
        for seed in cfg['data']['seeds']:
            child,_,run=settings(path,arm,seed);folder=Path(child['data']['prepared_dir']);folder.mkdir(parents=True,exist_ok=True)
            identity={'configuration':digest(child),'web_bundle_sha256':file_hash(web/'COMPLETED.json'),
                      'quality_bundle_sha256':file_hash(quality/'COMPLETED.json'),'implementation_sha256':file_hash(__file__)}
            if (folder/'COMPLETED.json').exists():
                saved=read_json(folder/'COMPLETED.json')
                if saved['identity']!=identity:raise RuntimeError('Existing recipe identity changed')
                for name,sha in saved['files'].items():
                    if file_hash(folder/name)!=sha:raise RuntimeError('Existing recipe changed')
                records.append(read_json(folder/'summary.json'));continue
            rows,batches=token_batches(pool,seed,cfg['training']['steps'],cfg['training']['tokens_per_step'])
            counts=Counter(r['document_id'] for r in rows)
            summary={'arm':arm,'seed':seed,'run_id':child['run_id'],'fragments':len(rows),'documents':len(counts),
                     'supervised_tokens':sum(r['response_tokens'] for r in rows),'input_tokens':sum(len(r['input_ids']) for r in rows),
                     'steps':len(batches),'tokens_per_step':cfg['training']['tokens_per_step'],'pool_windows':len(pool),
                     'pool_tokens':sum(w['scored_tokens'] for w in pool),'pool_documents':len({w['document_id'] for w in pool}),
                     'target_sha256':file_hash(web/cfg['data']['target_file'])}
            assert summary['supervised_tokens']==cfg['training']['total_supervised_tokens']
            assert all(sum(rows[i]['response_tokens'] for i in batch)==cfg['training']['tokens_per_step'] for batch in batches)
            pq.write_table(pa.Table.from_pylist(rows),folder/'train.parquet')
            write_json(folder/'batches.json',batches);write_json(folder/'summary.json',summary)
            write_json(folder/'COMPLETED.json',{'identity':identity,'files':{name:file_hash(folder/name) for name in ['train.parquet','batches.json','summary.json']}})
            records.append(summary);print('UTILITY_PREPARED',arm,seed,summary,flush=True)
    write_json(experiment/'data_preparation.json',{'runs':records,'config_sha256':file_hash(path),
        'target_path':str(web/cfg['data']['target_file']),'prepared_at_utc':datetime.now(timezone.utc).isoformat()})
    return records

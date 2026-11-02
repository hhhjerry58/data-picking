"""Validate web scoring in its actual unit: token-weighted document mean NLL.

Window-level deviations remain recorded. In particular, a one-token tail is
not treated as a complete example when deciding document-level tolerance.
The forward, CE, batching and stored scoring values are not changed here.
"""
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .common import file_hash, read_json, write_json
from . import web_scoring


def compare_document_scores(batch, repeat, single, tolerance):
    frames = [pd.DataFrame(rows).sort_values('id').reset_index(drop=True) for rows in [batch, repeat, single]]
    metadata = ['id', 'document_id', 'scored_tokens', 'tokenization_hash']
    for frame in frames[1:]: pd.testing.assert_frame_equal(frames[0][metadata], frame[metadata])
    documents = []
    for frame in frames:
        group = frame.groupby('document_id')[['sum_nll', 'scored_tokens']].sum()
        documents.append(group['sum_nll']/group['scored_tokens'])
    repeat_diff = float((documents[0]-documents[1]).abs().max())
    padding_diff = float((documents[0]-documents[2]).abs().max())
    window_repeat = float((frames[0]['mean_nll']-frames[1]['mean_nll']).abs().max())
    window_padding = float((frames[0]['mean_nll']-frames[2]['mean_nll']).abs().max())
    if not np.isfinite([repeat_diff,padding_diff,window_repeat,window_padding]).all() or max(repeat_diff,padding_diff)>tolerance:
        raise RuntimeError(f'Document NLL sanity failure: {repeat_diff}, {padding_diff}; tolerance {tolerance}')
    return {'unit':'whole_document_mean_nll','document_ids':documents[0].index.tolist(),'windows':len(frames[0]),
            'repeat_max_nll_difference':repeat_diff,'batch_padding_max_nll_difference':padding_diff,
            'window_repeat_max_nll_difference':window_repeat,'window_batch_padding_max_nll_difference':window_padding,
            'tolerance':tolerance}


def audit_documents(model, tokenizer, partition, all_windows, batch_size, tolerance):
    indexes=np.unique(np.linspace(0,len(partition)-1,min(8,len(partition)),dtype=int))
    ids={partition[int(i)]['document_id'] for i in indexes}
    rows=sorted([row for row in all_windows if row['document_id'] in ids],key=lambda r:(-len(r['input_ids']),r['id']))
    def run(size):
        return [r for i in range(0,len(rows),size) for r in web_scoring.score_batch(model,tokenizer,rows[i:i+size])]
    return compare_document_scores(run(batch_size),run(batch_size),run(1),tolerance)


def install_document_validator(config_path, model_label, checkpoint=None):
    ev,cfg,run=web_scoring.evaluation_config(config_path)
    all_windows=pq.read_table(Path(ev['bundle'])/'web/windows.parquet').to_pylist()
    if model_label=='reference':label='reference'
    else:
        checkpoint=Path(checkpoint or read_json(run/'training_complete.json')['checkpoint'])
        label=f"step-{read_json(checkpoint/'manifest.json')['global_step']:06d}"
    old=web_scoring.sanity_check
    completed_audits=False
    def callback(model,tokenizer,rows,batch_size,tolerance):
        nonlocal completed_audits
        if 'document_id' not in rows[0]:return old(model,tokenizer,rows,batch_size,tolerance)
        if not completed_audits:
            for shard in range(ev['scoring']['num_shards']):
                partition=sorted(all_windows[shard::ev['scoring']['num_shards']],key=lambda r:(-len(r['input_ids']),r['id']))
                result=audit_documents(model,tokenizer,partition,all_windows,batch_size,tolerance)
                result['validator_sha256']=file_hash(__file__)
                folder=run/'scores_web'/label/'web'/f"shard-{shard:02d}-of-{ev['scoring']['num_shards']:02d}"
                write_json(folder/'sanity_document.json',result)
                print('DOCUMENT_SCORING_SANITY',label,shard,result,flush=True)
            completed_audits=True
        own_ids={r['id'] for r in rows}
        shard=next(i for i in range(ev['scoring']['num_shards']) if all_windows[i]['id'] in own_ids)
        return read_json(run/'scores_web'/label/'web'/f"shard-{shard:02d}-of-{ev['scoring']['num_shards']:02d}"/'sanity_document.json')
    web_scoring.sanity_check=callback
    return old


def score_document_checked(config_path,model_label,shard_index=0,sets=('web','validation'),checkpoint=None):
    old=install_document_validator(config_path,model_label,checkpoint)
    try: web_scoring.score_web(config_path,model_label,shard_index,sets,checkpoint)
    finally:web_scoring.sanity_check=old

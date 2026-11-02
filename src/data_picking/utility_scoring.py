"""Identical held-out MATH answer scoring after document adaptation."""
from datetime import datetime, timezone
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer

from .common import digest,file_hash,read_json,read_jsonl,seed_all,write_json
from .training import load_model
from .utility_data import settings
from .web_scoring import checkpoint_metadata,load_staged_sft,sanity_check,score_batch


def score(path,arm,seed=42):
    cfg,experiment,run=settings(path) if arm=='reference' else settings(path,arm,seed)
    target=Path(cfg['data']['web_bundle'])/cfg['data']['target_file'];rows=read_jsonl(target)
    rows=sorted(rows,key=lambda r:(-len(r['input_ids']),r['id']))
    if len(rows)!=512:raise RuntimeError('Frozen target validation set changed')
    checkpoint=None;sha=None
    if arm!='reference':
        trained=read_json(run/'training_complete.json')
        if trained['global_step']!=cfg['training']['steps'] or trained['seen_response_tokens']!=cfg['training']['total_supervised_tokens']:
            raise RuntimeError('Incomplete training budget')
        checkpoint=Path(trained['checkpoint']);meta=checkpoint_metadata(checkpoint);sha=meta['files']['state.pt']['sha256']
    label='reference' if arm=='reference' else f'{arm}-seed{seed}'
    folder=experiment/'scores'/label;folder.mkdir(parents=True,exist_ok=True)
    identity={'target_sha256':file_hash(target),'model':cfg['model'],'checkpoint_sha256':sha,'label':label,
        'scoring':cfg['scoring'],'torch_version':torch.__version__,
        'implementation':{n:file_hash(Path(__file__).with_name(n)) for n in ['utility_scoring.py','web_scoring.py','loss.py','tokenization.py','runtime_model.py']}}
    fingerprint=digest(identity)
    if (folder/'scoring_config.json').exists() and read_json(folder/'scoring_config.json')['fingerprint']!=fingerprint:raise RuntimeError('Evaluation identity changed')
    write_json(folder/'scoring_config.json',{'identity':identity,'fingerprint':fingerprint})
    spec=cfg['scoring'];starts=set(range(0,len(rows),spec['chunk_size']));done=set()
    for p in folder.glob('chunk-*.complete.json'):
        r=read_json(p)
        if r['fingerprint']!=fingerprint or file_hash(folder/r['file'])!=r['sha256']:raise RuntimeError('Corrupt target score chunk')
        done.add(r['start'])
    if not done<=starts:raise RuntimeError('Unexpected score chunk')
    if done==starts and (folder/'sanity.json').exists():
        write_json(folder/'COMPLETED.json',{'rows':len(rows),'fingerprint':fingerprint});return
    seed_all(seed);tokenizer=AutoTokenizer.from_pretrained(cfg['runtime_model_dir'],local_files_only=True)
    model=load_model(cfg,train=False)
    if checkpoint:write_json(folder/'checkpoint_state_validation.json',load_staged_sft(model,checkpoint,run))
    sanity=sanity_check(model,tokenizer,rows,spec['batch_size'],spec['sanity_tolerance'])
    write_json(folder/'sanity.json',sanity)
    for start in range(0,len(rows),spec['chunk_size']):
        if start in done:continue
        chunk=rows[start:start+spec['chunk_size']]
        result=[r for i in range(0,len(chunk),spec['batch_size']) for r in score_batch(model,tokenizer,chunk[i:i+spec['batch_size']])]
        name=f'chunk-{start:06d}.parquet';pq.write_table(pa.Table.from_pylist(result),folder/name)
        write_json(folder/f'chunk-{start:06d}.complete.json',{'fingerprint':fingerprint,'start':start,'rows':len(result),'file':name,'sha256':file_hash(folder/name)})
        print('UTILITY_SCORED',label,start+len(chunk),len(rows),flush=True)
    write_json(folder/'COMPLETED.json',{'rows':len(rows),'fingerprint':fingerprint,'completed_at_utc':datetime.now(timezone.utc).isoformat()})

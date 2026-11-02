"""Token-budget document adaptation with complete, validated automatic resume."""
import gc
import json
import os
from pathlib import Path
import time

import pyarrow.parquet as pq
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .checkpoint import latest_checkpoint, load_training_checkpoint, save_checkpoint
from .checkpoint_transport import CheckpointReader
from .common import digest, file_hash, read_json, seed_all, write_json
from .loss import per_example_nll
from .tokenization import collate
from .training import load_model
from .utility_data import settings


def preflight(path, arm, seed):
    cfg,experiment,run=settings(path,arm,seed);t=cfg['training'];folder=Path(cfg['data']['prepared_dir'])
    if not t['save_optimizer'] or not t['auto_resume']:raise ValueError('Optimizer saving and auto-resume are required')
    if (t['parameter_dtype'],t['autocast_dtype'],t['attention_backend'])!=('float32','bfloat16','math'):raise ValueError('Invalid numerical configuration')
    if t['steps']*t['tokens_per_step']!=t['total_supervised_tokens']:raise ValueError('Token budget mismatch')
    if not os.path.ismount(Path(cfg['artifact_root']).parent):raise RuntimeError('Persistent mount missing')
    run.mkdir(parents=True,exist_ok=True)
    probe=run/f'.write-probe-{os.getpid()}'
    write_json(probe,{'ok':True});assert read_json(probe)=={'ok':True};probe.unlink()
    complete=read_json(folder/'COMPLETED.json')
    for name,sha in complete['files'].items():
        if file_hash(folder/name)!=sha:raise RuntimeError('Prepared training data changed')
    rows=pq.read_table(folder/'train.parquet').to_pylist();batches=read_json(folder/'batches.json')
    if len(batches)!=t['steps'] or sorted(i for b in batches for i in b)!=list(range(len(rows))):raise RuntimeError('Incomplete data iteration plan')
    for step,b in enumerate(batches,1):
        if sum(rows[i]['response_tokens'] for i in b)!=t['tokens_per_step']:raise RuntimeError('Unequal per-step token budget')
        if any(rows[i]['step']!=step for i in b):raise RuntimeError('Unexpected step assignment')
    files=['utility_training.py','checkpoint.py','loss.py','tokenization.py','training.py','runtime_model.py']
    identity={'config':cfg,'data_marker_sha256':file_hash(folder/'COMPLETED.json'),
        'implementation':{n:file_hash(Path(__file__).with_name(n)) for n in files}}
    fingerprint=digest(identity)
    effective={'config':cfg,'identity':identity,'training_fingerprint':fingerprint,'run_directory':str(run),
        'actual_train_size':len(rows),'total_steps':t['steps'],'warmup_steps':t['warmup_steps'],
        'supervised_tokens_per_step':t['tokens_per_step'],'total_supervised_tokens':t['total_supervised_tokens'],
        'micro_batch_size':t['micro_batch_size'],'gradient_accumulation':'variable fragments; fixed supervised-token budget',
        'save_optimizer':True,'auto_resume':True,'finite_gradient_check':True,
        'saved_state':['model','optimizer','scheduler','RNG','global_step','orders','batches','seen_response_tokens'],
        'output_dir':str(run/'checkpoints'),'data_summary':read_json(folder/'summary.json')}
    if (run/'effective_config.json').exists() and read_json(run/'effective_config.json')['training_fingerprint']!=fingerprint:
        raise RuntimeError('Existing training configuration/implementation mismatch')
    write_json(run/'training_preflight.json',effective)
    print('UTILITY_PREFLIGHT',arm,seed,'steps',t['steps'],'tokens',t['total_supervised_tokens'],'optimizer_saved',True,'auto_resume',True,flush=True)
    return cfg,run,rows,batches,effective


def verify_cursor(state, batches, nrows, tokens_per_step):
    step=state['global_step']
    if (state['orders']!=[list(range(nrows))] or state['batches']!=batches or
            state['seen_response_tokens']!=step*tokens_per_step or state['next_data_step']!=step):
        raise RuntimeError('Resume data-order/cursor/token-count mismatch')


def train(path, arm, seed, blob_config, stop_after_steps=None, check_only=False):
    cfg,run,rows,batches,effective=preflight(path,arm,seed)
    if check_only:return effective
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    t=cfg['training'];fingerprint=effective['training_fingerprint']
    reader=CheckpointReader(cfg,blob_config)
    with reader.install():
        checkpoint=latest_checkpoint(run/'checkpoints',fingerprint)
        if checkpoint and read_json(checkpoint/'manifest.json')['global_step']==t['steps']:
            if not (run/'training_complete.json').exists():
                payload=torch.load(checkpoint/'state.pt',map_location='cpu',weights_only=False)
                verify_cursor(payload['trainer'],batches,len(rows),t['tokens_per_step'])
                write_json(run/'training_complete.json',{'checkpoint':str(checkpoint),'global_step':t['steps'],
                    'seen_response_tokens':payload['trainer']['seen_response_tokens'],'fingerprint':fingerprint})
                del payload
            staged,record=reader.stage(checkpoint/'state.pt')
            write_json(run/f"runtime_weights_step-{t['steps']:06d}.json",{'model_dir':cfg['runtime_model_dir'],
                'checkpoint_file':str(staged),'checkpoint_ready':True,'ephemeral_only':True,'records':[record]})
            print('UTILITY_TRAINING_CACHED',arm,seed,flush=True);return checkpoint
        effective.update(cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),gpu=torch.cuda.get_device_name(),torch_version=torch.__version__)
        write_json(run/'effective_config.json',effective)
        seed_all(seed);tokenizer=AutoTokenizer.from_pretrained(cfg['runtime_model_dir'],local_files_only=True)
        model=load_model(cfg,train=True)
        decay,no_decay=[],[]
        for _,p in model.named_parameters():(decay if p.ndim>=2 else no_decay).append(p)
        optimizer=torch.optim.AdamW([{'params':decay,'weight_decay':t['weight_decay']},{'params':no_decay,'weight_decay':0.}],
            lr=t['learning_rate'],betas=tuple(t['betas']),eps=t['adam_epsilon'],foreach=False)
        scheduler=get_linear_schedule_with_warmup(optimizer,t['warmup_steps'],t['steps'])
        state={'global_step':0,'next_data_step':0,'orders':[list(range(len(rows)))],'batches':batches,
               'seen_response_tokens':0,'loss_history':[]}
        if checkpoint:
            state=load_training_checkpoint(checkpoint,model,optimizer,scheduler,fingerprint)
            verify_cursor(state,batches,len(rows),t['tokens_per_step'])
            steps={int(v['step']) for v in optimizer.state.values()}
            if steps!={state['global_step']} or scheduler.last_epoch!=state['global_step']:raise RuntimeError('Restored optimizer/scheduler mismatch')
            write_json(run/'resume_verification.json',{'checkpoint':str(checkpoint),'step':state['global_step'],
                'optimizer_steps':sorted(steps),'scheduler_step':scheduler.last_epoch,'rng_restored':True,
                'data_orders_and_batches_restored':True,'seen_response_tokens':state['seen_response_tokens']})
            print('UTILITY_RESUMED',arm,seed,state['global_step'],flush=True)
        logs=run/'logs';logs.mkdir(exist_ok=True)
        # Reconstruct the canonical log from checkpoint state when resuming.
        with (logs/'train.jsonl').open('w') as f:
            for entry in state['loss_history']:f.write(json.dumps(entry)+'\n')
        for step in range(state['global_step'],t['steps']):
            start=time.monotonic();optimizer.zero_grad(set_to_none=True);total_nll=0.;tokens=0;lr=scheduler.get_last_lr()[0]
            indexes=batches[step]
            for offset in range(0,len(indexes),t['micro_batch_size']):
                batch=[rows[i] for i in indexes[offset:offset+t['micro_batch_size']]]
                inputs,mask=collate(batch,tokenizer.pad_token_id)
                with sdpa_kernel(SDPBackend.MATH):
                    with torch.autocast('cuda',dtype=torch.bfloat16):output=model(**inputs,use_cache=False)
                    sums,means,counts=per_example_nll(output.logits,inputs['input_ids'],mask)
                    (sums.sum()/t['tokens_per_step']).backward()
                total_nll+=sums.detach().sum().item();tokens+=int(counts.sum())
                del output,inputs,mask,sums,means,counts
            if tokens!=t['tokens_per_step']:raise RuntimeError('Effective supervised-token count changed')
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),t['max_grad_norm'],error_if_nonfinite=True)
            optimizer.step();scheduler.step()
            state['global_step']=state['next_data_step']=step+1;state['seen_response_tokens']+=tokens
            entry={'step':step+1,'loss':total_nll/tokens,'lr':lr,'next_lr':scheduler.get_last_lr()[0],
                   'grad_norm':float(norm),'response_tokens':tokens,'fragments':len(indexes),'seconds':time.monotonic()-start}
            state['loss_history'].append(entry)
            with (logs/'train.jsonl').open('a') as f:f.write(json.dumps(entry)+'\n')
            print('UTILITY_TRAIN',arm,seed,json.dumps(entry),flush=True)
            stop=stop_after_steps is not None and step+1>=stop_after_steps
            if (step+1)%t['save_steps']==0 or step+1==t['steps'] or stop:
                checkpoint=save_checkpoint(run/'checkpoints',model,optimizer,scheduler,state,fingerprint)
                staged,record=reader.stage(checkpoint/'state.pt')
                write_json(run/f'runtime_weights_step-{step+1:06d}.json',{'model_dir':cfg['runtime_model_dir'],
                    'checkpoint_file':str(staged),'checkpoint_ready':True,'ephemeral_only':True,'records':[record]})
            if stop:break
        if state['global_step']==t['steps']:
            write_json(run/'training_complete.json',{'checkpoint':str(checkpoint),'global_step':state['global_step'],
                'seen_response_tokens':state['seen_response_tokens'],'fingerprint':fingerprint})
        del model,optimizer,scheduler
        gc.collect();torch.cuda.empty_cache()
    return checkpoint

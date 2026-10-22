"""Verified ephemeral RAM copies to avoid mmap/CUDA reads through blobfuse2.

Persistent sources remain in the configured artifact root. Credentials, when
needed for a direct read of the same blob, never enter logs or output metadata.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import urllib.error
import urllib.parse
import urllib.request

import yaml


def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/dolly_extension.yaml')
    parser.add_argument('--blob-config',required=True,type=Path)
    parser.add_argument('--workers',type=int,default=128,help='Concurrent bounded range reads')
    args=parser.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text())
    base=yaml.safe_load(Path(cfg['base_config']).read_text())
    root=Path(cfg['artifact_root'])
    run=root/'runs'/cfg['run_id']
    blob_cfg=yaml.safe_load(args.blob_config.read_text())
    cache=Path(blob_cfg['file_cache']['path'])
    mount=root.parent
    model=root/'cache/models'/base['model']['name'].replace('/','--')/base['model']['revision']
    completed=json.loads((Path(cfg['base_run'])/'training_complete.json').read_text())
    checkpoint=Path(completed['checkpoint'])
    manifest=json.loads((checkpoint/'manifest.json').read_text())
    runtime=Path('/dev/shm/data-picking-runtime/weights')/manifest['files']['state.pt']['sha256'][:16]
    runtime.mkdir(parents=True,exist_ok=True)
    records=[]

    def stage(source,target,expected,expected_size):
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and sha256(target)==expected:
            print('RAM_CACHED',target,flush=True)
        else:
            local=cache/source.relative_to(mount)
            if local.is_file() and local.stat().st_size==expected_size and local.stat().st_blocks*512>=expected_size*.98:
                print('READ_LOCAL_BLOB_CACHE',source.name,flush=True)
                shutil.copyfile(local,target)
            if not target.exists() or sha256(target)!=expected:
                print('READ_SOURCE_BLOB',source.name,flush=True)
                settings=blob_cfg['azstorage']
                endpoint=settings['endpoint'].rstrip('/')
                blob_name=str(source.relative_to(mount))
                url=(endpoint+'/'+urllib.parse.quote(settings['container'])+'/'+urllib.parse.quote(blob_name,safe='/')
                     +'?'+settings['sas'].lstrip('?'))
                # Bound requests: very large single streaming GETs can stall
                # behind the host's network path. Never log the signed URL.
                with target.open('wb') as out:
                    out.truncate(expected_size)
                fd=os.open(target,os.O_WRONLY)
                chunk=8*1024*1024
                def fetch(start):
                    end=min(expected_size,start+chunk)-1
                    for attempt in range(3):
                        try:
                            req=urllib.request.Request(url,headers={'Range':f'bytes={start}-{end}'})
                            with urllib.request.urlopen(req,timeout=30) as response:
                                if response.status!=206: raise RuntimeError('Expected range response')
                                content=response.read(end-start+2)
                            if len(content)!=end-start+1: raise RuntimeError('Wrong range length')
                            offset=0
                            while offset<len(content):
                                offset+=os.pwrite(fd,content[offset:],start+offset)
                            return len(content)
                        except Exception:
                            if attempt==2: raise RuntimeError(f'Blob range read failed at byte {start}; URL omitted') from None
                    raise AssertionError('unreachable')
                try:
                    total=0
                    with ThreadPoolExecutor(max_workers=args.workers) as pool:
                        futures=[pool.submit(fetch,start) for start in range(0,expected_size,chunk)]
                        for i,future in enumerate(as_completed(futures),1):
                            total+=future.result()
                            if i%128==0: print('RANGE_READ_PROGRESS_GIB',round(total/2**30,2),'/',round(expected_size/2**30,2),flush=True)
                finally:
                    os.close(fd)
                if sha256(target)!=expected:
                    raise RuntimeError(f'RAM copy checksum mismatch: {source.name}')
            print('RAM_VERIFIED',target,flush=True)
        records.append({'persistent_source':str(source),'runtime_copy':str(target),'sha256':expected,'bytes':target.stat().st_size})

    for name in ['config.json','generation_config.json','tokenizer_config.json','tokenizer.json','vocab.json','merges.txt','model.safetensors']:
        info=json.loads((model/(name+'.complete.json')).read_text())
        stage(model/name,runtime/'model'/name,info['sha256'],info['size'])
    partial={'model_dir':str(runtime/'model'),'checkpoint_file':str(runtime/'state.pt'),
             'ephemeral_only':True,'checkpoint_ready':False,'records':records}
    (run/'runtime_weights.json').write_text(json.dumps(partial,indent=2)+'\n')
    stage(checkpoint/'state.pt',runtime/'state.pt',manifest['files']['state.pt']['sha256'],manifest['files']['state.pt']['size'])
    result={'model_dir':str(runtime/'model'),'checkpoint_file':str(runtime/'state.pt'),
            'ephemeral_only':True,'checkpoint_ready':True,'records':records}
    (run/'runtime_weights.json').write_text(json.dumps(result,indent=2)+'\n')
    print('STAGING_COMPLETE',run/'runtime_weights.json',flush=True)


if __name__=='__main__':
    main()

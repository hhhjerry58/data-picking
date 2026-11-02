"""Direct block upload of full checkpoints to the SAME persistent mounted blobs.

Transport only: unchanged payload/optimizer/RNG state; server-validated per-block
MD5, committed block order/sizes and full local SHA-256 before completion.
Initial recovery/transport checks additionally use full remote range readback.
Credentials and signed URLs never enter output records.
"""
import base64
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import hashlib
import os
from pathlib import Path
import time
import threading
import urllib.request
import urllib.parse
import uuid
import xml.etree.ElementTree as ET

import torch
import requests

from .checkpoint import rng_state
from .checkpoint_transport import CheckpointReader
from .common import digest,file_hash,read_json,write_json


class DirectCheckpointReader(CheckpointReader):
    def persist(self,local,destination,verify_only=False,full_readback=False):
        local,destination=Path(local),Path(destination)
        if not destination.is_relative_to(self.root) or destination.name!='state.pt':raise ValueError('Checkpoint scope violation')
        size=local.stat().st_size;sha=file_hash(local);block=8*1024**2
        indexes=list(range(0,size,block));url=self._url(destination)
        ids=[base64.b64encode(f'{i:08d}'.encode()).decode() for i in range(len(indexes))]
        sessions=threading.local()
        def request(target,data=None,headers=None,method=None):
            if not hasattr(sessions,'http'):sessions.http=requests.Session()
            reason='unknown'
            for attempt in range(8):
                try:
                    with sessions.http.request(method or 'GET',target,data=data,headers=headers or {},timeout=(15,90)) as r:
                        if r.status_code>=400:
                            try:azure_code=ET.fromstring(r.content).findtext('Code') or 'unknown'
                            except Exception:azure_code='unknown'
                            reason=f'HTTP {r.status_code} / {azure_code}'
                            raise RuntimeError('Blob operation failed')
                        return r.status_code,r.content,dict(r.headers)
                except Exception as exc:
                    if not isinstance(exc,RuntimeError):reason=type(exc).__name__
                    print('CHECKPOINT_TRANSPORT_RETRY',destination.parent.name,attempt+1,reason,flush=True)
                    if attempt==7:raise RuntimeError(f'Checkpoint blob transport failed ({reason}); URL omitted') from None
                    time.sleep(min(2**attempt,15))
        expected=[]
        fd=os.open(local,os.O_RDONLY)
        def upload(i):
            data=os.pread(fd,min(block,size-indexes[i]),indexes[i])
            content_sha=hashlib.sha256(data).hexdigest()
            if verify_only:return content_sha
            md5=base64.b64encode(hashlib.md5(data).digest()).decode()
            status,_,headers=request(url+'&comp=block&blockid='+urllib.parse.quote(ids[i],safe=''),data,
                {'x-ms-version':'2021-12-02','Content-MD5':md5},'PUT')
            if status!=201:raise RuntimeError('Block upload was not accepted')
            returned=headers.get('Content-MD5',headers.get('content-md5'))
            if returned is not None and returned!=md5:raise RuntimeError('Server block MD5 differs')
            return content_sha
        try:
            with ThreadPoolExecutor(max_workers=64) as pool:
                for i,value in enumerate(pool.map(upload,range(len(indexes))),1):
                    expected.append(value)
                    if i%512==0:print('CHECKPOINT_LOCAL_BLOCK_HASHES' if verify_only else 'CHECKPOINT_BLOCKS_UPLOADED',destination.parent.name,i,len(indexes),flush=True)
        finally:os.close(fd)
        if not verify_only:
            body=('<BlockList>'+''.join('<Latest>'+i+'</Latest>' for i in ids)+'</BlockList>').encode()
            status,_,_=request(url+'&comp=blocklist',body,{'x-ms-version':'2021-12-02','x-ms-meta-sha256':sha},'PUT')
            if status!=201:raise RuntimeError('Checkpoint block-list commit failed')
        head=self._head(url)
        if head['bytes']!=size:raise RuntimeError('Committed checkpoint size mismatch')
        if not verify_only:
            _,listing,_=request(url+'&comp=blocklist&blocklisttype=committed')
            actual=[(b.findtext('Name'),int(b.findtext('Size'))) for b in ET.fromstring(listing).findall('./CommittedBlocks/Block')]
            wanted=[(ids[i],min(block,size-start)) for i,start in enumerate(indexes)]
            if actual!=wanted:raise RuntimeError('Committed checkpoint block order/size mismatch')
        def verify(i):
            start=indexes[i];end=min(size,start+block)-1
            status,data,_=request(url,headers={'Range':f'bytes={start}-{end}','If-Match':head['etag']})
            if status!=206 or len(data)!=end-start+1 or hashlib.sha256(data).hexdigest()!=expected[i]:
                raise RuntimeError('Checkpoint range SHA-256 verification failed')
        if verify_only or full_readback:
            with ThreadPoolExecutor(max_workers=32) as pool:
                for i,_ in enumerate(pool.map(verify,range(len(indexes))),1):
                    if i%512==0:print('CHECKPOINT_BLOCKS_VERIFIED',destination.parent.name,i,len(indexes),flush=True)
        if self._head(url)!=head:raise RuntimeError('Checkpoint changed during readback')
        if file_hash(local)!=sha:raise RuntimeError('Local checkpoint changed during persistence')
        record={'source':str(destination),'runtime_copy':str(local),'sha256':sha,'bytes':size,'etag':head['etag'],
            'adapter_sha256':file_hash(__file__),
            'verified_at_utc':datetime.now(timezone.utc).isoformat(),
            'full_remote_readback':bool(verify_only or full_readback),
            'transport':'Server-validated Content-MD5 per uploaded block; committed block order and sizes; stable ETag; local full-file SHA-256. Full If-Match range readback when full_remote_readback is true.'}
        write_json(local.with_suffix('.verified.json'),record)
        self.written_copies[destination]=local
        write_json(self.run/'checkpoint_read_records'/f'{destination.parent.name}.json',record)
        return record


def commit_recovered(reader,local,fingerprint,step):
    directory=reader.root/f'step-{step:06d}-recovered-{uuid.uuid4().hex[:8]}'
    directory.mkdir(parents=True)
    target=reader.scratch/directory.name/'state.pt';target.parent.mkdir(parents=True,exist_ok=True)
    # A RAM hardlink preserves the exact byte stream without duplicating 18 GB.
    os.link(local,target)
    record=reader.persist(target,directory/'state.pt')
    manifest={'global_step':step,'fingerprint':fingerprint,'save_optimizer':True,'auto_resume':True,
              'files':{'state.pt':{'size':record['bytes'],'sha256':record['sha256']}}}
    write_json(directory/'manifest.json',manifest)
    write_json(directory/'COMPLETED.json',{'manifest_hash':digest(manifest)})
    return directory


@contextmanager
def install_training_io():
    from . import utility_training as training
    old_reader,old_save,old_latest=training.CheckpointReader,training.save_checkpoint,training.latest_checkpoint
    active={}
    class Reader(DirectCheckpointReader):
        @contextmanager
        def install(self):
            active['reader']=self
            with super().install():yield
    def save(root,model,optimizer,scheduler,state,fingerprint):
        reader=active['reader'];directory=Path(root)/f"step-{state['global_step']:06d}-{uuid.uuid4().hex[:8]}"
        directory.mkdir(parents=True)
        local=reader.scratch/directory.name/'state.pt';local.parent.mkdir(parents=True,exist_ok=True)
        payload={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                 'rng':rng_state(),'trainer':state,'fingerprint':fingerprint}
        with local.open('wb') as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
        print('CHECKPOINT_RAM_CLOSED',directory,flush=True)
        record=reader.persist(local,directory/'state.pt')
        manifest={'global_step':state['global_step'],'fingerprint':fingerprint,'save_optimizer':True,'auto_resume':True,
                  'files':{'state.pt':{'size':record['bytes'],'sha256':record['sha256']}}}
        write_json(directory/'manifest.json',manifest)
        write_json(directory/'COMPLETED.json',{'manifest_hash':digest(manifest)})
        print('CHECKPOINT_COMPLETE',directory,flush=True)
        return directory
    def latest(root,fingerprint):
        root=Path(root);record=root.parent/'restart_record.json'
        if record.exists() and not any((p/'COMPLETED.json').exists() for p in root.glob('step-*')):
            if read_json(record)['fingerprint']!=fingerprint:raise RuntimeError('Restart identity mismatch')
            print('DOCUMENTED_RESTART_FROM_REFERENCE',root.parent.name,flush=True);return None
        return old_latest(root,fingerprint)
    training.CheckpointReader=Reader;training.save_checkpoint=save;training.latest_checkpoint=latest
    try:yield
    finally:training.CheckpointReader=old_reader;training.save_checkpoint=old_save;training.latest_checkpoint=old_latest

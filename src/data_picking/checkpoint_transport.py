"""Checkpoint I/O acceleration for this run, with ETag/SHA checks.

Persistent writes keep their destination and bytes, and are mirrored into RAM.
Only binary reads of state.pt beneath the selected checkpoint root are routed
to a verified copy. All numerical training code stays unchanged.
"""
import builtins
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import urllib.parse
import urllib.request

import yaml

from .common import file_hash, paths, read_json, write_json


class CheckpointReader:
    def __init__(self, cfg, blob_config, run_suffix="", workers=64):
        self.cfg = cfg
        self.run = paths(cfg)["run"].with_name(paths(cfg)["run"].name+run_suffix)
        self.root = self.run/"checkpoints"
        self.mount = Path(cfg["artifact_root"]).parent
        settings = yaml.safe_load(Path(blob_config).read_text())
        self.storage = settings["azstorage"]
        self.file_cache = Path(settings["file_cache"]["path"])
        self.scratch = Path('/dev/shm/data-picking-runtime/direct-checkpoints')/self.run.name
        self.workers = workers
        self.written_copies = {}

    def _url(self, path):
        s = self.storage
        return (s['endpoint'].rstrip('/')+'/'+urllib.parse.quote(s['container'])+'/'+
                urllib.parse.quote(str(path.relative_to(self.mount)),safe='/')+'?'+s['sas'].lstrip('?'))

    def _head(self, url):
        try:
            with urllib.request.urlopen(urllib.request.Request(url,method='HEAD'),timeout=45) as response:
                return {'etag':response.headers['ETag'],'bytes':int(response.headers['Content-Length'])}
        except Exception:
            raise RuntimeError('Checkpoint blob HEAD failed; URL omitted') from None

    def stage(self, source):
        source = Path(os.path.abspath(source))
        if not source.is_relative_to(self.root) or source.name != 'state.pt':
            raise ValueError('Direct reader is restricted to this run checkpoint state.pt files')
        url = self._url(source)
        head = self._head(url)
        target = self.scratch/source.parent.name/'state.pt'
        marker = target.with_suffix('.verified.json')
        if marker.is_file() and target.is_file():
            saved = read_json(marker)
            if (saved['source']==str(source) and all(saved[k]==head[k] for k in head)
                    and target.stat().st_size==head['bytes'] and file_hash(target)==saved['sha256']):
                return target, saved
        target.parent.mkdir(parents=True,exist_ok=True)
        mirrored = self.written_copies.get(source)
        if mirrored and mirrored.is_file() and mirrored.stat().st_size==head['bytes']:
            checksum=file_hash(mirrored)
            if self._head(url)!=head:raise RuntimeError('Checkpoint blob changed after its writer closed')
            record={'source':str(source),'runtime_copy':str(mirrored),**head,'sha256':checksum,
                    'verified_at_utc':datetime.now(timezone.utc).isoformat(),
                    'transport':'exact byte stream mirrored while writing persistent checkpoint; successful close and blob ETag/size checked'}
            write_json(marker,record)
            write_json(self.run/'checkpoint_read_records'/f'{source.parent.name}.json',record)
            return mirrored,record
        local = self.file_cache/source.relative_to(self.mount)
        manifest_path = source.parent/'manifest.json'
        closed = manifest_path.is_file() or any(
            f'CHECKPOINT_CLOSED_VERIFYING {source.parent}' in p.read_text() for p in (self.run/'logs').glob('train-launch*.log'))
        if (closed and local.is_file() and local.stat().st_size==head['bytes']
                and local.stat().st_blocks*512>=head['bytes']):
            # A fresh unique checkpoint has been successfully closed/uploaded.
            # Reuse the complete local file cache, just as a normal cached read
            # would, without asking blobfuse to discard and re-download it.
            shutil.copyfile(local,target)
            if self._head(url)!=head:raise RuntimeError('Checkpoint blob changed while copying its complete cache')
            checksum=file_hash(target)
            if manifest_path.is_file() and checksum!=read_json(manifest_path)['files']['state.pt']['sha256']:
                raise RuntimeError('Complete checkpoint cache checksum mismatch')
            record={'source':str(source),'runtime_copy':str(target),**head,'sha256':checksum,
                    'verified_at_utc':datetime.now(timezone.utc).isoformat(),
                    'transport':'complete blobfuse local cache after writer close; blob ETag/size checked before and after copy'}
            write_json(marker,record)
            write_json(self.run/'checkpoint_read_records'/f'{source.parent.name}.json',record)
            print('CHECKPOINT_CACHE_VERIFIED',source.parent.name,flush=True)
            return target,record
        with target.open('wb') as out:out.truncate(head['bytes'])
        fd = os.open(target,os.O_WRONLY)
        block = 8*1024**2
        def fetch(start):
            end = min(start+block,head['bytes'])-1
            for attempt in range(3):
                try:
                    request=urllib.request.Request(url,headers={'Range':f'bytes={start}-{end}','If-Match':head['etag']})
                    with urllib.request.urlopen(request,timeout=45) as response:
                        if response.status!=206:raise RuntimeError('Range response required')
                        content=response.read(end-start+2)
                    if len(content)!=end-start+1:raise RuntimeError('Wrong range length')
                    offset=0
                    while offset<len(content):offset+=os.pwrite(fd,content[offset:],start+offset)
                    return len(content)
                except Exception:
                    if attempt==2:raise RuntimeError(f'Checkpoint range read failed at {start}; URL omitted') from None
        try:
            done=0
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                futures=[pool.submit(fetch,start) for start in range(0,head['bytes'],block)]
                for i,future in enumerate(as_completed(futures),1):
                    done+=future.result()
                    if i%256==0:print('DIRECT_CHECKPOINT_READ_GIB',round(done/2**30,2),'/',round(head['bytes']/2**30,2),flush=True)
        finally:
            os.close(fd)
        if self._head(url)!=head:raise RuntimeError('Checkpoint blob changed while reading')
        record={'source':str(source),'runtime_copy':str(target),**head,'sha256':file_hash(target),
                'verified_at_utc':datetime.now(timezone.utc).isoformat(),'transport':'parallel same-blob range GET with If-Match'}
        write_json(marker,record)
        write_json(self.run/'checkpoint_read_records'/f'{source.parent.name}.json',record)
        return target,record

    @contextmanager
    def install(self):
        original = builtins.open
        original_io = io.open
        def checkpoint_open(file,mode='r',*args,**kwargs):
            if mode=='rb' and isinstance(file,(str,os.PathLike)):
                path=Path(os.path.abspath(file))
                if path.name=='state.pt' and path.is_relative_to(self.root):
                    file,_=self.stage(path)
            return original(file,mode,*args,**kwargs)
        def checkpoint_io_open(file,mode='r',*args,**kwargs):
            if mode=='wb' and isinstance(file,(str,os.PathLike)):
                path=Path(os.path.abspath(file))
                if path.name=='state.pt' and path.is_relative_to(self.root):
                    target=self.scratch/path.parent.name/'state.pt'
                    target.parent.mkdir(parents=True,exist_ok=True)
                    target.with_suffix('.verified.json').unlink(missing_ok=True)
                    persistent=original_io(file,mode,*args,**kwargs)
                    mirror=original_io(target,'wb')
                    return MirroredWriter(persistent,mirror,lambda:self.written_copies.__setitem__(path,target))
            return original_io(file,mode,*args,**kwargs)
        builtins.open=checkpoint_open
        io.open=checkpoint_io_open
        try:yield
        finally:
            builtins.open=original
            io.open=original_io


class MirroredWriter:
    """Preserve the persistent file API and mirror each successfully written byte."""
    def __init__(self,persistent,mirror,closed_callback):
        self.persistent,self.mirror,self.closed_callback=persistent,mirror,closed_callback
    def write(self,data):
        count=self.persistent.write(data)
        written=self.mirror.write(memoryview(data)[:count])
        if written!=count:raise OSError('Incomplete checkpoint RAM mirror write')
        return count
    def flush(self):
        self.persistent.flush();self.mirror.flush()
    def seek(self,*args):
        position=self.persistent.seek(*args)
        if self.mirror.seek(*args)!=position:raise OSError('Checkpoint mirror seek mismatch')
        return position
    def truncate(self,*args):
        size=self.persistent.truncate(*args)
        if self.mirror.truncate(*args)!=size:raise OSError('Checkpoint mirror truncate mismatch')
        return size
    def close(self):
        if self.persistent.closed:return
        self.flush()
        try:self.persistent.close()
        except BaseException:
            self.mirror.close();raise
        self.mirror.close()
        self.closed_callback()
    def __getattr__(self,name):return getattr(self.persistent,name)
    def __enter__(self):return self
    def __exit__(self,*args):self.close()

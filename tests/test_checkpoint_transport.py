import builtins
import hashlib
from pathlib import Path
import pytest

from data_picking.checkpoint_transport import CheckpointReader


def test_read_transport_is_scoped_and_leaves_writes_and_other_paths_unchanged(tmp_path):
    reader=CheckpointReader.__new__(CheckpointReader)
    reader.root=tmp_path/'checkpoints'
    source=reader.root/'step-000048-test/state.pt'
    source.parent.mkdir(parents=True)
    source.write_bytes(b'persistent state')
    cached=tmp_path/'ram/state.pt';cached.parent.mkdir();cached.write_bytes(b'verified copy')
    other=tmp_path/'other.pt';other.write_bytes(b'other data')
    calls=[]
    reader.stage=lambda p:(calls.append(p) or cached,{})
    original=builtins.open
    with reader.install():
        with open(source,'rb') as f:assert f.read()==b'verified copy'
        with open(other,'rb') as f:assert f.read()==b'other data'
        with open(source,'wb') as f:f.write(b'new persistent state')
    assert builtins.open is original
    assert source.read_bytes()==b'new persistent state'
    assert calls==[source]


def test_complete_cache_must_match_published_checkpoint_sha(tmp_path):
    from data_picking.common import write_json
    reader=CheckpointReader.__new__(CheckpointReader)
    reader.mount=tmp_path/'mount';reader.run=reader.mount/'data-picking/runs/test'
    reader.root=reader.run/'checkpoints';reader.scratch=tmp_path/'ram';reader.file_cache=tmp_path/'cache'
    source=reader.root/'step-000048-test/state.pt';source.parent.mkdir(parents=True)
    body=b'checkpoint bytes';expected=hashlib.sha256(body).hexdigest()
    source.write_bytes(body)
    write_json(source.parent/'manifest.json',{'files':{'state.pt':{'sha256':expected}}})
    local=reader.file_cache/source.relative_to(reader.mount);local.parent.mkdir(parents=True);local.write_bytes(b'X'*len(body))
    reader._url=lambda p:'unused'
    reader._head=lambda url:{'etag':'fixed-version','bytes':len(body)}
    reader.written_copies={}
    with pytest.raises(RuntimeError,match='checksum mismatch'):reader.stage(source)
    local.write_bytes(body)
    target,record=reader.stage(source)
    assert target.read_bytes()==body and record['sha256']==expected


def test_torch_checkpoint_writes_are_byte_identical_in_ram_mirror(tmp_path):
    import torch
    reader=CheckpointReader.__new__(CheckpointReader)
    reader.root=tmp_path/'checkpoints';reader.scratch=tmp_path/'ram';reader.written_copies={}
    source=reader.root/'step-000064-test/state.pt';source.parent.mkdir(parents=True)
    payload={'model':torch.arange(17),'optimizer':{'step':64},'rng':torch.get_rng_state()}
    with reader.install():
        with source.open('wb') as f:
            torch.save(payload,f);f.flush()
    mirror=reader.written_copies[source]
    assert source.read_bytes()==mirror.read_bytes()
    restored=torch.load(mirror,weights_only=False)
    torch.testing.assert_close(restored['model'],payload['model'])
    assert restored['optimizer']==payload['optimizer']

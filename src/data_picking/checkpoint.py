import os
import random
import uuid
from pathlib import Path

import numpy as np
import torch

from .common import digest, file_hash, read_json, write_json


def is_blobfuse_path(path):
    path = Path(path).absolute()
    for line in Path("/proc/mounts").read_text().splitlines():
        source, target, *_ = line.split()
        if source == "blobfuse2" and path.is_relative_to(target):
            return True
    return False


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def validate_checkpoint(directory):
    directory = Path(directory)
    marker = read_json(directory / "COMPLETED.json")
    manifest = read_json(directory / "manifest.json")
    if digest(manifest) != marker["manifest_hash"]:
        raise RuntimeError(f"Invalid checkpoint manifest: {directory}")
    for name, info in manifest["files"].items():
        path = directory / name
        if not path.is_file() or path.stat().st_size != info["size"] or file_hash(path) != info["sha256"]:
            raise RuntimeError(f"Corrupt checkpoint file: {path}")
    return manifest


def latest_checkpoint(root, fingerprint):
    root = Path(root)
    if not root.exists():
        return None
    entries = []
    for path in root.glob("step-*"):
        if not (path / "COMPLETED.json").exists():
            print("Ignoring incomplete checkpoint", path, flush=True)
            continue
        meta = read_json(path / "manifest.json")
        if meta["fingerprint"] != fingerprint:
            raise RuntimeError(f"Checkpoint configuration mismatch: {path}")
        entries.append((meta["global_step"], path))
    if not entries:
        if list(root.glob("step-*")):
            raise RuntimeError("Training artifacts exist but there is no complete resumable checkpoint")
        return None
    path = max(entries, key=lambda x: (x[0], str(x[1])))[1]
    validate_checkpoint(path)
    return path


def save_checkpoint(root, model, optimizer, scheduler, state, fingerprint):
    root = Path(root)
    directory = root / f'step-{state["global_step"]:06d}-{uuid.uuid4().hex[:8]}'
    directory.mkdir(parents=True)
    payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
               "scheduler": scheduler.state_dict(), "rng": rng_state(), "trainer": state, "fingerprint": fingerprint}
    filename = directory / "state.pt"
    with filename.open("wb") as f:
        torch.save(payload, f)
        f.flush()
        # blobfuse2 uploads dirty files on close. Its default fsync additionally
        # invalidates/deletes the cache, forcing an 18 GB download just to hash
        # the file we have written. Close, then verify every byte and size before
        # committing the manifest. Retain normal fsync on local filesystems.
        if not is_blobfuse_path(filename):
            os.fsync(f.fileno())
    print("CHECKPOINT_CLOSED_VERIFYING", directory, flush=True)
    manifest = {"global_step": state["global_step"], "fingerprint": fingerprint,
                "save_optimizer": True, "auto_resume": True,
                "files": {"state.pt": {"size": filename.stat().st_size, "sha256": file_hash(filename)}}}
    write_json(directory / "manifest.json", manifest)
    write_json(directory / "COMPLETED.json", {"manifest_hash": digest(manifest)})
    print("CHECKPOINT_COMPLETE", directory, flush=True)
    return directory


def load_training_checkpoint(directory, model, optimizer, scheduler, fingerprint):
    # Only load locally-created files already validated by latest_checkpoint.
    payload = torch.load(Path(directory) / "state.pt", map_location="cpu", weights_only=False)
    if payload["fingerprint"] != fingerprint:
        raise RuntimeError("Checkpoint fingerprint mismatch")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    restore_rng(payload["rng"])
    return payload["trainer"]

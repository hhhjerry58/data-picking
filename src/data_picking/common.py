import hashlib
import json
import os
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
import yaml


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def read_json(path):
    return json.loads(Path(path).read_text())


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(canonical(row) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_config(path):
    cfg = yaml.safe_load(Path(path).read_text())
    root = Path(cfg["artifact_root"])
    if not root.is_absolute() or not root.is_dir():
        raise ValueError(f"Artifact root unavailable: {root}")
    if not os.path.ismount(root.parent):
        raise RuntimeError(f"Expected a mounted persistent filesystem at {root.parent}")
    if not cfg["training"]["save_optimizer"] or not cfg["training"]["auto_resume"]:
        raise ValueError("This experiment requires optimizer saving and auto_resume")
    if cfg["training"]["effective_batch_size"] % cfg["training"]["micro_batch_size"]:
        raise ValueError("Effective batch must be divisible by micro batch size")
    probe = root / f".write-probe-{os.getpid()}"
    write_json(probe, {"ok": True})
    assert read_json(probe) == {"ok": True}
    probe.unlink()
    return cfg


def paths(cfg):
    root = Path(cfg["artifact_root"])
    data_key = digest({"data": cfg["data"], "model": cfg["model"]})[:16]
    return {
        "root": root,
        "model": root / "cache/models" / cfg["model"]["name"].replace("/", "--") / cfg["model"]["revision"],
        "data": root / "data/processed" / data_key,
        "run": root / "runs" / cfg["run_id"],
    }


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def code_state():
    root = Path(__file__).resolve().parents[2]
    files = sorted([*root.glob("src/**/*.py"), *root.glob("scripts/*.py"), *root.glob("configs/*.yaml"), root / "requirements.txt"])
    return {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()),
        "file_hashes": {str(p.relative_to(root)): file_hash(p) for p in files},
    }

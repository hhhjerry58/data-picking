"""Verify optional RAM copies against the pinned persistent model assets."""
from pathlib import Path

from .common import file_hash, paths, read_json


def verified_model_path(cfg):
    source = paths(cfg)["model"]
    runtime = cfg.get("runtime_model_dir")
    if not runtime:
        return source
    runtime = Path(runtime)
    for name in ["config.json", "generation_config.json", "tokenizer_config.json", "tokenizer.json",
                 "vocab.json", "merges.txt", "model.safetensors"]:
        expected = read_json(source / (name+".complete.json"))
        path = runtime / name
        if not path.is_file() or path.stat().st_size != expected["size"] or file_hash(path) != expected["sha256"]:
            raise RuntimeError(f"RAM model asset does not match the pinned source: {name}")
    return runtime

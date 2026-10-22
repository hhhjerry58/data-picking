"""Download pinned model and two train parquet files without rename/symlink assumptions."""
import hashlib
import json
from pathlib import Path
import urllib.request

import yaml


def get_json(url):
    with urllib.request.urlopen(url, timeout=60) as f:
        return json.load(f)


def download(url, destination, expected_sha=None, expected_size=None):
    destination.parent.mkdir(parents=True, exist_ok=True)
    marker = destination.with_name(destination.name + ".complete.json")
    if marker.exists():
        meta = json.loads(marker.read_text())
        if destination.exists() and destination.stat().st_size == meta["size"] and meta["url"] == url:
            print("CACHED", destination, flush=True)
            return
        raise RuntimeError(f"Invalid cached asset: {destination}")
    print("DOWNLOADING", destination, flush=True)
    h, size = hashlib.sha256(), 0
    with urllib.request.urlopen(url, timeout=120) as source, destination.open("wb") as target:
        while block := source.read(8 * 1024 * 1024):
            target.write(block)
            h.update(block)
            size += len(block)
    if expected_sha and h.hexdigest() != expected_sha:
        raise RuntimeError(f"SHA256 mismatch: {destination}")
    if expected_size is not None and size != expected_size:
        raise RuntimeError(f"Size mismatch: {destination}")
    marker.write_text(json.dumps({"url": url, "sha256": h.hexdigest(), "size": size}) + "\n")
    print("COMPLETE", destination, size, flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    root = Path(cfg["artifact_root"])
    spec = cfg["model"]
    model_root = root / "cache/models" / spec["name"].replace("/", "--") / spec["revision"]
    info = get_json(f'https://huggingface.co/api/models/{spec["name"]}/revision/{spec["revision"]}?blobs=true')
    allowed = {"config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json"}
    for item in info["siblings"]:
        name = item["rfilename"]
        if name in allowed or name.endswith(".safetensors"):
            download(f'https://huggingface.co/{spec["name"]}/resolve/{spec["revision"]}/{name}', model_root / name, item.get("lfs", {}).get("sha256"), item.get("size"))
    spec = cfg["data"]
    for subset in [spec["a_subset"], spec["b_subset"]]:
        filename = f"{subset}/train-00000-of-00001.parquet"
        download(f'https://huggingface.co/datasets/{spec["name"]}/resolve/{spec["revision"]}/{filename}', root / "data/raw" / spec["revision"] / filename)

"""Resumable FP32 scoring of frozen web windows and held-out MATH answers."""
from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoTokenizer
import yaml

from .common import digest, file_hash, load_config, paths, read_json, read_jsonl, seed_all, write_json
from .loss import per_example_nll
from .tokenization import collate
from .training import load_model


def evaluation_config(path):
    ev = yaml.safe_load(Path(path).read_text())
    cfg = load_config(ev["training_config"])
    return ev, cfg, paths(cfg)["run"]


def checkpoint_metadata(checkpoint):
    checkpoint = Path(checkpoint)
    meta = read_json(checkpoint/"manifest.json")
    if read_json(checkpoint/"COMPLETED.json")["manifest_hash"] != digest(meta):
        raise RuntimeError("Checkpoint is incomplete or its manifest changed")
    return meta


def load_staged_sft(model, checkpoint, run):
    meta = checkpoint_metadata(checkpoint)
    step = meta["global_step"]
    staging = read_json(run/f"runtime_weights_step-{step:06d}.json")
    path = Path(staging["checkpoint_file"])
    if not staging["checkpoint_ready"] or file_hash(path) != meta["files"]["state.pt"]["sha256"]:
        raise RuntimeError("RAM checkpoint does not match the saved checkpoint")
    # Fully materialize CPU tensors; do not copy mmap-backed blobfuse tensors to CUDA.
    payload = torch.load(path, map_location="cpu", mmap=False, weights_only=False)
    required = {"model", "optimizer", "scheduler", "rng", "trainer", "fingerprint"}
    if not required <= payload.keys() or payload["fingerprint"] != meta["fingerprint"]:
        raise RuntimeError("Missing/mismatched full training state")
    effective = read_json(run/"effective_config.json")
    optimizer_steps = {int(v["step"]) for v in payload["optimizer"]["state"].values()}
    if (payload["fingerprint"] != effective["training_fingerprint"] or optimizer_steps != {step}
            or payload["scheduler"]["last_epoch"] != step or payload["trainer"]["global_step"] != step):
        raise RuntimeError("Optimizer/scheduler/trainer state does not match checkpoint step")
    orders = payload["trainer"]["orders"]
    if any(sorted(order) != list(range(effective["actual_train_size"])) for order in orders):
        raise RuntimeError("Saved data iteration order is incomplete")
    validation = {"step": step, "optimizer_steps": sorted(optimizer_steps), "scheduler_step": payload["scheduler"]["last_epoch"],
                  "saved_rng_keys": sorted(payload["rng"]), "saved_orders": len(orders),
                  "seen_response_tokens": payload["trainer"]["seen_response_tokens"], "state_sha256": file_hash(path)}
    weights = payload["model"]
    del payload
    model.load_state_dict(weights)
    del weights
    return validation


@torch.inference_mode()
def score_batch(model, tokenizer, rows):
    inputs, mask = collate(rows, tokenizer.pad_token_id)
    with sdpa_kernel(SDPBackend.MATH), torch.autocast("cuda", enabled=False):
        output = model(**inputs, use_cache=False)
        sums, means, counts = per_example_nll(output.logits, inputs["input_ids"], mask)
    result = []
    for row, total, mean, count in zip(rows, sums.tolist(), means.tolist(), counts.tolist()):
        expected = row.get("scored_tokens", row.get("response_tokens"))
        if count != expected or not np.isfinite(total) or not np.isfinite(mean):
            raise RuntimeError(f"Invalid scoring mask/NLL for {row['id']}")
        keys = ["id", "document_id", "window_index", "token_start", "token_end", "tokenization_hash", "group_id", "subject", "level"]
        result.append({**{k: row[k] for k in keys if k in row}, "sum_nll": total, "mean_nll": mean, "scored_tokens": count})
    return result


def sanity_check(model, tokenizer, rows, batch_size, tolerance):
    indexes = np.unique(np.linspace(0, len(rows)-1, min(8, len(rows)), dtype=int))
    examples = [rows[int(i)] for i in indexes]
    def run(size):
        return [r for i in range(0, len(examples), size) for r in score_batch(model, tokenizer, examples[i:i+size])]
    first, repeat, singles = run(batch_size), run(batch_size), run(1)
    repeated = max(abs(a["mean_nll"]-b["mean_nll"]) for a, b in zip(first, repeat))
    padding = max(abs(a["mean_nll"]-b["mean_nll"]) for a, b in zip(first, singles))
    if max(repeated, padding) > tolerance:
        raise RuntimeError(f"FP32 repeat/padding sanity failure: {repeated}, {padding}; tolerance {tolerance}")
    return {"ids": [r["id"] for r in examples], "repeat_max_nll_difference": repeated,
            "batch_padding_max_nll_difference": padding, "tolerance": tolerance}


def score_web(config_path, model_label, shard_index=0, sets=("web", "validation"), checkpoint=None):
    ev, cfg, run = evaluation_config(config_path)
    spec = ev["scoring"]
    if spec["forward_dtype"] != "float32" or spec["attention_backend"] != "math":
        raise ValueError("Web scoring requires FP32 and SDPA math")
    n_shards = spec["num_shards"]
    if not 0 <= shard_index < n_shards: raise ValueError("Invalid scoring shard")
    if model_label not in {"reference", "sft"}: raise ValueError(model_label)
    meta = None
    if model_label == "sft":
        checkpoint = Path(checkpoint or read_json(run/"training_complete.json")["checkpoint"])
        meta = checkpoint_metadata(checkpoint)
    label = "reference" if meta is None else f"step-{meta['global_step']:06d}"
    bundle = Path(ev["bundle"])
    complete = read_json(bundle/"COMPLETED.json")
    sets = tuple(sets)
    if set(sets)-{"web", "validation"}: raise ValueError("Unknown scoring set")
    requested = {"web": "web/windows.parquet", "validation": "target/validation.jsonl"}
    pending = []
    for name in sets:
        source = bundle/requested[name]
        if file_hash(source) != complete["files"][requested[name]]: raise RuntimeError("Prepared scoring data changed")
        all_rows = pq.read_table(source).to_pylist() if name == "web" else read_jsonl(source)
        rows = sorted(all_rows[shard_index::n_shards], key=lambda r: (-len(r["input_ids"]), r["id"]))
        identity = {"evaluation": ev, "model": cfg["model"], "set": name, "shard_index": shard_index,
                    "data_sha256": file_hash(source), "label": label,
                    "checkpoint_sha256": meta["files"]["state.pt"]["sha256"] if meta else None,
                    "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
                    "implementation": {f: file_hash(Path(__file__).with_name(f)) for f in
                                       ["web_scoring.py", "training.py", "runtime_model.py", "loss.py", "tokenization.py"]}}
        fingerprint = digest(identity)
        folder = run/"scores_web"/label/name/f"shard-{shard_index:02d}-of-{n_shards:02d}"
        folder.mkdir(parents=True, exist_ok=True)
        config = folder/"scoring_config.json"
        if config.exists() and read_json(config)["fingerprint"] != fingerprint: raise RuntimeError("Scoring identity mismatch")
        write_json(config, {"identity": identity, "fingerprint": fingerprint})
        done = set()
        for marker in folder.glob("chunk-*.complete.json"):
            info = read_json(marker)
            if info["fingerprint"] != fingerprint or file_hash(folder/info["file"]) != info["sha256"]:
                raise RuntimeError("Corrupt or mismatched score chunk")
            done.add(info["start"])
        starts = set(range(0, len(rows), spec["chunk_size"]))
        if not done <= starts: raise RuntimeError("Unexpected chunk start")
        if done == starts:
            write_json(folder/"COMPLETED.json", {"fingerprint": fingerprint, "rows": len(rows), "label": label})
            print("SCORES_CACHED", label, name, shard_index, flush=True)
        else:
            pending.append((name, rows, folder, fingerprint, done))
    if not pending: return
    seed_all(cfg["training"]["seed"])
    tokenizer = AutoTokenizer.from_pretrained(cfg.get("runtime_model_dir", paths(cfg)["model"]), local_files_only=True)
    print("LOADING_SCORE_MODEL", label, shard_index, flush=True)
    model = load_model(cfg, train=False)
    state_validation = load_staged_sft(model, checkpoint, run) if meta else None
    for name, rows, folder, fingerprint, done in pending:
        if state_validation: write_json(folder/"checkpoint_state_validation.json", state_validation)
        sanity = sanity_check(model, tokenizer, rows, spec["batch_size"], spec["sanity_tolerance"])
        write_json(folder/"sanity.json", sanity)
        print("SCORING_SANITY", label, name, shard_index, sanity, flush=True)
        for start in range(0, len(rows), spec["chunk_size"]):
            if start in done: continue
            begin = time.monotonic()
            chunk = rows[start:start+spec["chunk_size"]]
            results = [r for i in range(0, len(chunk), spec["batch_size"])
                       for r in score_batch(model, tokenizer, chunk[i:i+spec["batch_size"]])]
            filename = f"chunk-{start:06d}.parquet"
            pq.write_table(pa.Table.from_pylist(results), folder/filename)
            write_json(folder/f"chunk-{start:06d}.complete.json", {"fingerprint": fingerprint, "start": start,
                       "rows": len(results), "file": filename, "sha256": file_hash(folder/filename)})
            print("WEB_SCORE_CHUNK", label, name, shard_index, start+len(chunk), "/", len(rows),
                  "seconds", round(time.monotonic()-begin, 2), flush=True)
        write_json(folder/"COMPLETED.json", {"fingerprint": fingerprint, "rows": len(rows), "label": label,
                   "completed_at_utc": datetime.now(timezone.utc).isoformat()})

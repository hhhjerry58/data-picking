"""Whole-document gradients on a fixed, source-stratified diagnostic sample."""
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import random

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoTokenizer
import yaml

from .common import digest, file_hash, load_config, paths, read_json, read_jsonl, seed_all, write_json
from .gradient_patterns import summarize_gradients
from .loss import per_example_nll
from .tokenization import collate
from .training import load_model
from .web_scoring import load_staged_sft


def freeze_gradient_plan(config_path):
    spec = yaml.safe_load(Path(config_path).read_text())
    cfg = load_config(spec["training_config"])
    run = paths(cfg)["run"]
    manifest = pq.read_table(Path(spec["bundle"])/"web/document_manifest.parquet").to_pylist()
    rng = random.Random(spec["sample_seed"])
    selected = []
    for source in ["math", "background"]:
        rows = sorted([r for r in manifest if r["source"] == source], key=lambda r: r["id"])
        selected.extend(rng.sample(rows, spec["examples_per_source"]))
    targets = read_jsonl(Path(spec["bundle"])/"target/train.jsonl")
    anchors = random.Random(spec["anchor_seed"]).sample(targets, spec["anchor_size"])
    plan = {"config": spec, "ids": [r["id"] for r in selected], "anchor_ids": [r["id"] for r in anchors],
            "bundle_sha256": file_hash(Path(spec["bundle"])/"COMPLETED.json")}
    path = run/"gradient_diagnostic_plan.json"
    if path.exists():
        saved = read_json(path)
        if any(saved[k] != v for k, v in plan.items()): raise RuntimeError("Gradient diagnostic plan changed")
    else:
        write_json(path, {**plan, "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
                         "scope": "Fixed random document subset; all its windows; no selection by scores or lengths"})
    return spec, cfg, run, selected, anchors


def backward_document(model, tokenizer, rows):
    model.zero_grad(set_to_none=True)
    total_tokens = sum(sum(row["response_mask"][1:]) for row in rows)
    total_nll = 0.
    for row in rows:
        inputs, mask = collate([row], tokenizer.pad_token_id)
        with sdpa_kernel(SDPBackend.MATH), torch.autocast("cuda", enabled=False):
            output = model(**inputs, use_cache=False)
            sums, means, counts = per_example_nll(output.logits, inputs["input_ids"], mask)
            if not torch.isfinite(sums).all() or counts.item() != sum(row["response_mask"][1:]):
                raise RuntimeError("Invalid document gradient loss")
            (sums[0]/total_tokens).backward()
        total_nll += sums[0].detach().item()
        del output, sums, means, counts, inputs, mask
    return total_nll/total_tokens


def measure_web_gradients(config_path, model_label):
    spec, cfg, run, docs, anchors = freeze_gradient_plan(config_path)
    if spec["forward_dtype"] != "float32" or spec["attention_backend"] != "math" or spec["clip"]:
        raise ValueError("Diagnostics require unclipped FP32 SDPA-math gradients")
    if model_label not in {"reference", "sft"}: raise ValueError(model_label)
    step = spec["checkpoint_step"] if model_label == "sft" else 0
    checkpoint = Path(read_json(run/"training_complete.json")["checkpoint"]) if step else None
    if step and read_json(checkpoint/"manifest.json")["global_step"] != step: raise RuntimeError("Wrong gradient checkpoint")
    identity = {"plan_sha256": file_hash(run/"gradient_diagnostic_plan.json"), "model": model_label,
                "checkpoint_sha256": read_json(checkpoint/"manifest.json")["files"]["state.pt"]["sha256"] if step else None,
                "implementation": {name: file_hash(Path(__file__).with_name(name)) for name in
                    ["web_gradients.py", "gradient_patterns.py", "web_scoring.py", "training.py", "runtime_model.py", "loss.py", "tokenization.py"]}}
    fingerprint = digest(identity)
    folder = run/"gradients_web"/model_label;folder.mkdir(parents=True, exist_ok=True)
    config = folder/"measurement_config.json"
    if config.exists() and read_json(config)["fingerprint"] != fingerprint: raise RuntimeError("Gradient identity mismatch")
    write_json(config, {"identity": identity, "fingerprint": fingerprint})
    done = set()
    for path in folder.glob("chunk-*.complete.json"):
        record = read_json(path)
        if record["fingerprint"] != fingerprint: raise RuntimeError("Gradient chunk identity mismatch")
        for name, sha in record["files"].items():
            if file_hash(folder/name) != sha: raise RuntimeError("Gradient chunk checksum mismatch")
        done.add(record["start"])
    starts = set(range(0, len(docs), spec["chunk_size"]))
    if not done <= starts: raise RuntimeError("Unexpected gradient chunk start")
    if done == starts:
        write_json(folder/"COMPLETED.json", {"fingerprint": fingerprint, "documents": len(docs), "optimizer_updates": 0,
                                             "completed_at_utc": datetime.now(timezone.utc).isoformat()})
        return
    selected = {r["id"] for r in docs}
    by_doc = defaultdict(list)
    for row in pq.read_table(Path(spec["bundle"])/"web/windows.parquet").to_pylist():
        if row["document_id"] in selected: by_doc[row["document_id"]].append(row)
    for rows in by_doc.values(): rows.sort(key=lambda r: r["window_index"])
    seed_all(cfg["training"]["seed"])
    tokenizer = AutoTokenizer.from_pretrained(cfg.get("runtime_model_dir", paths(cfg)["model"]), local_files_only=True)
    model = load_model(cfg, train=False)
    if checkpoint: load_staged_sft(model, checkpoint, run)
    named = list(model.named_parameters())
    anchor = {name: torch.zeros_like(p) for name, p in named}
    anchor_losses = []
    for i, row in enumerate(anchors):
        anchor_losses.append(backward_document(model, tokenizer, [row]))
        with torch.no_grad():
            for name, parameter in named:
                if parameter.grad is not None: anchor[name].add_(parameter.grad, alpha=1/len(anchors))
        if (i+1) % 32 == 0: print("WEB_GRAD_ANCHOR", model_label, i+1, flush=True)
    norm = torch.stack([torch.linalg.vector_norm(a).double().square() for a in anchor.values()]).sum().sqrt().item()
    write_json(folder/"anchor.json", {"ids": [r["id"] for r in anchors], "norm": norm, "mean_nll": float(np.mean(anchor_losses)),
                                    "parameter_count": sum(p.numel() for _, p in named), "fingerprint": fingerprint})
    for start in range(0, len(docs), spec["chunk_size"]):
        if start in done: continue
        global_rows, layers, parameters = [], [], []
        for doc in docs[start:start+spec["chunk_size"]]:
            mean_nll = backward_document(model, tokenizer, by_doc[doc["id"]])
            stats, per_layer, per_parameter = summarize_gradients(named, anchor, norm)
            metadata = {k: doc[k] for k in ["id", "group_id", "source", "token_count", "scored_tokens"]}
            global_rows.append({**metadata, "model": model_label, "mean_nll": mean_nll, **stats})
            for target, values in [(layers, per_layer), (parameters, per_parameter)]:
                target.extend({"id": doc["id"], "source": doc["source"], "model": model_label, **row} for row in values)
        files = {}
        for kind, values in [("global", global_rows), ("layers", layers), ("parameters", parameters)]:
            name = f"chunk-{start:06d}.{kind}.parquet"
            pq.write_table(pa.Table.from_pylist(values), folder/name);files[name] = file_hash(folder/name)
        write_json(folder/f"chunk-{start:06d}.complete.json", {"start": start, "fingerprint": fingerprint, "files": files})
        print("WEB_GRADIENT_CHUNK", model_label, start+len(global_rows), "/", len(docs), flush=True)
    write_json(folder/"COMPLETED.json", {"fingerprint": fingerprint, "documents": len(docs), "optimizer_updates": 0,
                                         "completed_at_utc": datetime.now(timezone.utc).isoformat()})

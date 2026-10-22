"""Per-example, unclipped FP32 gradient measurements without parameter updates."""
from collections import defaultdict
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import re
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .checkpoint import validate_checkpoint
from .common import digest, file_hash, paths, read_json, read_jsonl, seed_all, write_json
from .generalization_data import load_extension
from .loss import per_example_nll
from .tokenization import collate
from .training import load_model


def layer_name(name):
    match = re.match(r"model\.layers\.(\d+)\.", name)
    if match:
        return f"layer_{int(match.group(1)):02d}"
    if "embed_tokens" in name:
        return "embedding"
    if name.startswith("lm_head"):
        return "lm_head"
    return "final_norm" if name.startswith("model.norm") else "other"


@torch.no_grad()
def summarize_gradients(named, anchor, anchor_norm):
    """Deduplicate tied tensors; reduce each tensor in FP32, combine in FP64."""
    unique = []
    seen = set()
    for name, parameter in named:
        if id(parameter) not in seen:
            unique.append((name, parameter))
            seen.add(id(parameter))
    components = []
    for name, parameter in unique:
        g = parameter.grad
        if g is None:
            components.append(torch.zeros(3, dtype=torch.float64, device=parameter.device))
        else:
            norm = torch.linalg.vector_norm(g)
            minimum, maximum = torch.aminmax(g)
            dot = torch.dot(g.reshape(-1), anchor[name].reshape(-1))
            components.append(torch.stack([norm.double().square(), torch.maximum(-minimum, maximum).double(), dot.double()]))
    values = torch.stack(components).cpu().numpy()
    if not np.isfinite(values).all():
        raise RuntimeError("Non-finite per-parameter gradient statistic")
    details = []
    layers = defaultdict(lambda: {"squared_norm": 0.0, "max_abs": 0.0, "numel": 0})
    for (name, parameter), (square, maximum, dot) in zip(unique, values):
        size = parameter.numel()
        norm = math.sqrt(float(square))
        group = layer_name(name)
        details.append({"parameter": name, "layer": group, "numel": size,
                        "grad_l2": norm, "grad_rms": norm/math.sqrt(size), "grad_max_abs": float(maximum)})
        layers[group]["squared_norm"] += float(square)
        layers[group]["max_abs"] = max(layers[group]["max_abs"], float(maximum))
        layers[group]["numel"] += size
    total = sum(p.numel() for _, p in unique)
    norm = math.sqrt(float(values[:, 0].sum()))
    if norm <= 0 or anchor_norm <= 1e-12:
        raise RuntimeError("Zero gradient/anchor norm: cosine is undefined; do not silently assign a score")
    cosine = float(values[:, 2].sum() / (norm * anchor_norm))
    if not -1.00001 <= cosine <= 1.00001:
        raise RuntimeError(f"Invalid gradient cosine: {cosine}")
    imax, inorm = int(values[:, 1].argmax()), int(values[:, 0].argmax())
    global_stats = {"grad_l2": norm, "grad_rms": norm/math.sqrt(total), "grad_max_abs": float(values[imax, 1]),
                    "grad_max_abs_parameter": unique[imax][0], "max_parameter_l2": math.sqrt(float(values[inorm, 0])),
                    "max_parameter_l2_name": unique[inorm][0], "cos_to_a1": cosine,
                    "parameter_count": total, "missing_grad_tensors": sum(p.grad is None for _, p in unique)}
    by_layer = [{"layer": name, "numel": v["numel"], "grad_l2": math.sqrt(v["squared_norm"]),
                 "grad_rms": math.sqrt(v["squared_norm"]/v["numel"]), "grad_max_abs": v["max_abs"]}
                for name,v in sorted(layers.items())]
    return global_stats, by_layer, details


def backward_example(model, tokenizer, row):
    model.zero_grad(set_to_none=True)
    inputs, mask = collate([row], tokenizer.pad_token_id)
    with sdpa_kernel(SDPBackend.MATH), torch.autocast("cuda", enabled=False):
        output = model(**inputs, use_cache=False)
        sums, means, counts = per_example_nll(output.logits, inputs["input_ids"], mask)
        if counts.item() != row["response_tokens"] or not torch.isfinite(means).all():
            raise RuntimeError(f"Invalid NLL/mask for {row['id']}")
        means[0].backward()
    return {"mean_nll": means[0].item(), "sum_nll": sums[0].item()}


def measure(config_path, model_label):
    cfg, base, run = load_extension(config_path)
    if model_label not in {"reference", "sft"}:
        raise ValueError(model_label)
    spec = cfg["gradients"]
    if spec["forward_dtype"] != "float32" or spec["clip"] or spec["attention_backend"] != "math":
        raise ValueError("Measurements require unclipped FP32 gradients with SDPA math")
    data = run / "data"
    data_marker = read_json(data / "COMPLETED.json")
    for name, sha in data_marker["files"].items():
        if file_hash(data / name) != sha:
            raise RuntimeError(f"Prepared data changed: {name}")
    rows = read_jsonl(data / "evaluation.jsonl")
    anchor_rows = read_jsonl(data / "anchor_A1.jsonl")
    base_run = Path(cfg["base_run"])
    checkpoint = Path(read_json(base_run / "training_complete.json")["checkpoint"])
    checkpoint_meta = read_json(checkpoint / "manifest.json")
    out = run / cfg.get("gradient_directory", "gradients") / model_label
    out.mkdir(parents=True, exist_ok=True)
    identity = {"config": cfg, "data_manifest": file_hash(data / "COMPLETED.json"),
                "model_label": model_label, "model_revision": base["model"]["revision"],
                "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
                "checkpoint_sha256": checkpoint_meta["files"]["state.pt"]["sha256"] if model_label == "sft" else None,
                "implementation": {name: file_hash(Path(__file__).with_name(name)) for name in ["gradient_patterns.py", "loss.py", "tokenization.py"]}}
    fingerprint = digest(identity)
    config_file = out / "measurement_config.json"
    if config_file.exists() and read_json(config_file)["fingerprint"] != fingerprint:
        raise RuntimeError("Existing gradient measurement has a different identity")
    write_json(config_file, {"identity": identity, "fingerprint": fingerprint, "no_optimizer_updates": True,
                            "created_at_utc": datetime.now(timezone.utc).isoformat()})
    chunk_size = spec["chunk_size"]
    completed = set()
    for marker in sorted(out.glob("chunk-*.complete.json")):
        item = read_json(marker)
        if item["fingerprint"] != fingerprint:
            raise RuntimeError("Chunk configuration mismatch")
        for name, sha in item["files"].items():
            if file_hash(out / name) != sha:
                raise RuntimeError(f"Corrupt gradient chunk: {name}")
        completed.add(item["start"])
    if len(completed) == math.ceil(len(rows)/chunk_size):
        write_json(out / "COMPLETED.json", {"fingerprint": fingerprint, "examples": len(rows), "model": model_label,
                   "completed_at_utc": datetime.now(timezone.utc).isoformat(), "optimizer_updates": 0})
        print("ALL_GRADIENT_CHUNKS_COMPLETE", model_label, flush=True)
        return
    seed_all(base["training"]["seed"])
    staging = read_json(run / "runtime_weights.json") if (run / "runtime_weights.json").exists() else None
    runtime_model = Path(staging["model_dir"]) if staging else paths(base)["model"]
    runtime_checkpoint = Path(staging["checkpoint_file"]) if staging else checkpoint / "state.pt"
    if staging and model_label == "sft":
        if not staging.get("checkpoint_ready", True):
            raise RuntimeError("Staged SFT checkpoint is not ready")
        expected = checkpoint_meta["files"]["state.pt"]["sha256"]
        source_record = next(r for r in staging["records"] if r["runtime_copy"] == str(runtime_checkpoint))
        if source_record["sha256"] != expected or file_hash(runtime_checkpoint) != expected:
            raise RuntimeError("Staged checkpoint does not match the persistent checkpoint")
    if staging:
        for record in staging["records"]:
            if Path(record["runtime_copy"]).parent == runtime_model and file_hash(record["runtime_copy"]) != record["sha256"]:
                raise RuntimeError("Staged model asset checksum mismatch")
    tokenizer = AutoTokenizer.from_pretrained(runtime_model, local_files_only=True)
    print("LOADING_MODEL", model_label, flush=True)
    write_json(out / "device.json", {"visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
               "name": torch.cuda.get_device_name(), "capability": torch.cuda.get_device_capability()})
    if model_label == "sft" and not staging:
        validate_checkpoint(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(runtime_model,dtype=torch.float32,attn_implementation="sdpa",local_files_only=True)
    model.config.use_cache=False
    model.to("cuda").eval()
    if model_label == "sft":
        # Materialize CPU tensors before CUDA copies. A memory-mapped tensor on
        # blobfuse2 can stall in uninterruptible I/O when copied to CUDA.
        payload = torch.load(runtime_checkpoint, map_location="cpu", mmap=False, weights_only=False)
        model.load_state_dict(payload["model"])
        del payload
    named = list(model.named_parameters())
    if not all(p.requires_grad and p.dtype == torch.float32 for _,p in named):
        raise RuntimeError("All parameters must require FP32 gradients")
    anchor = {name: torch.zeros_like(p) for name,p in named}
    anchor_losses = []
    for i, row in enumerate(anchor_rows):
        score = backward_example(model, tokenizer, row)
        anchor_losses.append(score["mean_nll"])
        with torch.no_grad():
            for name,p in named:
                if p.grad is not None:
                    anchor[name].add_(p.grad, alpha=1/len(anchor_rows))
        if (i+1) % 16 == 0:
            print("ANCHOR", model_label, i+1, "/", len(anchor_rows), flush=True)
    anchor_norm = torch.stack([torch.linalg.vector_norm(a).double().square() for a in anchor.values()]).sum().sqrt().item()
    if not math.isfinite(anchor_norm) or anchor_norm <= 1e-12:
        raise RuntimeError("Non-finite or near-zero A1 mean gradient")
    write_json(out / "anchor.json", {"ids": [r["id"] for r in anchor_rows], "norm": anchor_norm,
               "mean_nll": float(np.mean(anchor_losses)), "parameter_count": sum(p.numel() for _,p in named),
               "tied_input_output_embeddings": model.get_input_embeddings().weight is model.get_output_embeddings().weight,
               "fingerprint": fingerprint})
    previous = {r["id"]: r for r in pq.read_table(base_run / "metrics/per_example.parquet").to_pylist()}
    prior_field = "ref_mean_nll" if model_label == "reference" else "sft_mean_nll"
    for start in range(0, len(rows), chunk_size):
        if start in completed:
            continue
        begin = time.monotonic()
        globals_, layers_, parameters_ = [], [], []
        max_old_difference = 0.0
        for row in rows[start:start+chunk_size]:
            score = backward_example(model, tokenizer, row)
            global_stats, layer_stats, parameter_stats = summarize_gradients(named, anchor, anchor_norm)
            if row["id"] in previous:
                old = previous[row["id"]]
                if old["tokenization_hash"] != row["tokenization_hash"]:
                    raise RuntimeError("A/B tokenization changed")
                difference = abs(score["mean_nll"] - old[prior_field])
                max_old_difference = max(max_old_difference, difference)
                if difference > spec["loss_crosscheck_tolerance"]:
                    raise RuntimeError(f"A/B NLL crosscheck failed: {row['id']} difference={difference}")
            metadata = {k: row[k] for k in ["id", "group_id", "domain", "split", "level", "text_hash", "tokenization_hash", "prompt_tokens", "response_tokens", "source", "source_revision"]}
            globals_.append({**metadata, **score, **global_stats, "model": model_label})
            for target, values in [(layers_, layer_stats), (parameters_, parameter_stats)]:
                target.extend({"id": row["id"], "domain": row["domain"], "model": model_label, **v} for v in values)
        files = {}
        for kind, values in [("global", globals_), ("layers", layers_), ("parameters", parameters_)]:
            filename = f"chunk-{start:06d}.{kind}.parquet"
            pq.write_table(pa.Table.from_pylist(values), out / filename)
            files[filename] = file_hash(out / filename)
        record = {"start": start, "count": len(globals_), "fingerprint": fingerprint, "files": files,
                  "max_old_loss_difference": max_old_difference, "seconds": time.monotonic()-begin}
        write_json(out / f"chunk-{start:06d}.complete.json", record)
        print("GRADIENT_CHUNK", model_label, start+len(globals_), "/", len(rows), "seconds", round(record["seconds"], 2), flush=True)
    write_json(out / "COMPLETED.json", {"fingerprint": fingerprint, "examples": len(rows), "model": model_label,
               "completed_at_utc": datetime.now(timezone.utc).isoformat(), "optimizer_updates": 0})


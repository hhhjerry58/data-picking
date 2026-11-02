import gc
import importlib.metadata
import json
import math
import os
import random
import time

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from .checkpoint import latest_checkpoint, load_training_checkpoint, save_checkpoint
from .common import code_state, digest, file_hash, paths, read_json, read_jsonl, seed_all, write_json
from .loss import per_example_nll
from .runtime_model import verified_model_path
from .tokenization import collate


def load_model(cfg, train=False):
    model = AutoModelForCausalLM.from_pretrained(verified_model_path(cfg), dtype=torch.float32,
                                                attn_implementation="sdpa", local_files_only=True)
    model.config.use_cache = False
    model.to("cuda")
    if train:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()
    else:
        model.eval()
    return model


def train(cfg, stop_after_steps=None, subset_size=None, run_suffix=""):
    p = paths(cfg)
    run = p["run"].with_name(p["run"].name + run_suffix)
    run.mkdir(parents=True, exist_ok=True)
    t = cfg["training"]
    if t["parameter_dtype"] != "float32" or t["autocast_dtype"] != "bfloat16":
        raise ValueError("Training requires float32 master weights with bf16 autocast")
    if t["attention_backend"] != "math":
        raise ValueError("The validated local training backend is SDPA math")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    rows = read_jsonl(p["data"] / "A1.jsonl")
    if subset_size is not None:
        rows = rows[:subset_size]
    assert all(r["split"] == "A1" for r in rows)
    epochs = t["epochs"]
    batch_size, micro = t["effective_batch_size"], t["micro_batch_size"]
    steps_per_epoch = math.ceil(len(rows) / batch_size)
    total_steps = epochs * steps_per_epoch
    warmup = math.ceil(total_steps * t["warmup_ratio"])
    current_code = code_state()
    # Scoring/analysis edits cannot change training semantics. These hashes do.
    training_files = ["src/data_picking/training.py", "src/data_picking/checkpoint.py", "src/data_picking/loss.py", "src/data_picking/tokenization.py"]
    if cfg.get("runtime_model_dir"):
        training_files.append("src/data_picking/runtime_model.py")
    identity = {"model": cfg["model"], "training": t, "data_manifest_hash": file_hash(p["data"] / "manifest.jsonl"),
                "train_data_hash": file_hash(p["data"] / "A1.jsonl"),
                "train_ids": [r["id"] for r in rows], "total_steps": total_steps,
                "implementation": {name: current_code["file_hashes"][name] for name in training_files}}
    fingerprint = digest(identity)
    effective = {"config": cfg, "run_directory": str(run), "training_fingerprint": fingerprint,
                 "identity": identity, "actual_train_size": len(rows), "total_steps": total_steps,
                 "warmup_steps": warmup, "gradient_accumulation_steps": batch_size // micro,
                 "save_optimizer": True, "auto_resume": True, "output_dir": str(run / "checkpoints"),
                 "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "gpu": torch.cuda.get_device_name(),
                 "code": current_code, "data_summary": read_json(p["data"] / "summary.json"),
                 "versions": {name: importlib.metadata.version(name) for name in
                              {d.metadata["Name"] for d in importlib.metadata.distributions()}}}
    existing = run / "effective_config.json"
    if existing.exists() and read_json(existing)["training_fingerprint"] != fingerprint:
        raise RuntimeError("Existing run configuration mismatch; use a different run ID")
    write_json(existing, effective)
    print("EFFECTIVE_CONFIG", json.dumps({k: effective[k] for k in ["actual_train_size", "total_steps", "save_optimizer", "auto_resume", "output_dir", "cuda_visible_devices"]}), flush=True)
    checkpoint = latest_checkpoint(run / "checkpoints", fingerprint)
    if checkpoint and read_json(checkpoint / "manifest.json")["global_step"] >= total_steps:
        if not (run / "training_complete.json").exists():
            payload = torch.load(checkpoint / "state.pt", map_location="cpu", weights_only=False)
            write_json(run / "training_complete.json", {"checkpoint": str(checkpoint), "global_step": total_steps,
                       "seen_response_tokens": payload["trainer"]["seen_response_tokens"], "fingerprint": fingerprint})
        print("Training already complete", checkpoint, flush=True)
        return checkpoint
    seed_all(t["seed"])
    tokenizer = AutoTokenizer.from_pretrained(p["model"], local_files_only=True)
    model = load_model(cfg, train=True)
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": t["weight_decay"]},
                                  {"params": no_decay, "weight_decay": 0.0}],
                                 lr=t["learning_rate"], betas=tuple(t["betas"]), eps=t["adam_epsilon"], foreach=False)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, total_steps)
    order_rng = random.Random(t["seed"])
    orders = []
    for _ in range(epochs):
        order = list(range(len(rows)))
        order_rng.shuffle(order)
        orders.append(order)
    state = {"global_step": 0, "orders": orders, "seen_response_tokens": 0, "loss_history": []}
    if checkpoint:
        state = load_training_checkpoint(checkpoint, model, optimizer, scheduler, fingerprint)
        optimizer_steps = sorted({int(value["step"]) for value in optimizer.state.values()})
        if optimizer_steps != [state["global_step"]] or scheduler.last_epoch != state["global_step"]:
            raise RuntimeError("Optimizer/scheduler step differs from restored trainer state")
        write_json(run / "resume_verification.json", {"checkpoint": str(checkpoint), "global_step": state["global_step"],
                   "optimizer_steps": optimizer_steps, "scheduler_last_epoch": scheduler.last_epoch,
                   "next_lr": scheduler.get_last_lr(), "saved_orders_restored": True, "rng_restored": True})
        print("RESUMED", checkpoint, "step", state["global_step"], "lr", scheduler.get_last_lr(), flush=True)
    log_path = run / "logs/train.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    for step in range(state["global_step"], total_steps):
        epoch, position = divmod(step, steps_per_epoch)
        indexes = state["orders"][epoch][position*batch_size:(position+1)*batch_size]
        start = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        loss_sum, tokens = 0.0, 0
        used_lr = scheduler.get_last_lr()[0]
        for offset in range(0, len(indexes), micro):
            batch = [rows[i] for i in indexes[offset:offset+micro]]
            inputs, mask = collate(batch, tokenizer.pad_token_id)
            # Keep the backend context through backward/checkpoint recomputation.
            # Automatic backend selection produced non-finite bf16 gradients on
            # a real padded batch in the local PyTorch/CUDA environment.
            with sdpa_kernel(SDPBackend.MATH):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = model(**inputs, use_cache=False)
                _, means, counts = per_example_nll(output.logits, inputs["input_ids"], mask)
                (means.sum() / len(indexes)).backward()
            loss_sum += means.detach().sum().item()
            tokens += int(counts.sum())
            del output, means, inputs, mask
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), t["max_grad_norm"], error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        state["global_step"] = step + 1
        state["seen_response_tokens"] += tokens
        entry = {"step": step+1, "loss": loss_sum/len(indexes), "lr": used_lr, "next_lr": scheduler.get_last_lr()[0],
                 "grad_norm": float(norm), "response_tokens": tokens, "seconds": time.monotonic()-start}
        state["loss_history"].append(entry)
        with log_path.open("a") as f:
            f.write(json.dumps(entry) + "\n")
        print("TRAIN", json.dumps(entry), flush=True)
        stop = stop_after_steps is not None and step+1 >= stop_after_steps
        if (step+1) % t["save_steps"] == 0 or step+1 == total_steps or stop:
            checkpoint = save_checkpoint(run / "checkpoints", model, optimizer, scheduler, state, fingerprint)
        if stop:
            break
    if state["global_step"] == total_steps:
        write_json(run / "training_complete.json", {"checkpoint": str(checkpoint), "global_step": total_steps,
                   "seen_response_tokens": state["seen_response_tokens"], "fingerprint": fingerprint})
    del model, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()
    return checkpoint

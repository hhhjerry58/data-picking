"""Validate the actual SFT data and effective schedule without loading weights."""
import math

from .common import file_hash, paths, read_json, read_jsonl, write_json


def training_plan(cfg, subset_size=None, run_suffix=""):
    p = paths(cfg)
    t = cfg["training"]
    if not t["save_optimizer"] or not t["auto_resume"]:
        raise ValueError("Training requires optimizer state and automatic resume")
    if (t["parameter_dtype"], t["autocast_dtype"], t["attention_backend"]) != ("float32", "bfloat16", "math"):
        raise ValueError("Use the validated FP32 master parameters / bf16 autocast / SDPA math settings")
    batch, micro, epochs, save = (t[k] for k in ["effective_batch_size", "micro_batch_size", "epochs", "save_steps"])
    if min(batch, micro, epochs, save) <= 0 or batch % micro:
        raise ValueError("Invalid batch, accumulation, epoch or checkpoint interval")
    complete = read_json(p["data"] / "COMPLETED.json")
    for name in ["A1.jsonl", "manifest.jsonl", "summary.json"]:
        if name not in complete["files"]:
            raise ValueError(f"Prepared training data is missing {name}")
    for name, sha in complete["files"].items():
        if file_hash(p["data"] / name) != sha:
            raise ValueError(f"Prepared training data checksum mismatch: {name}")
    summary = read_json(p["data"] / "summary.json")
    if summary.get("model", cfg["model"]) != cfg["model"]:
        raise ValueError("Prepared tokenizer/model revision differs from the training config")
    rows = read_jsonl(p["data"] / "A1.jsonl")
    if not rows or len({r["id"] for r in rows}) != len(rows) or any(r["split"] != "A1" for r in rows):
        raise ValueError("Training requires nonempty, unique A1 examples")
    if subset_size is None and len(rows) != cfg["data"]["a1_size"]:
        raise ValueError("Prepared training size differs from configured a1_size")
    if subset_size is not None:
        if not 0 < subset_size <= len(rows): raise ValueError("Invalid training subset size")
        rows = rows[:subset_size]
    steps = epochs * math.ceil(len(rows)/batch)
    if subset_size is None and t.get("expected_total_steps", steps) != steps:
        raise ValueError(f"Expected {t['expected_total_steps']} steps, actual data/batch/epochs give {steps}")
    checkpoints = list(range(save, steps+1, save))
    if not checkpoints or checkpoints[-1] != steps: checkpoints.append(steps)
    run = p["run"].with_name(p["run"].name+run_suffix)
    return {"run_id": run.name, "model": cfg["model"], "prepared_data": str(p["data"]),
            "train_examples": len(rows), "epochs": epochs, "effective_batch_size": batch,
            "micro_batch_size": micro, "gradient_accumulation_steps": batch//micro,
            "total_steps": steps, "warmup_steps": math.ceil(steps*t["warmup_ratio"]),
            "learning_rate": t["learning_rate"], "save_every_steps": save, "checkpoint_steps": checkpoints,
            "checkpoint_directory": str(run/"checkpoints"), "save_optimizer": True, "auto_resume": True,
            "saved_state": ["model", "optimizer", "scheduler", "RNG", "global_step", "data_order_and_position"],
            "parameter_dtype": t["parameter_dtype"], "autocast_dtype": t["autocast_dtype"],
            "attention_backend": t["attention_backend"], "finite_gradient_check": True,
            "train_data_sha256": file_hash(p["data"]/"A1.jsonl"),
            "validation_data_sha256": complete["files"].get("A2.jsonl"), "validation_only": True}


def write_training_plan(cfg, subset_size=None, run_suffix=""):
    plan = training_plan(cfg, subset_size, run_suffix)
    path = paths(cfg)["run"].with_name(paths(cfg)["run"].name+run_suffix)/"training_preflight.json"
    write_json(path, plan)
    return plan

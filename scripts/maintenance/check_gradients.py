"""Diagnose numerical kernels without updating any model parameters."""
import argparse
import json
import random

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoTokenizer

from data_picking.common import load_config, paths, read_jsonl, seed_all, write_json
from data_picking.loss import per_example_nll
from data_picking.tokenization import collate
from data_picking.training import load_model

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/pilot.yaml")
args = parser.parse_args()
cfg = load_config(args.config)
p = paths(cfg)
rows = read_jsonl(p["data"] / "A1.jsonl")[:64]
order = list(range(len(rows)))
random.Random(cfg["training"]["seed"]).shuffle(order)
tokenizer = AutoTokenizer.from_pretrained(p["model"], local_files_only=True)
model = load_model(cfg, train=True)
results = []
for checkpointing, precision, backend in [(True, "bf16", "auto"), (False, "bf16", "auto"), (True, "bf16", "math"), (True, "fp32", "math")]:
    seed_all(cfg["training"]["seed"])
    if checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        model.gradient_checkpointing_disable()
    model.zero_grad(set_to_none=True)
    for offset in range(0, 32, 2):
        batch = [rows[i] for i in order[offset:offset+2]]
        inputs, mask = collate(batch, tokenizer.pad_token_id)
        backends = [SDPBackend.MATH] if backend == "math" else [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH, SDPBackend.CUDNN_ATTENTION]
        with sdpa_kernel(backends):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision=="bf16"):
                output = model(**inputs, use_cache=False)
            _, means, _ = per_example_nll(output.logits, inputs["input_ids"], mask)
            (means.sum()/32).backward()
        bad = [name for name, value in model.named_parameters() if value.grad is not None and not torch.isfinite(value.grad).all()]
        record = {"checkpointing": checkpointing, "precision": precision, "backend": backend, "microbatch_offset": offset,
                  "loss": means.detach().tolist(), "bad_gradient_count": len(bad), "bad_gradient_examples": bad[:6],
                  "ids": [r["id"] for r in batch]}
        results.append(record)
        print(json.dumps(record), flush=True)
        del output, means, inputs, mask
        if bad:
            break
    if backend == "math" and precision == "bf16" and not bad:
        break
write_json(p["run"] / "logs/gradient_diagnostic.json", results)

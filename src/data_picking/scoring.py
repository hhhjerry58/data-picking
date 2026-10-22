import gc
import random
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer

from .checkpoint import validate_checkpoint
from .common import digest, file_hash, paths, read_json, read_jsonl, seed_all, write_json
from .loss import per_example_nll
from .tokenization import collate
from .training import load_model


@torch.inference_mode()
def score_batches(model, rows, tokenizer, batch_size):
    result = []
    # Full float32 scoring avoids bf16 batch-shape noise in small loss deltas.
    with torch.autocast("cuda", enabled=False):
        for offset in range(0, len(rows), batch_size):
            batch = rows[offset:offset+batch_size]
            inputs, mask = collate(batch, tokenizer.pad_token_id)
            output = model(**inputs, use_cache=False)
            sums, means, counts = per_example_nll(output.logits, inputs["input_ids"], mask)
            for row, total, mean, count in zip(batch, sums.tolist(), means.tolist(), counts.tolist()):
                assert count == row["response_tokens"]
                if not np.isfinite(mean):
                    raise ValueError(f'Non-finite score: {row["id"]}')
                result.append({k: row[k] for k in ["id", "group_id", "domain", "split", "level", "text_hash", "tokenization_hash", "prompt_tokens", "response_tokens", "source", "source_revision"]})
                result[-1].update(sum_nll=total, mean_nll=mean)
            if offset % 128 == 0:
                print("SCORE", offset+len(batch), "/", len(rows), flush=True)
    return result


def score(cfg, checkpoint=None, sanity_only=False):
    p = paths(cfg)
    if cfg["scoring"]["forward_dtype"] != "float32":
        raise ValueError("The validated scoring pipeline uses full float32 forward")
    seed_all(cfg["training"]["seed"])
    rows = read_jsonl(p["data"] / "A1.jsonl")
    diagnostic = random.Random(cfg["data"]["split_seed"]+1).sample(rows, min(len(rows), cfg["data"]["diagnostic_size"]))
    diagnostic = [dict(r, split="A1_diagnostic") for r in diagnostic]
    eval_rows = read_jsonl(p["data"] / "A2.jsonl") + read_jsonl(p["data"] / "B.jsonl")
    target_rows = eval_rows + diagnostic
    # A deterministic shuffle avoids domain ordering effects and metadata tie breaks.
    random.Random(cfg["data"]["split_seed"]+2).shuffle(target_rows)
    out = p["run"] / "scores"
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_step = 0
    checkpoint_hash = None
    if checkpoint:
        meta = validate_checkpoint(checkpoint)
        checkpoint_step = meta["global_step"]
        checkpoint_hash = meta["files"]["state.pt"]["sha256"]
    label = f"step-{checkpoint_step:06d}" if checkpoint else "reference"
    identity = {"model": cfg["model"], "data_hash": file_hash(p["data"] / "COMPLETED.json"),
                "checkpoint_hash": checkpoint_hash, "scoring_config": cfg["scoring"],
                "scoring_code": file_hash(Path(__file__)), "loss_code": file_hash(Path(__file__).with_name("loss.py"))}
    complete = out / f"{label}.complete.json"
    output_file = out / f"{label}.parquet"
    if not sanity_only and complete.exists():
        saved = read_json(complete)
        if saved["identity"] != identity or file_hash(output_file) != saved["sha256"]:
            raise RuntimeError("Existing scores do not match requested evaluation")
        print("Using complete scores", output_file, flush=True)
        return output_file
    tokenizer = AutoTokenizer.from_pretrained(p["model"], local_files_only=True)
    model = load_model(cfg)
    if checkpoint:
        payload = torch.load(Path(checkpoint) / "state.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"])
        del payload
    if not checkpoint:
        small = diagnostic[:8]
        one = score_batches(model, small, tokenizer, cfg["scoring"]["batch_size"])
        two = score_batches(model, small, tokenizer, cfg["scoring"]["batch_size"])
        singles = score_batches(model, small, tokenizer, 1)
        repeat_diff = max(abs(a["mean_nll"]-b["mean_nll"]) for a,b in zip(one,two))
        batch_diff = max(abs(a["mean_nll"]-b["mean_nll"]) for a,b in zip(one,singles))
        sanity = {"identical_model_max_delta": repeat_diff, "batch_padding_max_delta": batch_diff,
                  "tolerance": cfg["scoring"]["sanity_tolerance"], "sample_ids": [r["id"] for r in small]}
        write_json(out / "sanity.json", sanity)
        if repeat_diff > cfg["scoring"]["sanity_tolerance"] or batch_diff > cfg["scoring"]["sanity_tolerance"]:
            raise RuntimeError(f"Scoring sanity check failed: {sanity}")
        print("SCORING_SANITY_PASSED", sanity, flush=True)
    if sanity_only:
        return
    result = score_batches(model, target_rows, tokenizer, cfg["scoring"]["batch_size"])
    assert len({r["id"] for r in result}) == len(target_rows)
    for row in result:
        row.update(model_revision=cfg["model"]["revision"], checkpoint_step=checkpoint_step, run_id=cfg["run_id"])
    pq.write_table(pa.Table.from_pylist(result), output_file)
    write_json(complete, {"identity": identity, "sha256": file_hash(output_file), "examples": len(result)})
    print("SCORES_COMPLETE", output_file, flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return output_file

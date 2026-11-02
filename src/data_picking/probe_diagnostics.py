"""Frozen QA cross-evaluation and web checkpoint diagnostics; no training."""
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from transformers import AutoTokenizer

from .common import digest, file_hash, read_json, read_jsonl, seed_all, write_json
from .data import normalize
from .training import load_model
from .web_scoring import (checkpoint_metadata, evaluation_config, load_staged_sft,
                          sanity_check, score_batch)


QA_METADATA = ["id", "group_id", "domain", "split", "level", "text_hash",
               "tokenization_hash", "prompt_tokens", "response_tokens", "source", "source_revision"]


def settings(config_path):
    cfg = yaml.safe_load(Path(config_path).read_text())
    run = Path(cfg["artifact_root"])/"runs"/cfg["run_id"]
    ev, train, web_run = evaluation_config(cfg["web_evaluation_config"])
    return cfg, run, ev, train, web_run


def freeze(config_path):
    cfg, run, ev, train, web_run = settings(config_path)
    run.mkdir(parents=True, exist_ok=True)
    legacy, snapshot, bundle = Path(cfg["legacy_run"]), Path(cfg["legacy_snapshot"]), Path(ev["bundle"])
    files = [Path(config_path), Path(cfg["web_evaluation_config"]), Path(ev["training_config"]),
             legacy/"data/evaluation.jsonl", snapshot/"per_example.parquet",
             snapshot/"length_matched_ids.json", snapshot/"metrics.json", bundle/"target/train.jsonl",
             bundle/"web/windows.parquet", bundle/"web/document_manifest.parquet", bundle/"web/labels.jsonl",
             web_run/"metrics_web/length_matched_ids.json", web_run/"metrics_web/metrics.json"]
    checkpoints = {}
    for step in cfg["web_checkpoint_steps"]:
        candidates = list((web_run/"checkpoints").glob(f"step-{step:06d}-*"))
        if len(candidates) != 1:
            raise RuntimeError(f"Expected exactly one checkpoint at step {step}")
        meta = checkpoint_metadata(candidates[0])
        checkpoints[str(step)] = {"path": str(candidates[0]), "state_sha256": meta["files"]["state.pt"]["sha256"]}
    identity = {"config": cfg, "files": {str(f): file_hash(f) for f in files},
                "checkpoints": checkpoints, "web_run": str(web_run)}
    path = run/"diagnostic_plan.json"
    if path.exists():
        plan = read_json(path)
        if plan["identity"] != identity or plan["fingerprint"] != digest(identity):
            raise RuntimeError("Frozen diagnostic inputs changed")
    else:
        plan = {"identity": identity, "fingerprint": digest(identity),
                "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
                "known_before_diagnostics": "Original step-64 web result and legacy step-32 QA results already observed; these are follow-up diagnostics, not a new preregistered confirmation."}
        write_json(path, plan)
        write_json(run/"original_notebook_hashes.json", {
            str(p): file_hash(p) for p in Path("experiments-analysis").glob("*/analysis.ipynb")})
    (run/"logs").mkdir(exist_ok=True)
    return plan


def qa_rows(cfg):
    rows = read_jsonl(Path(cfg["legacy_run"])/"data/evaluation.jsonl")
    frame = pd.DataFrame([{k: r[k] for k in QA_METADATA} for r in rows])
    old = pd.read_parquet(Path(cfg["legacy_snapshot"])/"per_example.parquet")
    pd.testing.assert_frame_equal(frame.sort_values("id").reset_index(drop=True),
                                  old[QA_METADATA].sort_values("id").reset_index(drop=True))
    if not frame.id.is_unique or frame.groupby("domain").size().to_dict() != {"A": 512, "C": 512}:
        raise RuntimeError("The original fixed 512/512 QA pool changed")
    for row in rows:
        if digest({"input_ids": row["input_ids"], "response_mask": row["response_mask"]}) != row["tokenization_hash"]:
            raise RuntimeError(f"Tokenization checksum mismatch: {row['id']}")
    return sorted(rows, key=lambda r: (-len(r["input_ids"]), r["id"]))


def score_qa(config_path):
    cfg, run, ev, train, web_run = settings(config_path)
    plan = freeze(config_path)
    rows, spec = qa_rows(cfg), cfg["scoring"]
    if spec["forward_dtype"] != "float32" or spec["attention_backend"] != "math":
        raise ValueError("Diagnostics require full FP32 / math attention")
    # Explicitly audit overlap with the larger training set; never replace eval rows.
    train_rows = read_jsonl(Path(ev["bundle"])/"target/train.jsonl")
    train_questions = {normalize(r["prompt"]) for r in train_rows}
    write_json(run/"qa_overlap.json", {"training_examples": len(train_rows),
        "normalized_question_overlap_ids": [r["id"] for r in rows if normalize(r["prompt"]) in train_questions],
        "legacy_A2_reservation": "The bundle's pre-existing target-overlap quarantine also reserves legacy A2/B problem groups."})
    seed_all(train["training"]["seed"])
    tokenizer = AutoTokenizer.from_pretrained(train["runtime_model_dir"], local_files_only=True)
    model = None
    step = cfg["qa_checkpoint_step"]
    for label in ["reference", f"step-{step:06d}"]:
        folder = run/"scores_qa"/label
        folder.mkdir(parents=True, exist_ok=True)
        identity = {"plan": plan["fingerprint"], "label": label,
            "implementation": {name: file_hash(Path(__file__).with_name(name)) for name in
                ["probe_diagnostics.py", "web_scoring.py", "training.py", "runtime_model.py", "tokenization.py", "loss.py"]}}
        fingerprint = digest(identity)
        existing = folder/"scoring_config.json"
        if existing.exists() and read_json(existing)["fingerprint"] != fingerprint:
            raise RuntimeError("QA scoring implementation/identity changed")
        write_json(existing, {"identity": identity, "fingerprint": fingerprint})
        done = set()
        for marker in folder.glob("chunk-*.complete.json"):
            record = read_json(marker)
            if record["fingerprint"] != fingerprint or file_hash(folder/record["file"]) != record["sha256"]:
                raise RuntimeError("Corrupt QA score chunk")
            done.add(record["start"])
        starts = set(range(0, len(rows), spec["chunk_size"]))
        if not done <= starts:
            raise RuntimeError("Unexpected QA score chunk")
        if done == starts:
            print("QA_SCORES_CACHED", label, flush=True)
            continue
        if model is None:
            model = load_model(train, train=False)
        if label != "reference":
            state = load_staged_sft(model, plan["identity"]["checkpoints"][str(step)]["path"], web_run)
            write_json(folder/"checkpoint_state_validation.json", state)
        sanity = sanity_check(model, tokenizer, rows, spec["batch_size"], spec["sanity_tolerance"])
        write_json(folder/"sanity.json", sanity)
        print("QA_SANITY", label, sanity, flush=True)
        for start in range(0, len(rows), spec["chunk_size"]):
            if start in done:
                continue
            chunk = rows[start:start+spec["chunk_size"]]
            measured = [s for i in range(0, len(chunk), spec["batch_size"])
                        for s in score_batch(model, tokenizer, chunk[i:i+spec["batch_size"]])]
            result = [{**{k: row[k] for k in QA_METADATA}, **score} for row, score in zip(chunk, measured)]
            filename = f"chunk-{start:06d}.parquet"
            pq.write_table(pa.Table.from_pylist(result), folder/filename)
            write_json(folder/f"chunk-{start:06d}.complete.json", {"fingerprint": fingerprint,
                "start": start, "file": filename, "sha256": file_hash(folder/filename), "rows": len(result)})
            print("QA_CHUNK", label, start+len(chunk), len(rows), flush=True)
        write_json(folder/"COMPLETED.json", {"fingerprint": fingerprint, "rows": len(rows), "label": label,
            "completed_at_utc": datetime.now(timezone.utc).isoformat()})


def pair_qa(reference, sft):
    frames = [f.sort_values("id").reset_index(drop=True) for f in [reference, sft]]
    if any(not f.id.is_unique for f in frames):
        raise ValueError("Duplicate QA IDs")
    pd.testing.assert_frame_equal(frames[0][QA_METADATA+["scored_tokens"]], frames[1][QA_METADATA+["scored_tokens"]])
    result = frames[0][QA_METADATA].copy()
    for prefix, f in zip(["ref", "sft"], frames):
        if not np.isfinite(f[["mean_nll", "sum_nll"]]).all().all() or not (f.scored_tokens == f.response_tokens).all():
            raise ValueError("Invalid QA losses or response masks")
        result[f"{prefix}_mean_nll"] = f.mean_nll
        result[f"{prefix}_sum_nll"] = f.sum_nll
        result[f"neg_{prefix}_loss"] = -f.mean_nll
        result[f"pos_{prefix}_loss"] = f.mean_nll
    result["delta_loss"] = result.ref_mean_nll-result.sft_mean_nll
    result["relative_gain"] = result.delta_loss/np.maximum(result.ref_mean_nll, 1e-6)
    return result

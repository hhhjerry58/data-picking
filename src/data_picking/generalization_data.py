"""Length-valid random Dolly general_qa samples, with duplicates retained."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import urllib.request

from transformers import AutoTokenizer
import yaml

from .common import digest, file_hash, load_config, paths, read_json, read_jsonl, write_json, write_jsonl
from .data import normalize
from .tokenization import tokenize_example


def load_extension(path):
    cfg = yaml.safe_load(Path(path).read_text())
    base = load_config(cfg["base_config"])
    run = Path(cfg["artifact_root"]) / "runs" / cfg["run_id"]
    return cfg, base, run


def build_candidates(rows, spec, tokenizer):
    """Only exclude empty/unscorable content and overlong complete sequences."""
    metadata, candidates = [], []
    for i, item in enumerate(rows):
        if item["category"] != spec["category"]:
            continue
        # This pinned category has no context. Fail on a changed data contract,
        # rather than silently dropping context or introducing a new template.
        if item.get("context", "").strip():
            raise ValueError("Pinned general_qa context is expected to be empty")
        prompt, response = item["instruction"].strip(), item["response"].strip()
        row = {"id": f"dolly:{spec['category']}:{i}", "domain": "C", "split": "C", "level": "not_applicable",
               "source": spec["name"], "source_revision": spec["revision"], "source_row": i,
               "prompt": prompt, "response": response, "text_hash": digest([prompt, response]),
               "group_id": digest(normalize(prompt)), "response_hash": digest(normalize(response))}
        reason = None
        if not prompt:
            reason = "empty_prompt"
        elif not response:
            reason = "empty_response"
        else:
            try:
                row.update(tokenize_example(tokenizer, prompt, response))
            except ValueError:
                reason = "invalid_response_mask"
            if reason is None and row["sequence_tokens"] > spec["max_length"]:
                reason = "overlong"
        meta = {k: v for k, v in row.items() if k not in {"input_ids", "response_mask", "prompt", "response"}}
        meta.update(status=reason or "eligible", exclusion_reason=reason, selected=False)
        metadata.append(meta)
        if reason is None:
            candidates.append(row)
    return metadata, candidates


def sample_candidates(candidates, size, seed):
    if len(candidates) < size:
        raise RuntimeError(f"Only {len(candidates)} valid C samples; need {size}")
    # Sample rows uniformly. In particular, do not collapse question groups.
    return random.Random(seed).sample(sorted(candidates, key=lambda r: r["id"]), size)


def duplicate_summary(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    repeated = [{"group_id": key, "size": len(group), "ids": [r["id"] for r in group],
                 "distinct_answers": len({r["response_hash"] for r in group})}
                for key, group in sorted(groups.items()) if len(group) > 1]
    return {"examples": len(rows), "question_groups": len(groups), "repeated_question_groups": len(repeated),
            "examples_in_repeated_groups": sum(g["size"] for g in repeated),
            "same_question_different_answer_groups": sum(g["distinct_answers"] > 1 for g in repeated),
            "groups": repeated}


def overlap_report(metadata, original):
    by_question = defaultdict(list)
    for split, rows in original.items():
        for row in rows:
            by_question[digest(normalize(row["prompt"]))].append((split, row["id"]))
    return [{"c_id": row["id"], "original_id": original_id, "original_split": split,
             "group_id": row["group_id"], "eligible": row["exclusion_reason"] is None,
             "selected": row["selected"]}
            for row in metadata for split, original_id in by_question.get(row["group_id"], [])]


def prepare_extension(config_path):
    cfg, base, run = load_extension(config_path)
    data = run / "data"
    data.mkdir(parents=True, exist_ok=True)
    original_data = paths(base)["data"]
    original_hashes = {name: file_hash(original_data / f"{name}.jsonl") for name in ["A1", "A2", "B"]}
    identity = {"config": cfg, "preparation_code": file_hash(__file__), "original_split_hashes": original_hashes,
                "tokenization_code": file_hash(Path(__file__).with_name("tokenization.py")),
                "normalization_code": file_hash(Path(__file__).with_name("data.py"))}
    marker = data / "COMPLETED.json"
    if marker.exists():
        saved = read_json(marker)
        if saved["identity"] != identity:
            raise RuntimeError("Prepared extension configuration/code changed; use a new run ID")
        for name, sha in saved["files"].items():
            if file_hash(data / name) != sha:
                raise RuntimeError(f"Invalid prepared file {name}")
        print("PREPARED_DATA_CACHED", data, flush=True)
        return
    spec = cfg["c_data"]
    raw = Path(cfg["artifact_root"]) / "data/raw/dolly" / spec["revision"] / "databricks-dolly-15k.jsonl"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw_marker = raw.with_suffix(".complete.json")
    url = f'https://huggingface.co/datasets/{spec["name"]}/resolve/{spec["revision"]}/databricks-dolly-15k.jsonl'
    if not raw_marker.exists():
        with urllib.request.urlopen(url, timeout=60) as source, raw.open("wb") as target:
            while block := source.read(1024 * 1024):
                target.write(block)
        write_json(raw_marker, {"url": url, "sha256": file_hash(raw)})
    elif file_hash(raw) != read_json(raw_marker)["sha256"]:
        raise RuntimeError("Dolly raw file hash mismatch")
    tokenizer = AutoTokenizer.from_pretrained(paths(base)["model"], local_files_only=True, use_fast=True)
    rows = read_jsonl(raw)
    metadata, candidates = build_candidates(rows, spec, tokenizer)
    selected = sample_candidates(candidates, spec["size"], spec["seed"])
    selected_ids = {row["id"] for row in selected}
    for row in metadata:
        if row["id"] in selected_ids:
            row.update(status="selected", selected=True)
    original = {name: read_jsonl(original_data / f"{name}.jsonl") for name in original_hashes}
    overlaps = overlap_report(metadata, original)
    duplicates = {"question_normalization": "Unicode NFKC, lowercase, whitespace collapse; exact equality",
                  "scope": "Normalized question equality only; semantic paraphrases are not guaranteed detected",
                  "category": duplicate_summary(metadata), "eligible": duplicate_summary(candidates),
                  "selected": duplicate_summary(selected), "cross_split_matches": overlaps,
                  "selected_overlap_ids": sorted({r["c_id"] for r in overlaps if r["selected"]}),
                  "selection_effect": "None: duplicates and overlaps remain in the random sample"}
    anchors = random.Random(cfg["gradients"]["anchor_seed"]).sample(original["A1"], cfg["gradients"]["anchor_size"])
    evaluation = original["A2"] + original["B"] + selected
    random.Random(spec["seed"] + 1).shuffle(evaluation)
    assert len({r["id"] for r in evaluation}) == len(evaluation)
    for name, content in [("C.jsonl", selected), ("evaluation.jsonl", evaluation), ("anchor_A1.jsonl", anchors), ("c_manifest.jsonl", metadata)]:
        write_jsonl(data / name, content)
    write_json(data / "duplicate_report.json", duplicates)
    summary = {"raw_count": len(rows), "category_count": len(metadata), "eligible_count": len(candidates),
               "selected_count": len(selected), "nonempty_context_count": 0,
               "status_counts": dict(Counter(r["status"] for r in metadata)),
               "exclusion_counts": dict(Counter(r["exclusion_reason"] for r in metadata if r["exclusion_reason"])),
               "domains": {d: {"examples": sum(r["domain"] == d for r in evaluation),
                                "response_tokens": sum(r["response_tokens"] for r in evaluation if r["domain"] == d)} for d in ["A", "B", "C"]},
               "original_data_dir": str(original_data), "original_split_hashes": original_hashes,
               "raw_sha256": file_hash(raw), "template_hash": digest(tokenizer.chat_template),
               "tokenizer_revision": base["model"]["revision"], "anchor_ids": [r["id"] for r in anchors],
               "dataset_license": "CC-BY-SA-3.0", "dataset_attribution": "Databricks, Inc., databricks-dolly-15k",
               "selection_policy": "Uniform seeded row sample of general_qa with nonempty scorable content and complete sequence <= max_length; no truncation, minimum answer length, content filters or duplicate removal",
               "selected_question_groups": duplicates["selected"]["question_groups"],
               "selected_overlap_count": len(duplicates["selected_overlap_ids"])}
    write_json(data / "summary.json", summary)
    names = ["C.jsonl", "evaluation.jsonl", "anchor_A1.jsonl", "c_manifest.jsonl", "duplicate_report.json", "summary.json"]
    write_json(run / "extension_config.json", cfg)
    write_json(run / "evaluation_plan.json", {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "primary_comparison": "A_vs_C", "primary_score": cfg["analysis"]["primary_loss_score"],
        "positive_domains": cfg["analysis"]["positive_domains"],
        "gradient_metrics": "Exploratory; higher raw value predicts the first domain, no post-hoc sign flip",
        "length_control": "Three-way joint prompt/response length quartiles; equal domain counts per bin",
        "duplicate_control": "Retain rows; bootstrap normalized question groups within each domain",
        "overlap_control": "Keep main random sample; separately exclude overlapping C for sensitivity analysis",
        "selection_policy": summary["selection_policy"], "training_updates": 0,
        "model_selection": "Original reference and final step-32 SFT model"})
    write_json(run / "numerical_validation_amendment.json", {
        "previous_tolerance": 2e-5, "current_tolerance": cfg["gradients"]["loss_crosscheck_tolerance"],
        "prior_observed_difference": 2.384185791015625e-5, "prior_example": "algebra:train:162",
        "reason": "Single-example FP32 autograd/math versus previous batch inference, identical tokenization hashes",
        "scope": "Tolerance inherited from the earlier A/B crosscheck; C is newly sampled under the current policy",
        "score_and_gradient_definitions_unchanged": True})
    write_json(marker, {"identity": identity, "files": {name: file_hash(data / name) for name in names}})
    print("PREPARED", json.dumps(summary, ensure_ascii=False), flush=True)

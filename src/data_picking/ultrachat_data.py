"""Frozen first-turn UltraChat evaluation, with only length/validity filtering."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import random
import urllib.request

import pyarrow.parquet as pq
from transformers import AutoTokenizer

from .common import digest, file_hash, paths, read_json, read_jsonl, write_json, write_jsonl
from .data import normalize
from .generalization_data import duplicate_summary, load_extension, overlap_report, sample_candidates
from .tokenization import tokenize_example


def build_ultrachat_candidate(item, index, spec, tokenizer):
    messages = item.get("messages")
    valid_format = (isinstance(messages, list) and len(messages) >= 2
                    and all(isinstance(m, dict) for m in messages[:2])
                    and [m.get("role") for m in messages[:2]] == ["user", "assistant"]
                    and all(isinstance(m.get("content"), str) for m in messages[:2]))
    prompt, response = (messages[0]["content"].strip(), messages[1]["content"].strip()) if valid_format else ("", "")
    row = {"id": f"ultrachat:{spec['split']}:{index}", "domain": "C", "split": "C", "level": "not_applicable",
           "source": spec["name"], "source_revision": spec["revision"], "source_row": index,
           "source_split": spec["split"], "source_prompt_id": item.get("prompt_id"),
           "source_message_count": len(messages) if isinstance(messages, list) else 0,
           "conversation_view": "first_user_assistant_pair", "prompt": prompt, "response": response,
           "text_hash": digest([prompt, response]), "group_id": digest(normalize(prompt)),
           "response_hash": digest(normalize(response))}
    reason = None
    if not valid_format:
        reason = "invalid_first_pair"
    elif not prompt:
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
    metadata = {k: v for k, v in row.items() if k not in {"prompt", "response", "input_ids", "response_mask"}}
    metadata.update(status=reason or "eligible", exclusion_reason=reason, selected=False)
    return metadata, row if reason is None else None


def download_source(cfg):
    spec = cfg["c_data"]
    folder = Path(cfg["artifact_root"]) / "data/raw/ultrachat" / spec["revision"]
    folder.mkdir(parents=True, exist_ok=True)
    raw = folder / Path(spec["source_file"]).name
    if raw.exists() and raw.stat().st_size == spec["source_bytes"] and file_hash(raw) == spec["source_sha256"]:
        print("ULTRACHAT_RAW_CACHED", raw, flush=True)
        return raw
    url = f'https://huggingface.co/datasets/{spec["name"]}/resolve/{spec["revision"]}/{spec["source_file"]}'
    with urllib.request.urlopen(url, timeout=60) as source, raw.open("wb") as target:
        count = 0
        while block := source.read(4 * 1024 * 1024):
            target.write(block)
            count += len(block)
            if count % (16 * 1024 * 1024) == 0:
                print("DOWNLOAD_MIB", count // 1024**2, flush=True)
    if raw.stat().st_size != spec["source_bytes"] or file_hash(raw) != spec["source_sha256"]:
        raise RuntimeError("UltraChat source size/SHA256 mismatch")
    write_json(raw.with_suffix(".complete.json"), {"url": url, "bytes": raw.stat().st_size, "sha256": file_hash(raw)})
    with urllib.request.urlopen(f'https://huggingface.co/datasets/{spec["name"]}/raw/{spec["revision"]}/README.md', timeout=60) as source:
        (folder / "README.md").write_bytes(source.read())
    return raw


def prepare_ultrachat(config_path):
    cfg, base, run = load_extension(config_path)
    spec = cfg["c_data"]
    if spec["conversation_view"] != "first_user_assistant_pair":
        raise ValueError("The registered experiment uses the first complete question/answer pair")
    raw = download_source(cfg)
    data = run / "data"
    data.mkdir(parents=True, exist_ok=True)
    original_data = paths(base)["data"]
    split_hashes = {name: file_hash(original_data / f"{name}.jsonl") for name in ["A1", "A2", "B"]}
    identity = {"config": cfg, "raw_sha256": file_hash(raw), "original_split_hashes": split_hashes,
                "implementation": {name: file_hash(Path(__file__).with_name(name)) for name in
                                   ["ultrachat_data.py", "generalization_data.py", "tokenization.py", "data.py"]}}
    marker = data / "COMPLETED.json"
    if marker.exists():
        saved = read_json(marker)
        if saved["identity"] != identity:
            raise RuntimeError("Prepared configuration/code changed; use a new run ID")
        for name, sha in saved["files"].items():
            if file_hash(data / name) != sha: raise RuntimeError(f"Prepared file changed: {name}")
        print("ULTRACHAT_PREPARED_CACHED", data, flush=True)
        return
    tokenizer = AutoTokenizer.from_pretrained(paths(base)["model"], local_files_only=True, use_fast=True)
    metadata, candidates = [], []
    for batch in pq.ParquetFile(raw).iter_batches(batch_size=512):
        for item in batch.to_pylist():
            meta, row = build_ultrachat_candidate(item, len(metadata), spec, tokenizer)
            metadata.append(meta)
            if row is not None: candidates.append(row)
        print("TOKENIZED", len(metadata), "eligible", len(candidates), flush=True)
    selected = sample_candidates(candidates, spec["size"], spec["seed"])
    selected_ids = {row["id"] for row in selected}
    for row in metadata:
        if row["id"] in selected_ids: row.update(status="selected", selected=True)
    original = {name: read_jsonl(original_data / f"{name}.jsonl") for name in split_hashes}
    overlaps = overlap_report(metadata, original)
    duplicates = {"question_normalization": "Unicode NFKC, lowercase, whitespace collapse; exact equality",
                  "scope": "Normalized first-question equality; semantic paraphrases are not guaranteed detected",
                  "category": duplicate_summary(metadata), "eligible": duplicate_summary(candidates),
                  "selected": duplicate_summary(selected), "cross_split_matches": overlaps,
                  "selected_overlap_ids": sorted({r["c_id"] for r in overlaps if r["selected"]}),
                  "selection_effect": "None: duplicates and overlaps remain in the random sample"}
    anchors = random.Random(cfg["gradients"]["anchor_seed"]).sample(original["A1"], cfg["gradients"]["anchor_size"])
    evaluation = original["A2"] + selected
    random.Random(spec["seed"] + 1).shuffle(evaluation)
    assert len({r["id"] for r in evaluation}) == len(evaluation)
    for name, rows in [("C.jsonl", selected), ("evaluation.jsonl", evaluation), ("anchor_A1.jsonl", anchors),
                       ("c_manifest.jsonl", metadata), ("c_selected_manifest.jsonl", [r for r in metadata if r["selected"]])]:
        write_jsonl(data / name, rows)
    write_json(data / "duplicate_report.json", duplicates)
    policy = "Uniform seeded row sample of complete first user/assistant pairs with nonempty scorable content and sequence <= 2048 tokens; no truncation, minimum answer length, content filters or duplicate removal"
    summary = {"raw_count": len(metadata), "source_split": spec["split"], "category_count": len(metadata),
               "eligible_count": len(candidates), "selected_count": len(selected), "conversation_view": spec["conversation_view"],
               "status_counts": dict(Counter(r["status"] for r in metadata)),
               "exclusion_counts": dict(Counter(r["exclusion_reason"] for r in metadata if r["exclusion_reason"])),
               "source_message_counts": dict(Counter(r["source_message_count"] for r in metadata)),
               "domains": {d: {"examples": sum(r["domain"] == d for r in evaluation),
                                "response_tokens": sum(r["response_tokens"] for r in evaluation if r["domain"] == d)} for d in ["A", "C"]},
               "original_data_dir": str(original_data), "original_split_hashes": split_hashes,
               "raw_sha256": file_hash(raw), "template_hash": digest(tokenizer.chat_template),
               "tokenizer_revision": base["model"]["revision"], "anchor_ids": [r["id"] for r in anchors],
               "dataset_license": "MIT", "dataset_attribution": "HuggingFaceH4; UltraChat by Ding et al. (2023)",
               "upstream_processing": "UltraChat 200k is an upstream filtered/truecased subset of synthetic ChatGPT dialogues; this experiment adds no topic/language/code filters",
               "selection_policy": policy, "selected_question_groups": duplicates["selected"]["question_groups"],
               "selected_overlap_count": len(duplicates["selected_overlap_ids"])}
    write_json(data / "summary.json", summary)
    write_json(run / "extension_config.json", cfg)
    write_json(run / "evaluation_plan.json", {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(), "primary_comparison": "A_vs_C",
        "primary_score": "delta_loss", "positive_domain": "A", "selection_policy": policy,
        "source_role": "External-source replication; one fixed C sample, not a multiple-seed estimate",
        "length_control": "Joint A2/C prompt/response quartiles; balanced per bin; no score-based selection",
        "bootstrap": "2000 domain-stratified question-group resamples", "training_updates": 0,
        "model_selection": "Original reference and final step-32 SFT, no checkpoint selection",
        "gradient_metrics": "Unclipped per-example FP32 global/layer/parameter statistics and same-model fixed A1 cosine; record, do not display in notebook"})
    names = ["C.jsonl", "evaluation.jsonl", "anchor_A1.jsonl", "c_manifest.jsonl", "c_selected_manifest.jsonl", "duplicate_report.json", "summary.json"]
    write_json(marker, {"identity": identity, "files": {name: file_hash(data / name) for name in names}})
    print("ULTRACHAT_PREPARED", {k: summary[k] for k in ["raw_count", "eligible_count", "selected_count", "exclusion_counts", "selected_question_groups", "selected_overlap_count"]}, flush=True)

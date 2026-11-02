"""Prepare a frozen math-document retrieval pool and independent SFT data."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import re
import shutil
import urllib.request

import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer
import yaml

from .common import digest, file_hash, read_json, read_jsonl, write_json, write_jsonl
from .data import quotas
from .document_utils import TargetOverlap, canonical_url, document_windows, grouped_duplicates
from .tokenization import tokenize_example
from .ultrachat_data import build_ultrachat_candidate


def raw_path(root, spec, record):
    if spec["name"] == "EleutherAI/hendrycks_math":
        return root / "data/raw" / spec["revision"] / record["path"]
    if spec["name"] == "HuggingFaceH4/ultrachat_200k":
        return root / "data/raw/ultrachat" / spec["revision"] / Path(record["path"]).name
    name = "openwebmath" if spec["name"].startswith("open-web-math/") else "fineweb"
    relative = Path(record["path"]).name if name == "openwebmath" else record["path"]
    return root / "data/raw/web-retrieval" / name / spec["revision"] / relative


def download_file(root, spec, record):
    path = raw_path(root, spec, record)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size == record["bytes"] and file_hash(path) == record["sha256"]:
        return path
    url = f'https://huggingface.co/datasets/{spec["name"]}/resolve/{spec["revision"]}/{record["path"]}'
    print("DOWNLOADING", spec["name"], record["path"], flush=True)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as source, path.open("wb") as target:
                size = 0
                while block := source.read(8 * 1024**2):
                    target.write(block); size += len(block)
                    if size % (256 * 1024**2) == 0: print("DOWNLOAD_MIB", path.name, size // 1024**2, flush=True)
            if path.stat().st_size != record["bytes"] or file_hash(path) != record["sha256"]:
                raise RuntimeError("Downloaded file size/SHA256 mismatch")
            write_json(path.with_suffix(".complete.json"), {**record, "url": url})
            return path
        except Exception:
            if attempt == 2: raise
            print("RETRY_DOWNLOAD", path.name, attempt+1, flush=True)


def download_sources(cfg):
    root = Path(cfg["artifact_root"])
    specs = [cfg["target"], cfg["web"]["math"], cfg["web"]["background"], cfg["control_sft"]]
    jobs = [(spec, record) for spec in specs for record in spec["files"]]
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda job: download_file(root, *job), jobs))
    for spec in specs:
        card = raw_path(root, spec, spec["files"][0]).parent / "SOURCE_CARD.md"
        if not card.exists():
            url = f'https://huggingface.co/datasets/{spec["name"]}/raw/{spec["revision"]}/README.md'
            with urllib.request.urlopen(url, timeout=45) as source: card.write_bytes(source.read())
    print("ALL_SOURCES_VERIFIED", len(jobs), flush=True)


def selected_rows(path, indexes):
    """Read only row groups touched by the selected source row numbers."""
    pf = pq.ParquetFile(path)
    wanted = sorted(indexes)
    cursor, offset = 0, 0
    result = {}
    for group in range(pf.num_row_groups):
        end = offset + pf.metadata.row_group(group).num_rows
        take = []
        while cursor < len(wanted) and wanted[cursor] < end:
            take.append(wanted[cursor]); cursor += 1
        if take:
            table = pf.read_row_group(group).take(pa.array([i-offset for i in take], type=pa.int64()))
            result.update(zip(take, table.to_pylist()))
        offset = end
    if len(result) != len(set(indexes)): raise RuntimeError("Missing selected source rows")
    return result


def split_target_pools(pools, spec):
    """Optionally grow training while keeping the parent validation split fixed."""
    rng = random.Random(spec["seed"])
    parent = spec.get("preserve_splits_from")
    if parent:
        parent = Path(parent)
        old_train = read_jsonl(parent/"train.jsonl")
        old_validation = read_jsonl(parent/"validation.jsonl")
        if len(old_validation) != spec["validation_size"] or len(old_train) > spec["train_size"]:
            raise ValueError("Preserved split sizes do not match the requested expansion")
        eligible = {r["id"]: r for rows in pools.values() for r in rows}
        for old in old_train+old_validation:
            current = eligible.get(old["id"])
            if current is None or any(current[k] != old[k] for k in
                                      ["group_id", "text_hash", "input_ids", "response_mask"]):
                raise ValueError(f"Parent sample/group/tokenization changed: {old['id']}")
        excluded = {r["id"] for r in old_train+old_validation}
        remaining = {key: [r for r in rows if r["id"] not in excluded] for key, rows in pools.items()}
        capacity = {key: len(rows) for key, rows in remaining.items()}
        additions = quotas(spec["train_size"]-len(old_train), capacity, capacity)
        train = [eligible[r["id"]] for r in old_train]
        for key in sorted(remaining):
            rng.shuffle(remaining[key])
            train.extend(remaining[key][:additions[key]])
        rng.shuffle(train)
        validation = [eligible[r["id"]] for r in old_validation]
    else:
        capacity = {key: len(rows) for key, rows in pools.items()}
        combined = quotas(spec["train_size"]+spec["validation_size"], capacity, capacity)
        training = quotas(spec["train_size"], combined, combined)
        train, validation = [], []
        for key in sorted(pools):
            rng.shuffle(pools[key])
            selected = pools[key][:combined[key]]
            train.extend(selected[:training[key]])
            validation.extend(selected[training[key]:])
        rng.shuffle(train)
        rng.shuffle(validation)
    return train, validation


def prepare_targets(cfg, out, tokenizer):
    spec, root = cfg["target"], Path(cfg["artifact_root"])
    reserved = {f"math:{r['source_subset']}:train:{r['source_row']}"
                for path in spec["reserved_evaluation_files"] for r in read_jsonl(path)}
    rows, manifest = [], []
    image = re.compile(r"\[asy\]|<img\b|\\includegraphics|(?:diagram|figure|graph)\s+(?:below|above)", re.I)
    for record in spec["files"]:
        subject = record["path"].split("/")[0]
        for i, raw in enumerate(pq.read_table(raw_path(root, spec, record)).to_pylist()):
            prompt, response = raw["problem"].strip(), raw["solution"].strip()
            row = {"id": f"math:{subject}:train:{i}", "domain": "A", "subject": subject,
                   "level": raw["level"], "source": spec["name"], "source_revision": spec["revision"],
                   "source_file": record["path"], "source_row": i, "prompt": prompt, "response": response,
                   "text_hash": digest([prompt, response])}
            reason = None
            if not prompt or not response: reason = "empty"
            elif image.search(prompt+"\n"+response): reason = "requires_image_or_diagram"
            else:
                try: row.update(tokenize_example(tokenizer, prompt, response))
                except ValueError: reason = "invalid_response_mask"
                if reason is None and row["sequence_tokens"] > spec["max_length"]: reason = "overlong"
            meta = {k: v for k, v in row.items() if k not in {"prompt", "response", "input_ids", "response_mask"}}
            meta["status"] = reason or "eligible"
            manifest.append(meta)
            if reason is None: rows.append(row)
    print("GROUPING_TARGET", len(rows), flush=True)
    groups, edges = grouped_duplicates(rows, "prompt", cfg["web"]["near_duplicate_threshold"], cfg["statistics"]["minhash_seed"])
    meta = {r["id"]: r for r in manifest}
    pools = defaultdict(list)
    for group in groups:
        ordered = sorted(group, key=lambda r: r["id"])
        if any(row["id"] in reserved for row in ordered):
            for row in ordered:
                meta[row["id"]].update(group_id=row["group_id"], status="reserved_legacy_evaluation_group")
            continue
        for i, row in enumerate(ordered):
            meta[row["id"]]["group_id"] = row["group_id"]
            if i: meta[row["id"]]["status"] = "duplicate_problem_group"
        row = ordered[0]
        pools[row["subject"]+"|"+row["level"]].append(row)
    train, validation = split_target_pools(pools, spec)
    for split, selected in [("A1", train), ("A2", validation)]:
        for row in selected:
            row["split"] = split
            meta[row["id"]]["status"] = "target_train" if split == "A1" else "target_validation"
    assert not {r["group_id"] for r in train} & {r["group_id"] for r in validation}
    assert not {r["id"] for r in train+validation} & reserved
    folder = out / "target"
    folder.mkdir(exist_ok=True)
    write_jsonl(folder/"train.jsonl", train)
    write_jsonl(folder/"validation.jsonl", validation)
    if spec.get("preserve_splits_from"):
        parent = Path(spec["preserve_splits_from"])
        assert file_hash(folder/"validation.jsonl") == file_hash(parent/"validation.jsonl")
    write_jsonl(folder/"manifest.jsonl", manifest)
    write_json(folder/"duplicate_edges.json", edges)
    summary = {"raw_examples": len(manifest), "status_counts": dict(Counter(r["status"] for r in manifest)),
               "splits": {name: {"examples": len(data), "response_tokens": sum(r["response_tokens"] for r in data),
                                  "subjects": dict(Counter(r["subject"] for r in data)),
                                  "levels": dict(Counter(r["level"] for r in data))} for name, data in [("train", train), ("validation", validation)]},
               "duplicate_edges": len(edges), "official_test_used": False, "legacy_evaluation_ids_reserved": len(reserved),
               "policy": "Official train only; text-only/scorable <=2048-token QA; one representative per problem group; subject/level-stratified split"}
    if spec.get("preserve_splits_from"):
        parent = Path(spec["preserve_splits_from"])
        summary["expansion"] = {"parent": str(parent), "parent_train_examples": len(read_jsonl(parent/"train.jsonl")),
                                "validation_sha256": file_hash(parent/"validation.jsonl"),
                                "parent_train_is_subset": True, "validation_unchanged": True}
    write_json(folder/"summary.json", summary)
    return train, validation, summary


def web_candidate_order(cfg, key):
    spec, root = cfg["web"][key], Path(cfg["artifact_root"])
    population = []
    for index, record in enumerate(spec["files"]):
        pf = pq.ParquetFile(raw_path(root, spec, record))
        if key == "background":
            population.extend((index, i) for i in range(pf.metadata.num_rows))
        else:
            offset = 0
            for batch in pf.iter_batches(columns=["metadata"], batch_size=4096):
                for i, value in enumerate(batch.column(0).to_pylist()):
                    match = re.search(r"CC-MAIN-\d{4}-\d{2}", value)
                    if match and match.group() == cfg["web"]["crawl"]: population.append((index, offset+i))
                offset += len(batch)
    order = random.Random(spec["seed"]).sample(population, min(len(population), spec["candidate_limit"]))
    by_file = defaultdict(list)
    for index, row in order: by_file[index].append(row)
    loaded = {index: selected_rows(raw_path(root, spec, spec["files"][index]), indexes) for index, indexes in by_file.items()}
    print("WEB_CANDIDATES", key, "pool", len(population), "drawn", len(order), flush=True)
    return [(index, row, loaded[index][row]) for index, row in order], len(population)


def select_documents(cfg, key, tokenizer, guard):
    spec = cfg["web"][key]
    candidates, population = web_candidate_order(cfg, key)
    selected, audit, quarantined = [], [], []
    special = set(tokenizer.all_special_ids)
    for priority, (file_index, row_index, raw) in enumerate(candidates):
        if priority and priority % 1000 == 0: print("WEB_SELECT_PROGRESS", key, priority, "selected", len(selected), flush=True)
        record = spec["files"][file_index]
        identifier = "doc:"+digest([spec["name"], spec["revision"], record["path"], row_index])[:24]
        text = raw["text"].replace("\r\n", "\n").replace("\r", "\n").strip()
        url = raw.get("url", "")
        source_metadata = json.loads(raw["metadata"]) if key == "math" else {k: v for k, v in raw.items() if k != "text"}
        row = {"id": identifier, "text": text, "url": url, "source": key,
               "source_dataset": spec["name"], "source_revision": spec["revision"], "source_file": record["path"],
               "source_row": row_index, "source_date": raw.get("date"), "source_metadata": json.dumps(source_metadata, ensure_ascii=False),
               "crawl": cfg["web"]["crawl"], "canonical_url": canonical_url(url),
               "text_hash": digest(text), "source_priority": priority}
        reason = None
        matches = []
        if not text: reason = "empty"
        else:
            matches = guard.matches(text)
            if matches: reason = "possible_target_overlap"
            else:
                # Full documents are tokenized here; only bounded windows reach a model.
                ids = tokenizer.encode(text, add_special_tokens=False, verbose=False)
                if len(ids) < 2: reason = "insufficient_tokens"
                elif special.intersection(ids): reason = "control_token_in_body"
                else:
                    row.update(input_ids=ids, token_count=len(ids), scored_tokens=len(ids)-1)
        audit.append({"id": identifier, "source": key, "source_file": record["path"], "source_row": row_index,
                      "priority": priority, "status": reason or "selected", "text_hash": row["text_hash"]})
        if matches: quarantined.append({**row, "target_matches": matches})
        if reason is None: selected.append(row)
        if len(selected) == spec["size"]: break
    if len(selected) != spec["size"]:
        raise RuntimeError(f"Insufficient {key} candidates: {len(selected)} / {spec['size']}; do not silently shrink the pool")
    return selected, audit, quarantined, {"population_in_fixed_shards_and_crawl": population,
        "random_priority_candidates": len(candidates), "examined": len(audit),
        "status_counts": dict(Counter(r["status"] for r in audit))}


def prepare_control(cfg, out, tokenizer, guard):
    spec, root = cfg["control_sft"], Path(cfg["artifact_root"])
    record = spec["files"][0]
    path = raw_path(root, spec, record)
    n = pq.ParquetFile(path).metadata.num_rows
    order = random.Random(spec["seed"]).sample(range(n), min(n, spec["pool_size"]*2))
    raw = selected_rows(path, order)
    selected, audit = [], []
    for index in order:
        meta, row = build_ultrachat_candidate(raw[index], index, spec, tokenizer)
        meta["source_file"] = record["path"]
        if row is not None and (guard.matches(row["prompt"]) or guard.matches(row["response"])):
            meta["status"] = "possible_target_overlap"
            row = None
        if row is not None:
            row.update(split="control_pool", source_file=record["path"])
            selected.append(row); meta["status"] = "selected"
        audit.append(meta)
        if len(selected) == spec["pool_size"]: break
    if len(selected) != spec["pool_size"]: raise RuntimeError("Not enough valid general SFT control rows")
    folder = out/"control_sft"; folder.mkdir(exist_ok=True)
    write_jsonl(folder/"pool.jsonl", selected)
    write_jsonl(folder/"manifest.jsonl", audit)
    summary = {"examples": len(selected), "source_split": spec["split"],
               "response_tokens": sum(r["response_tokens"] for r in selected),
               "status_counts": dict(Counter(r["status"] for r in audit)),
               "purpose": "General SFT control pool; equal-token training subset/budget must be fixed before future training"}
    write_json(folder/"summary.json", summary)
    return summary


def prepare_bundle(config_path):
    cfg = yaml.safe_load(Path(config_path).read_text())
    stats = cfg["statistics"]
    if (stats["minhash_permutations"], stats["lsh_bands"], stats["lsh_rows"]) != (64, 16, 4):
        raise ValueError("This preparation implementation uses 64 MinHash permutations and 16x4 bands")
    root = Path(cfg["artifact_root"])
    if not root.is_dir() or not os.path.ismount(root.parent): raise RuntimeError("Persistent artifact mount unavailable")
    out = root/"data/processed"/cfg["dataset_id"]
    out.mkdir(parents=True, exist_ok=True)
    identity = {"config": cfg, "datasketch_version": "1.6.5",
                "reserved_evaluation_hashes": {p: file_hash(p) for p in cfg["target"]["reserved_evaluation_files"]},
                "implementation": {name: file_hash(Path(__file__).with_name(name)) for name in
        ["web_retrieval.py", "document_utils.py", "tokenization.py", "data.py", "ultrachat_data.py"]}}
    if cfg["target"].get("preserve_splits_from"):
        parent = Path(cfg["target"]["preserve_splits_from"])
        identity["parent_split_hashes"] = {name: file_hash(parent/name) for name in ["train.jsonl", "validation.jsonl"]}
    complete = out/"COMPLETED.json"
    if complete.exists():
        saved = read_json(complete)
        if saved["identity"] != identity: raise RuntimeError("Prepared data identity changed; use a new dataset_id")
        for name, sha in saved["files"].items():
            if file_hash(out/name) != sha: raise RuntimeError(f"Artifact hash mismatch: {name}")
        print("WEB_RETRIEVAL_CACHED", out, flush=True)
        return out
    write_json(out/"effective_data_config.json", cfg)
    download_sources(cfg)
    tokenizer_path = root/"cache/models"/cfg["model"]["name"].replace("/", "--")/cfg["model"]["revision"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, use_fast=True)
    train, validation, target_summary = prepare_targets(cfg, out, tokenizer)
    legacy_evaluation = [r for p in cfg["target"]["reserved_evaluation_files"] for r in read_jsonl(p)]
    guard = TargetOverlap(train+validation+legacy_evaluation, cfg["overlap"])
    documents, audits, quarantined, pools = [], [], [], {}
    for key in ["math", "background"]:
        chosen, audit, blocked, info = select_documents(cfg, key, tokenizer, guard)
        documents.extend(chosen); audits.extend(audit); quarantined.extend(blocked); pools[key] = info
    print("GROUPING_DOCUMENTS", len(documents), flush=True)
    groups, edges = grouped_duplicates(documents, "text", cfg["web"]["near_duplicate_threshold"],
                                        cfg["statistics"]["minhash_seed"], url_field="url")
    random.Random(cfg["web"]["shuffle_seed"]).shuffle(documents)
    web = out/"web"; web.mkdir(exist_ok=True)
    pq.write_table(pa.Table.from_pylist([{k: r[k] for k in ["id", "text", "group_id", "token_count", "scored_tokens"]} for r in documents]), web/"documents.parquet")
    manifest = [{k: v for k, v in r.items() if k not in {"text", "input_ids"}} for r in documents]
    pq.write_table(pa.Table.from_pylist(manifest), web/"document_manifest.parquet")
    write_jsonl(web/"labels.jsonl", [{"id": r["id"], "injected_math_source": r["source"] == "math",
                                     "semantic_math_label": None} for r in documents])
    write_jsonl(web/"sampling_audit.jsonl", audits)
    write_jsonl(web/"quarantine.jsonl", quarantined)
    write_json(web/"duplicate_edges.json", edges)
    windows = []
    for row in documents:
        parts = list(document_windows(row["id"], row["input_ids"], cfg["web"]["window_tokens"]))
        assert sum(p["scored_tokens"] for p in parts) == row["token_count"]-1
        rebuilt = [parts[0]["input_ids"][0]] + [t for p in parts for t in p["input_ids"][1:]]
        assert rebuilt == row["input_ids"]
        windows.extend(parts)
    pq.write_table(pa.Table.from_pylist(windows), web/"windows.parquet")
    control_summary = prepare_control(cfg, out, tokenizer, guard)
    summary = {"dataset_id": cfg["dataset_id"], "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_only": True, "model_training_performed": False, "model_scoring_performed": False,
        "target": target_summary, "control_sft": control_summary, "web_pools": pools,
        "web": {"documents": len(documents), "injected_math": sum(r["source"] == "math" for r in documents),
                "unlabeled_background": sum(r["source"] == "background" for r in documents),
                "windows": len(windows), "scored_tokens": sum(w["scored_tokens"] for w in windows),
                "document_groups": len(groups), "duplicate_edges": len(edges),
                "cross_source_groups": sum(len({r["source"] for r in g}) > 1 for g in groups),
                "quarantined_for_possible_target_overlap": len(quarantined), "crawl": cfg["web"]["crawl"],
                "per_source": {s: {"documents": sum(r["source"] == s for r in documents),
                                    "tokens": sum(r["token_count"] for r in documents if r["source"] == s)} for s in ["math", "background"]}},
        "tokenization": {"model": cfg["model"], "chat_template_hash_for_sft": digest(tokenizer.chat_template),
            "web_chat_template": False, "web_added_special_tokens": False, "window_tokens": cfg["web"]["window_tokens"],
            "window_overlap": 1, "coverage": "Every body token after the document's first is scored exactly once; aggregate token-weighted NLL before computing both loss-change metrics"},
        "overlap_checks": {"indexed_target_passages": len(guard.passages), "skipped_short_passages": dict(guard.skipped_short),
            "scope": "Selected target train/validation plus legacy A2/B; exact normalized substring and >=0.8 5-token containment via rare anchors; short generic answers and semantic paraphrases are not certified clean"},
        "duplicate_method": "Normalized exact text/URL plus 64-permutation MinHash (16x4 bands) and verified >=0.8 five-token Jaccard; approximate, all web rows retained",
        "limitations": ["Uniform draws within fixed source shards and one crawl, not a uniform sample of all Common Crawl",
            "OpenWebMath and FineWeb have different upstream extraction/quality filters; identical minimal local newline normalization does not remove that source confound",
            "OpenWebMath is a math-related source including physics/CS; injection labels are source labels, not manually verified mathematical relevance",
            "Background pages may contain mathematics; semantic negatives have not been asserted",
            "No training, ranking metric, retrieval quality or training benefit has been measured"]}
    write_json(out/"summary.json", summary)
    if cfg.get("publish_sft_view"):
        sft = out/"sft"; sft.mkdir(exist_ok=True)
        mapping = {"A1.jsonl": "train.jsonl", "A2.jsonl": "validation.jsonl", "manifest.jsonl": "manifest.jsonl"}
        for name, source in mapping.items(): shutil.copy2(out/"target"/source, sft/name)
        write_json(sft/"summary.json", {**target_summary, "source_bundle": str(out), "model": cfg["model"]})
        names = [*mapping, "summary.json"]
        write_json(sft/"COMPLETED.json", {"files": {name: file_hash(sft/name) for name in names},
                                         "source_bundle": str(out), "train_examples": len(train), "validation_examples": len(validation)})
    write_json(out/"source_manifest.json", {key: cfg[key] for key in ["model", "target", "web", "control_sft"]})
    files = sorted(p for p in out.rglob("*") if p.is_file() and p != complete)
    write_json(complete, {"identity": identity, "files": {str(p.relative_to(out)): file_hash(p) for p in files}})
    print("WEB_RETRIEVAL_READY", out, json.dumps(summary["web"], ensure_ascii=False), flush=True)
    return out

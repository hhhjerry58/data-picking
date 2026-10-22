import math
import random
import re
import unicodedata
from collections import Counter, defaultdict

import pyarrow.parquet as pq
from transformers import AutoTokenizer

from .common import digest, file_hash, paths, read_json, write_json, write_jsonl
from .tokenization import tokenize_example


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).lower().split())


def duplicate_groups(rows, threshold):
    """Exhaustive length-pruned character 5-gram Jaccard, followed by union-find."""
    parents = list(range(len(rows)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parents[max(a, b)] = min(a, b)

    texts = [normalize(r["prompt"]) for r in rows]
    grams = [set(t[j:j+5] for j in range(max(1, len(t)-4))) for t in texts]
    order = sorted(range(len(rows)), key=lambda i: len(grams[i]))
    edges = []
    for pos, i in enumerate(order):
        ni = len(grams[i])
        for j in order[pos+1:]:
            nj = len(grams[j])
            if ni < threshold * nj:
                break
            intersection = len(grams[i] & grams[j])
            similarity = intersection / max(1, ni + nj - intersection)
            if texts[i] == texts[j] or similarity >= threshold:
                union(i, j)
                edges.append({"a": rows[i]["id"], "b": rows[j]["id"], "similarity": similarity})
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[find(i)].append(row)
    return list(groups.values()), edges


def quotas(total, capacities, weights):
    if total > sum(capacities.values()):
        raise ValueError("Insufficient capacity")
    keys = sorted(capacities)
    denom = sum(weights.get(k, 0) for k in keys)
    ideals = {k: total * weights.get(k, 0) / max(denom, 1) for k in keys}
    out = {k: min(capacities[k], math.floor(ideals[k])) for k in keys}
    while sum(out.values()) < total:
        available = [k for k in keys if out[k] < capacities[k]]
        k = max(available, key=lambda x: (ideals[x] - out[x], x))
        out[k] += 1
    return out


def prepare(cfg):
    p = paths(cfg)
    p["data"].mkdir(parents=True, exist_ok=True)
    complete = p["data"] / "COMPLETED.json"
    if complete.exists():
        record = read_json(complete)
        for name, sha in record["files"].items():
            if file_hash(p["data"] / name) != sha:
                raise RuntimeError(f"Prepared data hash mismatch: {name}")
        print("Using complete prepared data", p["data"], flush=True)
        return
    tokenizer = AutoTokenizer.from_pretrained(p["model"], local_files_only=True, use_fast=True)
    manifest, valid, stats = [], [], {}
    image_pattern = re.compile(r"\[asy\]|<img\b|\\includegraphics|https?://\S+\.(?:png|jpg|gif)|(?:diagram|figure|graph)\s+(?:below|above)", re.I)
    for domain, subset in [("A", cfg["data"]["a_subset"]), ("B", cfg["data"]["b_subset"])]:
        raw_path = p["root"] / "data/raw" / cfg["data"]["revision"] / subset / "train-00000-of-00001.parquet"
        raw = pq.read_table(raw_path).to_pylist()
        counts = Counter(raw=len(raw))
        for index, item in enumerate(raw):
            prompt, response = item["problem"].strip(), item["solution"].strip()
            row = {"id": f"{subset}:train:{index}", "domain": domain, "prompt": prompt, "response": response,
                   "level": item["level"], "type": item["type"], "source": cfg["data"]["name"],
                   "source_revision": cfg["data"]["revision"], "source_subset": subset, "source_row": index,
                   "text_hash": digest([prompt, response]), "prompt_hash": digest(normalize(prompt))}
            reason = None
            if not prompt or not response:
                reason = "empty"
            elif image_pattern.search(prompt + "\n" + response):
                reason = "image_or_diagram"
            else:
                row.update(tokenize_example(tokenizer, prompt, response))
                if row["sequence_tokens"] > cfg["data"]["max_length"]:
                    reason = "overlong"
            meta = {k: v for k, v in row.items() if k not in {"prompt", "response", "input_ids", "response_mask"}}
            meta["status"] = reason or "eligible"
            manifest.append(meta)
            if reason:
                counts[reason] += 1
            else:
                valid.append(row)
        stats[domain] = dict(counts)
    print("Filtered source counts", stats, "checking near duplicates", flush=True)
    groups, edges = duplicate_groups(valid, cfg["data"]["near_duplicate_threshold"])
    metadata = {r["id"]: r for r in manifest}
    unique = []
    for group in groups:
        group_id = digest(sorted(r["id"] for r in group))
        cross_domain = len({r["domain"] for r in group}) > 1
        for i, row in enumerate(sorted(group, key=lambda r: r["id"])):
            row["group_id"] = group_id
            metadata[row["id"]]["group_id"] = group_id
            if cross_domain:
                metadata[row["id"]]["status"] = "cross_domain_duplicate"
            elif i:
                metadata[row["id"]]["status"] = "duplicate_group_member"
            else:
                unique.append(row)
    pools = {d: defaultdict(list) for d in ["A", "B"]}
    for row in unique:
        pools[row["domain"]][row["level"]].append(row)
    rng = random.Random(cfg["data"]["split_seed"])
    for domain in ["A", "B"]:
        for level in sorted(pools[domain]):
            rng.shuffle(pools[domain][level])
    levels = sorted(set(pools["A"]) & set(pools["B"]))
    a_counts = {k: len(v) for k, v in pools["A"].items()}
    capacity = {k: min(len(pools["A"][k]), len(pools["B"][k])) for k in levels}
    n_eval = min(cfg["data"]["eval_size"], sum(capacity.values()), sum(a_counts.values()) // 3)
    if n_eval < 2:
        raise ValueError("Insufficient evaluation data")
    eval_quotas = quotas(n_eval, capacity, a_counts)
    splits = {"A1": [], "A2": [], "B": []}
    for level, n in eval_quotas.items():
        splits["A2"].extend(pools["A"][level][:n])
        splits["B"].extend(pools["B"][level][:n])
        pools["A"][level] = pools["A"][level][n:]
    train_capacity = {k: len(v) for k, v in pools["A"].items()}
    n_train = min(cfg["data"]["a1_size"], sum(train_capacity.values()))
    train_quotas = quotas(n_train, train_capacity, a_counts)
    for level, n in train_quotas.items():
        splits["A1"].extend(pools["A"][level][:n])
    for split, rows in splits.items():
        rng.shuffle(rows)
        for row in rows:
            row["split"] = split
            metadata[row["id"]].update({"status": "selected", "split": split})
        write_jsonl(p["data"] / f"{split}.jsonl", rows)
    ids = [{r["id"] for r in rows} for rows in splits.values()]
    group_ids = [{r["group_id"] for r in rows} for rows in splits.values()]
    for i in range(3):
        for j in range(i):
            assert not ids[i] & ids[j]
            assert not group_ids[i] & group_ids[j]
    assert Counter(r["level"] for r in splits["A2"]) == Counter(r["level"] for r in splits["B"])
    write_jsonl(p["data"] / "manifest.jsonl", manifest)
    write_json(p["data"] / "near_duplicate_edges.json", edges)
    summary = {"data_config": cfg["data"], "model": cfg["model"], "template_hash": digest(tokenizer.chat_template),
               "filter_counts": {d: dict(Counter(r["status"] for r in manifest if r["domain"] == d)) for d in ["A", "B"]},
               "raw_counts": stats, "near_duplicate_edges": len(edges),
               "splits": {s: {"examples": len(rows), "response_tokens": sum(r["response_tokens"] for r in rows),
                               "level_counts": dict(Counter(r["level"] for r in rows))} for s, rows in splits.items()},
               "duplicate_policy": "One representative per near-duplicate component; exclude cross-domain components",
               "official_test_used": False}
    write_json(p["data"] / "summary.json", summary)
    names = ["A1.jsonl", "A2.jsonl", "B.jsonl", "manifest.jsonl", "summary.json", "near_duplicate_edges.json"]
    write_json(complete, {"files": {name: file_hash(p["data"] / name) for name in names}})
    print("Prepared", summary, flush=True)

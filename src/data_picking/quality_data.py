"""Frozen OpenStax / FineMath document preparation for the existing math probe."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import re
import xml.etree.ElementTree as ET

import pyarrow as pa
import pyarrow.parquet as pq
import requests
from transformers import AutoTokenizer
import yaml

from .common import digest, file_hash, read_json, read_jsonl, write_json, write_jsonl
from .document_utils import TargetOverlap, canonical_url, document_windows, grouped_duplicates
from .web_scoring import evaluation_config


COL = "{http://cnx.rice.edu/collxml}"
MD = "{http://cnx.rice.edu/mdml}"
CNX = "{http://cnx.rice.edu/cnxml}"


def settings(path):
    cfg = yaml.safe_load(Path(path).read_text())
    root = Path(cfg["artifact_root"])
    return cfg, root/"runs"/cfg["run_id"], root/"data/processed"/cfg["bundle_id"]


def fetch(url, destination, expected_sha=None, expected_size=None):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    marker = destination.with_suffix(destination.suffix+".complete.json")
    if marker.exists():
        info = read_json(marker)
        if (file_hash(destination) == info["sha256"] and (expected_sha is None or info["sha256"] == expected_sha)
                and (expected_size is None or destination.stat().st_size == expected_size)):
            return info
    for attempt in range(3):
        try:
            with requests.get(url, stream=True, timeout=(15, 60)) as response:
                if response.status_code != 200:
                    raise RuntimeError(f"Download HTTP {response.status_code}: {destination.name}")
                with destination.open("wb") as out:
                    for block in response.iter_content(8*1024**2): out.write(block)
            sha, size = file_hash(destination), destination.stat().st_size
            if expected_sha and sha != expected_sha: raise RuntimeError("Downloaded SHA mismatch")
            if expected_size and size != expected_size: raise RuntimeError("Downloaded size mismatch")
            info = {"url": url, "sha256": sha, "bytes": size}
            write_json(marker, info)
            return info
        except Exception:
            if attempt == 2: raise
            print("DOWNLOAD_RETRY", destination.name, attempt+1, flush=True)


def collection_modules(text, collection_path):
    root = ET.fromstring(text)
    book = root.findtext(f"{COL}metadata/{MD}title") or Path(collection_path).stem
    found = []
    def walk(content, chapters):
        for item in content:
            if item.tag == COL+"subcollection":
                title = item.findtext(MD+"title") or "untitled"
                walk(item.find(COL+"content"), chapters+[title])
            elif item.tag == COL+"module" and chapters:
                found.append({"module": item.attrib["document"], "book": book,
                              "chapter": " / ".join(chapters), "collection": collection_path})
    walk(root.find(COL+"content"), [])
    return found


def download(path):
    cfg, run, bundle = settings(path)
    run.mkdir(parents=True, exist_ok=True)
    root = Path(cfg["artifact_root"])/"data/raw/document-quality"
    spec = cfg["sources"]["openstax"]
    directory = root/"openstax"/spec["revision"]
    base = f"https://raw.githubusercontent.com/{spec['repository']}/{spec['revision']}"
    members = []
    for name in spec["collections"]:
        fetch(f"{base}/{name}", directory/name)
        members.extend(collection_modules((directory/name).read_text(), name))
    write_json(directory/"collection_memberships.json", members)
    jobs = [(f"{base}/modules/{module}/index.cnxml", directory/"modules"/module/"index.cnxml", None, None)
            for module in sorted({m["module"] for m in members})]
    jobs += [(f"{base}/{name}", directory/name, None, None) for name in ["README.md", "LICENSE"]]
    fm = cfg["sources"]["finemath"]
    for rec in fm["files"]:
        jobs.append((f"https://huggingface.co/datasets/{fm['name']}/resolve/{fm['revision']}/{rec['path']}",
                     root/"finemath"/fm["revision"]/Path(rec["path"]).name, rec["sha256"], rec["bytes"]))
    jobs.append((f"https://huggingface.co/datasets/{fm['name']}/raw/{fm['revision']}/README.md",
                 root/"finemath"/fm["revision"]/"README.md", None, None))
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, record in enumerate(pool.map(lambda args: fetch(*args), jobs), 1):
            if i % 20 == 0 or i == len(jobs): print("DOWNLOADS", i, len(jobs), flush=True)
    print("QUALITY_SOURCES_DOWNLOADED", len(jobs), "OpenStax unique modules", len({m['module'] for m in members}), flush=True)
    return directory


def load_candidates(cfg):
    from .openstax_text import cnxml_document
    root = Path(cfg["artifact_root"])/"data/raw/document-quality"
    ospec = cfg["sources"]["openstax"]
    directory = root/"openstax"/ospec["revision"]
    members = read_json(directory/"collection_memberships.json")
    by_module = defaultdict(list)
    for row in members: by_module[row["module"]].append(row)
    rows = []
    for module, membership in sorted(by_module.items()):
        path = directory/"modules"/module/"index.cnxml"
        if file_hash(path) != read_json(path.with_suffix(".cnxml.complete.json"))["sha256"]:
            raise RuntimeError("OpenStax source checksum mismatch")
        text, stats = cnxml_document(path.read_text())
        rows.append({"id": "openstax:"+module, "corpus": "openstax", "source": "math", "text": text,
            "url": f"https://github.com/{ospec['repository']}/blob/{ospec['revision']}/modules/{module}/index.cnxml",
            "source_revision": ospec["revision"], "source_file": str(path), "source_row": 0,
            "chapter_id": digest([membership[0]["book"], membership[0]["chapter"]]),
            "book": membership[0]["book"], "chapter": membership[0]["chapter"],
            "memberships": json.dumps(membership), "crawl": "not_applicable", **stats})
    fspec = cfg["sources"]["finemath"]
    counts = Counter()
    for rec in fspec["files"]:
        path = root/"finemath"/fspec["revision"]/Path(rec["path"]).name
        if file_hash(path) != rec["sha256"]: raise RuntimeError("FineMath source checksum mismatch")
        offset = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=2048, columns=["text", "url", "crawl", "score", "int_score"]):
            for i, raw in enumerate(batch.to_pylist()):
                counts[raw["crawl"]] += 1
                if raw["crawl"] != fspec["crawl"]: continue
                text = (raw["text"] or "").replace("\r\n", "\n").replace("\r", "\n").strip()
                rows.append({"id": "finemath:"+digest([rec["path"], offset+i])[:24],
                    "corpus": "finemath", "source": "math", "text": text, "url": raw["url"],
                    "source_revision": fspec["revision"], "source_file": rec["path"], "source_row": offset+i,
                    "crawl": raw["crawl"], "quality_score": raw["score"], "quality_int_score": raw["int_score"]})
            offset += len(batch)
        print("FINEMATH_SOURCE_READ", rec["path"], offset, "matching crawl", counts[fspec["crawl"]], flush=True)
    return rows, dict(counts)


def chapter_groups(rows):
    """Keep whole-chapter bootstrap blocks, unioning duplicate-linked chapters."""
    parent = {r["chapter_id"]: r["chapter_id"] for r in rows}
    def find(k):
        while parent[k] != k: parent[k] = parent[parent[k]]; k = parent[k]
        return k
    duplicate_chapters = defaultdict(list)
    for row in rows: duplicate_chapters[row["group_id"]].append(row["chapter_id"])
    for chapters in duplicate_chapters.values():
        for a,b in zip(chapters,chapters[1:]):
            a,b=find(a),find(b);parent[max(a,b)]=min(a,b)
    for row in rows:
        row["duplicate_group_id"] = row["group_id"]
        row["group_id"] = "openstax-chapter:"+find(row["chapter_id"])


def prepare(path):
    from .data import normalize
    cfg, run, bundle = settings(path)
    plan_path=run/"preparation_plan_v2.json"
    if read_json(plan_path if plan_path.exists() else run/"preparation_plan.json")["config"] != cfg:
        raise RuntimeError("Frozen preparation config changed")
    if (bundle/"COMPLETED.json").exists():
        marker=read_json(bundle/"COMPLETED.json")
        if marker["config_sha256"] != file_hash(path): raise RuntimeError("Prepared configuration changed")
        for name, sha in marker["files"].items():
            if file_hash(bundle/name) != sha: raise RuntimeError("Prepared data changed")
        return bundle
    bundle.mkdir(parents=True, exist_ok=True)
    ev, train, base_run = evaluation_config(cfg["base_evaluation_config"])
    original = Path(ev["bundle"])
    manifest = pq.read_table(original/"web/document_manifest.parquet").to_pandas()
    labels = {r["id"]: r for r in read_jsonl(original/"web/labels.jsonl")}
    indexed_manifest = manifest.set_index("id")
    backgrounds = [dict(r, source="background", corpus="background", url=indexed_manifest.loc[r["id"],"url"])
                   for r in pq.read_table(original/"web/documents.parquet").to_pylist() if not labels[r["id"]]["injected_math_source"]]
    assert len(backgrounds)==9000
    targets = read_jsonl(original/"target/train.jsonl")+read_jsonl(original/"target/validation.jsonl")
    preparation = yaml.safe_load(Path(train["data"]["preparation_config"]).read_text())
    for f in preparation["target"]["reserved_evaluation_files"]: targets.extend(read_jsonl(f))
    guard = TargetOverlap(targets, cfg["data"]["target_overlap"])
    candidates, crawl_counts = load_candidates(cfg)
    eligible, audit, quarantine = [], [], []
    for row in candidates:
        hits = guard.matches(row["text"]) if row["text"].strip() else []
        reason = ("empty" if not row["text"].strip() else "empty_source_formula" if row.get("empty_math_elements",0)>0
                  else "target_overlap" if hits else None)
        if hits: quarantine.append({"id": row["id"], "matches": hits})
        row["text_hash"] = digest(row["text"])
        if not reason: eligible.append(row)
        audit.append({k:v for k,v in row.items() if k != "text"}|{"status":reason or "eligible"})
    print("QUALITY_ELIGIBILITY", Counter(r["corpus"] for r in eligible), "target quarantined", len(quarantine), flush=True)
    # Background body/IDs remain fixed. Any new source connected to it by exact
    # text/URL or verified near-duplicate detection is quarantined before sampling.
    combined = [dict(r) for r in backgrounds]+eligible
    groups, edges = grouped_duplicates(combined,"text",cfg["data"]["near_duplicate_threshold"],cfg["data"]["minhash_seed"],"url")
    ambiguous = {r["id"] for g in groups if any(x["source"]=="background" for x in g)
                 for r in g if r["source"]!="background"}
    selected = [r for r in eligible if r["corpus"]=="openstax" and r["id"] not in ambiguous]
    pool = sorted([r for r in eligible if r["corpus"]=="finemath" and r["id"] not in ambiguous],key=lambda r:r["id"])
    if len(pool)<cfg["sources"]["finemath"]["size"]: raise RuntimeError("Insufficient matching-crawl FineMath pool")
    selected += random.Random(cfg["seed"]).sample(pool,cfg["sources"]["finemath"]["size"])
    chosen = {r["id"] for r in selected}
    for row in audit:
        if row["id"] in ambiguous: row["status"]="background_overlap"
        if row["id"] in chosen: row["status"]="selected"
    chapter_groups([r for r in selected if r["corpus"]=="openstax"])
    tokenizer=AutoTokenizer.from_pretrained(train["runtime_model_dir"],local_files_only=True)
    windows=[]
    for row in sorted(selected,key=lambda r:r["id"]):
        tokens=tokenizer.encode(row["text"],add_special_tokens=False)
        if len(tokens)<2: raise RuntimeError("Selected document is unscorable; freeze an explicit amendment before replacement")
        row.update(token_count=len(tokens),scored_tokens=len(tokens)-1,injected_math_source=True)
        windows.extend(document_windows(row["id"],tokens,cfg["data"]["window_length"]))
    # Pandas union-of-columns prevents Arrow from dropping source-specific fields.
    import pandas as pd
    pd.DataFrame(selected).to_parquet(bundle/"documents.parquet",index=False)
    pd.DataFrame([{k:v for k,v in r.items() if k!="text"} for r in selected]).to_parquet(bundle/"document_manifest.parquet",index=False)
    pq.write_table(pa.Table.from_pylist(windows),bundle/"windows.parquet")
    write_jsonl(bundle/"candidate_manifest.jsonl",audit)
    write_json(bundle/"target_overlap_quarantine.json",quarantine)
    write_json(bundle/"duplicate_edges.json",edges)
    write_json(bundle/"background_overlap_quarantine.json",sorted(ambiguous))
    selected_audit=[]
    for corpus in ["openstax","finemath"]:
        source=sorted([r for r in selected if r["corpus"]==corpus],key=lambda r:r["id"])
        sampled=random.Random(cfg["data"]["audit_seed"]).sample(source,min(len(source),cfg["data"]["audit_size_per_source"]))
        selected_audit.extend(sampled)
        with (bundle/f"audit_{corpus}.txt").open("w") as f:
            for r in sampled:f.write(json.dumps({k:v for k,v in r.items() if k!="text"},ensure_ascii=False)+"\n\n"+r["text"]+"\n\n"+"="*80+"\n")
    write_json(bundle/"audit_ids.json",[r["id"] for r in selected_audit])
    summary={"sources":{c:{"documents":len(s),"tokens":sum(r["token_count"] for r in s),
        "bootstrap_groups":len({r["group_id"] for r in s}),"mean_tokens":sum(r["token_count"] for r in s)/len(s)}
        for c in ["openstax","finemath"] if (s:=[r for r in selected if r["corpus"]==c])},
        "status_counts":dict(Counter(r["status"] for r in audit)),"finemath_crawl_counts":crawl_counts,
        "background_documents":9000,"windows":len(windows),"scored_tokens":sum(r["scored_tokens"] for r in selected),
        "openstax_formula_counts":dict(sum((Counter({k:r.get(k,0) for k in ["mathml_elements","formulas","empty_math_elements","images","images_without_description"]}) for r in selected if r["corpus"]=="openstax"),Counter())),
        "source_confounds":"OpenStax is editorial textbook content with MathML-to-LaTeX serialization and image descriptions, not an independent CC sample. FineMath uses upstream classifier filtering and a separate extraction pipeline; fixed shards are not a uniform sample of all FineMath. Background is unlabeled.",
        "prepared_at_utc":datetime.now(timezone.utc).isoformat()}
    write_json(bundle/"summary.json",summary)
    write_json(bundle/"COMPLETED.json",{"config_sha256":file_hash(path),"files":{p.name:file_hash(p) for p in bundle.iterdir() if p.name!="COMPLETED.json"}})
    print("QUALITY_PREPARED",json.dumps(summary,ensure_ascii=False),flush=True)
    return bundle

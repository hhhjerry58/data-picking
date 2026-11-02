"""Score new document sources with the existing reference/final math probe."""
from datetime import datetime, timezone
from pathlib import Path
import time

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer

from .common import digest, file_hash, read_json, seed_all, write_json
from .document_sanity import audit_documents
from .quality_data import settings
from .training import load_model
from .web_scoring import checkpoint_metadata, evaluation_config, load_staged_sft, score_batch


def freeze_evaluation(path):
    cfg, run, bundle = settings(path)
    ev, train, base_run = evaluation_config(cfg["base_evaluation_config"])
    complete=read_json(bundle/"COMPLETED.json")
    if complete["config_sha256"] != file_hash(path): raise RuntimeError("Data configuration changed")
    for name,sha in complete["files"].items():
        if file_hash(bundle/name)!=sha: raise RuntimeError("Prepared artifact changed")
    checkpoint=Path(read_json(base_run/"training_complete.json")["checkpoint"])
    meta=checkpoint_metadata(checkpoint)
    if meta["global_step"]!=cfg["checkpoint_step"]:raise RuntimeError("Wrong checkpoint")
    # A human-readable sample audit must be finished before examining new losses.
    audit=read_json(run/"quality_audit_review.json")
    if not audit["completed_before_scoring"]:raise RuntimeError("Quality audit pending")
    identity={"config":cfg,"data_marker_sha256":file_hash(bundle/"COMPLETED.json"),
        "background_scores_sha256":file_hash(base_run/"metrics_web/per_document.parquet"),
        "background_bundle_sha256":file_hash(Path(ev["bundle"])/"COMPLETED.json"),
        "checkpoint":str(checkpoint),"checkpoint_sha256":meta["files"]["state.pt"]["sha256"],
        "quality_audit_sha256":file_hash(run/"quality_audit_review.json")}
    plan={"identity":identity,"fingerprint":digest(identity),"frozen_at_utc":datetime.now(timezone.utc).isoformat()}
    p=run/"evaluation_plan.json"
    if p.exists():
        if read_json(p)["identity"]!=identity:raise RuntimeError("Frozen evaluation plan changed")
    else:write_json(p,plan)
    return read_json(p)


def score(path, model_label):
    cfg, run, bundle = settings(path)
    ev, train, base_run = evaluation_config(cfg["base_evaluation_config"])
    plan=read_json(run/"evaluation_plan.json")
    if plan["identity"]["config"]!=cfg or plan["identity"]["data_marker_sha256"]!=file_hash(bundle/"COMPLETED.json"):
        raise RuntimeError("Frozen scoring inputs changed")
    spec=cfg["scoring"]
    if spec["forward_dtype"]!="float32" or spec["attention_backend"]!="math":raise ValueError("FP32/math required")
    if model_label not in {"reference","sft"}:raise ValueError(model_label)
    marker=read_json(bundle/"COMPLETED.json")
    if file_hash(bundle/"windows.parquet")!=marker["files"]["windows.parquet"]:raise RuntimeError("Window checksum mismatch")
    rows=sorted(pq.read_table(bundle/"windows.parquet").to_pylist(),key=lambda r:(-len(r["input_ids"]),r["id"]))
    identity={"plan":plan["fingerprint"],"model":model_label,"torch_version":torch.__version__,
        "cuda_version":torch.version.cuda,"code":{n:file_hash(Path(__file__).with_name(n)) for n in
            ["quality_scoring.py","web_scoring.py","document_sanity.py","training.py","runtime_model.py","loss.py","tokenization.py"]}}
    fingerprint=digest(identity)
    folder=run/"scores"/model_label;folder.mkdir(parents=True,exist_ok=True)
    if (folder/"scoring_config.json").exists() and read_json(folder/"scoring_config.json")["fingerprint"]!=fingerprint:
        raise RuntimeError("Scoring implementation changed")
    write_json(folder/"scoring_config.json",{"identity":identity,"fingerprint":fingerprint})
    done=set()
    for p in folder.glob("chunk-*.complete.json"):
        record=read_json(p)
        if record["fingerprint"]!=fingerprint or file_hash(folder/record["file"])!=record["sha256"]:raise RuntimeError("Corrupt score chunk")
        done.add(record["start"])
    starts=set(range(0,len(rows),spec["chunk_size"]))
    if not done<=starts:raise RuntimeError("Unexpected scoring chunk")
    if done==starts and (folder/"sanity_document.json").exists():
        write_json(folder/"COMPLETED.json",{"fingerprint":fingerprint,"rows":len(rows),"model":model_label})
        return
    seed_all(train["training"]["seed"])
    tokenizer=AutoTokenizer.from_pretrained(train["runtime_model_dir"],local_files_only=True)
    model=load_model(train,train=False)
    if model_label=="sft":
        validation=load_staged_sft(model,plan["identity"]["checkpoint"],base_run)
        write_json(folder/"checkpoint_state_validation.json",validation)
    sanity=audit_documents(model,tokenizer,rows,rows,spec["batch_size"],spec["sanity_tolerance"])
    write_json(folder/"sanity_document.json",sanity)
    print("QUALITY_SANITY",model_label,sanity,flush=True)
    for start in range(0,len(rows),spec["chunk_size"]):
        if start in done:continue
        begin=time.monotonic();chunk=rows[start:start+spec["chunk_size"]]
        results=[r for i in range(0,len(chunk),spec["batch_size"]) for r in score_batch(model,tokenizer,chunk[i:i+spec["batch_size"]])]
        name=f"chunk-{start:06d}.parquet";pq.write_table(pa.Table.from_pylist(results),folder/name)
        write_json(folder/f"chunk-{start:06d}.complete.json",{"fingerprint":fingerprint,"start":start,"file":name,
            "sha256":file_hash(folder/name),"rows":len(results)})
        print("QUALITY_CHUNK",model_label,start+len(chunk),len(rows),"seconds",round(time.monotonic()-begin,2),flush=True)
    write_json(folder/"COMPLETED.json",{"fingerprint":fingerprint,"rows":len(rows),"model":model_label,
        "completed_at_utc":datetime.now(timezone.utc).isoformat()})

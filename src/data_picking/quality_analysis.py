"""Separate OpenStax/FineMath comparisons against the unchanged background."""
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .common import file_hash, read_json, write_json
from .quality_data import settings
from .web_analysis import aggregate_documents, pair_windows, retrieval_metrics, summarize_documents
from .web_scoring import evaluation_config


def load_scores(run, label):
    folder=run/"scores"/label
    completed=read_json(folder/"COMPLETED.json")
    parts=[]
    for path in sorted(folder.glob("chunk-*.complete.json")):
        record=read_json(path)
        if record["fingerprint"]!=completed["fingerprint"] or file_hash(folder/record["file"])!=record["sha256"]:
            raise RuntimeError("Corrupt score chunk")
        parts.append(pd.read_parquet(folder/record["file"]))
    frame=pd.concat(parts,ignore_index=True).sort_values("id").reset_index(drop=True)
    if len(frame)!=completed["rows"] or not frame.id.is_unique:raise RuntimeError("Missing/duplicate window scores")
    return frame


def match_lengths(frame, cfg):
    frame=frame.sort_values("id").reset_index(drop=True)
    bins=np.floor(np.log2(frame.token_count)).astype(int)
    groups=defaultdict(lambda:{0:[],1:[]})
    for i,(b,y) in enumerate(zip(bins,frame.injected_math_source)):groups[int(b)][int(y)].append(i)
    rng=np.random.default_rng(cfg["length_matching_seed"]);ratio=cfg["matched_background_per_math"]
    selected=[];details=[]
    for b,g in sorted(groups.items()):
        n=min(len(g[1]),len(g[0])//ratio)
        details.append({"token_lower":2**b,"token_upper_exclusive":2**(b+1),"math":len(g[1]),
                        "background":len(g[0]),"retained_math":n,"retained_background":ratio*n})
        for label,size in [(1,n),(0,ratio*n)]:
            if size:selected.extend(rng.choice(g[label],size,replace=False).tolist())
    matched=frame.iloc[sorted(selected)].copy().reset_index(drop=True)
    return matched,{"method":"fixed powers-of-two length bins; 1:9 per bin","bins":details,
        "documents":len(matched),"math":int(matched.injected_math_source.sum()),
        "retained_math_fraction":float(matched.injected_math_source.sum()/frame.injected_math_source.sum()),
        "bootstrap_groups":matched.groupby("source").group_id.nunique().to_dict()}


def analyze(path):
    cfg,run,bundle=settings(path)
    ev,train,base_run=evaluation_config(cfg["base_evaluation_config"])
    plan=read_json(run/"evaluation_plan.json")
    if file_hash(base_run/"metrics_web/per_document.parquet")!=plan["identity"]["background_scores_sha256"]:
        raise RuntimeError("Background scores changed")
    windows=pair_windows(load_scores(run,"reference"),load_scores(run,"sft"))
    prepared=pd.read_parquet(bundle/"windows.parquet",columns=["id","tokenization_hash","scored_tokens"]).sort_values("id").reset_index(drop=True)
    pd.testing.assert_frame_equal(windows[["id","tokenization_hash","scored_tokens"]],prepared)
    manifest=pd.read_parquet(bundle/"document_manifest.parquet")
    labels=manifest[["id","injected_math_source"]].copy()
    new=aggregate_documents(windows,manifest.drop(columns="injected_math_source"),labels)
    background=pd.read_parquet(base_run/"metrics_web/per_document.parquet")
    background=background.loc[background.source=="background"].copy()
    if len(background)!=9000:raise RuntimeError("Background pool changed")
    background["corpus"]="background"
    new.to_parquet(run/"new_per_document.parquet",index=False)
    out=run/"metrics";out.mkdir(exist_ok=True)
    results={};pools=[]
    columns=["id","group_id","source","corpus","token_count","scored_tokens","windows","injected_math_source",
             "ref_mean_nll","sft_mean_nll",*cfg["analysis"]["reported_loss_scores"],*cfg["analysis"]["raw_loss_baselines"]]
    for corpus in ["openstax","finemath"]:
        frame=pd.concat([new.loc[new.corpus==corpus,columns],background[columns]],ignore_index=True).sort_values("id").reset_index(drop=True)
        if not frame.id.is_unique:raise RuntimeError("Duplicate document IDs across sources")
        matched,matching=match_lengths(frame,cfg["analysis"])
        frozen_lengths=read_json(run/"length_control_plan.json")[corpus]
        if matched.id.tolist()!=frozen_lengths["ids"] or matching!=frozen_lengths["summary"]:
            raise RuntimeError("Length-matching sample differs from the pre-scoring plan")
        print("QUALITY_ANALYSIS",corpus,"positive",int(frame.injected_math_source.sum()),"matched",matching,flush=True)
        main=retrieval_metrics(frame,cfg["analysis"])
        main["bootstrap_unit"]=cfg["analysis"][f"{corpus}_bootstrap"]
        control=None
        if matching["math"]>=2 and matching["bootstrap_groups"].get("math",0)>=2:
            control=retrieval_metrics(matched,cfg["analysis"],1)
            control["bootstrap_unit"]=main["bootstrap_unit"]
        results[corpus]={"main":main,"length_matched":control,"length_matching":matching,
            "document_summary":summarize_documents(frame),"relative_gain_denominator_clamps":int((frame.ref_mean_nll<1e-6).sum())}
        frame["comparison"]=corpus;pools.append(frame)
        write_json(out/f"{corpus}_length_matched_ids.json",matched.id.tolist())
        write_json(out/f"{corpus}_metrics.json",results[corpus])
        print("QUALITY_RESULT",corpus,{k:main["scores"][k]["roc_auc"] for k in cfg["analysis"]["reported_loss_scores"]},flush=True)
    pd.concat(pools,ignore_index=True).to_parquet(out/"per_document.parquet",index=False)
    new[[c for c in new.columns if c not in {"ref_sum_nll","sft_sum_nll"}]].to_parquet(out/"new_source_metadata.parquet",index=False)
    metrics={"analysis_type":"document_quality_sources","run_id":cfg["run_id"],"comparisons":results,
        "reported_loss_scores":cfg["analysis"]["reported_loss_scores"],"checkpoint_step":cfg["checkpoint_step"],
        "dataset_summary":read_json(bundle/"summary.json"),"quality_audit":read_json(run/"quality_audit_review.json"),
        "sanity":{label:read_json(run/"scores"/label/"sanity_document.json") for label in ["reference","sft"]},
        "original_web_reference":read_json(base_run/"metrics_web/metrics.json")["main"],
        "optimizer_updates":0,"new_gradient_measurements":False,"plan_fingerprint":plan["fingerprint"],
        "analysis_code_sha256":file_hash(__file__),"completed_at_utc":datetime.now(timezone.utc).isoformat()}
    write_json(out/"metrics.json",metrics)
    write_json(out/"COMPLETED.json",{"files":{p.name:file_hash(p) for p in out.iterdir() if p.name!="COMPLETED.json"}})
    return metrics

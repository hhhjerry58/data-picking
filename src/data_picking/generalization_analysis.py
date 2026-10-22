"""Paired loss/gradient analysis for fixed target and general-domain examples."""
from collections import defaultdict
from datetime import datetime, timezone
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

from .analysis import auc, precision_at
from .common import file_hash, read_json, write_json
from .generalization_data import load_extension
from .loss_review import match_a2_c_lengths


LOSS_SCORES = ["delta_loss", "relative_gain", "neg_ref_loss", "neg_sft_loss", "pos_ref_loss", "pos_sft_loss"]
GRADIENT_SCORES = ["ref_grad_l2", "sft_grad_l2", "grad_l2_delta", "ref_grad_max_abs", "sft_grad_max_abs",
                   "grad_max_abs_delta", "ref_cos_to_a1", "sft_cos_to_a1"]
COMPARISONS = [("A_vs_B", "A", "B"), ("A_vs_C", "A", "C"), ("B_vs_C", "B", "C")]


def load_measurements(run, label, gradient_directory="gradients"):
    folder = run / gradient_directory / label
    completed = read_json(folder / "COMPLETED.json")
    by_kind = {key: [] for key in ["global", "layers", "parameters"]}
    maximum_difference = 0.0
    for path in sorted(folder.glob("chunk-*.complete.json")):
        marker = read_json(path)
        if marker["fingerprint"] != completed["fingerprint"]:
            raise RuntimeError("Chunk has a different measurement fingerprint")
        maximum_difference = max(maximum_difference, marker["max_old_loss_difference"])
        for filename, sha in marker["files"].items():
            if file_hash(folder / filename) != sha:
                raise RuntimeError(f"Chunk hash mismatch: {filename}")
            kind = filename.split(".")[-2]
            by_kind[kind].append(pq.read_table(folder / filename).to_pandas())
    frames = {key: pd.concat(parts, ignore_index=True) for key,parts in by_kind.items()}
    if not frames["global"]["id"].is_unique or len(frames["global"]) != completed["examples"]:
        raise RuntimeError("Missing or duplicate measured examples")
    return frames, maximum_difference


def match_three_domains(frame, seed):
    edges = [np.unique(np.quantile(frame[c], [.25,.5,.75])) for c in ["prompt_tokens", "response_tokens"]]
    bins = np.column_stack([np.searchsorted(e, frame[c], side="right") for e,c in zip(edges,["prompt_tokens", "response_tokens"])])
    buckets = defaultdict(lambda: {"A": [], "B": [], "C": []})
    for i, (domain, group) in enumerate(zip(frame["domain"], bins)):
        buckets[tuple(group)][domain].append(i)
    rng = np.random.default_rng(seed)
    selected, details = [], []
    for key in sorted(buckets):
        counts = {d:len(buckets[key][d]) for d in ["A","B","C"]}
        n = min(counts.values())
        details.append({"bin": [int(x) for x in key], "counts": counts, "retained_per_domain": n})
        if n:
            for domain in ["A","B","C"]:
                selected.extend(rng.choice(buckets[key][domain], n, replace=False).tolist())
    matched = frame.iloc[sorted(selected)].copy().reset_index(drop=True)
    return matched, {"quartile_edges": [e.tolist() for e in edges], "retained_fraction": len(matched)/len(frame),
                     "retained_counts": {d:int((matched['domain']==d).sum()) for d in ["A","B","C"]}, "bins": details}


def evaluate_pair(frame, positive, negative, cfg, seed_offset=0):
    frame = frame.loc[frame["domain"].isin([positive,negative])].reset_index(drop=True)
    y = (frame["domain"] == positive).to_numpy(dtype=int)
    if len(set(y)) != 2:
        raise ValueError("Two nonempty domains are required")
    keys = LOSS_SCORES + GRADIENT_SCORES
    scores = {key:frame[key].to_numpy(dtype=float) for key in keys}
    if not all(np.isfinite(s).all() for s in scores.values()):
        raise ValueError("Non-finite score; no silent sample dropping")
    seed = cfg["analysis"]["bootstrap_seed"] + seed_offset
    ties = np.random.default_rng(seed+100).random(len(frame))
    result = {}
    for key,s in scores.items():
        result[key] = {"roc_auc": auc(y,s)}
        if key in LOSS_SCORES:
            result[key].update(average_precision=float(average_precision_score(y,s)),
                               precision_top_10pct=precision_at(y,s,.1,ties), precision_top_20pct=precision_at(y,s,.2,ties))
    groups = {label: [g.index.to_numpy() for _,g in frame.loc[y==label].groupby("group_id",sort=True)] for label in [0,1]}
    unique_groups = all(len(g)==1 for blocks in groups.values() for g in blocks)
    rng = np.random.default_rng(seed)
    n_boot = cfg["analysis"]["bootstrap_samples"]
    bootstrap_auc = {key:[] for key in keys}
    bootstrap_precision = {key:{p:[] for p in [.1,.2]} for key in LOSS_SCORES}
    for _ in range(n_boot):
        pieces = []
        for label in [0,1]:
            blocks = groups[label]
            if unique_groups:
                candidates = np.concatenate(blocks)
                pieces.append(rng.choice(candidates, len(candidates), replace=True))
            else:
                pieces.append(np.concatenate([blocks[i] for i in rng.integers(len(blocks),size=len(blocks))]))
        index = np.concatenate(pieces)
        for key,s in scores.items():
            bootstrap_auc[key].append(auc(y[index],s[index]))
            if key in LOSS_SCORES:
                for fraction in [.1,.2]:
                    bootstrap_precision[key][fraction].append(precision_at(y[index],s[index],fraction,ties[index]))
    for key in keys:
        result[key]["roc_auc_ci95"] = np.quantile(bootstrap_auc[key],[.025,.975]).tolist()
        if key in LOSS_SCORES:
            for fraction,label in [(.1,"10"),(.2,"20")]:
                result[key][f"precision_top_{label}pct_ci95"] = np.quantile(bootstrap_precision[key][fraction],[.025,.975]).tolist()
    differences = {}
    for baseline in ["neg_ref_loss","neg_sft_loss"]:
        values = np.array(bootstrap_auc["delta_loss"]) - np.array(bootstrap_auc[baseline])
        differences[baseline] = {"difference": result["delta_loss"]["roc_auc"]-result[baseline]["roc_auc"],
                                 "ci95": np.quantile(values,[.025,.975]).tolist()}
    return {"positive_domain":positive,"negative_domain":negative,"examples":len(frame),
            "counts": {d:int((frame['domain']==d).sum()) for d in [positive,negative]},
            "question_group_counts": {d:int(frame.loc[frame['domain']==d,'group_id'].nunique()) for d in [positive,negative]},
            "loss_scores": {key:result[key] for key in LOSS_SCORES},
            "gradient_scores": {key:result[key] for key in GRADIENT_SCORES},
            "paired_delta_auc_minus_baseline": differences,
            "length_baselines": {key:{"positive_auc":auc(y,frame[key].to_numpy()),"negative_auc":auc(y,-frame[key].to_numpy())}
                                 for key in ["prompt_tokens","response_tokens"]},
            "bootstrap_samples":n_boot,"bootstrap_seed":seed,"tie_seed":seed+100,"bootstrap_unit":"group_id, stratified by domain",
            "gradient_direction":"Higher raw feature predicts positive domain; exploratory; no sign flipping"}


def summarize_groups(frame):
    columns = ["ref_mean_nll","sft_mean_nll","delta_loss","relative_gain","prompt_tokens","response_tokens",
               "ref_grad_l2","sft_grad_l2","grad_l2_delta","grad_l2_relative_drop","ref_grad_max_abs","sft_grad_max_abs",
               "ref_cos_to_a1","sft_cos_to_a1"]
    result = {}
    for domain, group in frame.groupby("domain"):
        result[domain] = {"examples":len(group),"positive_delta_fraction":float((group['delta_loss']>0).mean())}
        for key in columns:
            v=group[key].to_numpy()
            result[domain][key] = {"mean":float(v.mean()),"median":float(np.median(v)),"q10_q25_q75_q90":np.quantile(v,[.1,.25,.75,.9]).tolist()}
    return result


def analyze_extension(config_path):
    cfg, base, run = load_extension(config_path)
    ref, ref_error = load_measurements(run,"reference",cfg.get("gradient_directory","gradients"))
    sft, sft_error = load_measurements(run,"sft",cfg.get("gradient_directory","gradients"))
    metadata = ["id","group_id","domain","split","level","text_hash","tokenization_hash","prompt_tokens","response_tokens","source","source_revision"]
    a = ref["global"].sort_values("id").reset_index(drop=True)
    b = sft["global"].sort_values("id").reset_index(drop=True)
    pd.testing.assert_frame_equal(a[metadata],b[metadata])
    paired = a[metadata].copy()
    numeric = ["mean_nll","sum_nll","grad_l2","grad_rms","grad_max_abs","cos_to_a1","max_parameter_l2"]
    for prefix, frame in [("ref",a),("sft",b)]:
        for key in numeric + ["grad_max_abs_parameter","max_parameter_l2_name"]:
            paired[f"{prefix}_{key}"] = frame[key]
    paired["delta_loss"] = paired["ref_mean_nll"]-paired["sft_mean_nll"]
    paired["relative_gain"] = paired["delta_loss"]/np.maximum(paired["ref_mean_nll"],1e-6)
    for prefix in ["ref","sft"]:
        paired[f"neg_{prefix}_loss"] = -paired[f"{prefix}_mean_nll"]
        paired[f"pos_{prefix}_loss"] = paired[f"{prefix}_mean_nll"]
    paired["grad_l2_delta"] = paired["ref_grad_l2"]-paired["sft_grad_l2"]
    paired["grad_l2_relative_drop"] = paired["grad_l2_delta"]/paired["ref_grad_l2"]
    paired["grad_max_abs_delta"] = paired["ref_grad_max_abs"]-paired["sft_grad_max_abs"]
    domains = cfg.get("evaluation_domains", ["A", "B", "C"])
    if set(paired["domain"]) != set(domains):
        raise RuntimeError("Measured domains differ from the frozen evaluation configuration")
    matching_function = match_a2_c_lengths if domains == ["A", "C"] else match_three_domains
    matched, matching = matching_function(paired,cfg["analysis"]["length_matching_seed"])
    if min(matching["retained_counts"].values())<2:
        raise RuntimeError("Insufficient length overlap for the planned control")
    comparisons = {}
    for i,(name,positive,negative) in enumerate(COMPARISONS):
        if not {positive, negative} <= set(domains):
            continue
        print("ANALYZING",name,"raw",flush=True)
        raw_metrics = evaluate_pair(paired,positive,negative,cfg,seed_offset=10*i)
        print("ANALYZING",name,"length_matched",flush=True)
        matched_metrics = evaluate_pair(matched,positive,negative,cfg,seed_offset=10*i+1)
        comparisons[name] = {"raw":raw_metrics,"length_matched":matched_metrics}
    duplicates = read_json(run / "data/duplicate_report.json")
    overlap_ids = duplicates["selected_overlap_ids"]
    overlap_sensitivity = {"excluded_c_ids": overlap_ids, "status": "no_normalized_question_overlap",
                           "comparisons": {}}
    if overlap_ids:
        overlap_sensitivity["status"] = "reported_separately_main_sample_unchanged"
        for i, (name, positive, negative) in enumerate(COMPARISONS):
            if "C" not in (positive, negative) or name not in comparisons:
                continue
            pools = {}
            for pool_name, pool in [("raw", paired), ("length_matched", matched)]:
                clean = pool.loc[~pool["id"].isin(overlap_ids)]
                counts = clean.loc[clean["domain"].isin([positive, negative])].groupby("domain").size()
                pools[pool_name] = (evaluate_pair(clean, positive, negative, cfg, seed_offset=100+10*i)
                                    if all(counts.get(d, 0) >= 2 for d in [positive, negative])
                                    else {"status": "insufficient_examples"})
            overlap_sensitivity["comparisons"][name] = pools
    layers = pd.concat([ref["layers"],sft["layers"]],ignore_index=True)
    parameter_frames = pd.concat([ref["parameters"],sft["parameters"]],ignore_index=True)
    parameter_summary = parameter_frames.groupby(["model","domain","parameter","layer"],sort=True).agg(
        examples=("id","size"),numel=("numel","first"),grad_l2_mean=("grad_l2","mean"),grad_l2_median=("grad_l2","median"),
        grad_rms_median=("grad_rms","median"),grad_max_abs_median=("grad_max_abs","median"),grad_max_abs_max=("grad_max_abs","max")).reset_index()
    correlations = []
    for domain, group in paired.groupby("domain"):
        for prefix in ["ref","sft"]:
            for feature in [f"{prefix}_mean_nll","prompt_tokens","response_tokens"]:
                correlations.append({"domain":domain,"model":prefix,"feature":feature,
                                     "spearman_with_grad_l2":float(spearmanr(group[feature],group[f"{prefix}_grad_l2"]).statistic)})
    anchors = {label:read_json(run/cfg.get("gradient_directory","gradients")/label/"anchor.json") for label in ["reference","sft"]}
    metrics = {"analysis_type":cfg["c_data"].get("kind", "dolly")+"_gradient_extension",
               "c_display_name":cfg["c_data"].get("display_name", "Dolly"),
               "run_id":cfg["run_id"],"base_run":cfg["base_run"],
               "analysis_implementation_sha256":file_hash(__file__),
               "primary_comparison":"A_vs_C","primary_score":"delta_loss","comparisons":comparisons,
               "length_matching":matching,"group_summary":summarize_groups(paired),"matched_group_summary":summarize_groups(matched),
               "anchor_summary":anchors,"gradient_loss_length_correlations":correlations,
               "duplicate_report":duplicates,"overlap_sensitivity":overlap_sensitivity,
               "crosscheck_original_max_nll_difference":{"reference":ref_error,"sft":sft_error},
               "parameter_count":int(a['parameter_count'].iloc[0]),"gradient_clipping":False,"optimizer_updates":0,
               "relative_gain_denominator_clamps":int((paired['ref_mean_nll']<1e-6).sum()),
               "completed_at_utc":datetime.now(timezone.utc).isoformat()}
    directory = run/"metrics"
    directory.mkdir(exist_ok=True)
    paired.to_parquet(directory/"per_example.parquet",index=False)
    layers.to_parquet(directory/"per_layer.parquet",index=False)
    parameter_summary.to_parquet(directory/"parameter_summary.parquet",index=False)
    write_json(directory/"length_matched_ids.json",matched['id'].tolist())
    write_json(directory/"metrics.json",metrics)
    effective = read_json(Path(cfg["base_run"])/"effective_config.json")
    effective["source_training_run"] = cfg["base_run"]
    effective["evaluation_only"] = True
    effective["extension_config"] = cfg
    effective["config"]["run_id"] = cfg["run_id"]
    effective["config"]["c_data"] = cfg["c_data"]
    effective["config"]["analysis"] = cfg["analysis"]
    write_json(run/"effective_config.json",effective)
    print("EXTENSION_ANALYSIS_COMPLETE",directory,flush=True)
    for name,value in comparisons.items():
        print(name,"delta AUC",value["raw"]["loss_scores"]["delta_loss"]["roc_auc"],"matched",value["length_matched"]["loss_scores"]["delta_loss"]["roc_auc"],flush=True)
    return metrics

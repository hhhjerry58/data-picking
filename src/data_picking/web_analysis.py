"""Document-weighted retrieval analysis for the frozen math/CC injection pool."""
from collections import defaultdict
from datetime import datetime, timezone
import math
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score

from .analysis import auc
from .common import file_hash, read_json, read_jsonl, write_json
from .web_scoring import evaluation_config


def load_score_set(run, label, name, ev):
    frames, validation = [], []
    for shard in range(ev["scoring"]["num_shards"]):
        folder = run/"scores_web"/label/name/f"shard-{shard:02d}-of-{ev['scoring']['num_shards']:02d}"
        done = read_json(folder/"COMPLETED.json")
        parts = []
        for marker in sorted(folder.glob("chunk-*.complete.json")):
            record = read_json(marker)
            if record["fingerprint"] != done["fingerprint"] or file_hash(folder/record["file"]) != record["sha256"]:
                raise RuntimeError("Corrupt score chunk")
            parts.append(pq.read_table(folder/record["file"]).to_pandas())
        frame = pd.concat(parts, ignore_index=True)
        if len(frame) != done["rows"] or not frame["id"].is_unique: raise RuntimeError("Missing/duplicate scored examples")
        frames.append(frame)
        sanity_file=folder/"sanity_document.json" if name=="web" and (folder/"sanity_document.json").exists() else folder/"sanity.json"
        validation.append({"shard": shard, **read_json(sanity_file)})
    result = pd.concat(frames, ignore_index=True).sort_values("id").reset_index(drop=True)
    if not result["id"].is_unique: raise RuntimeError("Duplicate examples across score shards")
    return result, validation


def pair_windows(reference, sft):
    a = reference.sort_values("id").reset_index(drop=True)
    b = sft.sort_values("id").reset_index(drop=True)
    metadata = ["id", "document_id", "window_index", "token_start", "token_end", "tokenization_hash", "scored_tokens"]
    pd.testing.assert_frame_equal(a[metadata], b[metadata])
    result = a[metadata].copy()
    result["ref_sum_nll"] = a["sum_nll"]
    result["sft_sum_nll"] = b["sum_nll"]
    return result


def aggregate_documents(windows, manifest, labels):
    aggregated = windows.groupby("document_id", sort=True).agg(
        ref_sum_nll=("ref_sum_nll", "sum"), sft_sum_nll=("sft_sum_nll", "sum"),
        scored_tokens=("scored_tokens", "sum"), windows=("id", "size")).reset_index().rename(columns={"document_id": "id"})
    frame = manifest.merge(aggregated, on="id", validate="one_to_one", suffixes=("_prepared", ""))
    if len(frame) != len(manifest) or (frame["scored_tokens"] != frame["scored_tokens_prepared"]).any():
        raise RuntimeError("Incomplete document token coverage")
    frame = frame.merge(labels, on="id", validate="one_to_one")
    if frame["injected_math_source"].isna().any(): raise RuntimeError("Missing source label")
    frame["ref_mean_nll"] = frame["ref_sum_nll"]/frame["scored_tokens"]
    frame["sft_mean_nll"] = frame["sft_sum_nll"]/frame["scored_tokens"]
    frame["delta_loss"] = frame["ref_mean_nll"]-frame["sft_mean_nll"]
    frame["relative_gain"] = frame["delta_loss"]/np.maximum(frame["ref_mean_nll"], 1e-6)
    for prefix in ["ref", "sft"]:
        frame[f"neg_{prefix}_loss"] = -frame[f"{prefix}_mean_nll"]
        frame[f"pos_{prefix}_loss"] = frame[f"{prefix}_mean_nll"]
    frame["hostname"] = frame["url"].map(lambda url: urlsplit(url).hostname or "unknown")
    return frame.sort_values("id").reset_index(drop=True)


def match_document_lengths(frame, config):
    edges = np.unique(np.quantile(frame["token_count"], config["length_quantiles"]))
    bins = np.searchsorted(edges, frame["token_count"], side="right")
    buckets = defaultdict(lambda: {0: [], 1: []})
    for index, (bucket, positive) in enumerate(zip(bins, frame["injected_math_source"])):
        buckets[int(bucket)][int(positive)].append(index)
    ratio = config["matched_background_per_math"]
    rng = np.random.default_rng(config["length_matching_seed"])
    selected, details = [], []
    for bucket, groups in sorted(buckets.items()):
        n = min(len(groups[1]), len(groups[0])//ratio)
        details.append({"bin": bucket, "math": len(groups[1]), "background": len(groups[0]), "retained_math": n, "retained_background": ratio*n})
        if n:
            selected.extend(rng.choice(groups[1], n, replace=False).tolist())
            selected.extend(rng.choice(groups[0], ratio*n, replace=False).tolist())
    matched = frame.iloc[sorted(selected)].copy().reset_index(drop=True)
    if int(matched["injected_math_source"].sum()) < 2: raise RuntimeError("Insufficient length-matched source overlap")
    return matched, {"quantile_edges": edges.tolist(), "background_per_math": ratio, "bins": details,
                     "retained_documents": len(matched), "retained_math": int(matched["injected_math_source"].sum()),
                     "retained_fraction": len(matched)/len(frame)}


def ranking_points(y, score, ties, fractions):
    order = np.lexsort((ties, -score))
    cumulative = np.cumsum(y[order])
    prevalence = float(y.mean())
    points = {}
    for fraction in fractions:
        k = max(1, math.ceil(len(y)*fraction))
        positives = int(cumulative[k-1])
        points[str(fraction)] = {"k": k, "retrieved_math_source": positives,
            "source_precision": positives/k, "source_recall": positives/int(y.sum()),
            "source_enrichment": (positives/k)/prevalence}
    return points


def retrieval_metrics(frame, config, seed_offset=0):
    frame = frame.reset_index(drop=True)
    y = frame["injected_math_source"].to_numpy(dtype=int)
    if set(y) != {0, 1}: raise ValueError("Both source classes are required")
    keys = config["reported_loss_scores"]+config["raw_loss_baselines"]
    values = {key: frame[key].to_numpy(dtype=float) for key in keys}
    if not all(np.isfinite(v).all() for v in values.values()): raise ValueError("Non-finite score; no silent sample removal")
    if frame.groupby("group_id")["injected_math_source"].nunique().max() > 1:
        raise ValueError("Cross-source duplicate groups require a separately defined evaluation")
    seed = config["bootstrap_seed"]+seed_offset
    ties = np.random.default_rng(config["tie_seed"]+seed_offset).random(len(frame))
    fractions = config["top_fractions"]
    results = {key: {"roc_auc": auc(y, s), "average_precision": float(average_precision_score(y, s)),
                     "top": ranking_points(y, s, ties, fractions)} for key, s in values.items()}
    groups = {label: [g.index.to_numpy() for _, g in frame.loc[y == label].groupby("group_id", sort=True)] for label in [0, 1]}
    singleton = all(len(g) == 1 for blocks in groups.values() for g in blocks)
    source_indexes = {label: np.concatenate(groups[label]) for label in [0, 1]}
    rng = np.random.default_rng(seed)
    samples = {key: {"auc": [], "top": {str(f): {stat: [] for stat in ["source_precision", "source_recall", "source_enrichment"]} for f in fractions}} for key in keys}
    for iteration in range(config["bootstrap_samples"]):
        index = np.concatenate([rng.choice(source_indexes[label], len(source_indexes[label]), replace=True) if singleton else
                  np.concatenate([groups[label][j] for j in rng.integers(len(groups[label]), size=len(groups[label]))]) for label in [0, 1]])
        for key, score in values.items():
            samples[key]["auc"].append(auc(y[index], score[index]))
            top = ranking_points(y[index], score[index], ties[index], fractions)
            for f, stats in samples[key]["top"].items():
                for stat in stats: stats[stat].append(top[f][stat])
        if (iteration+1) % 500 == 0: print("RETRIEVAL_BOOTSTRAP", len(frame), iteration+1, flush=True)
    for key in keys:
        results[key]["roc_auc_ci95"] = np.quantile(samples[key]["auc"], [.025, .975]).tolist()
        for f, stats in samples[key]["top"].items():
            for stat, draws in stats.items(): results[key]["top"][f][stat+"_ci95"] = np.quantile(draws, [.025, .975]).tolist()
    paired = {}
    for key in config["reported_loss_scores"]:
        paired[key] = {}
        for baseline in config["raw_loss_baselines"]:
            draws = np.array(samples[key]["auc"])-np.array(samples[baseline]["auc"])
            paired[key][baseline] = {"difference": results[key]["roc_auc"]-results[baseline]["roc_auc"],
                                     "ci95": np.quantile(draws, [.025, .975]).tolist()}
    return {"documents": len(frame), "math_source": int(y.sum()), "unlabeled_background": int((1-y).sum()),
            "source_prevalence": float(y.mean()), "scores": results, "paired_auc_differences": paired,
            "length_baselines": {sign: auc(y, mul*frame["token_count"].to_numpy()) for sign, mul in [("positive", 1), ("negative", -1)]},
            "bootstrap_samples": config["bootstrap_samples"], "bootstrap_unit": "duplicate document group, stratified by source",
            "bootstrap_seed": seed, "tie_seed": config["tie_seed"]+seed_offset}


def summarize_documents(frame):
    return {source: {"documents": len(group), "positive_delta_fraction": float((group["delta_loss"] > 0).mean()),
            **{key: {"mean": float(group[key].mean()), "median": float(group[key].median()),
                       "q10_q90": group[key].quantile([.1, .9]).tolist()} for key in
              ["ref_mean_nll", "sft_mean_nll", "delta_loss", "relative_gain", "token_count"]}}
            for source, group in frame.groupby("source")}


def prepare_gradient_snapshots(run):
    plan = read_json(run/"gradient_diagnostic_plan.json")
    scores = pd.read_parquet(run/"metrics_web/per_document.parquet").set_index("id")
    frames = {kind: [] for kind in ["global", "layers", "parameters"]}
    anchors, errors = {}, {}
    for label, prefix in [("reference", "ref"), ("sft", "sft")]:
        folder = run/"gradients_web"/label
        completed = read_json(folder/"COMPLETED.json")
        pieces = {kind: [] for kind in frames}
        for marker in sorted(folder.glob("chunk-*.complete.json")):
            record = read_json(marker)
            if record["fingerprint"] != completed["fingerprint"]: raise RuntimeError("Gradient chunk identity mismatch")
            for name, sha in record["files"].items():
                if file_hash(folder/name) != sha: raise RuntimeError("Gradient checksum mismatch")
                pieces[name.split('.')[-2]].append(pd.read_parquet(folder/name))
        combined = {k: pd.concat(v, ignore_index=True) for k, v in pieces.items()}
        measured = combined["global"].set_index("id")
        if not measured.index.is_unique or set(measured.index) != set(plan["ids"]): raise RuntimeError("Gradient sample changed")
        target = scores.loc[measured.index]
        if not np.array_equal(target["scored_tokens"], measured["scored_tokens"]): raise RuntimeError("Gradient token count changed")
        difference = float((measured["mean_nll"]-target[f"{prefix}_mean_nll"]).abs().max())
        if difference > plan["config"]["loss_crosscheck_tolerance"]: raise RuntimeError("Document gradient/score NLL mismatch")
        errors[label] = difference
        anchors[label] = read_json(folder/"anchor.json")
        for kind in frames: frames[kind].append(combined[kind])
    all_frames = {k: pd.concat(v, ignore_index=True) for k, v in frames.items()}
    folder = run/"metrics_web"
    all_frames["global"].to_parquet(folder/"gradient_per_example.parquet", index=False)
    all_frames["layers"].to_parquet(folder/"gradient_per_layer.parquet", index=False)
    parameters = all_frames["parameters"].groupby(["model", "source", "parameter", "layer"]).agg(
        examples=("id", "size"), grad_l2_median=("grad_l2", "median"), grad_rms_median=("grad_rms", "median"),
        grad_max_abs_median=("grad_max_abs", "median")).reset_index()
    parameters.to_parquet(folder/"gradient_parameter_summary.parquet", index=False)
    write_json(folder/"gradient_summary.json", {"plan": plan, "anchors": anchors, "max_nll_crosscheck_error": errors,
                                               "documents_per_model": len(plan["ids"]), "optimizer_updates": 0})


def analyze_web(config_path):
    ev, cfg, run = evaluation_config(config_path)
    final = f"step-{ev['final_checkpoint_step']:06d}"
    ref, ref_sanity = load_score_set(run, "reference", "web", ev)
    sft, sft_sanity = load_score_set(run, final, "web", ev)
    windows = pair_windows(ref, sft)
    bundle = Path(ev["bundle"])
    prepared = pq.read_table(bundle/"web/windows.parquet", columns=["id", "tokenization_hash", "scored_tokens"]).to_pandas().sort_values("id").reset_index(drop=True)
    pd.testing.assert_frame_equal(windows[["id", "tokenization_hash", "scored_tokens"]], prepared)
    manifest = pq.read_table(bundle/"web/document_manifest.parquet").to_pandas()
    labels = pd.DataFrame(read_jsonl(bundle/"web/labels.jsonl"))
    frame = aggregate_documents(windows, manifest, labels)
    if len(frame) != 10000 or int(frame["injected_math_source"].sum()) != 1000:
        raise RuntimeError("The registered 1000/9000 pool changed")
    matched, matching = match_document_lengths(frame, ev["analysis"])
    main = retrieval_metrics(frame, ev["analysis"])
    control = retrieval_metrics(matched, ev["analysis"], seed_offset=1)
    baseline, val_sanity = load_score_set(run, "reference", "validation", ev)
    target_frames, trends = [], []
    for step in [0, *ev["diagnostic_validation_steps"], ev["final_checkpoint_step"]]:
        scored = baseline.copy() if step == 0 else load_score_set(run, f"step-{step:06d}", "validation", ev)[0]
        pd.testing.assert_frame_equal(baseline[["id", "tokenization_hash", "scored_tokens"]], scored[["id", "tokenization_hash", "scored_tokens"]])
        scored["checkpoint_step"] = step
        target_frames.append(scored)
        trends.append({"checkpoint_step": step, "examples": len(scored), "mean_nll": float(scored["mean_nll"].mean()),
                       "mean_drop_from_ref": float((baseline["mean_nll"]-scored["mean_nll"]).mean()),
                       "fraction_improved": float((baseline["mean_nll"] > scored["mean_nll"]).mean())})
    metrics = {"analysis_type": "web_document_retrieval", "run_id": cfg["run_id"], "checkpoint_step": ev["final_checkpoint_step"],
               "reported_loss_scores": ev["analysis"]["reported_loss_scores"], "main": main, "length_matched": control,
               "length_matching": matching, "document_summary": summarize_documents(frame), "target_validation": trends,
               "score_sanity": {"reference_web": ref_sanity, "sft_web": sft_sanity, "reference_validation": val_sanity},
               "relative_gain_denominator_clamps": int((frame["ref_mean_nll"] < 1e-6).sum()),
               "scored_tokens": int(frame["scored_tokens"].sum()), "windows": len(windows),
               "label_policy": ev["label_policy"], "checkpoint_rule": "Final step 64 fixed before evaluation; intermediate target losses are diagnostic",
               "general_sft_control_trained": False, "document_selection_training_tested": False,
               "analysis_sha256": file_hash(__file__), "completed_at_utc": datetime.now(timezone.utc).isoformat()}
    folder = run/"metrics_web"; folder.mkdir(exist_ok=True)
    keep = ["id", "group_id", "source", "hostname", "url", "token_count", "scored_tokens", "windows", "injected_math_source",
            "ref_mean_nll", "sft_mean_nll", *ev["analysis"]["reported_loss_scores"], *ev["analysis"]["raw_loss_baselines"]]
    frame[keep].to_parquet(folder/"per_document.parquet", index=False)
    windows.to_parquet(folder/"per_window.parquet", index=False)
    pd.concat(target_frames, ignore_index=True).to_parquet(folder/"target_validation.parquet", index=False)
    write_json(folder/"length_matched_ids.json", matched["id"].tolist())
    write_json(folder/"metrics.json", metrics)
    for key in ev["analysis"]["reported_loss_scores"]:
        print("WEB_RETRIEVAL_RESULT", key, main["scores"][key], flush=True)
    return metrics

"""Analyze fixed-pool diagnostics without changing the registered final result."""
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import auc
from .common import digest, file_hash, read_json, read_jsonl, write_json
from .probe_diagnostics import QA_METADATA, freeze, pair_qa, qa_rows, settings
from .web_analysis import (aggregate_documents, load_score_set, pair_windows,
                           retrieval_metrics, summarize_documents)


def load_qa(run, label):
    folder = run/"scores_qa"/label
    completed = read_json(folder/"COMPLETED.json")
    frames = []
    for path in sorted(folder.glob("chunk-*.complete.json")):
        record = read_json(path)
        if record["fingerprint"] != completed["fingerprint"] or file_hash(folder/record["file"]) != record["sha256"]:
            raise RuntimeError("Corrupt QA scores")
        frame = pd.read_parquet(folder/record["file"])
        if len(frame) != record["rows"]:
            raise RuntimeError("Incomplete QA chunk")
        frames.append(frame)
    frame = pd.concat(frames, ignore_index=True).sort_values("id").reset_index(drop=True)
    if not frame.id.is_unique or len(frame) != completed["rows"]:
        raise RuntimeError("Missing/duplicate QA examples")
    return frame, read_json(folder/"sanity.json")


def fixed_subset(frame, ids):
    if not frame.id.is_unique or len(ids) != len(set(ids)) or not set(ids) <= set(frame.id):
        raise ValueError("Invalid fixed length-matching IDs")
    return frame.set_index("id", drop=False).loc[ids].reset_index(drop=True)


def compare_auc(before, after, keys, samples, seed):
    """Paired, source-stratified duplicate-group bootstrap of AUC changes."""
    a, b = [f.sort_values("id").reset_index(drop=True) for f in [before, after]]
    pd.testing.assert_frame_equal(a[["id", "group_id", "injected_math_source"]],
                                  b[["id", "group_id", "injected_math_source"]])
    y = a.injected_math_source.to_numpy(dtype=int)
    groups = {c: [g.index.to_numpy() for _, g in a.loc[y == c].groupby("group_id", sort=True)] for c in [0, 1]}
    if a.groupby("group_id").injected_math_source.nunique().max() > 1:
        raise ValueError("Duplicate group crosses classes")
    singleton = all(len(g) == 1 for blocks in groups.values() for g in blocks)
    indexes = {c: np.concatenate(groups[c]) for c in [0, 1]}
    rng = np.random.default_rng(seed)
    values = {k: (a[k].to_numpy(), b[k].to_numpy()) for k in keys}
    estimates = {k: auc(y, new)-auc(y, old) for k, (old, new) in values.items()}
    draws = {k: [] for k in keys}
    for _ in range(samples):
        ix = np.concatenate([rng.choice(indexes[c], len(indexes[c]), replace=True) if singleton else
            np.concatenate([groups[c][i] for i in rng.integers(len(groups[c]), size=len(groups[c]))]) for c in [0, 1]])
        for key, (old, new) in values.items():
            draws[key].append(auc(y[ix], new[ix])-auc(y[ix], old[ix]))
    return {k: {"auc_difference_after_minus_before": estimates[k],
                "ci95": np.quantile(draws[k], [.025, .975]).tolist()} for k in keys}


def qa_statistics(frame, cfg, offset=0):
    adapted = frame.assign(injected_math_source=(frame.domain == "A").astype(int),
                           token_count=frame.prompt_tokens+frame.response_tokens)
    spec = {**cfg["analysis"], "top_fractions": cfg["analysis"]["qa_top_fractions"]}
    result = retrieval_metrics(adapted, spec, seed_offset=offset)
    # The shared ranking implementation calls its positive source "math"; expose
    # the actual QA class names and keep the generic statistics for this view.
    return {"examples": len(frame), "positive_class": "A2", "negative_class": "UltraChat",
            "domain_counts": frame.groupby("domain").size().to_dict(),
            **{k: result[k] for k in ["scores", "paired_auc_differences", "bootstrap_samples", "bootstrap_seed", "tie_seed"]},
            "bootstrap_unit": "normalized question group, stratified by domain",
            "length_baselines": {col: {"positive_auc": auc(adapted.injected_math_source.to_numpy(), frame[col].to_numpy()),
                                       "negative_auc": auc(adapted.injected_math_source.to_numpy(), -frame[col].to_numpy())}
                                 for col in ["prompt_tokens", "response_tokens"]}}


def qa_summary(frame):
    return {name: {"examples": len(g), "fraction_improved": float((g.delta_loss > 0).mean()),
            **{key: float(g[key].mean()) for key in ["ref_mean_nll", "sft_mean_nll", "delta_loss", "relative_gain"]}}
            for name, g in frame.groupby("domain")}


def analyze(config_path, qa_only=False):
    cfg, run, ev, train, web_run = settings(config_path)
    plan = freeze(config_path)
    out = run/"metrics"; out.mkdir(exist_ok=True)
    rows = qa_rows(cfg)
    ref, ref_sanity = load_qa(run, "reference")
    sft, sft_sanity = load_qa(run, f"step-{cfg['qa_checkpoint_step']:06d}")
    fresh = pair_qa(ref, sft)
    old = pd.read_parquet(Path(cfg["legacy_snapshot"])/"per_example.parquet").sort_values("id").reset_index(drop=True)
    pd.testing.assert_frame_equal(fresh[QA_METADATA], old[QA_METADATA])
    reference_error = float((fresh.ref_mean_nll-old.ref_mean_nll).abs().max())
    if reference_error > cfg["scoring"]["legacy_reference_tolerance"]:
        raise RuntimeError("Fresh reference scores do not reproduce the old QA experiment")
    ids = read_json(Path(cfg["legacy_snapshot"])/"length_matched_ids.json")
    qa_results, qa_frames = {}, []
    for name, frame in [("legacy_algebra1024_step32", old), ("math4096_step64", fresh)]:
        selected = frame[QA_METADATA+["ref_mean_nll", "sft_mean_nll", *cfg["analysis"]["reported_loss_scores"],
                                      *cfg["analysis"]["raw_loss_baselines"]]].copy()
        qa_frames.append(selected.assign(probe=name))
        qa_results[name] = {"main": qa_statistics(selected, cfg),
                           "length_matched": qa_statistics(fixed_subset(selected, ids), cfg, 1),
                           "summary": qa_summary(selected)}
    adapted = [f.assign(injected_math_source=(f.domain == "A").astype(int)) for f in [old, fresh]]
    qa_difference = {"main": compare_auc(*adapted, cfg["analysis"]["reported_loss_scores"],
                                          cfg["analysis"]["bootstrap_samples"], cfg["analysis"]["bootstrap_seed"]),
        "length_matched": compare_auc(*[fixed_subset(f, ids) for f in adapted], cfg["analysis"]["reported_loss_scores"],
                                       cfg["analysis"]["bootstrap_samples"], cfg["analysis"]["bootstrap_seed"]+1)}
    pd.concat(qa_frames, ignore_index=True).to_parquet(out/"qa_per_example.parquet", index=False)
    write_json(out/"qa_length_matched_ids.json", ids)
    legacy_ref_delta = old.ref_mean_nll-fresh.sft_mean_nll
    y_qa = (fresh.domain == "A").to_numpy(dtype=int)
    reuse_sensitivity = {"delta_loss": auc(y_qa, legacy_ref_delta.to_numpy()),
                         "relative_gain": auc(y_qa, (legacy_ref_delta/np.maximum(old.ref_mean_nll, 1e-6)).to_numpy())}
    write_json(out/"qa_metrics.json", {"probes": qa_results, "paired_new_minus_old": qa_difference,
        "reference_max_nll_difference": reference_error, "sanity": {"reference": ref_sanity, "sft": sft_sanity},
        "new_probe_auc_with_legacy_reference": reuse_sensitivity,
        "overlap": read_json(run/"qa_overlap.json")})
    print("QA_DIAGNOSTIC_RESULT", {name: {k: v["main"]["scores"][k]["roc_auc"] for k in cfg["analysis"]["reported_loss_scores"]}
                                   for name, v in qa_results.items()}, flush=True)
    if qa_only:
        return read_json(out/"qa_metrics.json")

    baseline, ref_web_sanity = load_score_set(web_run, "reference", "web", ev)
    bundle = Path(ev["bundle"])
    prepared = pd.read_parquet(bundle/"web/windows.parquet", columns=["id", "tokenization_hash", "scored_tokens"]).sort_values("id").reset_index(drop=True)
    manifest = pd.read_parquet(bundle/"web/document_manifest.parquet")
    labels = pd.DataFrame(read_jsonl(bundle/"web/labels.jsonl"))
    matched_ids = read_json(web_run/"metrics_web/length_matched_ids.json")
    old_metrics = read_json(web_run/"metrics_web/metrics.json")
    frames, web_results = {}, {}
    for step in cfg["web_checkpoint_steps"]:
        scored, sanity = load_score_set(web_run, f"step-{step:06d}", "web", ev)
        windows = pair_windows(baseline, scored)
        pd.testing.assert_frame_equal(windows[["id", "tokenization_hash", "scored_tokens"]], prepared)
        frame = aggregate_documents(windows, manifest, labels)
        if len(frame) != 10000 or int(frame.injected_math_source.sum()) != 1000:
            raise RuntimeError("Web pool changed")
        keep = ["id", "group_id", "source", "token_count", "scored_tokens", "windows", "injected_math_source",
                "ref_mean_nll", "sft_mean_nll", *cfg["analysis"]["reported_loss_scores"], *cfg["analysis"]["raw_loss_baselines"]]
        frames[step] = frame[keep].copy()
        if step == ev["final_checkpoint_step"]:
            original = pd.read_parquet(web_run/"metrics_web/per_document.parquet").sort_values("id").reset_index(drop=True)
            pd.testing.assert_frame_equal(frame[keep], original[keep])
            result = {k: old_metrics[k] for k in ["main", "length_matched", "document_summary"]}
        else:
            cache_path = out/f"web_step-{step:06d}.json"
            identity = {"plan": plan["fingerprint"], "step": step, "scored": digest(scored.to_dict("records")),
                        "analysis_code": file_hash(__file__), "retrieval_code": file_hash(Path(__file__).with_name("web_analysis.py"))}
            if cache_path.exists() and read_json(cache_path)["identity"] == identity:
                result = read_json(cache_path)["result"]
            else:
                result = {"main": retrieval_metrics(frame, ev["analysis"]),
                          "length_matched": retrieval_metrics(fixed_subset(frame, matched_ids), ev["analysis"], 1),
                          "document_summary": summarize_documents(frame)}
                write_json(cache_path, {"identity": identity, "result": result})
        web_results[str(step)] = {**result, "sanity": sanity}
        print("WEB_DIAGNOSTIC_RESULT", step, {k: result["main"]["scores"][k]["roc_auc"]
                                              for k in cfg["analysis"]["reported_loss_scores"]}, flush=True)
    web_differences = {str(step): compare_auc(frames[step], frames[64], cfg["analysis"]["reported_loss_scores"],
                        cfg["analysis"]["bootstrap_samples"], cfg["analysis"]["bootstrap_seed"])
                       for step in cfg["web_checkpoint_steps"] if step != 64}
    pd.concat([frame.assign(checkpoint_step=step) for step, frame in frames.items()], ignore_index=True).to_parquet(out/"web_per_document.parquet", index=False)
    write_json(out/"web_length_matched_ids.json", matched_ids)
    original_hashes = read_json(run/"original_notebook_hashes.json")
    if any(file_hash(path) != sha for path, sha in original_hashes.items()):
        raise RuntimeError("An existing review notebook changed during diagnostics")
    metrics = {"analysis_type": "probe_transfer_diagnostics", "run_id": cfg["run_id"],
        "reported_loss_scores": cfg["analysis"]["reported_loss_scores"],
        "qa": read_json(out/"qa_metrics.json"), "web": web_results,
        "web_paired_auc_final_minus_intermediate": web_differences,
        "target_validation": old_metrics["target_validation"],
        "web_length_matching": old_metrics["length_matching"], "qa_matched_examples": len(ids),
        "main": qa_results["math4096_step64"]["main"],
        "reference_web_sanity": ref_web_sanity, "optimizer_updates": 0,
        "checkpoint_policy": cfg["checkpoint_policy"], "original_notebooks_unchanged": True,
        "analysis_code_sha256": file_hash(__file__),
        "scoring_code_sha256": file_hash(Path(__file__).with_name("probe_diagnostics.py")),
        "plan_fingerprint": plan["fingerprint"], "completed_at_utc": datetime.now(timezone.utc).isoformat()}
    write_json(out/"metrics.json", metrics)
    write_json(out/"COMPLETED.json", {"plan_fingerprint": plan["fingerprint"],
        "files": {p.name: file_hash(p) for p in out.iterdir() if p.name != "COMPLETED.json"}})
    return metrics

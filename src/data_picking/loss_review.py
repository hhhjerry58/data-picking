"""A2/general-C loss review from existing, immutable model scores."""
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import SCORES, evaluate, loss_summary
from .common import file_hash, read_json, write_json


def match_a2_c_lengths(frame, seed):
    """Construct quartiles and matches using only A2 and C, ignoring other rows."""
    frame = frame.loc[frame["domain"].isin(["A", "C"])].reset_index(drop=True)
    if set(frame["domain"]) != {"A", "C"}:
        raise ValueError("Both A2 and general C are required")
    columns = ["prompt_tokens", "response_tokens"]
    edges = [np.unique(np.quantile(frame[c], [.25, .5, .75])) for c in columns]
    bins = np.column_stack([np.searchsorted(e, frame[c], side="right") for e, c in zip(edges, columns)])
    buckets = defaultdict(lambda: {"A": [], "C": []})
    for i, (domain, group) in enumerate(zip(frame["domain"], bins)):
        buckets[tuple(group)][domain].append(i)
    rng = np.random.default_rng(seed)
    selected, details = [], []
    for key in sorted(buckets):
        counts = {d: len(buckets[key][d]) for d in ["A", "C"]}
        n = min(counts.values())
        details.append({"bin": [int(x) for x in key], "counts": counts, "retained_per_domain": n})
        for domain in ["A", "C"]:
            selected.extend(rng.choice(buckets[key][domain], n, replace=False).tolist())
    matched = frame.iloc[sorted(selected)].copy().reset_index(drop=True)
    if len(matched) < 4:
        raise ValueError("Insufficient A2/C length overlap")
    return matched, {"domains": ["A", "C"], "quartile_edges": [e.tolist() for e in edges],
                     "retained_fraction": len(matched) / len(frame),
                     "retained_counts": matched.groupby("domain").size().to_dict(), "bins": details}


# Keep the previous entry point for existing callers.
match_a2_dolly_lengths = match_a2_c_lengths


def loss_result_from_pair(result, fallback_tie_seed):
    view = {key: deepcopy(result[key]) for key in
            ["examples", "positive_domain", "negative_domain", "question_group_counts",
             "paired_delta_auc_minus_baseline", "length_baselines", "bootstrap_samples", "bootstrap_unit"]}
    view["scores"] = deepcopy(result["loss_scores"])
    view["domain_counts"] = deepcopy(result["counts"])
    view["tie_seed"] = result.get("tie_seed", fallback_tie_seed)
    return view


def prepare_loss_review(run):
    """Preserve the original A2/C estimate; recompute the two-domain control."""
    run = Path(run)
    cfg = read_json(run / "extension_config.json")
    original_metrics = read_json(run / "metrics/metrics.json")
    kind = cfg["c_data"].get("kind", "dolly")
    display_name = cfg["c_data"].get("display_name", "Dolly")
    directory = run / "metrics" / f"a2_{kind}_loss"
    identity = {"scores": file_hash(run / "metrics/per_example.parquet"),
                "metrics": file_hash(run / "metrics/metrics.json"),
                "config": file_hash(run / "extension_config.json"),
                "implementation": file_hash(__file__),
                "statistics_implementation": file_hash(Path(__file__).with_name("analysis.py"))}
    marker = directory / "COMPLETED.json"
    if marker.is_file():
        saved = read_json(marker)
        if saved["identity"] == identity:
            for name, sha in saved["files"].items():
                if file_hash(directory / name) != sha:
                    raise RuntimeError(f"Review artifact changed: {name}")
            return directory, read_json(directory / "metrics.json")
    columns = ["id", "group_id", "domain", "split", "level", "text_hash", "tokenization_hash",
               "prompt_tokens", "response_tokens", "source", "source_revision",
               "ref_mean_nll", "sft_mean_nll", *SCORES]
    scores = pd.read_parquet(run / "metrics/per_example.parquet", columns=columns)
    frame = scores.loc[scores["domain"].isin(["A", "C"])].reset_index(drop=True)
    assert frame["id"].is_unique and frame.groupby("domain").size().to_dict() == {"A": 512, "C": 512}
    completed = read_json(Path(cfg["base_run"]) / "training_complete.json")
    frame["checkpoint_step"] = completed["global_step"]
    matched, matching = match_a2_c_lengths(frame, cfg["analysis"]["length_matching_seed"])

    original_pair = original_metrics["comparisons"]["A_vs_C"]["raw"]
    main = loss_result_from_pair(original_pair, cfg["analysis"]["bootstrap_seed"] + 10 + 100)

    control_cfg = deepcopy(cfg)
    control_cfg["analysis"]["bootstrap_seed"] += 11
    same_control = (original_metrics["length_matching"].get("domains") == ["A", "C"]
                    and matched["id"].tolist() == read_json(run / "metrics/length_matched_ids.json"))
    if same_control:
        controlled = loss_result_from_pair(original_metrics["comparisons"]["A_vs_C"]["length_matched"],
                                           cfg["analysis"]["bootstrap_seed"] + 11 + 100)
        print("REUSING_A2_C_LENGTH_CONTROL", display_name, matching["retained_counts"], flush=True)
    else:
        print("ANALYZING_A2_C_LENGTH_CONTROL", display_name, matching["retained_counts"], flush=True)
        controlled = evaluate(matched.to_dict("records"), control_cfg)
        controlled["tie_seed"] = control_cfg["analysis"]["bootstrap_seed"] + 1
        controlled["question_group_counts"] = matched.groupby("domain")["group_id"].nunique().to_dict()

    previous = pd.read_parquet(Path(cfg["base_run"]) / "metrics/per_example.parquet")
    crosscheck = frame.loc[frame["domain"] == "A"].merge(previous, on="id", suffixes=("", "_original"), validate="one_to_one")
    assert len(crosscheck) == 512
    assert (crosscheck["tokenization_hash"] == crosscheck["tokenization_hash_original"]).all()
    errors = {p: float((crosscheck[f"{p}_mean_nll"] - crosscheck[f"{p}_mean_nll_original"]).abs().max())
              for p in ["ref", "sft"]}
    overlaps = read_json(run / "data/duplicate_report.json")["selected_overlap_ids"]
    sensitivity = {"excluded_c_ids": overlaps, "status": "no_normalized_question_overlap"}
    if overlaps:
        sensitivity["status"] = "reported_separately_main_sample_unchanged"
        for name, pool in [("main", frame), ("length_matched", matched)]:
            clean = pool.loc[~pool["id"].isin(overlaps)]
            sensitivity[name] = evaluate(clean.to_dict("records"), control_cfg)
    metrics = {"analysis_type": f"{kind}_loss_comparison", "c_display_name": display_name,
               "run_id": cfg["run_id"], "base_run": cfg["base_run"],
               "primary_comparison": "A_vs_C", "primary_score": "delta_loss", "primary_checkpoint": True,
               "checkpoint_step": completed["global_step"], "main": main, "length_matched": controlled,
               "length_matching": matching, "loss_summary": loss_summary(frame.to_dict("records")),
               "relative_gain_denominator_clamps": int((frame["ref_mean_nll"] < 1e-6).sum()),
               "overlap_sensitivity": sensitivity, "a2_original_max_nll_difference": errors,
               "result_time_utc": original_metrics["completed_at_utc"],
               "updated_at_utc": datetime.now(timezone.utc).isoformat(),
               "main_estimates": "Unchanged original A2/C raw loss results; no model rescoring",
               "control_definition": "Joint length quartiles fitted only to A2 and C; equal domain counts per bin",
               "source_identity": identity}
    directory.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(directory / "per_example.parquet", index=False)
    write_json(directory / "length_matched_ids.json", matched["id"].tolist())
    write_json(directory / "metrics.json", metrics)
    files = ["metrics.json", "per_example.parquet", "length_matched_ids.json"]
    write_json(marker, {"identity": identity, "files": {name: file_hash(directory / name) for name in files}})
    return directory, metrics

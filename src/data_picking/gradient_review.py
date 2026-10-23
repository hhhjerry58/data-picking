"""Prepare paired gradient appendix snapshots without changing model scores."""
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .common import file_hash, read_json, read_jsonl, write_json
from .generalization_analysis import GRADIENT_SCORES, evaluate_pair


def align_gradient_rows(target, measured, tolerance):
    """Require the same examples, grouping, tokens and scoring definition."""
    if not target["id"].is_unique or not measured["id"].is_unique:
        raise ValueError("Duplicate example IDs in loss or gradient scores")
    rows = measured.set_index("id").loc[target["id"]].reset_index()
    metadata = ["id", "group_id", "domain", "split", "text_hash", "tokenization_hash", "prompt_tokens", "response_tokens"]
    pd.testing.assert_frame_equal(target[metadata].reset_index(drop=True), rows[metadata], check_dtype=False)
    errors = {}
    for prefix in ["ref", "sft"]:
        field = f"{prefix}_mean_nll"
        error = float(np.max(np.abs(rows[field].to_numpy() - target[field].to_numpy())))
        if not np.isfinite(error) or error > tolerance:
            raise ValueError(f"Gradient/loss NLL mismatch for {prefix}: {error}")
        errors[prefix] = error
    return rows, errors


def prepare_gradient_review(notebook_folder, measurement_run):
    folder, source = Path(notebook_folder), Path(measurement_run)
    manifest = read_json(folder / "manifest.json")
    effective = read_json(folder / "snapshot/effective_config.json")
    source_cfg = read_json(source / "extension_config.json")
    measured_metrics = read_json(source / "metrics/metrics.json")
    identity = {
        "target_scores": file_hash(folder / "snapshot/per_example.parquet"),
        "length_matched_ids": file_hash(folder / "snapshot/length_matched_ids.json"),
        "measurement_run": str(source.resolve()),
        "source_files": {name: file_hash(source / name) for name in
                         ["metrics/per_example.parquet", "metrics/per_layer.parquet", "metrics/parameter_summary.parquet", "metrics/metrics.json"]},
        "implementation": {name: file_hash(Path(__file__).with_name(name)) for name in
                           ["gradient_review.py", "generalization_analysis.py", "analysis.py"]},
        "bootstrap_samples": 2000, "bootstrap_seed": 20261023,
    }
    directory = Path(manifest["source_run_dir"]) / "metrics/gradient_review"
    marker = directory / "COMPLETED.json"
    if marker.is_file():
        saved = read_json(marker)
        if saved["identity"] == identity:
            for name, sha in saved["files"].items():
                if file_hash(directory / name) != sha: raise ValueError(f"Invalid gradient review file: {name}")
            return directory, read_json(directory / "gradient_metrics.json")

    target = pd.read_parquet(folder / "snapshot/per_example.parquet")
    target = target.loc[target["split"].isin(["A2", "B", "C"])].reset_index(drop=True)
    domains = sorted(target["domain"].unique())
    if domains not in [["A", "B"], ["A", "C"]] or target.groupby("domain").size().to_dict() != {d: 512 for d in domains}:
        raise ValueError("The appendix expects the original two 512-example evaluation groups")
    negative = domains[1]
    target_metrics = read_json(folder / "snapshot/metrics.json")
    other_name = "B: Number Theory" if negative == "B" else "C: " + target_metrics.get("c_display_name", "Dolly")
    names = {"A": "A2: Algebra", negative: other_name}
    measured = pd.read_parquet(source / "metrics/per_example.parquet")
    tolerance = source_cfg["gradients"]["loss_crosscheck_tolerance"]
    rows, errors = align_gradient_rows(target, measured, tolerance)
    for domain in domains:
        # The parameter-level aggregate is valid only for this exact domain pool.
        if set(rows.loc[rows.domain == domain, "id"]) != set(measured.loc[measured.domain == domain, "id"]):
            raise ValueError("Parameter summary needs reaggregation for a subset of a measured domain")
    ids = set(rows["id"])
    layers = pd.read_parquet(source / "metrics/per_layer.parquet")
    layers = layers.loc[layers["id"].isin(ids)].reset_index(drop=True)
    parameters = pd.read_parquet(source / "metrics/parameter_summary.parquet")
    parameters = parameters.loc[parameters["domain"].isin(domains)].reset_index(drop=True)
    count = measured_metrics["parameter_count"]
    if layers.duplicated(["id", "model", "layer"]).any(): raise ValueError("Duplicate layer measurements")
    assert (layers.groupby(["id", "model"])["numel"].sum() == count).all()
    assert (parameters.groupby(["model", "domain"])["numel"].sum() == count).all()
    assert (parameters["examples"] == 512).all()
    assert set(layers["model"]) == set(parameters["model"]) == {"reference", "sft"}
    for field in ["grad_l2", "grad_rms", "grad_max_abs"]:
        assert np.isfinite(layers[field]).all() and (layers[field] >= 0).all()
    for prefix, model in [("ref", "reference"), ("sft", "sft")]:
        layer_norms = layers.loc[layers.model == model].assign(square=lambda x:x.grad_l2**2).groupby("id")["square"].sum()
        np.testing.assert_allclose(layer_norms.loc[rows.id], rows[f"{prefix}_grad_l2"]**2, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(rows[f"{prefix}_grad_rms"], rows[f"{prefix}_grad_l2"] / np.sqrt(count), rtol=1e-6)
        assert np.isfinite(rows[f"{prefix}_cos_to_a1"]).all()
        assert rows[f"{prefix}_cos_to_a1"].between(-1.00001, 1.00001).all()

    checkpoint = Path(read_json(folder / "snapshot/training_complete.json")["checkpoint"])
    checkpoint_sha = read_json(checkpoint / "manifest.json")["files"]["state.pt"]["sha256"]
    anchor_train_dir = Path(read_json(source / "data/summary.json")["original_data_dir"])
    a1_ids = {row["id"] for row in read_jsonl(anchor_train_dir / "A1.jsonl")}
    anchors, provenance = {}, {}
    gradient_dir = source / source_cfg.get("gradient_directory", "gradients")
    for model in ["reference", "sft"]:
        measured_config = read_json(gradient_dir / model / "measurement_config.json")
        complete = read_json(gradient_dir / model / "COMPLETED.json")
        anchor = read_json(gradient_dir / model / "anchor.json")
        config_identity = measured_config["identity"]
        spec = config_identity["config"]["gradients"]
        assert spec["per_example"] and not spec["clip"]
        assert spec["parameter_dtype"] == spec["forward_dtype"] == "float32" and spec["attention_backend"] == "math"
        assert config_identity["model_revision"] == effective["config"]["model"]["revision"]
        assert complete["optimizer_updates"] == 0
        assert complete["fingerprint"] == anchor["fingerprint"] == measured_config["fingerprint"]
        assert set(anchor["ids"]) <= a1_ids and not set(anchor["ids"]) & ids
        assert len(anchor["ids"]) == spec["anchor_size"] == 128
        assert anchor["parameter_count"] == count and anchor["norm"] > 0
        if model == "sft": assert config_identity["checkpoint_sha256"] == checkpoint_sha
        anchors[model] = anchor
        provenance[model] = {"measurement_config_sha256": file_hash(gradient_dir / model / "measurement_config.json"),
                             "fingerprint": complete["fingerprint"], "completed_at_utc": complete["completed_at_utc"],
                             "measurement_spec": spec, "model_revision": config_identity["model_revision"],
                             "checkpoint_sha256": config_identity["checkpoint_sha256"]}
    assert anchors["reference"]["ids"] == anchors["sft"]["ids"]
    matched_ids = read_json(folder / "snapshot/length_matched_ids.json")
    if len(set(matched_ids)) != len(matched_ids) or not set(matched_ids) <= ids: raise ValueError("Invalid length-matched IDs")
    matched = rows.set_index("id").loc[matched_ids].reset_index()
    metrics_cfg = {"analysis": {"bootstrap_samples": identity["bootstrap_samples"], "bootstrap_seed": identity["bootstrap_seed"]}}
    comparisons = {}
    for offset, (pool, frame) in enumerate([("raw", rows), ("length_matched", matched)]):
        print("GRADIENT_REVIEW_STATS", folder.name, pool, len(frame), flush=True)
        value = evaluate_pair(frame, "A", negative, metrics_cfg, seed_offset=offset)
        comparisons[pool] = {key: value[key] for key in ["counts", "question_group_counts", "gradient_scores", "length_baselines",
                                                        "bootstrap_samples", "bootstrap_seed", "bootstrap_unit", "gradient_direction"]}
    correlations, loss_change_correlations = [], []
    for domain, group in rows.groupby("domain"):
        for prefix in ["ref", "sft"]:
            correlations.append({"domain": domain, "model": prefix,
                                 **{feature: float(spearmanr(group[f"{prefix}_grad_l2"], group[column]).statistic)
                                    for feature, column in [("nll", f"{prefix}_mean_nll"), ("prompt_tokens", "prompt_tokens"), ("response_tokens", "response_tokens")]}})
        loss_change_correlations.append({"domain": domain,
            **{key: float(spearmanr(group["grad_l2_delta"], group[key]).statistic) for key in ["delta_loss", "relative_gain"]}})
    result = {"target_run_id": manifest["run_id"], "measurement_run": str(source.resolve()), "domains": domains,
              "domain_names": names, "parameter_count": count, "anchor_summary": anchors, "model_provenance": provenance,
              "checkpoint_step": target_metrics["checkpoint_step"], "examples": len(rows),
              "length_matched_counts": matched.groupby("domain").size().to_dict(), "length_matched_ids": matched_ids,
              "comparisons": comparisons, "norm_correlations": correlations, "norm_drop_loss_correlations": loss_change_correlations,
              "nll_difference_from_loss_notebook": errors, "nll_crosscheck_tolerance": tolerance,
              "source_identity": identity, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
              "interpretation": "Exploratory gradient diagnostics; larger raw feature predicts A2; no direction flipping; not actual training utility"}
    directory.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(directory / "gradient_per_example.parquet", index=False)
    layers.to_parquet(directory / "gradient_per_layer.parquet", index=False)
    parameters.to_parquet(directory / "gradient_parameter_summary.parquet", index=False)
    write_json(directory / "gradient_metrics.json", result)
    files = ["gradient_per_example.parquet", "gradient_per_layer.parquet", "gradient_parameter_summary.parquet", "gradient_metrics.json"]
    write_json(marker, {"identity": identity, "files": {name: file_hash(directory / name) for name in files}})
    return directory, result

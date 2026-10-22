import math
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_curve

from .common import file_hash, paths, read_json, write_json


SCORES = ["delta_loss", "relative_gain", "neg_ref_loss", "neg_sft_loss", "pos_ref_loss", "pos_sft_loss"]


def auc(labels, scores):
    n1 = int(labels.sum())
    n0 = len(labels)-n1
    return float((rankdata(scores)[labels == 1].sum() - n1*(n1+1)/2) / (n0*n1))


def precision_at(labels, scores, fraction, ties):
    n = max(1, math.ceil(len(labels)*fraction))
    selected = np.lexsort((ties, -scores))[:n]
    return float(labels[selected].mean())


def bootstrap_indices(rows, rng):
    groups = {0: defaultdict(list), 1: defaultdict(list)}
    for i, row in enumerate(rows):
        groups[int(row["domain"] == "A")][row["group_id"]].append(i)
    blocks = [[np.array(v) for v in group.values()] for group in groups.values()]
    while True:
        sampled = []
        for domain in blocks:
            sampled.extend(domain[j] for j in rng.integers(len(domain), size=len(domain)))
        yield np.concatenate(sampled)


def evaluate(rows, cfg):
    y = np.array([int(r["domain"] == "A") for r in rows])
    if set(y) != {0, 1}:
        raise ValueError("Both domains required")
    scores = {key: np.array([r[key] for r in rows]) for key in SCORES}
    ties = np.random.default_rng(cfg["analysis"]["bootstrap_seed"]+1).random(len(rows))
    metrics = {key: {"roc_auc": auc(y, s), "average_precision": float(average_precision_score(y, s)),
                     "precision_top_10pct": precision_at(y, s, 0.1, ties),
                     "precision_top_20pct": precision_at(y, s, 0.2, ties)} for key,s in scores.items()}
    n_boot = cfg["analysis"]["bootstrap_samples"]
    samples = {key: {metric: [] for metric in ["roc_auc", "precision_top_10pct", "precision_top_20pct"]} for key in SCORES}
    stream = bootstrap_indices(rows, np.random.default_rng(cfg["analysis"]["bootstrap_seed"]))
    for _ in range(n_boot):
        idx = next(stream)
        for key, s in scores.items():
            samples[key]["roc_auc"].append(auc(y[idx], s[idx]))
            samples[key]["precision_top_10pct"].append(precision_at(y[idx], s[idx], 0.1, ties[idx]))
            samples[key]["precision_top_20pct"].append(precision_at(y[idx], s[idx], 0.2, ties[idx]))
    for key in SCORES:
        for metric, values in samples[key].items():
            metrics[key][metric+"_ci95"] = np.quantile(values, [0.025,0.975]).tolist()
    paired = {}
    for key in SCORES[2:]:
        values = np.array(samples["delta_loss"]["roc_auc"]) - np.array(samples[key]["roc_auc"])
        paired[key] = {"difference": metrics["delta_loss"]["roc_auc"]-metrics[key]["roc_auc"],
                       "ci95": np.quantile(values, [0.025,0.975]).tolist()}
    length_metrics = {}
    for column in ["prompt_tokens", "response_tokens"]:
        values = np.array([r[column] for r in rows])
        length_metrics[column] = {"positive_auc": auc(y, values), "negative_auc": auc(y, -values)}
    return {"examples": len(rows), "domain_counts": dict(Counter(r["domain"] for r in rows)),
            "scores": metrics, "paired_delta_auc_minus_baseline": paired, "length_baselines": length_metrics,
            "bootstrap_samples": n_boot, "bootstrap_unit": "group_id, stratified by domain"}


def match_lengths(rows, seed):
    columns = [np.array([r[c] for r in rows]) for c in ["prompt_tokens", "response_tokens"]]
    edges = [np.unique(np.quantile(v, [0.25, 0.5, 0.75])) for v in columns]
    buckets = defaultdict(lambda: {"A": [], "B": []})
    for i, row in enumerate(rows):
        key = tuple(int(np.searchsorted(e, col[i], side="right")) for e,col in zip(edges,columns))
        buckets[key][row["domain"]].append(i)
    rng = np.random.default_rng(seed)
    selected = []
    for key in sorted(buckets):
        group = buckets[key]
        n = min(len(group["A"]), len(group["B"]))
        if n:
            selected.extend(rng.choice(group["A"], n, replace=False).tolist())
            selected.extend(rng.choice(group["B"], n, replace=False).tolist())
    return [rows[i] for i in sorted(selected)], {"quartile_edges": [e.tolist() for e in edges], "retained_fraction": len(selected)/len(rows)}


def loss_summary(rows):
    result = {}
    for split in sorted({r["split"] for r in rows}):
        selected = [r for r in rows if r["split"] == split]
        result[split] = {"examples": len(selected), "positive_delta_fraction": float(np.mean([r["delta_loss"]>0 for r in selected])),
                         "level_counts": dict(Counter(r["level"] for r in selected))}
        for key in ["ref_mean_nll", "sft_mean_nll", "delta_loss", "response_tokens", "prompt_tokens"]:
            values = np.array([r[key] for r in selected])
            result[split][key] = {"mean": float(values.mean()), "median": float(np.median(values)),
                                  "quantiles_10_25_75_90": np.quantile(values, [0.1,0.25,0.75,0.9]).tolist()}
    return result


def plots(rows, metrics, matched_metrics, directory):
    directory.mkdir(parents=True, exist_ok=True)
    colors = {"A": "#1674B8", "B": "#DF7931"}
    labels = {"A": "A2: Algebra", "B": "B: Number Theory"}
    fig, ax = plt.subplots(figsize=(7,4.5))
    edges = np.histogram_bin_edges([r["delta_loss"] for r in rows], bins=40)
    for domain in ["A", "B"]:
        ax.hist([r["delta_loss"] for r in rows if r["domain"] == domain], bins=edges, density=True,
                alpha=0.5, label=labels[domain], color=colors[domain])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set(xlabel="Reference NLL - SFT NLL (nats/token)", ylabel="Density")
    ax.legend()
    fig.tight_layout(); fig.savefig(directory / "delta_distribution.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6,5))
    y = np.array([r["domain"] == "A" for r in rows])
    for key in SCORES[:4]:
        x, tpr, _ = roc_curve(y, [r[key] for r in rows])
        ax.plot(x, tpr, label=f'{key} (AUC {metrics["scores"][key]["roc_auc"]:.3f})')
    ax.plot([0,1],[0,1], "k--", linewidth=0.8)
    ax.set(xlabel="False positive rate", ylabel="True positive rate")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(directory / "roc.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6,5))
    for domain in ["A", "B"]:
        group = [r for r in rows if r["domain"] == domain]
        ax.scatter([r["ref_mean_nll"] for r in group], [r["sft_mean_nll"] for r in group], s=10, alpha=0.4, color=colors[domain], label=labels[domain])
    bound = max(max(r["ref_mean_nll"],r["sft_mean_nll"]) for r in rows)*1.02
    ax.plot([0,bound],[0,bound], "k--", linewidth=0.8)
    ax.set(xlabel="Reference NLL (nats/token)", ylabel="SFT NLL (nats/token)")
    ax.legend(); fig.tight_layout(); fig.savefig(directory / "loss_scatter.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8,4.5))
    for offset, result, label in [(-0.18,metrics,"Original"), (0.18,matched_metrics,"Length matched")]:
        if result is None:
            continue
        values = [result["scores"][k]["roc_auc"] for k in SCORES[:4]]
        intervals = [result["scores"][k]["roc_auc_ci95"] for k in SCORES[:4]]
        ax.bar(np.arange(4)+offset, values, width=0.36, label=label)
        # Plot percentile interval endpoints directly; the point estimate need not lie inside them.
        for x, interval in zip(np.arange(4)+offset, intervals):
            ax.vlines(x, *interval, color="black", linewidth=1)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8)
    ax.set(xticks=np.arange(4), xticklabels=SCORES[:4], ylabel="ROC-AUC", ylim=(0,1))
    ax.legend(); fig.tight_layout(); fig.savefig(directory / "length_matching.png", dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(10,4))
    for ax, key in zip(axes, ["prompt_tokens", "response_tokens"]):
        for domain in ["A", "B"]:
            ax.hist([r[key] for r in rows if r["domain"] == domain], bins=30, alpha=0.5, label=labels[domain], color=colors[domain])
        ax.set(xlabel=key, ylabel="Examples")
    axes[0].legend(); fig.tight_layout(); fig.savefig(directory / "length_distributions.png", dpi=180); plt.close(fig)


def analyze(cfg, step=None):
    p = paths(cfg)
    final_step = read_json(p["run"] / "training_complete.json")["global_step"]
    step = final_step if step is None else step
    tables = []
    for label in ["reference", f"step-{step:06d}"]:
        filename = p["run"] / "scores" / f"{label}.parquet"
        marker = read_json(filename.with_name(label + ".complete.json"))
        if marker["sha256"] != file_hash(filename):
            raise RuntimeError("Score hash mismatch")
        rows = pq.read_table(filename).to_pylist()
        if len({r["id"] for r in rows}) != len(rows):
            raise RuntimeError("Duplicate score IDs")
        tables.append({r["id"]: r for r in rows})
    ref, tuned = tables
    if ref.keys() != tuned.keys():
        raise RuntimeError("Reference and SFT scores have different coverage")
    joined = []
    for id in sorted(ref):
        a,b = ref[id],tuned[id]
        for key in ["text_hash", "tokenization_hash", "response_tokens", "prompt_tokens", "split", "domain", "group_id", "level", "model_revision"]:
            if a[key] != b[key]:
                raise RuntimeError(f"Paired score metadata mismatch: {id} {key}")
        row = {k:v for k,v in a.items() if k not in {"sum_nll","mean_nll","checkpoint_step"}}
        l0, la = a["mean_nll"],b["mean_nll"]
        row.update(ref_mean_nll=l0, sft_mean_nll=la, ref_sum_nll=a["sum_nll"], sft_sum_nll=b["sum_nll"],
                   delta_loss=l0-la, relative_gain=(l0-la)/max(l0,1e-6), neg_ref_loss=-l0, neg_sft_loss=-la,
                   pos_ref_loss=l0, pos_sft_loss=la, checkpoint_step=step)
        joined.append(row)
    evaluation = [r for r in joined if r["split"] in {"A2", "B"}]
    main = evaluate(evaluation, cfg)
    matched, matching = match_lengths(evaluation, cfg["data"]["split_seed"]+3)
    matched_metrics = evaluate(matched, cfg) if len(matched) >= 4 else None
    metrics = {"checkpoint_step": step, "primary_checkpoint": step == final_step, "main": main,
               "length_matching": matching, "length_matched": matched_metrics, "loss_summary": loss_summary(joined),
               "matched_loss_summary": loss_summary(matched), "relative_gain_denominator_clamps": sum(r["ref_mean_nll"]<1e-6 for r in joined),
               "scoring_sanity": read_json(p["run"] / "scores/sanity.json")}
    directory = p["run"] if step == final_step else p["run"] / f"diagnostic-step-{step:06d}"
    (directory / "metrics").mkdir(parents=True, exist_ok=True)
    write_json(directory / "metrics/metrics.json", metrics)
    pq.write_table(pa.Table.from_pylist(joined), directory / "metrics/per_example.parquet")
    write_json(directory / "metrics/length_matched_ids.json", [r["id"] for r in matched])
    plots(evaluation, main, matched_metrics, directory / "figures")
    m = main["scores"]["delta_loss"]
    if m["roc_auc_ci95"][0] > 0.5:
        conclusion = "本轮在预先指定的方向上观察到 A2/B 区分信号。"
    elif m["roc_auc_ci95"][1] < 0.5:
        conclusion = "本轮主指标的区分方向与预设假设相反；没有事后翻转分数。"
    else:
        conclusion = "本轮主指标的置信区间覆盖 0.5，尚无稳定的同域区分证据。"
    lines = ["# MATH Algebra / Number Theory 预实验结果", "", conclusion, "",
             f"模型：{cfg['model']['name']}；训练 seed={cfg['training']['seed']}；报告 checkpoint step={step}。",
             "", "| 排序指标 | ROC-AUC（95% CI） | AP | top 10% A2 占比 | top 20% A2 占比 |",
             "| --- | --- | --- | --- | --- |"]
    for key in SCORES:
        v = main["scores"][key]
        low, high = v["roc_auc_ci95"]
        lines.append(f"| {key} | {v['roc_auc']:.4f} [{low:.4f}, {high:.4f}] | {v['average_precision']:.4f} | {v['precision_top_10pct']:.4f} | {v['precision_top_20pct']:.4f} |")
    lines += ["", "| 数据 | 条数 | ref NLL | SFT NLL | 平均下降 |", "| --- | --- | --- | --- | --- |"]
    for split,v in metrics["loss_summary"].items():
        lines.append(f"| {split} | {v['examples']} | {v['ref_mean_nll']['mean']:.4f} | {v['sft_mean_nll']['mean']:.4f} | {v['delta_loss']['mean']:.4f} |")
    difference = main["paired_delta_auc_minus_baseline"]["neg_ref_loss"]
    lines += ["", f"主指标相对 neg_ref_loss 的 AUC 差：{difference['difference']:.4f}，配对 95% CI {difference['ci95']}。",
              f"长度匹配保留 {len(matched)}/{len(evaluation)} 条（{matching['retained_fraction']:.1%}）。"]
    sft_difference = main["paired_delta_auc_minus_baseline"]["neg_sft_loss"]
    lines += [f"主指标相对 neg_sft_loss 的 AUC 差：{sft_difference['difference']:.4f}，配对 95% CI {sft_difference['ci95']}。"]
    if matched_metrics:
        value = matched_metrics["scores"]["delta_loss"]
        lines += [f"长度匹配后的主指标 AUC：{value['roc_auc']:.4f}，95% CI {value['roc_auc_ci95']}。"]
        matched_difference = matched_metrics["paired_delta_auc_minus_baseline"]["neg_ref_loss"]
        lines += [f"长度匹配后，主指标相对 neg_ref_loss 的 AUC 差：{matched_difference['difference']:.4f}，配对 95% CI {matched_difference['ci95']}。"]
    lines += ["", "![Loss 下降分布](figures/delta_distribution.png)", "", "![ROC](figures/roc.png)", "",
              "![长度匹配对照](figures/length_matching.png)", "",
              "统计约定：A2 为正类，分数方向预先固定；分组、按域分层 bootstrap 2,000 次。A2/B 各占一半，随机 top-k A2 占比为 50%。",
              "", "局限：单模型、单训练 seed、单数据划分；置信区间不涵盖训练随机性。这里只验证同域识别，不证明选出的数据训练效用更高。官方 test 未用于本轮训练或评分，初始模型预训练是否包含 MATH 未核验。",
              "", "数值设置：fp32 参数及 AdamW 状态、bf16 autocast 训练；当前本地环境固定 SDPA math 后端以避免自动后端的非有限梯度。评分使用完整 fp32，重复评分及 batch/padding 误差检查见 scores/sanity.json。",
              "", "复现配置、逐样本分数、长度匹配样本 ID 和完整指标都保存在本 run 目录；checkpoint 含模型、optimizer、scheduler、RNG 和数据顺序。"]
    (directory / "report.md").write_text("\n".join(lines)+"\n")
    print("ANALYSIS_COMPLETE", directory / "report.md", flush=True)
    print("PRIMARY", m, flush=True)
    return metrics

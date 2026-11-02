"""Shared notebook summaries, snapshot checksums and experiment indexing."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import textwrap


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def loss_metric_overview(scores):
    parts = []
    for key, label in [("delta_loss", "loss 下降量"), ("relative_gain", "相对下降比例")]:
        value = scores[key]
        low, high = value["roc_auc_ci95"]
        parts.append(f"{label} ROC-AUC **{value['roc_auc']:.4f}**（95% CI {low:.4f}–{high:.4f}）")
    return "两项指标并列考察，不设主次：" + "；".join(parts) + "。"


def metric_conclusion_prefix():
    return textwrap.dedent(r"""
    texts = []
    for key in ['delta_loss', 'relative_gain']:
        value = metrics['main']['scores'][key]
        lo, hi = value['roc_auc_ci95']
        signal = ('在固定方向上有区分信号' if lo > .5 else
                  '区分方向与预设假设相反，没有事后翻转分数' if hi < .5 else
                  '区间覆盖 0.5，尚无稳定的区分证据')
        texts.append(f"{score_labels[key]}：AUC {value['roc_auc']:.4f}，95% CI {interval(value['roc_auc_ci95'])}；{signal}。")
    """).strip()


def update_index(root):
    readme = root / "README.md"
    if not readme.exists():
        readme.write_text("# 实验分析\n\n<!-- experiments-index:start -->\n<!-- experiments-index:end -->\n")
    start, end = "<!-- experiments-index:start -->", "<!-- experiments-index:end -->"
    content = readme.read_text()
    if start not in content or end not in content:
        raise ValueError(f"Preserving README without index markers: {readme}")
    lines = [start, "", "| 结果时间（UTC） | 实验 notebook | 模型 | 下降量 AUC | 相对下降比例 AUC |",
             "| --- | --- | --- | --- | --- |"]
    utility_rows = []
    for folder in sorted(root.iterdir()):
        if not (folder / "manifest.json").is_file():
            continue
        manifest = read_json(folder / "manifest.json")
        if manifest.get("execution", {}).get("status") != "completed":
            continue
        cfg = read_json(folder / "snapshot/effective_config.json")["config"]
        metrics = read_json(folder / "snapshot/metrics.json")
        if metrics.get("analysis_type") == "document_training_utility":
            timestamp = datetime.fromisoformat(manifest["result_time_utc"]).strftime("%Y-%m-%d %H:%M:%S")
            for arm, result in metrics["arms"].items():
                utility_rows.append(f"| {timestamp} | [{arm} · 3 seeds]({folder.name}/analysis.ipynb) | "
                    f"{result['mean_target_nll']:.5f} | {result['scores']['delta_loss']['mean']:.5f} | "
                    f"{result['scores']['relative_gain']['mean']:.5f} |")
            continue
        if metrics.get("analysis_type") == "document_quality_sources":
            timestamp = datetime.fromisoformat(manifest["result_time_utc"]).strftime("%Y-%m-%d %H:%M:%S")
            model = cfg["model"]["name"].split("/")[-1]
            for corpus, label in [("openstax", "OpenStax"), ("finemath", "FineMath-4+")]:
                scores = metrics["comparisons"][corpus]["main"]["scores"]
                lines.append(f"| {timestamp} | [{label} / CC · step 64]({folder.name}/analysis.ipynb) | {model} | "
                             f"{scores['delta_loss']['roc_auc']:.4f} | {scores['relative_gain']['roc_auc']:.4f} |")
            continue
        title = ("MATH probe 诊断：A2/UltraChat AUC；网页 checkpoint 轨迹" if metrics.get("analysis_type") == "probe_transfer_diagnostics"
                 else f"MATH target / OpenWebMath + CC · seed {cfg['training']['seed']}" if metrics.get("analysis_type") == "web_document_retrieval"
                 else f"{cfg['data']['a_subset']} / {cfg['data']['b_subset']} · seed {cfg['training']['seed']}")
        timestamp = datetime.fromisoformat(manifest["result_time_utc"]).strftime("%Y-%m-%d %H:%M:%S")
        model = cfg["model"]["name"].split("/")[-1]
        if metrics.get("analysis_type") in {"dolly_loss_comparison", "ultrachat_loss_comparison"}:
            title = f"{cfg['data']['a_subset']} / {metrics.get('c_display_name', 'Dolly')} C · loss"
            scores = metrics["main"]["scores"]
        elif metrics.get("analysis_type") == "dolly_gradient_extension":
            title = f"{cfg['data']['a_subset']} / {cfg['data']['b_subset']} / Dolly C · loss + gradients"
            comparison = metrics["primary_comparison"]
            scores = metrics["comparisons"][comparison]["raw"]["loss_scores"]
        else:
            scores = metrics["main"]["scores"]
        if manifest.get("gradient_appendix"):
            title = title.replace(" · loss", "") + " · loss + grad"
        lines.append(f"| {timestamp} | [{title}]({folder.name}/analysis.ipynb) | {model} | "
                     f"{scores['delta_loss']['roc_auc']:.4f} | {scores['relative_gain']['roc_auc']:.4f} |")
    if utility_rows:
        lines += ["", "直接训练效用（下表是目标 loss 变化，不是来源 AUC）：", "",
                  "| 结果时间（UTC） | 实验 notebook / 来源 | 最终目标 NLL | loss 下降量 | 相对下降比例 |",
                  "| --- | --- | --- | --- | --- |", *utility_rows]
    lines += ["", end]
    prefix, rest = content.split(start, 1)
    _, suffix = rest.split(end, 1)
    readme.write_text(prefix + "\n".join(lines) + suffix)

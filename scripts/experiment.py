"""One entry point for data preparation, training, evaluation and review artifacts.

Run from the repository root after sourcing scripts/env.sh.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys


SCRIPTS = Path(__file__).resolve().parent
DELEGATED = {
    "download": ("maintenance/download_assets.py", "Download the pinned pilot model and data"),
    "run-pilot": ("workflows/run_pilot.py", "Run or resume the complete A1/A2/B pilot"),
    "append-gradients": ("notebooks/append_gradients.py", "Append gradient results while preserving loss cells"),
    "stage-weights": ("maintenance/stage_weights.py", "Make verified temporary RAM copies of evaluation weights"),
    "check-gradients": ("maintenance/check_gradients.py", "Diagnose numerical kernels without optimizer updates"),
    "export-web": ("notebooks/export_web.py", "Export completed web retrieval verification and gradient diagnostics"),
    "export-probe": ("notebooks/export_probe.py", "Export QA cross-check and web checkpoint diagnostics"),
    "export-quality": ("notebooks/export_quality.py", "Export OpenStax and FineMath document-source comparisons"),
    "export-utility": ("notebooks/export_utility.py", "Export direct document training-utility comparisons"),
}


def delegate(script, arguments):
    """Pass argument boundaries and the selected Python environment unchanged."""
    return subprocess.call([sys.executable, str(SCRIPTS / script), *arguments])


def is_extension(config):
    import yaml
    spec = yaml.safe_load(Path(config).read_text())
    return "c_data" in spec, spec


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    for name, (_, help_text) in DELEGATED.items():
        # The implementation owns these options, including --help.
        commands.add_parser(name, help=help_text, add_help=False)
    prepare = commands.add_parser("prepare", help="Prepare the dataset selected by --config")
    prepare.add_argument("--config", default="configs/pilot.yaml")
    train = commands.add_parser("train", help="Train or resume SFT on A1")
    train.add_argument("--config", default="configs/pilot.yaml")
    train.add_argument("--stop-after-steps", type=int)
    train.add_argument("--subset-size", type=int)
    train.add_argument("--run-suffix", default="")
    train.add_argument("--check-only", action="store_true", help="Validate data, effective steps and checkpoint/resume settings on CPU")
    train.add_argument("--checkpoint-blob-config", type=Path, help="Use verified parallel reads from the same blob for checkpoint reads")
    score = commands.add_parser("score", help="Score pilot examples with the reference or an SFT checkpoint")
    score.add_argument("--config", default="configs/pilot.yaml")
    score.add_argument("--checkpoint", help="Checkpoint path, or 'final'; omit for reference")
    score.add_argument("--sanity-only", action="store_true")
    analyze = commands.add_parser("analyze", help="Analyze pilot or general-domain measurements")
    analyze.add_argument("--config", default="configs/pilot.yaml")
    analyze.add_argument("--step", type=int, help="Pilot diagnostic checkpoint only")
    gradients = commands.add_parser("gradients", help="Measure per-example FP32 loss and gradients")
    gradients.add_argument("--config", required=True)
    gradients.add_argument("--model", required=True, choices=["reference", "sft"])
    web_score = commands.add_parser("score-web", help="Score frozen web windows / target validation with ref or an SFT checkpoint")
    web_score.add_argument("--config", default="configs/web_retrieval_eval.yaml")
    web_score.add_argument("--model", required=True, choices=["reference", "sft"])
    web_score.add_argument("--shard-index", type=int, default=0)
    web_score.add_argument("--sets", nargs="+", choices=["web", "validation"], default=["web", "validation"])
    web_score.add_argument("--checkpoint")
    web_score.add_argument("--document-sanity",action="store_true",help="Validate complete document NLL while retaining per-window deviations")
    web_analysis = commands.add_parser("analyze-web", help="Analyze completed document retrieval and target-validation scores")
    web_analysis.add_argument("--config", default="configs/web_retrieval_eval.yaml")
    web_grad = commands.add_parser("gradients-web", help="Whole-document gradient diagnostics on a fixed random subset")
    web_grad.add_argument("--config", default="configs/web_retrieval_gradients.yaml")
    web_grad.add_argument("--model", required=True, choices=["reference", "sft"])
    qa_diagnostic = commands.add_parser("score-probe-qa", help="Cross-evaluate the web probe on frozen legacy A2/UltraChat")
    qa_diagnostic.add_argument("--config", default="configs/probe_diagnostics.yaml")
    probe_analysis = commands.add_parser("analyze-probe", help="Analyze QA cross-evaluation and web checkpoint diagnostics")
    probe_analysis.add_argument("--config", default="configs/probe_diagnostics.yaml")
    probe_analysis.add_argument("--qa-only", action="store_true", help="Summarize completed QA scores while web scoring continues")
    quality_prepare = commands.add_parser("prepare-quality", help="Prepare fixed OpenStax/FineMath document sources")
    quality_prepare.add_argument("--config", default="configs/document_quality.yaml")
    quality_score = commands.add_parser("score-quality", help="Score prepared OpenStax/FineMath with the existing models")
    quality_score.add_argument("--config", default="configs/document_quality.yaml")
    quality_score.add_argument("--model", required=True, choices=["reference", "sft"])
    quality_analysis = commands.add_parser("analyze-quality", help="Analyze the two document-source comparisons")
    quality_analysis.add_argument("--config", default="configs/document_quality.yaml")
    utility_prepare = commands.add_parser("prepare-utility", help="Prepare exact-token document training recipes")
    utility_prepare.add_argument("--config", default="configs/document_utility.yaml")
    utility_train = commands.add_parser("train-utility", help="Train/resume one document-source utility arm")
    utility_train.add_argument("--config", default="configs/document_utility.yaml")
    utility_train.add_argument("--arm", required=True, choices=["general","openwebmath","openstax","finemath"])
    utility_train.add_argument("--seed", required=True, type=int)
    utility_train.add_argument("--checkpoint-blob-config", type=Path)
    utility_train.add_argument("--stop-after-steps", type=int)
    utility_train.add_argument("--check-only", action="store_true")
    utility_train.add_argument("--direct-checkpoint-io", action="store_true", help="Use verified block upload/readback for checkpoint transport")
    utility_score = commands.add_parser("score-utility", help="Score held-out MATH answers after document adaptation")
    utility_score.add_argument("--config", default="configs/document_utility.yaml")
    utility_score.add_argument("--arm", choices=["reference","general","openwebmath","openstax","finemath"], required=True)
    utility_score.add_argument("--seed", type=int, default=42)
    utility_analysis = commands.add_parser("analyze-utility", help="Compare math/general direct training utility across seeds")
    utility_analysis.add_argument("--config", default="configs/document_utility.yaml")
    export = commands.add_parser("export", help="Export a run's loss notebook; append gradients separately")
    export.add_argument("--run-dir", required=True, type=Path)
    export.add_argument("--output-root", type=Path, help="Pilot exporter only")
    export.add_argument("--overwrite", action="store_true")
    return root


def main(argv=None):
    root = parser()
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] in DELEGATED:
        return delegate(DELEGATED[arguments[0]][0], arguments[1:])
    args = root.parse_args(arguments)
    if args.command == "prepare-utility":
        from data_picking.utility_data import prepare
        prepare(args.config)
        return 0
    if args.command == "train-utility":
        from data_picking.utility_training import train
        if not args.check_only and args.checkpoint_blob_config is None:
            root.error("--checkpoint-blob-config is required for validated checkpoint I/O on this mount")
        if args.direct_checkpoint_io:
            from data_picking.utility_checkpoint_io import install_training_io
            with install_training_io():
                train(args.config,args.arm,args.seed,args.checkpoint_blob_config,args.stop_after_steps,args.check_only)
        else:
            train(args.config,args.arm,args.seed,args.checkpoint_blob_config,args.stop_after_steps,args.check_only)
        return 0
    if args.command == "score-utility":
        from data_picking.utility_scoring import score
        score(args.config,args.arm,args.seed)
        return 0
    if args.command == "analyze-utility":
        from data_picking.utility_analysis import analyze
        analyze(args.config)
        return 0
    if args.command == "prepare-quality":
        from data_picking.quality_data import download, prepare
        download(args.config)
        prepare(args.config)
        return 0
    if args.command == "score-quality":
        from data_picking.quality_scoring import score
        score(args.config, args.model)
        return 0
    if args.command == "analyze-quality":
        from data_picking.quality_analysis import analyze
        analyze(args.config)
        return 0
    if args.command == "score-probe-qa":
        from data_picking.probe_diagnostics import score_qa
        score_qa(args.config)
        return 0
    if args.command == "analyze-probe":
        from data_picking.probe_diagnostic_analysis import analyze
        analyze(args.config, qa_only=args.qa_only)
        return 0
    if args.command == "score-web":
        if args.document_sanity:
            from data_picking.document_sanity import score_document_checked
            score_document_checked(args.config,args.model,args.shard_index,args.sets,args.checkpoint)
        else:
            from data_picking.web_scoring import score_web
            score_web(args.config, args.model, args.shard_index, args.sets, args.checkpoint)
        return 0
    if args.command == "analyze-web":
        from data_picking.web_analysis import analyze_web
        analyze_web(args.config)
        return 0
    if args.command == "gradients-web":
        from data_picking.web_gradients import measure_web_gradients
        measure_web_gradients(args.config, args.model)
        return 0
    if args.command == "export":
        metrics = json.loads((args.run_dir / "metrics/metrics.json").read_text())
        extension = "comparisons" in metrics
        if extension and args.output_root:
            root.error("--output-root is only supported for pilot exports")
        forwarded = ["--run-dir", str(args.run_dir)]
        if args.overwrite:
            forwarded.append("--overwrite")
        if args.output_root:
            forwarded += ["--output-root", str(args.output_root)]
        script = "notebooks/export_generalization.py" if extension else "notebooks/export_pilot.py"
        return delegate(script, forwarded)

    extension, spec = is_extension(args.config)
    if spec.get("dataset_kind") == "web_retrieval":
        if args.command != "prepare":
            root.error("web_retrieval is a data-only bundle; use prepare, then configure training/evaluation separately")
        from data_picking.web_retrieval import prepare_bundle
        prepare_bundle(args.config)
        return 0
    if args.command in {"train", "score"} and extension:
        root.error(f"{args.command} requires a pilot/training config, not an evaluation extension")
    if args.command == "gradients" and not extension:
        root.error("gradients requires an evaluation extension config")
    if args.command == "prepare":
        if not extension:
            from data_picking.common import load_config
            from data_picking.data import prepare
            prepare(load_config(args.config))
        elif spec["c_data"].get("kind", "dolly") == "dolly":
            from data_picking.generalization_data import prepare_extension
            prepare_extension(args.config)
        elif spec["c_data"].get("kind") == "ultrachat":
            from data_picking.ultrachat_data import prepare_ultrachat
            prepare_ultrachat(args.config)
        else:
            root.error(f"Unsupported c_data.kind: {spec['c_data'].get('kind')}")
    elif args.command == "train":
        from data_picking.common import load_config
        from data_picking.training_plan import write_training_plan
        cfg = load_config(args.config)
        plan = write_training_plan(cfg, args.subset_size, args.run_suffix)
        print("TRAINING_PREFLIGHT", json.dumps(plan, ensure_ascii=False), flush=True)
        if args.check_only:
            return 0
        from data_picking.training import train
        if args.checkpoint_blob_config:
            from data_picking.checkpoint_transport import CheckpointReader
            with CheckpointReader(cfg,args.checkpoint_blob_config,args.run_suffix).install():
                train(cfg,args.stop_after_steps,args.subset_size,args.run_suffix)
        else:
            train(cfg, args.stop_after_steps, args.subset_size, args.run_suffix)
    elif args.command == "score":
        from data_picking.common import load_config, paths, read_json
        from data_picking.scoring import score
        cfg = load_config(args.config)
        checkpoint = args.checkpoint
        if checkpoint == "final":
            checkpoint = read_json(paths(cfg)["run"] / "training_complete.json")["checkpoint"]
        score(cfg, checkpoint, args.sanity_only)
    elif args.command == "analyze":
        if extension:
            if args.step is not None:
                root.error("--step is only supported for pilot analyses")
            from data_picking.generalization_analysis import analyze_extension
            analyze_extension(args.config)
        else:
            from data_picking.common import load_config
            from data_picking.analysis import analyze
            analyze(load_config(args.config), args.step)
    elif args.command == "gradients":
        from data_picking.gradient_patterns import measure
        measure(args.config, args.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Run/restart the entire pilot. Source scripts/env.sh before invocation."""
import argparse
import fcntl
import hashlib
import os
import subprocess
import sys
from pathlib import Path

from data_picking.common import load_config, paths, read_json


parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/pilot.yaml")
parser.add_argument("--score-intermediate", action="store_true", help="Also score saved intermediate checkpoints as diagnostics")
args = parser.parse_args()
cfg = load_config(args.config)
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    raise SystemExit("Set CUDA_VISIBLE_DEVICES explicitly to an available local GPU")
p = paths(cfg)
lock_path = Path(os.environ["TMPDIR"]) / ("run-" + hashlib.sha256(str(p["run"]).encode()).hexdigest() + ".lock")
run_lock = lock_path.open("a+")
try:
    fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    raise SystemExit(f"A local pipeline for this run is already active: {p['run']}")
run_lock.seek(0)
run_lock.truncate()
run_lock.write(str(os.getpid()) + "\n")
run_lock.flush()
(p["run"] / "logs").mkdir(parents=True, exist_ok=True)


def run(script, *extra):
    command = [sys.executable, f"scripts/{script}.py", "--config", args.config, *extra]
    with (p["run"] / "logs/pipeline.log").open("a") as log:
        log.write("COMMAND " + repr(command) + "\n")
        log.flush()
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        if process.wait():
            raise SystemExit(f"Failed: {script}; inspect {p['run'] / 'logs/pipeline.log'}")


run("download_assets")
run("prepare_data")
run("score")
run("train_sft")
run("score", "--checkpoint", "final")
run("analyze")
final_step = read_json(p["run"] / "training_complete.json")["global_step"]
for checkpoint in sorted((p["run"] / "checkpoints").glob("step-*")) if args.score_intermediate else []:
    if not (checkpoint / "COMPLETED.json").exists():
        continue
    step = read_json(checkpoint / "manifest.json")["global_step"]
    if step != final_step:
        run("score", "--checkpoint", str(checkpoint))
        run("analyze", "--step", str(step))

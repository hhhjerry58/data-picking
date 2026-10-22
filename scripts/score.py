import argparse
from data_picking.common import load_config, paths, read_json
from data_picking.scoring import score

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/pilot.yaml")
parser.add_argument("--checkpoint", help="Checkpoint path, or 'final'; omit to score reference")
parser.add_argument("--sanity-only", action="store_true")
args = parser.parse_args()
cfg = load_config(args.config)
checkpoint = args.checkpoint
if checkpoint == "final":
    checkpoint = read_json(paths(cfg)["run"] / "training_complete.json")["checkpoint"]
score(cfg, checkpoint, args.sanity_only)

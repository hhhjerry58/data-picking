import argparse
from data_picking.common import load_config
from data_picking.training import train

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/pilot.yaml")
parser.add_argument("--stop-after-steps", type=int)
parser.add_argument("--subset-size", type=int)
parser.add_argument("--run-suffix", default="")
args = parser.parse_args()
train(load_config(args.config), args.stop_after_steps, args.subset_size, args.run_suffix)

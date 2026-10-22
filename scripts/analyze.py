import argparse
from data_picking.common import load_config
from data_picking.analysis import analyze

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/pilot.yaml")
parser.add_argument("--step", type=int)
args = parser.parse_args()
analyze(load_config(args.config), args.step)

import argparse
from data_picking.common import load_config
from data_picking.data import prepare

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/pilot.yaml")
args = parser.parse_args()
prepare(load_config(args.config))

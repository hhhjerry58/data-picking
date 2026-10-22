import argparse
from data_picking.ultrachat_data import prepare_ultrachat

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/ultrachat_extension.yaml")
args = parser.parse_args()
prepare_ultrachat(args.config)

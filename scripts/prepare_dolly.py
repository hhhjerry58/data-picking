import argparse
from data_picking.generalization_data import prepare_extension

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/dolly_extension.yaml")
args = parser.parse_args()
prepare_extension(args.config)

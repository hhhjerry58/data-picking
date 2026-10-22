import argparse
from data_picking.generalization_analysis import analyze_extension

parser=argparse.ArgumentParser()
parser.add_argument("--config",default="configs/dolly_extension.yaml")
args=parser.parse_args()
analyze_extension(args.config)

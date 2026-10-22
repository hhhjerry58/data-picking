import argparse
from data_picking.gradient_patterns import measure

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/dolly_extension.yaml")
parser.add_argument("--model", required=True, choices=["reference", "sft"])
args = parser.parse_args()
measure(args.config, args.model)

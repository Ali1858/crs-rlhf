from utils import read_yaml, parse_additional_args, print_yaml_config
import argparse
from training_datasets.dataset_utils import load_rm_dataset

def main(conf):
    print_yaml_config(conf)
    train_ds,eval_ds = load_rm_dataset(conf)

if __name__ == "__main__":
    config = {}
    parser = argparse.ArgumentParser(description="Parse configuration")
    parser.add_argument("--overrides", nargs='+', help="Override configurations (key=value)", default=[])
    args, remaining = parser.parse_known_args()

    overrides = dict(override.split('=') for override in args.overrides)
    conf = read_yaml('./configs/rm_config.yaml')
    config.update(conf["default"])
    config.update(overrides)

    parser = parse_additional_args(config)
    args = parser.parse_args(remaining)

    main(args)
import yaml
from distutils.util import strtobool
import argparse


def print_yaml_config(config):
    print(yaml.dump(config, default_flow_style=False))


def read_yaml(config_path):
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)
    return config


"""Taken from:
https://github.com/LAION-AI/Open-Assistant/blob/main/model/model_training/trainer_sft.py#L169"""
def parse_additional_args(conf):
    # Override config from command-line
    parser = argparse.ArgumentParser()
    for key, value in conf.items():
        type_ = type(value) if value is not None else str
        if type_ == bool:
            type_ = _strtobool
        parser.add_argument(f"--{key}", type=type_, default=value)
        # Allow --no-{key}  to remove it completely
        parser.add_argument(f"--no-{key}", dest=key, action="store_const", const=None)
    return parser

def _strtobool(x):
    return bool(strtobool(x))


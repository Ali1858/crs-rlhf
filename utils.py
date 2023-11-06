import os
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


def init_or_resume_from(args):
    if args.init_from_adapter is not None:
        args.init_from_adapter = os.path.join(args.output_dir,args.init_from_adapter,args.adapter_number) 
        print(f'{"==="*10} Initialize from adapter {args.init_from_adapter}')
    elif args.checkpoint_name is not None:
        args.resume_from_checkpoint = os.path.join(args.output_dir,args.checkpoint_name,args.checkpoint_number) 
        print(f'{"==="*10} resuming from checkpoint {args.resume_from_checkpoint}')

    if args.adpater_name is not None:
        args.adpater_path = os.path.join(args.adpater_name,args.adapter_number)
        args.merged_adapter_path = os.path.join(args.adpater_name,'merged')


def parse_arguments(config_path='config.yaml'):
    "Parse cli arguments and load the yaml configs"
    config = {}
    parser = argparse.ArgumentParser(description="Parse configuration")
    parser.add_argument("--config_subset", type=str, help="Subset of the configs to use")
    parser.add_argument("--name_suffix", type=str, default="", help="Suffix name while performing multiple experiment. Keep it simple because by default wandb store configs of each train")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args, remaining = parser.parse_known_args()
    conf = read_yaml(config_path)
    config.update(conf["default"])
    config.update(conf[args.config_subset])
    config["name_suffix"] = args.name_suffix
    config["debug"] = args.debug
    config["subset"] = args.config_subset


    return config, remaining


def debug_configurations(args):
    if args.debug:
        args.report_to = "none"
        args.train_batch = 1
        args.eval_batch = 1
        args.gradient_accumulation_steps = 1
        args.num_train_epochs = 1
        args.log_steps = 100
        args.eval_steps = 100
        args.save_steps = 100


def save_trained_model(trainer, output_dir):
    trainer.model.save_pretrained(os.path.join(output_dir, "final_checkpoint/"))
    trainer.tokenizer.save_pretrained(os.path.join(output_dir, "final_checkpoint/"))


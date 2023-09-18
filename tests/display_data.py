import os
import argparse

from utils import read_yaml, parse_additional_args
from training_datasets.dataset_utils import load_rm_dataset, load_sft_dataset
from training_datasets.collators import RankingDataCollator, DialogueDataCollator
from model_training.training_utils import get_tokenizer
from constants import TOKENIZER_SEPECIAL_TOKENS

def main(conf):
    if conf.subset in ["pre_sft","sft"]:
        display_sft(conf)
    elif conf.subset in ["rm"]:
        display_rm(conf)


def display_data(data,collator,tokenizer,each_fn):
    for k,v in data.items():
        print(f"{'==='*10} Displaying one instance for {k} in ...")
        item = collator([v[0]])
        if len(item) > 1:
            item = item[0]
        each_fn(item,tokenizer)

"====== display sft"  
def display_sft(config_ns):
    tokenizer, eos_token= get_tokenizer(config_ns,TOKENIZER_SEPECIAL_TOKENS)
    train, eval = load_sft_dataset(config_ns,eos_token)

    train_collate_fn = DialogueDataCollator(
        tokenizer,
        max_length=config_ns.collator["max_length"],
        random_offset_probability=config_ns.collator["random_offset_probability"],
        label_masking=config_ns.collator["label_masking"],
        samples_mixing=config_ns.collator["samples_mixing"],
        pad_to_multiple_of=16,
        use_system_prefix=config_ns.collator["use_system_prefix"],
        system_prefix=config_ns.collator["system_prefix"],
    )
    
    if config_ns.collator.get("val_max_length") is None:
        config_ns.collator["val_max_length"] = config_ns.collator["max_length"]
    
    eval_collate_fn = DialogueDataCollator(
        tokenizer,
        max_length=config_ns.collator["val_max_length"],
        random_offset_probability=config_ns.collator["random_offset_probability"],
        label_masking=config_ns.collator["label_masking"],
        samples_mixing=False,
        use_system_prefix=config_ns.collator["use_system_prefix"],
        system_prefix=config_ns.collator["system_prefix"],
        )
    
    display_data(eval,eval_collate_fn,tokenizer,display_each_sft)
    display_data({"train":train},train_collate_fn,tokenizer,display_each_sft)


def display_each_sft(item,tokenizer):
    inp = item["input_ids"][0].view(-1)
    target = item["targets"][0].view(-1)
    mask = item["label_masks"][0].view(-1).bool()
    print(f'inp {tokenizer.decode(inp)} -->\n masked target {tokenizer.decode(inp[mask])}')
    print('***'*10)
    print(f'target {tokenizer.decode(target)} -->\n masked target {tokenizer.decode(target[mask])}')


"====== display rm"
"====== display sft"  
def display_rm(config_ns):
    tokenizer, eos_token= get_tokenizer(config_ns,TOKENIZER_SEPECIAL_TOKENS)
    train, eval = load_rm_dataset(config_ns)

    train_collate_fn = RankingDataCollator(
        tokenizer,
        max_length=config_ns.collator["max_length"],
        pad_to_multiple_of=16,
        max_replies=config_ns.max_replies
    )
    eval_collate_fn = RankingDataCollator(
        tokenizer,
        max_length=config_ns.collator["max_length"],
        pad_to_multiple_of=16,
        max_replies=config_ns.max_replies
    )
    
    display_data(eval,eval_collate_fn,tokenizer,display_each_rm)
    display_data({"train":train},train_collate_fn,tokenizer,display_each_rm)


def display_each_rm(item,tokenizer):
    inp = item["input_ids"]
    for i in inp:
        print(f' {tokenizer.decode(i)}')
        print('====')


if __name__ == "__main__":
    config = {}
    parser = argparse.ArgumentParser(description="Parse configuration")
    parser.add_argument("--config_subset",type=str, help="Subset of the configs to use")

    args, remaining = parser.parse_known_args()
    subset = args.config_subset
    conf = read_yaml('./configs/config.yaml')
    config.update(conf[subset])
    config.update(conf["common"])
    config["subset"] = subset


    parser = parse_additional_args(config)
    args = parser.parse_args(remaining)
    main(args)
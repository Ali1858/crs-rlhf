import os
import argparse

from utils import read_yaml, parse_additional_args,parse_arguments,init_or_resume_from
from training_datasets.dataset_utils import load_rm_dataset, load_sft_dataset
from training_datasets.collators import RankingDataCollator, DialogueDataCollator
from model_training.training_utils import get_model_and_tokenizer
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
        if type(item)==int and len(item) > 1:
            item = item[0]
        each_fn(item,tokenizer)

"====== display sft"  
def display_sft(config_ns):
    train, eval = load_sft_dataset(config_ns,special_tokens["eos_token"])

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
    config, remaining_args = parse_arguments()
    parser = parse_additional_args(config)
    args = parser.parse_args(remaining_args)

    init_or_resume_from(args)

    device_map = "auto"#"{"":0}"
    assert "llama" in args.model_name.lower(), "Currently only llama model supported"
    special_tokens = TOKENIZER_SEPECIAL_TOKENS["llama"]
    tokenizer = get_model_and_tokenizer(device_map,args,special_tokens,only_tokenizer=True)

    main(args)
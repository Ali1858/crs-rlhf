"""rewritten from:
https://github.com/LAION-AI/Open-Assistant/blob/main/model/model_training/trainer_sft.py
""" 
import os
import argparse
from functools import partial

import torch
import evaluate
from transformers import TrainingArguments
from transformers.training_args import OptimizerNames

from training_datasets.dataset_utils import load_sft_dataset
from training_datasets.collators import DialogueDataCollator
from model_training.trainers import SFTTrainer
from model_training.training_utils import get_model, get_tokenizer
from utils import read_yaml, parse_additional_args, print_yaml_config
from constants import TOKENIZER_SEPECIAL_TOKENS


def compute_accuracy(eval_pred, accuracy):
    preds, labels = eval_pred
    mask = labels > 0
    preds, labels = preds[mask], labels[mask]
    return {"accuracy":accuracy.compute(predictions=preds, references=labels)}


def preprocess_logits_for_metrics(logits, labels):
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids


def main(conf):
    print(f"\n{'==='*10} Following are the configuration for training{'==='*10}")
    print_yaml_config(conf)
    # needs to happen before model loading in case of stage 3 training
    optimizer =  OptimizerNames.ADAMW_BNB if conf.int8_training else OptimizerNames.ADAMW_HF
    accuracy = evaluate.load("accuracy")
    # device_map = "auto"#{"":1}

    device_map = {'model.embed_tokens': 0, 'model.layers.0': 1, 'model.layers.1': 1,
            'model.layers.2': 1, 'model.layers.3': 1, 'model.layers.4': 1, 
            'model.layers.5': 1, 'model.layers.6': 1, 'model.layers.7': 1, 
            'model.layers.8': 1, 'model.layers.9': 1, 'model.layers.10': 1,
            'model.layers.11': 1, 'model.layers.12': 1, 'model.layers.13': 1,
            'model.layers.14': 1, 'model.layers.15': 1, 'model.layers.16': 1,
            'model.layers.17': 1, 'model.layers.18': 1, 'model.layers.19': 1,
            'model.layers.20': 1, 'model.layers.21': 1, 'model.layers.22': 1,
            'model.layers.23': 1, 'model.layers.24': 1, 'model.layers.25': 1,
            'model.layers.26': 1, 'model.layers.27': 1, 'model.layers.28': 1,
            'model.layers.29': 1, 'model.layers.30': 1, 'model.layers.31': 1,
            'model.norm': 1, 'lm_head': 1}
    args = TrainingArguments(
        output_dir=conf.output_dir,
        num_train_epochs=conf.num_train_epochs,
        lr_scheduler_type=conf.lr_scheduler_type,
        warmup_steps=conf.warmup_steps,
        learning_rate=float(conf.lr),
        optim=optimizer,
        fp16=conf.dtype in ["fp16", "float16"],
        bf16=conf.dtype in ["bf16", "bfloat16"],
        gradient_checkpointing=conf.gradient_checkpointing,
        gradient_accumulation_steps=conf.gradient_accumulation_steps,
        per_device_train_batch_size=conf.train_batch,
        per_device_eval_batch_size=conf.eval_batch,
        adam_beta1=conf.adam_beta1,
        adam_beta2=conf.adam_beta2,
        adam_epsilon=float(conf.adam_epsilon),
        weight_decay=conf.weight_decay,
        logging_steps=conf.log_steps,
        evaluation_strategy="steps",
        eval_steps=conf.eval_steps,
        save_strategy="steps",
        save_steps=conf.save_steps,
        eval_accumulation_steps=conf.eval_accumulation_steps,
        resume_from_checkpoint=conf.resume_from_checkpoint,
        report_to=conf.report_to
    )

    tokenizer, eos_token= get_tokenizer(conf,TOKENIZER_SEPECIAL_TOKENS)
    train_ds , eval_ds = load_sft_dataset(conf,eos_token)
    model = get_model(tokenizer, device=device_map,config=conf)
    
    train_collate_fn = DialogueDataCollator(
        tokenizer,
        max_length=conf.collator["max_length"],
        random_offset_probability=conf.collator["random_offset_probability"],
        label_masking=conf.collator["label_masking"],
        samples_mixing=conf.collator["samples_mixing"],
        pad_to_multiple_of=16,
        use_system_prefix=conf.collator["use_system_prefix"],
        system_prefix=conf.collator["system_prefix"],
    )
    
    if conf.collator.get("val_max_length") is None:
        conf.collator["val_max_length"] = conf.collator["max_length"]
    
    eval_collate_fn = DialogueDataCollator(
        tokenizer,
        max_length=conf.collator["val_max_length"],
        random_offset_probability=conf.collator["random_offset_probability"],
        label_masking=conf.collator["label_masking"],
        samples_mixing=False,
        use_system_prefix=conf.collator["use_system_prefix"],
        system_prefix=conf.collator["system_prefix"],
        )

    wandb_suffix = ""
    if conf.debug:
        wandb_suffix = "_debug"
        for module in model.modules():
            if isinstance(module, torch.nn.Embedding):
                for para in module.parameters():
                    print(para.requires_grad)
                print(f'Embedding {module.weight.shape} and {module.weight.dtype}')
    
    if not conf.debug:
        import wandb
        wandb_project_name = f"supervised-finetuning{wandb_suffix}"
        wandb.init(
            project=wandb_project_name,
            entity=None,
            resume=conf.resume_from_checkpoint,
            name=conf.name,
            config=conf,
        )

    trainer = SFTTrainer(
    model=model,
    args=args,
    train_collate_fn=train_collate_fn,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=eval_collate_fn,
    tokenizer=tokenizer,
    compute_metrics=partial(compute_accuracy, accuracy=accuracy),
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,)
    return trainer


def train(trainer,conf):
    trainer.train(resume_from_checkpoint=conf.resume_from_checkpoint)
    trainer.model.save_pretrained(os.path.join(conf.output_dir, "final_checkpoint/"))
    trainer.tokenizer.save_pretrained(os.path.join(conf.output_dir, "final_checkpoint/"))


if __name__ == "__main__":
    config = {}
    parser = argparse.ArgumentParser(description="Parse configuration")
    parser.add_argument("--config_subset", type=str, help="Subset of the configs to use")
    parser.add_argument("--name_suffix", type=str, default="", help="Suffix name while performing multiple experiment. Keep it  simple because by default wandb store configs of each train")

    args, remaining = parser.parse_known_args()

    config_subset = args.config_subset
    conf = read_yaml('./configs/config.yaml')
    config.update(conf["common"])
    config.update(conf[config_subset])
    config["name_suffix"] = args.name_suffix

    for k,v in config.pop("peft_config_additional").items():
        config["peft_config"][k]=v


    parser = parse_additional_args(config)
    args = parser.parse_args(remaining)


    if config_subset == "sft":
        args.resume_from_checkpoint = os.path.join(args.output_dir,args.checkpoint_name,"final_checkpoint") 

    debug_tag = "_dbug" if args.debug else ""
    args.name = f"{args.name}{debug_tag}{args.name_suffix}"
    args.output_dir = os.path.join(args.output_dir,args.name)


    if args.debug:
        args.report_to="none"
        args.train_batch=1
        args.eval_batch=1
        args.gradient_accumulation_steps = 1
        args.num_train_epochs=1
        args.log_steps=100
        args.eval_steps=100
        args.save_steps=100

    trainer = main(args)
    train(trainer,args)

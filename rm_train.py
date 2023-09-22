import os
import argparse

import torch
from transformers import TrainingArguments
from transformers.training_args import OptimizerNames

from training_datasets.dataset_utils import load_rm_dataset
from training_datasets.collators import RankingDataCollator
from model_training.trainers import RMTrainer
from model_training.training_utils import merge_and_save_peft_model, get_model, get_tokenizer
from utils import read_yaml, parse_additional_args, print_yaml_config
from constants import TOKENIZER_SEPECIAL_TOKENS


def main(conf):
    print(f"\n{'==='*10} Following are the configuration for training{'==='*10}")
    print_yaml_config(conf)

    # needs to happen before model loading in case of stage 3 training
    optimizer =  OptimizerNames.ADAMW_BNB if conf.int8_training else OptimizerNames.ADAMW_HF
    device_map = "auto"#"{"":0}"

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
        report_to=conf.report_to,
    )
    conf.model_name = conf.merged_adapter_path
    merge_and_save_peft_model(conf)
    tokenizer, eos_token= get_tokenizer(conf,TOKENIZER_SEPECIAL_TOKENS,add_additional_special_tokens=False)
    train_ds , eval_ds = load_rm_dataset(conf)
    model = get_model(tokenizer, device=device_map,config=conf, need_embedding_resize=False,reward_model=True)
    # metrics,preprocess_function = get_sft_metrics(conf.metrics)
    
    train_collate_fn = RankingDataCollator(
        tokenizer,
        max_length=conf.collator["max_length"],
        pad_to_multiple_of=16,
        max_replies=conf.max_replies
    )
    eval_collate_fn = RankingDataCollator(
        tokenizer,
        max_length=conf.collator["max_length"],
        pad_to_multiple_of=16,
        max_replies=conf.max_replies
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
        resume = None
        if conf.checkpoint_name:
            resume = conf.checkpoint_name +'_'+ conf.checkpoint_number

        os.environ["WANDB_WATCH"] = "all"
        wandb_project_name = f"reward-model{wandb_suffix}"
        wandb.init(
            project=wandb_project_name,
            entity=None,
            resume=resume,
            name=conf.name,
            config=conf,
            save_code=True,

        )

    trainer = RMTrainer(
    model=model,
    args=args,
    train_collate_fn=train_collate_fn,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=eval_collate_fn,
    tokenizer=tokenizer,
    # compute_metrics=partial(compute_metrics, metrics=metrics, preprocess_fns=preprocess_function))
    )
    return trainer


def train(trainer,conf):
    trainer.train(resume_from_checkpoint=conf.resume_from_checkpoint is not None)
    trainer.model.save_pretrained(os.path.join(conf.output_dir, "final_checkpoint/"))
    trainer.tokenizer.save_pretrained(os.path.join(conf.output_dir, "final_checkpoint/"))

if __name__ == "__main__":
    config = {}
    parser = argparse.ArgumentParser(description="Parse configuration")
    parser.add_argument("--config_subset",type=str, help="Subset of the configs to use")
    parser.add_argument("--name_suffix", type=str, default="", help="Suffix name while performing multiple experiment. Keep it  simple because by default wandb store configs of each train")

    args, remaining = parser.parse_known_args()

    config_subset = args.config_subset
    conf = read_yaml('./configs/config.yaml')
    config.update(conf["common"])
    config.update(conf[args.config_subset])
    config["name_suffix"] = args.name_suffix


    for k,v in config.pop("peft_config_additional").items():
        config["peft_config"][k]=v


    parser = parse_additional_args(config)
    args = parser.parse_args(remaining)

    if args.checkpoint_name is not None:
        if args.checkpoint_number is None:
            checkpoint_number="final_checkpoint"
        else:
            checkpoint_number = args.checkpoint_number
        args.resume_from_checkpoint = os.path.join(args.output_dir,args.checkpoint_name,checkpoint_number) 
        print(f'{"==="*10} resuming from checkpoint {args.resume_from_checkpoint}')

    args.adpater_path = os.path.join(args.output_dir,config["adpater_name"],'final_checkpoint')
    args.merged_adapter_path = os.path.join(args.output_dir,config["adpater_name"],'merged')

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
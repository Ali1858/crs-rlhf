import os
import argparse

import torch
from transformers import TrainingArguments
from transformers.training_args import OptimizerNames

from training_datasets.dataset_utils import load_rm_dataset
from training_datasets.collators import RankingDataCollator, AbsoluteScoreDataCollator
from model_training.trainers import RMTrainer, AbsRMTrainer
from model_training.training_utils import merge_and_save_peft_model, get_model_and_tokenizer
from utils import (parse_additional_args, print_yaml_config, 
                   parse_arguments, init_or_resume_from,
                    debug_configurations, save_trained_model)
from constants import TOKENIZER_SEPECIAL_TOKENS


def create_trainer(conf):
    print(f"\n{'==='*10} Following are the configuration for training{'==='*10}")
    print_yaml_config(conf)

    # needs to happen before model loading in case of stage 3 training
    optimizer =  OptimizerNames.ADAMW_BNB
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
    assert "llama" in conf.model_name.lower(), "Currently only llama model supported"
    special_tokens = TOKENIZER_SEPECIAL_TOKENS["llama"]
    merge_and_save_peft_model(conf)
    train_ds , eval_ds = load_rm_dataset(conf)
    model,tokenizer = get_model_and_tokenizer(device_map,conf,special_tokens, need_embedding_resize=False,reward_model=True)

    # metrics,preprocess_function = get_sft_metrics(conf.metrics)
    
    if conf.is_abs_rm:
        Trainer = AbsRMTrainer
        collate_fn = AbsoluteScoreDataCollator(
            tokenizer,
            max_length=conf.collator["max_length"],
            pad_to_multiple_of=16,
        )
    else:
        Trainer = RMTrainer
        collate_fn = RankingDataCollator(
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

    trainer = Trainer(
    model=model,
    args=args,
    train_collate_fn=collate_fn,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=collate_fn,
    tokenizer=tokenizer,
    # compute_metrics=partial(compute_metrics, metrics=metrics, preprocess_fns=preprocess_function))
    )
    return trainer

if __name__ == "__main__":
    config, remaining_args = parse_arguments()
    parser = parse_additional_args(config)
    args = parser.parse_args(remaining_args)

    init_or_resume_from(args)

    debug_tag = "_dbug" if args.debug else ""
    args.name = f"{args.name}{debug_tag}{args.name_suffix}"
    args.output_dir = os.path.join(args.output_dir, args.name)

    debug_configurations(args)

    trainer = create_trainer(args)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint is not None)
    save_trained_model(trainer, args.output_dir)
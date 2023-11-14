import os
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

import torch
from torch.utils.data import ConcatDataset
from transformers import TrainingArguments, EarlyStoppingCallback
from transformers.training_args import OptimizerNames
import wandb

from training_datasets.dataset_utils import load_rm_dataset
from training_datasets.collators import RankingDataCollator, AbsoluteScoreDataCollator
from model_training.trainers import RMTrainer, AbsRMTrainer
from model_training.training_utils import merge_and_save_peft_model, get_model_and_tokenizer
from utils import (parse_additional_args, print_yaml_config, 
                   parse_arguments, init_or_resume_from,
                    debug_configurations, save_trained_model)
from constants import TOKENIZER_SEPECIAL_TOKENS


def ranking_reward_accuracy(eval_pred):
    logits = eval_pred.predictions
    labels = eval_pred.label_ids
    pos_scores, neg_scores = [], []
    for b_logits, b_labels in zip(logits, labels):
        b_labels = b_labels[b_labels != -100]
        b_logits = b_logits[b_logits != -100]
        for i in np.unique(b_labels):
            logits_batch = b_logits[b_labels == i]
            pos_scores.append(logits_batch[0])
            neg_scores.append(logits_batch[-1])
    pos_scores = np.array(pos_scores).reshape(-1, 1)
    neg_scores = np.array(neg_scores).reshape(-1, 1)

    metrics = {
        "pos_score": np.mean(pos_scores),
        "neg_score": np.mean(neg_scores),
        "score_diff": np.mean(pos_scores - neg_scores),
        "accuracy": np.mean(pos_scores > neg_scores),
    }
    return metrics


def abs_reward_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.squeeze()
    return {
        'mse': mean_squared_error(labels, predictions),
        'mae': mean_absolute_error(labels, predictions)
    }


def create_trainer(conf):
    print(f"\n{'==='*10} Following are the configuration for training{'==='*10}")
    print_yaml_config(conf)

    # needs to happen before model loading in case of stage 3 training
    optimizer =  OptimizerNames.ADAMW_TORCH
    device_map = "auto" #{"":0} #

    args = TrainingArguments(
        output_dir=conf.output_dir,
        num_train_epochs=conf.num_train_epochs,
        max_steps=conf.max_steps,
        lr_scheduler_type=conf.lr_scheduler_type,
        warmup_ratio=conf.warmup_ratio,
        max_grad_norm=conf.max_grad_norm,
        learning_rate=float(conf.lr),
        optim=optimizer,
        fp16=conf.dtype in ["fp16", "float16"],
        bf16=conf.dtype in ["bf16", "bfloat16"],
        gradient_checkpointing=conf.gradient_checkpointing,
        gradient_accumulation_steps=conf.gradient_accumulation_steps,
        per_device_train_batch_size=conf.train_batch,
        per_device_eval_batch_size=conf.eval_batch,
        adam_beta2=conf.adam_beta2,
        weight_decay=float(conf.weight_decay),
        logging_steps=conf.log_steps,
        evaluation_strategy="steps",
        eval_steps=conf.eval_steps,
        save_strategy="steps",
        save_steps=conf.save_steps,
        eval_accumulation_steps=conf.eval_accumulation_steps,
        resume_from_checkpoint=conf.resume_from_checkpoint,
        report_to=conf.report_to,
    )

    conf.model_name = conf.merged_adapter_path if conf.merged_adapter_path else conf.base_model_name
    assert conf.model_name, "Model name can't be null"
    assert "llama" in conf.model_name.lower(), "Currently only llama model supported"
    special_tokens = TOKENIZER_SEPECIAL_TOKENS["llama"]
    merge_and_save_peft_model(conf)
    train_ds , eval_ds = load_rm_dataset(conf)
    model,tokenizer = get_model_and_tokenizer(device_map,conf,special_tokens, need_embedding_resize=False,reward_model=True)
    if conf.is_abs_rm:
        from functools import partial
        scores = []
        abs_oversample_threshold = conf.dataset["oasst_export_abs"]["abs_oversample_threshold"]
        for i in train_ds:
            _, _, score = i
            if score <= abs_oversample_threshold:
                scores.append(score)
        loss_weight = len(train_ds)/len(scores)

        # 38852/3354 + oversampling leads to 6.30 if not use 11.5
        # only using quality 38852/9292 4.1
        # oversampled + v4 at 0.5 30690/10348 --> 2.96
        # oversampled + v3 at 0.5 30175/9912 --> 3.04
        
        metrics = abs_reward_metrics
        Trainer = partial(AbsRMTrainer,loss_wgt_and_threshold=(loss_weight, abs_oversample_threshold))
        collate_fn = AbsoluteScoreDataCollator(
            tokenizer,
            max_length=conf.collator["max_length"],
            pad_to_multiple_of=16,
        )
    else:
        metrics = ranking_reward_accuracy
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

    os.environ["WANDB_WATCH"] = "all"
    wandb_project_name = f"reward-model{wandb_suffix}"
    wandb.init(
        project=wandb_project_name,
        entity=None,
        name=conf.name,
        config=conf,
        save_code=True,
    )

    callbacks = None
    if conf.early_stopping:
        print(f'{"==="*10}Concating the eval dataset and setting EarlyStopping callback')
        args.metric_for_best_model = 'eval_accuracy'
        args.load_best_model_at_end =True
        callbacks = [EarlyStoppingCallback(early_stopping_patience = 3,early_stopping_threshold=0.001)]
        eval_ds_list = []
        for k,v in eval_ds.items():
            eval_ds_list.append(v)
        eval_ds = ConcatDataset(eval_ds_list)

    trainer = Trainer(
        model=model,
        args=args,
        train_collate_fn=collate_fn,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collate_fn,
        tokenizer=tokenizer,
        compute_metrics=metrics,
        callbacks=callbacks

    )
    return trainer

if __name__ == "__main__":
    config, remaining_args = parse_arguments()
    parser = parse_additional_args(config)
    args = parser.parse_args(remaining_args)

    init_or_resume_from(args)

    debug_tag = "_dbug" if args.debug else ""
    reward_type = "_abs" if args.is_abs_rm else "_ranking"
    args.name = f"{args.name}{reward_type}{debug_tag}{args.name_suffix}"
    args.output_dir = os.path.join(args.output_dir, args.name)

    debug_configurations(args)

    trainer = create_trainer(args)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint is not None)
    save_trained_model(trainer, args.output_dir)
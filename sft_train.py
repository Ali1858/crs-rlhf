"""rewritten from:
https://github.com/LAION-AI/Open-Assistant/blob/main/model/model_training/trainer_sft.py
""" 
import os
from functools import partial

import torch
import evaluate
from torch.utils.data import ConcatDataset
from transformers import TrainingArguments
from transformers.training_args import OptimizerNames

from training_datasets.dataset_utils import load_sft_dataset
from training_datasets.collators import DialogueDataCollator
from model_training.trainers import SFTTrainer
from model_training.training_utils import get_model_and_tokenizer
from utils import (parse_additional_args, print_yaml_config, 
                   parse_arguments, init_or_resume_from,
                    debug_configurations, save_trained_model)
from constants import TOKENIZER_SEPECIAL_TOKENS


def compute_accuracy(eval_pred, accuracy):
    preds, labels = eval_pred
    mask = labels > 0
    preds, labels = preds[mask], labels[mask]
    return {"accuracy":accuracy.compute(predictions=preds, references=labels)}


def preprocess_logits_for_metrics(logits, labels):
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids


def optuna_hp_space(trial):
    return {
        "num_train_epochs" :trial.suggest_categorical("num_train_epochs", [1,2,3]),
        "gradient_accumulation_steps": trial.suggest_categorical("gradient_accumulation_steps", [8, 16, 32]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "lr_scheduler_type": trial.suggest_categorical("lr_scheduler_type", ["cosine","linear"]),
        "warmup_steps": trial.suggest_categorical("warmup_steps", [30,50,100,300]),
        "weight_decay": trial.suggest_float("weight_decay", 0.000001, 0.05, log=True),
    }


def create_trainer(conf):
    print(f"\n{'==='*10} Following are the configuration for training{'==='*10}")
    print_yaml_config(conf)
    # needs to happen before model loading in case of stage 3 training
    optimizer =  OptimizerNames.ADAMW_HF
    accuracy = evaluate.load("accuracy")
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

    assert "llama" in conf.model_name.lower(), "Currently only llama model supported"
    special_tokens = TOKENIZER_SEPECIAL_TOKENS["llama"]
    train_ds , eval_ds = load_sft_dataset(conf,special_tokens["eos_token"])
    model, tokenizer = get_model_and_tokenizer(device_map,conf,special_tokens)
    
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
    
    if not conf.debug and not conf.optuna:
        import wandb        
        os.environ["WANDB_WATCH"] = "all"
        wandb_project_name = f"supervised-finetuning{wandb_suffix}"
        wandb.init(
            project=wandb_project_name,
            entity=None,
            name=conf.name,
            config=conf,
            save_code=True,
        )

    model_init = None
    if conf.optuna:
        print(f'setting model to None for hyper parameter sweeping using optuna')
        model=None
        eval_ds_list = []
        for k,v in eval_ds.items():
            eval_ds_list.append(v)
        eval_ds = ConcatDataset(eval_ds_list)

        def model_init(trail):
            model, _ = get_model_and_tokenizer(device_map,conf,special_tokens)
            return model

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_collate_fn=train_collate_fn,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=eval_collate_fn,
        tokenizer=tokenizer,
        compute_metrics=partial(compute_accuracy, accuracy=accuracy),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        model_init=model_init
    )

    return trainer

def compute_objective(metrics):
    return metrics["eval_accuracy"]["accuracy"]

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
    if args.optuna:
        best_trial = trainer.hyperparameter_search(
            direction="maximize",
            backend="optuna",
            hp_space=optuna_hp_space,
            n_trials=20,
            compute_objective=compute_objective,
            )
        print(best_trial)
    else:
        print(trainer.optimizer)
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint is not None)
        save_trained_model(trainer, args.output_dir)

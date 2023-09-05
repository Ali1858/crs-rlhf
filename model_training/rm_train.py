import os
import argparse
from functools import partial

import torch
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments
from transformers.training_args import OptimizerNames
from transformers.utils import is_datasets_available
from transformers.trainer_utils import seed_worker
import datasets

from training_datasets.dataset_utils import load_rm_dataset
from training_datasets.collators import RankingDataCollator
from model_training.training_utils import merge_and_save_peft_model, get_model, get_sft_tokenizer
from model_training.losses import RMLoss
from constants import TOKENIZER_SEPECIAL_TOKENS
from utils import read_yaml, parse_additional_args, print_yaml_config


def compute_metrics(eval_pred, preprocess_fns, metrics):
    out = {}
    for metric, preprocess_fn in zip(metrics, preprocess_fns):
        preds, labels = preprocess_fn(eval_pred)
        out = dict(**out, **metric.compute(predictions=preds, references=labels))

    return out


"""Taken from:
https://github.com/LAION-AI/Open-Assistant/blob/main/model/model_training/trainer_rm.py#L31
"""
class RMTrainer(Trainer):
    def __init__(self, model, args, train_collate_fn,**kwargs):
        super().__init__(model=model,args=args,**kwargs)
        self.train_collate_fn = train_collate_fn
        self.loss_fct = RMLoss(beta=0.001)

    def compute_loss(self, model, inputs, return_logits=False):
        batch, cu_lens = inputs

        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        ).logits

        loss = self.loss_fct(logits, cu_lens)

        return (loss, logits) if return_logits else loss
    
    
    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys,
    ):
        batch, cu_lens = inputs
        with torch.no_grad():
            batch = self._prepare_inputs(batch)
            loss, logits = self.compute_loss(model, (batch, cu_lens), return_logits=True)

        loss = loss.mean().detach()

        labels = []
        for i, (s, e) in enumerate(zip(cu_lens[:-1], cu_lens[1:])):
            labels.extend([i] * (e - s))
        # make sure labels are same as logits, needed for deepspeed
        labels = torch.tensor(labels, device=logits.device, requires_grad=False).view(-1, 1)
        return (loss, logits.T, labels.T)  # transposed to avoid truncation in evaluation_loop

    
    def get_train_dataloader(self):
        data_collator = self.train_collate_fn
        train_dataset = self.train_dataset

        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        
        train_sampler = self._get_train_sampler()
        dataloader = DataLoader(
            train_dataset,
            batch_size=self._train_batch_size,
            sampler=train_sampler,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            worker_init_fn=seed_worker,
        )
        return dataloader


def main(conf,output_dir):
    print(f"\n{'==='*10} Following are the configuration for training{'==='*10}")
    print_yaml_config(conf)

    # needs to happen before model loading in case of stage 3 training
    optimizer =  OptimizerNames.ADAMW_BNB if conf.int8_training else OptimizerNames.ADAMW_HF
    
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=conf.num_train_epochs,
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
    adpater_name = conf.base_model_name+conf.sft_adapter_suffix
    output_dir = os.path.join(conf.base_model_name+conf.sft_adapter_suffix.split('/')[0],"merged/")
    conf.model_name = output_dir

    merge_and_save_peft_model(conf,adpater_name,output_dir)
    tokenizer, eos_token= get_sft_tokenizer(conf,TOKENIZER_SEPECIAL_TOKENS,add_additional_special_tokens=False)
    train_ds , eval_ds = load_rm_dataset(conf)
    model = get_model(tokenizer, conf, need_embedding_resize=False,reward_model=True)
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

    if conf.debug:
        for module in model.modules():
            if isinstance(module, torch.nn.Embedding):
                for para in module.parameters():
                    print(para.requires_grad)
                print(f'Embedding {module.weight.shape} and {module.weight.dtype}')
            elif isinstance(module, torch.nn.Linear):
                for para in module.parameters():
                    print(para.requires_grad)
                print(f'Linear {module.weight.shape} and {module.weight.dtype}')
        
    import wandb

    wandb_name = conf.base_model_name
    wandb.init(
        project="reward-model",
        entity=None,
        resume=conf.resume_from_checkpoint,
        name=f"{wandb_name}--rm",
        config=conf,
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


def train(trainer,output_dir,conf):
    trainer.train(resume_from_checkpoint=conf.resume_from_checkpoint)
    trainer.model.save_pretrained(os.path.join(output_dir, "final_checkpoint/"))
    trainer.tokenizer.save_pretrained(os.path.join(output_dir, "final_checkpoint/"))

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

    output_dir=f"{args.base_model_name}-lora-rm"

    if args.debug:
        args.train_batch=1
        args.eval_batch=1
        args.num_train_epochs=1
        args.log_steps=100
        args.eval_steps=100
        args.save_steps=100

    trainer = main(args,output_dir)
    train(trainer,output_dir,args)
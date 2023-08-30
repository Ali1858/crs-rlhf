"""rewritten from:
https://github.com/LAION-AI/Open-Assistant/blob/main/model/model_training/trainer_sft.py
"""
        
import argparse
from functools import partial

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments
from transformers.training_args import OptimizerNames
from transformers.utils import is_datasets_available
from transformers.trainer_utils import seed_worker
import datasets

from training_datasets.dataset_utils import load_sft_dataset
from training_datasets.collators import DialogueDataCollator
from model.training_utils import get_sft_model, get_sft_tokenizer, get_sft_metrics
from constants import TOKENIZER_SEPECIAL_TOKENS
from utils import read_yaml, parse_additional_args, print_yaml_config


def compute_metrics(eval_pred, preprocess_fns, metrics):
    out = {}
    for metric, preprocess_fn in zip(metrics, preprocess_fns):
        preds, labels = preprocess_fn(eval_pred)
        out = dict(**out, **metric.compute(predictions=preds, references=labels))

    return out


def preprocess_logits_for_metrics(logits, labels):
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids


class SFTTrainer(Trainer):
    def __init__(self, model, args, train_collate_fn,**kwargs):
        super().__init__(model=model,args=args,**kwargs)
        self.train_collate_fn = train_collate_fn
        self.loss_fct = nn.CrossEntropyLoss()

    def compute_loss(self, model, inputs, return_outputs=False):
        #batch,max_length
        targets = inputs.pop("targets")
        #batch,max_length
        labels_mask = inputs.pop("label_masks")
        
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask", None),
            use_cache=False,
        )
        
        # *** Intially logits are of 3 dim
        #batch, max_length, vocab size
        logits = outputs.get("logits")
        
        # TEXT:             Question: Hello, how are you? Answer: I am fine. Question: What is your name? Answer: My name is John.
        # LABEL_MASK:       0         0      0   0   0    1       1 1  1     0         0    0  0    0     1       1  1    1  0
        # TARGET:           Hello, how are you? Answer: I am fine. Question: What is your name? Answer: My name is John. Question

        # then flatten it to be 2 dim
        #batch*max_length, vocab size
        logits = logits.view(-1, logits.size(-1))

        # making target and label of one dim
        # batch*max_length
        targets = targets.view(-1)
        # batch*max_length
        mask = labels_mask.view(-1).bool()
        
        # **** By using the target mask 
        #NEW INPUT: [Answer: I am fine.], [Answer: My name is.]
        #Technically after each Q and A there will be eos token. So that will come here
        #NEW TARGET: [I am fine. Question (eos)], [My name is John.]
        
        inp = logits[mask]
        targ = targets[mask]

        loss = self.loss_fct(inp, targ)

        return (loss, outputs) if return_outputs else loss
    
    
    def _compute_loss(self, model, inputs):
        inputs = self._prepare_inputs(inputs)
        targets = inputs.pop("targets")
        labels_mask = inputs.pop("label_masks")
        
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask", None),
            use_cache=False,
        )
        
        logits = outputs.get("logits")
        logits_ = logits.view(-1, logits.size(-1))
        targets_ = targets.view(-1)
        mask = labels_mask.view(-1).bool()
        
        inp = logits_[mask]
        targ = targets_[mask]

        loss = self.loss_fct(inp, targ)

        return loss, logits, targets,labels_mask
    
    
    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys=None,
    ):
        with torch.no_grad():
            loss, logits, labels, labels_mask = self._compute_loss(model, inputs)
            labels[~labels_mask.bool()] = -100  # padding_index

        loss = loss.mean().detach()

        if self.args.prediction_loss_only:
            return (loss, None, None)

        return (loss, logits, labels)
    

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

    tokenizer, eos_token= get_sft_tokenizer(conf,TOKENIZER_SEPECIAL_TOKENS)
    train_ds , eval_ds = load_sft_dataset(conf,eos_token)
    model = get_sft_model(tokenizer, conf)
    metrics,preprocess_function = get_sft_metrics(conf.metrics)
    
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
    

    # if conf.int8_training:
    #     import bitsandbytes  # This is noisy, so delay importing until after argument parsing so it doesn't make --help noisy
    #     for module in model.modules():
    #         if isinstance(module, torch.nn.Embedding):
    #             bitsandbytes.optim.GlobalOptimManager.get_instance().register_module_override(
    #                 module, "weight", {"optim_bits": 32}
    #             )

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

    wandb_name = conf.model_name
    wandb.init(
        project="supervised-finetuning",
        entity=None,
        resume=conf.resume_from_checkpoint,
        name=f"{wandb_name}--finetuned",
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
    compute_metrics=partial(compute_metrics, metrics=metrics, preprocess_fns=preprocess_function),
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,)
    return trainer


def train(trainer,output_dir,conf):
    trainer.train(resume_from_checkpoint=conf.resume_from_checkpoint)
    trainer.save_model(output_dir)
    trainer.tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    config = {}
    parser = argparse.ArgumentParser(description="Parse configuration")
    parser.add_argument("--overrides", nargs='+', help="Override configurations (key=value)", default=[])
    args, remaining = parser.parse_known_args()

    overrides = dict(override.split('=') for override in args.overrides)
    conf = read_yaml('./configs/sft_config.yaml')
    config.update(conf["default"])
    config.update(overrides)

    parser = parse_additional_args(config)
    args = parser.parse_args(remaining)

    output_dir=f"{args.model_name}-lora-finetuned"

    if args.debug:
        args.train_batch=1
        args.eval_batch=1
        args.num_train_epochs=1
        args.log_steps=100
        args.eval_steps=100
        args.save_steps=100

    trainer = main(args,output_dir)
    train(trainer,output_dir,args)

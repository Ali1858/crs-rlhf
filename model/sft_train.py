"""rewritten from:
https://github.com/LAION-AI/Open-Assistant/blob/main/model/model_training/trainer_sft.py
"""
from functools import partial
import math

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
from config import SFT_TRAINING_CONFIG, DIALOGUE_COLLATOR_CONFIG, TOKENIZER_CONFIG



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


def train():
    # needs to happen before model loading in case of stage 3 training
    quantization = True
    optimizer = OptimizerNames.ADAMW_BNB if quantization else OptimizerNames.ADAMW_HF
    
    output_dir=f"{SFT_TRAINING_CONFIG['model_name']}-lora-finetuned"

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=SFT_TRAINING_CONFIG["num_train_epochs"],
        warmup_steps=SFT_TRAINING_CONFIG["warmup_steps"],
        learning_rate=float(SFT_TRAINING_CONFIG["lr"]),
        optim=optimizer,
        fp16=SFT_TRAINING_CONFIG["dtype"] in ["fp16", "float16"],
        bf16=SFT_TRAINING_CONFIG["dtype"] in ["bf16", "bfloat16"],
        gradient_checkpointing=SFT_TRAINING_CONFIG["gradient_checkpointing"],
        gradient_accumulation_steps=SFT_TRAINING_CONFIG["gradient_accumulation_steps"],
        per_device_train_batch_size=SFT_TRAINING_CONFIG["train_batch"],
        per_device_eval_batch_size=SFT_TRAINING_CONFIG["eval_batch"],
        adam_beta1=SFT_TRAINING_CONFIG["adam_beta1"],
        adam_beta2=SFT_TRAINING_CONFIG["adam_beta2"],
        adam_epsilon=float(SFT_TRAINING_CONFIG["adam_epsilon"]),
        weight_decay=SFT_TRAINING_CONFIG["weight_decay"],
        logging_steps=SFT_TRAINING_CONFIG["log_steps"],
        evaluation_strategy="steps",
        eval_steps=SFT_TRAINING_CONFIG["eval_steps"],
        save_strategy="steps",
        save_steps=SFT_TRAINING_CONFIG["save_steps"],
        eval_accumulation_steps=SFT_TRAINING_CONFIG["eval_accumulation_steps"],
        resume_from_checkpoint=SFT_TRAINING_CONFIG["resume_from_checkpoint"],
        report_to="wandb",
    )

    tokenizer, eos_token= get_sft_tokenizer(SFT_TRAINING_CONFIG,TOKENIZER_CONFIG)
    train_ds , eval_ds = load_sft_dataset(eos_token)
    model = get_sft_model(SFT_TRAINING_CONFIG)
    metrics,preprocess_function = get_sft_metrics(SFT_TRAINING_CONFIG["metrics"])
    
    train_collate_fn = DialogueDataCollator(
        tokenizer,
        max_length=DIALOGUE_COLLATOR_CONFIG["max_length"],
        random_offset_probability=DIALOGUE_COLLATOR_CONFIG["random_offset_probability"],
        label_masking=DIALOGUE_COLLATOR_CONFIG["label_masking"],
        samples_mixing=DIALOGUE_COLLATOR_CONFIG["samples_mixing"],
        pad_to_multiple_of=16,
        use_system_prefix=DIALOGUE_COLLATOR_CONFIG["use_system_prefix"],
        system_prefix=DIALOGUE_COLLATOR_CONFIG["system_prefix"],
    )
    
    if DIALOGUE_COLLATOR_CONFIG.get("val_max_length") is None:
        DIALOGUE_COLLATOR_CONFIG["val_max_length"] = DIALOGUE_COLLATOR_CONFIG["max_length"]
    
    eval_collate_fn = DialogueDataCollator(
        tokenizer,
        max_length=DIALOGUE_COLLATOR_CONFIG["val_max_length"],
        random_offset_probability=DIALOGUE_COLLATOR_CONFIG["random_offset_probability"],
        label_masking=DIALOGUE_COLLATOR_CONFIG["label_masking"],
        samples_mixing=False,
        use_system_prefix=DIALOGUE_COLLATOR_CONFIG["use_system_prefix"],
        system_prefix=DIALOGUE_COLLATOR_CONFIG["system_prefix"],
        )
    
    if quantization:
        import bitsandbytes  # This is noisy, so delay importing until after argument parsing so it doesn't make --help noisy
        
        for module in model.modules():
            if isinstance(module, torch.nn.Embedding):
                bitsandbytes.optim.GlobalOptimManager.get_instance().register_module_override(
                    module, "weight", {"optim_bits": 32}
                )
        
        import wandb

        wandb_name = SFT_TRAINING_CONFIG["model_name"]
        wandb.init(
            project="supervised-finetuning",
            entity=None,
            resume=SFT_TRAINING_CONFIG["resume_from_checkpoint"],
            name=f"{wandb_name}--finetuned",
            config=dict(**DIALOGUE_COLLATOR_CONFIG,**SFT_TRAINING_CONFIG),
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
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    return trainer
    # trainer.train(resume_from_checkpoint=SFT_TRAINING_CONFIG["resume_from_checkpoint"])
    # trainer.save_model()
    # tokenizer.save_pretrained(output_dir)





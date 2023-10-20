import datasets
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import Trainer
from transformers.utils import is_datasets_available
from transformers.trainer_utils import seed_worker

from model_training.losses import RMLoss


"""Rewritten from:
https://github.com/LAION-AI/Open-Assistant/blob/main/model/model_training/trainer_sft.py#L51C2-L51C2
"""
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
            use_cache=False,
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
    

"""Taken from:
https://github.com/LAION-AI/Open-Assistant/blob/main/model/model_training/trainer_rm.py#L31
"""
class AbsRMTrainer(Trainer):
    def __init__(self, model, args, train_collate_fn,**kwargs):
        super().__init__(model=model,args=args,**kwargs)
        self.train_collate_fn = train_collate_fn
        self.loss_fct = nn.MSELoss()
        self.sigmoid = nn.Sigmoid()

    def compute_loss(self, model, inputs, return_logits=False):
        labels = inputs.pop("labels")
        logits = model(input_ids=inputs["input_ids"],
              attention_mask=inputs["attention_mask"],
              use_cache=False,
              ).logits
        pred = self.sigmoid(logits)
        loss = self.loss_fct(pred.view(-1).float(), labels.view(-1).float())
        return (loss, logits) if return_logits else loss
    

    def _compute_loss(self, model, inputs):
        inputs = self._prepare_inputs(inputs)
        labels = inputs.pop("labels")
        logits = model(input_ids=inputs["input_ids"],
              attention_mask=inputs["attention_mask"],
              use_cache=False,
              ).logits
        pred = self.sigmoid(logits)
        loss = self.loss_fct(pred.view(-1).float(), labels.view(-1).float())
        return loss, pred, labels
    
    
    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys,
    ):
        with torch.no_grad():
            loss, pred, labels = self._compute_loss(model, inputs)

        loss = loss.mean().detach()
        return (loss, pred, labels)  # transposed to avoid truncation in evaluation_loop

    
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
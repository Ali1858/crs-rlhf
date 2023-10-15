from dataclasses import dataclass
from typing import Optional, Union
import random

import torch
import numpy as np
from torch.nn import functional as F
from transformers.tokenization_utils_base import PaddingStrategy,TruncationStrategy,PreTrainedTokenizerBase

from training_datasets.dataset_utils import get_rm_formatted
from constants import QA_SPECIAL_TOKENS


"Taken from "
"https://github.com/LAION-AI/Open-Assistant/blob/main/model/model_training/custom_datasets/dialogue_collator.py#L19"
@dataclass
class DialogueDataCollator:
    """
    Expects a list of texts corresponding to a sequence of [question, answer, question, answer, ...] pairs.
    """
    tokenizer: PreTrainedTokenizerBase
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    padding: Union[bool, str, PaddingStrategy] = True
    random_offset_probability: Optional[float] = 0.5
    label_masking: bool = True
    min_length_threshold: Optional[int] = 256
    mix_probability: Optional[float] = 0.6
    samples_mixing: Optional[bool] = False
    system_prefix: str = None
    use_system_prefix: bool = False


    def __post_init__(self):
        assert self.tokenizer.eos_token
        if self.use_system_prefix:
            assert self.system_prefix
            formatted_prefix = "{}{}{}".format(QA_SPECIAL_TOKENS["System"],self.system_prefix,self.tokenizer.eos_token,)
            self.system_prefix = self.tokenizer.encode(formatted_prefix, add_special_tokens=False, return_tensors="np",)[0]
            self.max_length = self.max_length - len(self.system_prefix)


    def process_one(self, messages):
        total_short_context_one = 0
        if random.random() < self.random_offset_probability:
            truncation = TruncationStrategy.DO_NOT_TRUNCATE
            max_length = None
        else:
            truncation = TruncationStrategy.LONGEST_FIRST
            max_length = self.max_length

        flatten_message = self.tokenizer(
            "".join(messages),
            max_length=max_length,
            truncation=truncation,
            padding=False,
        )
        
        message_indices = None
        if self.label_masking:
            # message_change_indices = np.cumsum([len(x) for x in messages])
            # for each token an integer indicating the index of the message it belongs to. Just to create the label mask.
            # Label mask is true when predicting a token that is part of the answer, false otherwise.
            # TEXT:             <s> Question: Hello, how are you? Answer: I am fine. Question: What is your name? Answer: My name is John.
            # MESSAGE_INDICES:  0         0      0     0   0   0      1   1 1  1     2         2    2  2    2     3       3  3    3  3
            # LABEL_MASK:       0         0      0     0   0   0      1   1 1  1     0         0    0  0    0     1       1  1    1  1
            # TARGET:           Question: Hello, how are you? Answer: I am fine. Question: What is your name? Answer: My name is John.<s>


            # If no result in next, we are predicting the last termination token(s)
            # message_indices = list(
            #     map(
            #         lambda x: next((i for i, val in enumerate(message_change_indices) if val >= x)),
            #         list(map(lambda x: x[1], flatten_message.offset_mapping)),
            #     )
            # )

            prompter_token_id = self.tokenizer.convert_tokens_to_ids(QA_SPECIAL_TOKENS["Question"])
            assistant_token_id = self.tokenizer.convert_tokens_to_ids(QA_SPECIAL_TOKENS["Answer"])
            sep_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token)
            assert prompter_token_id >= 0 and assistant_token_id >= 0
            
            message_indices = []
            i = -1
            for x in flatten_message.input_ids:
                if x in (prompter_token_id, assistant_token_id):
                    i += 1
                message_indices.append(i)

        if flatten_message.input_ids[0] == sep_id and message_indices[0] == -1:
            message_indices[0] = 0

        input_length = len(flatten_message.input_ids)
        if self.max_length and input_length > self.max_length:
            offset = random.randint(0, input_length - self.max_length)
            for k in flatten_message.keys():
                v = flatten_message[k]
                if isinstance(v, list) and len(v) == input_length:
                    flatten_message[k] = v[offset : offset + self.max_length]
            if message_indices:
                message_indices = message_indices[offset : offset + self.max_length]

        if self.label_masking:
            label_mask = np.array(list(map(lambda x: x % 2 == 1, message_indices)))
        else:
            label_mask = np.ones(len(flatten_message.input_ids), dtype=bool)

        label_mask[-1] = False  # make sure last token is inactive, has an effect only when truncating

        if len(flatten_message.input_ids) < self.min_length_threshold and self.samples_mixing:
            total_short_context_one += len(flatten_message.input_ids)

        return {k: v for k, v in flatten_message.items() if k != "offset_mapping"}, label_mask, total_short_context_one
    

    def __call__(self, features):
        flatten_messages = []
        label_masks = []
        total_short_context = 0
        for messages in features:
            flatten_message, label_mask, total_short_context_one = self.process_one(messages)
            flatten_messages.append(flatten_message)
            label_masks.append(label_mask)
            total_short_context += total_short_context_one

        # packing
        if total_short_context > 2 and self.samples_mixing:
            _flatten_messages, _label_masks = [], []
            prev_short_msg, prev_short_mask = None, None
            for flatten_msg, label_mask in zip(flatten_messages, label_masks):
                if len(flatten_msg["input_ids"]) < self.min_length_threshold and random.random() > self.mix_probability:
                    if prev_short_msg is not None:
                        for key in flatten_msg.keys():
                            flatten_msg[key] += prev_short_msg[key]
                            flatten_msg[key] = flatten_msg[key][: self.max_length]
                        label_mask = np.concatenate([label_mask, prev_short_mask])
                        _label_masks.append(label_mask[: self.max_length])
                        _flatten_messages.append(flatten_msg)
                        # reset
                        prev_short_msg, prev_short_mask = None, None
                    else:
                        # prime
                        prev_short_msg, prev_short_mask = flatten_msg, label_mask
                else:
                    _label_masks.append(label_mask)
                    _flatten_messages.append(flatten_msg)
            if prev_short_msg is not None:
                for key in flatten_msg.keys():
                    flatten_msg[key] += prev_short_msg[key]
                    flatten_msg[key] = flatten_msg[key][: self.max_length]
                label_mask = np.concatenate([label_mask, prev_short_mask])[: self.max_length]
                _label_masks.append(label_mask)
                _flatten_messages.append(flatten_msg)

            label_masks = _label_masks
            flatten_messages = _flatten_messages
        
        if self.use_system_prefix:
            flatten_messages = [
                {
                    "input_ids": np.concatenate([self.system_prefix, flatten_msg["input_ids"]]),
                    "attention_mask": np.concatenate(
                        [np.ones_like(self.system_prefix).astype(bool), flatten_msg["attention_mask"]]
                    ),
                }
                for flatten_msg in flatten_messages
            ]
            label_masks = [
                np.concatenate([np.zeros_like(self.system_prefix).astype(bool), label_mask])
                for label_mask in label_masks
            ]

        
        batch = self.tokenizer.pad(
            flatten_messages,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        dim = batch.input_ids.shape[-1]

        batch["label_masks"] = torch.stack(
            [F.pad(torch.tensor(x), (0, dim - len(x)), value=False) for x in label_masks]
        )
        # roll in reverse order for last dim(input_ids_size)
        # shape --> batch * input_ids_size
        #x
        #>>tensor([[1, 2],
        #         [3, 4],
        #         [5, 6],
        #         [7, 8]])
        #torch.roll(x,-1,-1)
        #>> tensor([[2, 1],
        #          [4, 3],
        #          [6, 5],
        #          [8, 7]])
        batch["targets"] = torch.roll(batch.input_ids, -1, -1)
        return batch


"""Taken from:
https://github.com/LAION-AI/Open-Assistant/blob/main/model/model_training/custom_datasets/ranking_collator.py#L11
"""
@dataclass
class RankingDataCollator:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    min_prefix_length: int = 256
    pad_to_multiple_of: Optional[int] = None
    max_replies: Optional[int] = 5

    def process_one(self,example):
        assert self.tokenizer.eos_token
        eos = self.tokenizer.eos_token

        prefix, replies = example

        if self.max_replies:
            assert self.max_replies > 1, "max_replies parameter must be > 1 or None"
            if len(replies) > self.max_replies:
                replies = replies[: self.max_replies]

        if prefix is None or len(prefix) == 1 and prefix[0] is None:
            # special handling for non-dialogue datasets like Hellaswag
            prefix = ""
            replies = [r + eos for r in replies]
        else:
            # append eos token to each messages
            prefix = "".join(get_rm_formatted(eos,prefix))
            replies = [get_rm_formatted(eos,r,is_replies=True) for r in replies]
            
        prefix_tokens = self.tokenizer(prefix, padding=False, truncation=False)
        reply_tokens = [self.tokenizer(r, padding=False, truncation=False) for r in replies]

        prefix_len = len(prefix_tokens.input_ids)
        for r in reply_tokens:
            max_prefix_len = (
                prefix_len
                if self.max_length is None
                else max(self.min_prefix_length, self.max_length - len(r.input_ids))
            )
            max_suffix_len = len(r.input_ids) if self.max_length is None else self.max_length - max_prefix_len

            for k in r.keys():
                r[k] = prefix_tokens[k][-max_prefix_len:] + r[k][:max_suffix_len]

        return reply_tokens

    def __call__(self, examples):
        flat_tokenized, cu_lens = [], [0]
        n_samples = 0
        for example in examples:
            tokenized = self.process_one(example)
            flat_tokenized.extend(tokenized)

            n_samples += len(tokenized)
            cu_lens.append(n_samples)

        batch = self.tokenizer.pad(
            flat_tokenized,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        if "token_type_ids" in batch:
            batch.pop("token_type_ids")
        return batch, cu_lens


@dataclass
class AbsoluteScoreDataCollator:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    min_prefix_length: int = 256
    pad_to_multiple_of: Optional[int] = None

    def process_one(self,example):
        assert self.tokenizer.eos_token
        eos = self.tokenizer.eos_token

        prefix, reply, score = example
        # append eos token to each messages
        prefix = "".join(get_rm_formatted(eos,prefix))
        reply = get_rm_formatted(eos,reply,is_replies=True)
            
        prefix_tokens = self.tokenizer(prefix, padding=False, truncation=False)
        reply_tokens = self.tokenizer(reply, padding=False, truncation=False)

        prefix_len = len(prefix_tokens.input_ids)
        max_prefix_len = (
            prefix_len
            if self.max_length is None
            else max(self.min_prefix_length, self.max_length - len(reply_tokens.input_ids))
        )
        max_suffix_len = len(reply_tokens.input_ids) if self.max_length is None else self.max_length - max_prefix_len

        for k in reply_tokens.keys():
            reply_tokens[k] = prefix_tokens[k][-max_prefix_len:] + reply_tokens[k][:max_suffix_len]

        return reply_tokens, score

    def __call__(self, examples):
        tokenized,scores = [], []
        for example in examples:
            tokens,score = self.process_one(example)
            tokenized.append(tokens)
            scores.append(score)

        batch = self.tokenizer.pad(
            tokenized,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch["labels"] = torch.tensor(scores,dtype=float)

        if "token_type_ids" in batch:
            batch.pop("token_type_ids")
        return batch

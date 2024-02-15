"""Rewritten from:
https://github.com/LAION-AI/Open-Assistant/blob/main/model/model_training/custom_datasets"""

import random
from typing import List
from collections import defaultdict
import numpy as np
import re
import pandas as pd
from torch.utils.data import Dataset, random_split
from datasets import load_dataset
from torch import Generator
from sklearn.utils import resample

from training_datasets.dataset_utils import load_oasst, ListDataset
from constants import QA_SPECIAL_TOKENS


SEED = 2020

SUMMARIZATION_SPECIAL_TOKENS = {"Text": "", "Summary": ["TL;DR:", "Summarize this", "Give me the summary"]}

SUMMARY_SPECIAL_PROMPT = {
    "multi_news": ["Summarize in bullet points", "Generate summary in list of points"],
    "xsum": ["Give me summary in one sentence", "Short TLDR", "Give me a concise summary"],
    "samsum": ["TLDR;", "Summarize this dialogue", "Summarize dialogue"],
}



# @agoryuno contributed this
re_reference_remove = re.compile(r"\[\d+(?:,\s*\d+)*?\]")
re_single_reference_remove = re.compile(r"\[\s?\d+\s?\]")

# check if the whole string is just a combination of (multiple) whitespaces and newlines
re_whitespace_newline_match = re.compile(r"^[\s\n]*$")


"Taken from https://github.com/LAION-AI/Open-Assistant/blob/main/model/model_training/custom_datasets/rank_datasets.py#L182"
class AnthropicRLHF(Dataset):

    @staticmethod
    def _split_dialogue(text):
        lines = text.split("\n\n")

        dialogue = []

        # go over messages and combine consecutive messages from the
        # same speaker (OA v1 expects alternating roles)
        role = None
        messages = []
        for line in lines:
            if line.startswith("Human:"):
                speaker = "Human"
                message = line[7:]
            elif line.startswith("Assistant:"):
                speaker = "Assistant"
                message = line[11:]
            else:
                continue
            if role != speaker:
                if role is not None:
                    dialogue.append((role, "\n".join(messages)))
                    messages = []
                role = speaker
            messages.append(message.strip())

        if role is not None and len(messages) > 0:
            dialogue.append((role, "\n".join(messages)))

        return dialogue
    

    def __init__(self,cache_dir,split="train"):
        super().__init__()
        self.data = []
        dataset = load_dataset("Anthropic/hh-rlhf",cache_dir=cache_dir)[split]

        for entry in dataset:

            chosen = entry["chosen"]
            if "Assistant" not in chosen:
                continue

            rejected = entry["rejected"]
            chosen = self._split_dialogue(chosen)
            rejected = self._split_dialogue(rejected)
            assert rejected[0][0] == "Human" and chosen[0][0] == "Human"

            # only very few items have non matching lengths
            if len(rejected) == len(chosen):
                prefix = [line for (speaker, line) in chosen[:-1]]
                good_reply = chosen[-1][1]  # last part of dialog, the text
                bad_reply = rejected[-1][1]  # last part of dialog, the text
                self.data.append((prefix, [good_reply, bad_reply]))


    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[str, list[str]]:
        return self.data[index]


"Taken from https://github.com/LAION-AI/Open-Assistant/blob/main/model/model_training/custom_datasets/rank_datasets.py#L12"
class SHPDataset(Dataset):
    """
    Dataset class to load stanfordnlp/SHP for Reward Modeling
    """

    name = "SHP"

    def __init__(self, cache_dir,split="train", max_answers: int = 5):
        super().__init__()

        self.questions = []
        self.answers = []

        if not isinstance(split, list):
            split = [split]
        dataset_splits = load_dataset("stanfordnlp/SHP", cache_dir=cache_dir,split=split)

        answers_by_id = defaultdict(dict)
        history_by_id = dict()
        for split in dataset_splits:
            for row in split:
                post_id = row["post_id"]
                history_by_id[post_id] = row["history"]
                answers_by_id[post_id][row["human_ref_A"]] = row["score_A"]
                answers_by_id[post_id][row["human_ref_B"]] = row["score_B"]

        for post_id, history in history_by_id.items():
            self.questions.append(history)
            answers = answers_by_id[post_id]
            # Sort answer dict with the highest score first (hence the prefactor -1).
            # Then take only the first `max_answers` elements (usually there are just
            # 2, but there are examples where we have more)
            answers_sorted = [x[0] for x in sorted(answers.items(), key=lambda x: -1 * x[1])]
            self.answers.append(answers_sorted[:max_answers])

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, index):
        return [self.questions[index]], self.answers[index]
    

"Taken from https://github.com/LAION-AI/Open-Assistant/blob/main/model/model_training/custom_datasets/rank_datasets.py#L12"
class HellaSwagDataset(Dataset):
    """
    Dataset class to use data from https://arxiv.org/pdf/1905.07830.pdf
    for Reward modeling

    Note: In order to disable dialog-formatting None is returned as context.
    """

    name = "hellaswag"

    def __init__(self,cache_dir, split="train", seed = SEED):
        super().__init__()

        np.random.seed(seed)
        self.dataset_list = []
        if not isinstance(split, List):
            split = [split]
        dataset = load_dataset("AlekseyKorshuk/hellaswag", cache_dir=cache_dir,split=split)
        for data in dataset:
            for item in data:
                context = item.get("ctx")
                endings = item.get("endings")
                selected = endings.pop(item.get("label"))
                ordered_ends = [selected, np.random.choice(endings)]
                self.dataset_list.append({"context": context, "completions": ordered_ends})

    def __len__(self):
        return len(self.dataset_list)

    def __getitem__(self, idx):
        context, completions = self.dataset_list[idx].values()
        return None, [context + c for c in completions]
    

class RMDataset(Dataset):
    def __init__(self, data: list,eos_token:str=''):
        super().__init__()
        self.data = data
        self.eos_token = eos_token

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        prefix, replies = self.data[index]
        return prefix,replies


class AbsoluteRMDataset(Dataset):
    def __init__(self, dataset: list, label_weight,abs_oversample_threshold,split=None):
        super().__init__()
        self.data = []
        oversample_data = []
        none_count = 0
        skip_count = 0
        skip_count_neg = 0
        desired_labels = ["violence","creativity","helpfulness","humor","toxicity","quality"]
        print(label_weight)

        for row in dataset:
            message,replies = row
            first_score = 0
            for r_idx,r in enumerate(replies):
                reply_text = r.text
                skip = False
                labels = {}
                for l in desired_labels:
                    v = r.get_label_value(l)
                    labels[l] = v

                labels_list = list(labels.values())
                if any(item is None for item in labels_list):
                    none_count += 1
                    skip = True

                if not skip:
                    # Ensure labels match between the two dictionaries
                    if set(labels.keys()) != set(label_weight.keys()):
                        raise ValueError("Labels do not match between the weight dictionary")
                    if sum([abs(v) for v in label_weight.values()]) != 1.0:
                        raise ValueError("Sum of label weight is not equal one")

                    reward_score = sum(labels[label] * label_weight[label] for label in labels)
                    if reward_score < 0:
                        skip_count_neg += 1
                        continue

                    self.data.append((message,reply_text,reward_score))

                    # if r_idx == 0:
                    #     self.data.append((message,reply_text,reward_score))
                    #     first_score = reward_score
                    # elif reward_score >= 0.6 and abs(first_score-reward_score) >= 0.05:
                    #     self.data.append((message,reply_text,reward_score))
                    # elif reward_score < 0.6:
                    #     self.data.append((message,reply_text,reward_score))
                    # else:
                    #     skip_count +=1

                    # if abs_oversample_threshold and reward_score <= abs_oversample_threshold:
                    #     oversample_data.append((message,reply_text,reward_score))
        # Create a DataFrame
        df = pd.DataFrame(self.data, columns=["message", "reply", "score"])
        print(f'{"==="*10} total {none_count} has None value for atleast 1 label')
        # print(f'{"==="*10} total {skip_count} has been skipped because low difference threshold')
        print(f'{"==="*10} total {skip_count_neg} has been skipped because reward is negative')

        if split == 'train' and abs_oversample_threshold:
            # print(f'{"==="*10} Oversample count {len(oversample_data)}')
            # self.data.extend(oversample_data)
            self.data = self.oversampling(df,abs_oversample_threshold)

    def __len__(self):
        return len(self.data)
    

    def oversampling(self, df,lower_threshold):
        high_threshold = 0.9
        # Separate the dataset into three groups
        low_score_df = df[df['score'] <= lower_threshold]
        print(f'size of low score df: {low_score_df.shape}')
        
        high_score_df = df[df['score'] >= high_threshold]
        print(f'size of high score df: {high_score_df.shape}')
        
        mid_score_df = df[(df['score'] > lower_threshold) & (df['score'] < high_threshold)]
        print(f'size of mid score df: {mid_score_df.shape}')

        # Oversample the low scores
        low_score_oversampled = low_score_df
        # resample(low_score_df,
        #                          replace=True,      # sample with replacement
        #                          n_samples=int(low_score_df.shape[0]*2),    # to match desired count
        #                          random_state=123)  # for reproducibility
        print(f'size of low score df after sampling: {low_score_oversampled.shape}')
        
        # Stratified undersampling for mid-range scores
        step = 0.1
        bins = [round(i, 2)/10 for i in range(int(lower_threshold * 10), int(high_threshold * 10) + 1, int(step * 10))]
        print(f'bins for mid score sampling {bins}')

        mid_score_stratified = pd.DataFrame()
        
        for i in range(len(bins)-1):
            bin_df = mid_score_df[(mid_score_df['score'] > bins[i]) & (mid_score_df['score'] <= bins[i+1])]
            sampled_bin_df = resample(bin_df,
                              replace=False,    # sample without replacement
                              n_samples=int(8000/(len(bins)-1)), # evenly distribute samples
                              random_state=123) # for reproducibility
            mid_score_stratified = pd.concat([mid_score_stratified, sampled_bin_df])
        print(f'size of mid score df after sampling: {mid_score_stratified.shape}')

        high_score_undersample_df = high_score_df
        # resample(high_score_df,
        #                                      replace=False, 
        #                                      n_samples=7000,
        #                                      random_state=123)  # for reproducibility
        print(f'size of high score df after sampling: {high_score_undersample_df.shape}')


        # Combine oversampled, stratified undersampled, and high score data
        balanced_df = pd.concat([low_score_oversampled, mid_score_stratified, high_score_undersample_df])
        print(f'size of final df after sampling: {balanced_df.shape}')
        
        # Shuffle the combined dataset
        balanced_df = balanced_df.sample(frac=1, random_state=123).reset_index(drop=True)
        return balanced_df.values.tolist()


    def __getitem__(self, index):
        prefix, reply,reward_score = self.data[index]
        return prefix,reply,reward_score


"""Rewritten from:
https://github.com/LAION-AI/Open-Assistant/blob/main/model/model_training/custom_datasets/oasst_dataset.py#L23
"""
def get_oasst_rm(val_split,cache_dir,lang,manual_seed=90,top_k=None,**kwargs):
    generator = Generator()
    generator.manual_seed(manual_seed)
    threads_per_tree = load_oasst(mode="rm",top_k=top_k,lang=lang)
    def process_thread(thread):
        prefix = [m.text for m in thread]
        replies = [r for r in thread[-1].replies if r.role == "assistant" and r.rank is not None]
        replies = sorted(replies, key=lambda r: r.rank)
        replies = [r.text for r in replies]
        return (prefix, replies)
    
    # split on tree basis, messages from same tree must not end up in different splits
    trees = ListDataset(threads_per_tree,)
    splits = random_split(trees, lengths=[1.0 - val_split, val_split], generator=generator)

    def flatten(ds: ListDataset) -> RMDataset:
        return RMDataset([process_thread(thread) for tree_threads in ds for thread in tree_threads])

    train = flatten(splits[0])
    val = flatten(splits[1])
    print(f"OASST HF dataset: {len(train)=}, {len(val)=}")
    return train,val


def get_oasst_abs_rm(val_split,cache_dir,lang,abs_oversample_threshold,manual_seed=90,top_k=None,label_weight=None,**kwargs):
    generator = Generator()
    generator.manual_seed(manual_seed)
    threads_per_tree = load_oasst(mode="rm",top_k=top_k,lang=lang)

    def process_thread(thread):
        prefix = [m.text for m in thread]
        replies = [r for r in thread[-1].replies if r.role == "assistant" and r.rank is not None and r.labels is not None]
        replies = sorted(replies, key=lambda r: r.rank)
        return (prefix, replies)
    
    # split on tree basis, messages from same tree must not end up in different splits
    trees = ListDataset(threads_per_tree,)
    splits = random_split(trees, lengths=[1.0 - val_split, val_split], generator=generator)

    def flatten(ds: ListDataset,split='train') -> AbsoluteRMDataset:
        return AbsoluteRMDataset([process_thread(thread) for tree_threads in ds for thread in tree_threads],label_weight,abs_oversample_threshold,split)

    train = flatten(splits[0])
    val = flatten(splits[1],'eval')
    print(f"OASST HF dataset: {len(train)=}, {len(val)=}")
    return train,val

"""Taken from:
https://github.com/LAION-AI/Open-Assistant/blob/main/model/model_training/custom_datasets/qa_datasets.py#L188
"""
class WebGPT(Dataset):
    name = "webgpt"

    def __init__(self, cache_dir, max_answers: int = 5) -> None:
        super().__init__()
        self.rows = []

        dataset = load_dataset("openai/webgpt_comparisons",cache_dir=cache_dir)
        question_answer_dict = defaultdict(dict)

        for row in dataset["train"]:
            question = row["question"]["full_text"]
            answer_0 = re_reference_remove.sub("", row["answer_0"])
            answer_1 = re_reference_remove.sub("", row["answer_1"])
            if answer_0 != "" and answer_1 != "" and answer_0 != answer_1:
                question_answer_dict[question][answer_0] = row["score_0"]
                question_answer_dict[question][answer_1] = row["score_1"]

        for question, answers in question_answer_dict.items():
            # Sort answer dict with the highest score first (hence the prefactor -1).
            # Then take only the first `max_answers` elements (usually there are just
            # 2, but there are examples where we have more)
            answers_sorted = [x[0] for x in sorted(answers.items(), key=lambda x: -1 * x[1])]
            self.rows.append((question,answers_sorted[:max_answers]))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def get_webgpt_rm(val_split,cache_dir,manual_seed=90,**kwargs):
    generator = Generator()
    generator.manual_seed(manual_seed)
    webgpt = WebGPT(cache_dir=cache_dir)
    splits = random_split(webgpt, lengths=[1.0 - val_split, val_split], generator=generator)

    train = splits[0]
    val = splits[1]
    print(f"WebGPT: {len(train)=}, {len(val)=}")
    return train,val

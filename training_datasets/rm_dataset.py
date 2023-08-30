"""Rewritten from:
https://github.com/LAION-AI/Open-Assistant/blob/main/model/model_training/custom_datasets"""

from typing import List
from collections import defaultdict
import numpy as np

from torch.utils.data import Dataset, random_split
from datasets import load_dataset
from torch import Generator

from training_datasets.dataset_utils import load_oasst, ListDataset
from constants import QA_SPECIAL_TOKENS


SEED = 2020


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


"""Rewritten from:
https://github.com/LAION-AI/Open-Assistant/blob/main/model/model_training/custom_datasets/oasst_dataset.py#L23
"""
def get_oasst_rm(val_split,cache_dir,lang,manual_seed=90):
    generator = Generator()
    generator.manual_seed(manual_seed)
    threads_per_tree = load_oasst(mode="rm",lang=lang)
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
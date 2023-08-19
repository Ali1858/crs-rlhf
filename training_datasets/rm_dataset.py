"""Rewritten from:
https://github.com/LAION-AI/Open-Assistant/blob/main/model/model_training/custom_datasets"""

from torch.utils.data import Dataset
from datasets import load_dataset

from typing import List

from collections import defaultdict
import numpy as np

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
    

    def __init__(self,split="train"):
        super().__init__()
        self.data = []
        dataset = load_dataset("Anthropic/hh-rlhf")[split]

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

    def __init__(self, split="train", max_answers: int = 5):
        super().__init__()

        self.questions = []
        self.answers = []

        if not isinstance(split, list):
            split = [split]
        dataset_splits = load_dataset("stanfordnlp/SHP", split=split)

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

    def __init__(self, split="train", seed = SEED):
        super().__init__()

        np.random.seed(seed)
        self.dataset_list = []
        if not isinstance(split, List):
            split = [split]
        dataset = load_dataset("AlekseyKorshuk/hellaswag", split=split)
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
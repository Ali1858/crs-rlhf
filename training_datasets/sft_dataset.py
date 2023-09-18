"""Rewritten from:
https://github.com/LAION-AI/Open-Assistant/blob/main/model/model_training/custom_datasets/"""

import re
import random
from collections import defaultdict


import numpy as np
from torch.utils.data import Dataset, Subset, random_split
from torch import Generator
from datasets import load_dataset

from training_datasets.dataset_utils import filter_by_words, load_oasst, ListDataset
from constants import QA_SPECIAL_TOKENS

# @agoryuno contributed this
re_reference_remove = re.compile(r"\[\d+(?:,\s*\d+)*?\]")
re_single_reference_remove = re.compile(r"\[\s?\d+\s?\]")

# check if the whole string is just a combination of (multiple) whitespaces and newlines
re_whitespace_newline_match = re.compile(r"^[\s\n]*$")


LINKING_CHARS = ["\n", "\n\n", " "]


def get_qa_formatted(
        eos_token,
        questions=None,
        answers=None,
        context=None,
        system_msg=None,
        conversations=None
    ):
        output: list[str] = []
        
        if conversations is None:
            assert answers,""
            assert questions,""
            conversations = []
            for q,a in zip(questions,answers):
                conversations.append([q,"prompter"])
                conversations.append([a,"assistant"])
        

        for i, m in enumerate(conversations):
            if m[1] == "prompter":
                if context:
                    system_tag = f"{QA_SPECIAL_TOKENS['System']}{context}\n{eos_token}"
                else:
                    system_tag = ""
                if i==0 and system_msg:
                    output.append(f"{QA_SPECIAL_TOKENS['System']}{system_msg}{eos_token}{QA_SPECIAL_TOKENS['Question']}{m[0]}{eos_token}{system_tag}")
                else:
                    output.append(f"{QA_SPECIAL_TOKENS['Question']}{m[0]}{eos_token}{system_tag}")
            else:
                output.append(f"{QA_SPECIAL_TOKENS['Answer']}{m[0]}{eos_token}")

        return output


"Taken from https://github.com/LAION-AI/Open-Assistant/blob/main/model/model_training/custom_datasets/qa_datasets.py#L24"
class Vicuna(Dataset):
    name = "vicuna"

    @staticmethod
    def process_vicuna_conversation(data,input_max_length):
        role = None
        messages = []
        questions = []
        answers = []

        if len(data["conversations"]) == 0 or data["conversations"][0]["from"] != "human":
            return None
        
        for line in data["conversations"]:
            speaker = line["from"]  # 'human' or 'gpt'
            message = line["value"]
            if message is None or message == "":
                if speaker == "gpt":
                    return None
                elif speaker == "human":
                    # replace empty messages with one of the following
                    message = random.choice(["...", "Please continue", "Go on", ""])

            if role != speaker:
                if role is not None:
                    if role == "human":
                        questions.append("\n".join(messages)[:input_max_length])
                    if role == "gpt":
                        answers.append("\n".join(messages)[:input_max_length])
                    messages = []
                role = speaker
            messages.append(message.strip())

        if role is not None and len(messages) > 0:
            if role == "human":
                questions.append("\n".join(messages)[:input_max_length])
            if role == "gpt":
                answers.append("\n".join(messages)[:input_max_length])
        return questions,answers


    def __init__(self,cache_dir,eos_token,input_max_length=32*1024,**kwargs) -> None:
        super().__init__()
        
        dataset = load_dataset(
            "anon8231489123/ShareGPT_Vicuna_unfiltered",
            cache_dir=cache_dir,
            data_files=["ShareGPT_V3_unfiltered_cleaned_split.json"],
        )["train"]

        self.pairs = []
        for data in dataset:
            qa = self.process_vicuna_conversation(data,input_max_length=input_max_length)
            if qa is not None:
                if len(qa[0]) > 0 and len(qa[0]) == len(qa[1]):
                    self.pairs.append(
                        get_qa_formatted(
                        eos_token,
                        questions=qa[0],
                        answers=qa[1]
                        )
                    )
    
    def __len__(self):
        return len(self.pairs)


    def __getitem__(self, index):
        return self.pairs[index]


"Taken https://github.com/LAION-AI/Open-Assistant/blob/main/model/model_training/custom_datasets/qa_datasets.py#L570"
class DatabrickDolly15k(Dataset):
    def __init__(self,cache_dir,eos_token,**kwargs):
        super().__init__()

        self.rows = []
        self.citation_regex = re.compile(r"\[[a-zA-Z]\]")  # removes citations in the form of e.g. [a] or [A]

        data = load_dataset("OllieStanley/oa_dolly_15k", cache_dir=cache_dir)
        for line in data["train"]:
            instruction = self._process_instruction(line,eos_token)
            if instruction is not None:
                self.rows.append(instruction)


    def _process_instruction(self,row,eos_token):
        context = re_reference_remove.sub("",row["METADATA"]["CONTEXT"])
        context = context.replace("[citation needed]", "")
        context = self.citation_regex.sub("", context)
        if filter_by_words(row["INSTRUCTION"]) and filter_by_words(row["RESPONSE"]):
            return get_qa_formatted(eos_token,
                                    questions=[row["INSTRUCTION"]],
                                    answers=[row["RESPONSE"]],
                                    context=context)


    def __len__(self):
        return len(self.rows)


    def __getitem__(self, index):
        return self.rows[index]


"Taken from https://github.com/LAION-AI/Open-Assistant/blob/main/model/model_training/custom_datasets/instruction.py#L38"
class MathInstruction(Dataset):
    def __init__(self,cache_dir,eos_token,fill_min_length=None,**kwargs):
        super().__init__()
        self.rows = []
        self.eos_token = eos_token
        questions = []
        answers = []

        dataset = load_dataset("qwedsacf/grade-school-math-instructions", cache_dir=cache_dir, split="train")
        
        rng = random.Random(42)
        order = list(range(len(dataset)))
        rng.shuffle(order)
        num_invalid = 0
        item_len = 0
    

        # filter entries and optionally combine multiple entries
        for i in order:
            entry = dataset[i]
            q = entry["INSTRUCTION"]
            a = entry["RESPONSE"]
            if (
                q is not None
                and len(q.strip()) > 0
                and a is not None
                and len(a.strip()) > 0
                and filter_by_words(q)
                and filter_by_words(a)
            ):
                questions.append(q)
                answers.append(a)
                item_len += len(a) + len(q)

                if fill_min_length is None or fill_min_length < item_len:
                    self.rows.append((questions, answers))
                    item_len = 0
                    questions, answers = [], []
            else:
                num_invalid += 1
        
        if len(questions) > 0 and len(answers) > 0:
            self.rows.append((questions, answers))

        if num_invalid > 0:
            print(f"[Warning] {num_invalid} entries of {dataset} were invalid.")

    
    def __len__(self):
        return len(self.rows)
    

    def __getitem__(self, index):
        questions, answers =  self.rows[index]

        return get_qa_formatted(self.eos_token,
                         questions=questions,
                         answers=answers
                         )


"Taken from https://github.com/LAION-AI/Open-Assistant/blob/main/model/model_training/custom_datasets/qa_datasets.py#L426"
class AlpacaBaseDataset(Dataset):
    def __init__(self, cache_dir, eos_token, dataset_name="code_alpaca",**kwargs):
        super().__init__()
        if dataset_name == "alpaca":
            dataset = load_dataset("yahma/alpaca-cleaned", cache_dir=cache_dir)
        elif dataset_name == "code_alpaca":
            dataset = load_dataset("sahil2801/CodeAlpaca-20k", cache_dir=cache_dir)
        else:
            raise ValueError(f"Expected dataset_name to be 'alapaca' or 'code_alpaca'. Received {dataset_name}.")
        
        self.data = self._process(dataset["train"],eos_token)

    
    def _process(self,dataset,eos_token):
        data = []
        for row in dataset:
            question = row["instruction"]
            if len(row["input"]) > 0:
                input_ = "{}\n{}".format(question, row["input"])
            else:
                input_ = question

            if (filter_by_words(input_) is None) or (filter_by_words(row["output"]) is None):
                continue

            ds_entry = get_qa_formatted(eos_token,
                         questions=[input_],
                         answers=[row["output"]]
                         )
            data.append(ds_entry)
        return data


    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        dialogue = self.data[index]
        return dialogue


class QADataset(Dataset):
    def __init__(self, data: list,eos_token:str=''):
        super().__init__()
        self.data = data
        self.eos_token = eos_token

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        conversations= self.data[index]
        return get_qa_formatted(self.eos_token,
                         conversations=conversations,
                         )


"""Taken from:
https://github.com/LAION-AI/Open-Assistant/blob/main/model/model_training/custom_datasets/qa_datasets.py#L188
"""
class WebGPT(Dataset):
    name = "webgpt"

    def __init__(self, cache_dir,eos_token) -> None:
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
            answers = [x[0] for x in sorted(answers.items(), key=lambda x: -1 * x[1])]
            ds_entry = get_qa_formatted(eos_token,
                         questions=[question],
                         answers=[answers[0]]
                         )

            self.rows.append(ds_entry)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]

"""Rewritten from:
https://github.com/LAION-AI/Open-Assistant/blob/main/model/model_training/custom_datasets/oasst_dataset.py#L23
"""
def get_oasst_sft(val_split,eos_token,lang,manual_seed=90,max_val_set=None,**kwargs):
    generator = Generator()
    generator.manual_seed(manual_seed)
    threads_per_tree = load_oasst(mode="sft",lang=lang)
    def process_thread(thread):
            # ensure roles are strictly alternating between prompter and assistant
            assert all(m.role == "prompter" for m in thread[0::2]) and all(m.role == "assistant" for m in thread[1::2])
            conversation: list[list] = [[m.text,m.role]for m in thread]
            return conversation
    
    # split on tree basis, messages from same tree must not end up in different splits
    trees = ListDataset(threads_per_tree,)
    splits = random_split(trees, lengths=[1.0 - val_split, val_split], generator=generator)

    def flatten(ds: ListDataset) -> QADataset:
        return QADataset([process_thread(thread) for tree_threads in ds for thread in tree_threads],eos_token=eos_token)

    train = flatten(splits[0])
    eval = flatten(splits[1])

    if max_val_set and len(eval) > max_val_set:
        subset_indices = np.random.choice(len(eval), size=max_val_set, replace=False)
        eval = Subset(eval, subset_indices)

    print(f"OASST HF dataset: {len(train)=}, {len(eval)=}")
    return train,eval


def get_webgpt_sft(val_split,eos_token,cache_dir,manual_seed=90,max_val_set=None,**kwargs):
    generator = Generator()
    generator.manual_seed(manual_seed)
    webgpt = WebGPT(cache_dir=cache_dir,eos_token=eos_token)
    splits = random_split(webgpt, lengths=[1.0 - val_split, val_split], generator=generator)

    train = splits[0]
    eval = splits[1]

    if max_val_set and len(eval) > max_val_set:
        subset_indices = np.random.choice(len(eval), size=max_val_set, replace=False)
        eval = Subset(eval, subset_indices)

    print(f"WebGPT: {len(train)=}, {len(eval)=}")
    return train,eval

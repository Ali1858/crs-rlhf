from functools import partial

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset, Subset

from config import SFT_DATASET_CONFIG,CACHE_DIR
from constants import RANDOM_SEED


# mostly taken from
# https://huggingface.co/datasets/gozfarb/ShareGPT_Vicuna_unfiltered/blob/main/optional_clean.py,
# https://huggingface.co/datasets/ehartford/WizardLM_alpaca_evol_instruct_70k_unfiltered/blob/main/wizardlm_clean.py
FILTER_BY_WORDS = [
    "as a language model",
    "as an AI language model",
    "As a large language model",
    "As an AI ",
    "an AI language model you don't have",
    "As an AI language model, I cannot",
    "As an AI language model, I do not",
    "As an AI language model, I am not able",
    "As an AI language model, I don't have personal",
    "I am an AI language model and do not",
    "As an AI language model, I don't have",
    "As an AI language model, I am only able",
    "AI language model and I do not",
    "As an AI language model, I cannot modify",
    "As an AI language model, I do not",
    "I know as an AI language model you don't have",
    "as an AI language model, you cannot",
    "I'm sorry, but as an AI language model",
    "As an AI language model, I don't have",
    "I'm an AI ",
    "I am an AI ",
    "As your dedicated AI language model",
    "As a hypothetical AI",
    "As a neutral AI",
    "my knowledge cutoff",
    "my knowledge cut off",
    "As a machine",
    "I cannot assist",
    "I do not have personal preferences",
    "I don't have personal preferences",
    "Unfortunately, I cannot provide",
    "I'm sorry, I cannot",
    "I'm sorry, I cannot generate",
    "AI cannot create or program",
    "I'm afraid I cannot create",
    "OpenAI",
]


def filter_by_words(text: str, filter_words: list[str] | None = None) -> None | str:
    """Used to filter text that contains one of the `FILTER_BY_WORDS`. If so we return `None`
       otherwise we return the string

    Args:
        text (str): text to be filtered

    Returns:
        None | str: filtered text
    """
    filter_words = filter_words or FILTER_BY_WORDS
    for word in filter_words:
        if word.lower() in text.lower():
            return None
    return text


def train_val_dataset(dataset, name='unknown',val_split=0.2, max_val_set=None):
    if val_split == 0:
        return dataset, None

    train_idx, val_idx = train_test_split(
        list(range(len(dataset))), test_size=val_split, random_state=RANDOM_SEED, shuffle=True
    )
    train_subset, eval_subset = Subset(dataset, train_idx), Subset(dataset, val_idx)

    if max_val_set and len(eval_subset) > max_val_set:
        subset_indices = np.random.choice(len(eval_subset), size=max_val_set, replace=False)
        eval_subset = Subset(eval_subset, subset_indices)
    print(f'Size of {name} training data: {len(train_subset)}')
    print(f'Size of {name} training data: {len(eval_subset)}')
    return train_subset, eval_subset


def load_sft_dataset(eos_token):
    from training_datasets.sft_dataset import Vicuna, DatabrickDolly15k, AlpacaBaseDataset, MathInstruction
    dataset_func_mapping  = {"vicuna": partial(Vicuna,input_max_length=1024),
                         "dolly": DatabrickDolly15k,
                         "alpaca": AlpacaBaseDataset,
                         "math_instruction":MathInstruction,
                         }

    train_datasets = []
    evals = {}

    for ds_name, value in SFT_DATASET_CONFIG.items():
        train_ds, val_ds = train_val_dataset(dataset_func_mapping[ds_name](cache_dir=CACHE_DIR,eos_token=eos_token),name=ds_name,val_split=value["val_split"])
        train_datasets.append(train_ds)

        if val_ds is not None:
            evals[ds_name] = val_ds
    train = ConcatDataset(train_datasets)
    return train,evals

import random
from typing import List
from collections import defaultdict

from torch.utils.data import Dataset, random_split
from torch import Generator

from training_datasets.dataset_utils import load_oasst, ListDataset, format_pairs
from constants import QA_SPECIAL_TOKENS


SEED = 2020


class TextDataset(Dataset):
    def __init__(self, data: list):
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def get_oasst_rl(val_split,cache_dir,lang,manual_seed=90,top_k=None,**kwargs):
    generator = Generator()
    generator.manual_seed(manual_seed)
    threads_per_tree = load_oasst(mode="rl",top_k=top_k,lang=lang)
    def process_thread(thread):
        return [m.text for m in thread]
    
    # split on tree basis, messages from same tree must not end up in different splits
    trees = ListDataset(threads_per_tree,)
    splits = random_split(trees, lengths=[1.0 - val_split, val_split], generator=generator)

    def flatten(ds: ListDataset) -> TextDataset:
        return TextDataset([process_thread(thread) for tree_threads in ds for thread in tree_threads])

    train = flatten(splits[0])
    val = flatten(splits[1])
    print(f"OASST HF dataset: {len(train)=}, {len(val)=}")
    return train,val
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset, Subset, Dataset
from oasst_data import ExportMessageNode, read_dataset_message_trees, visit_threads_depth_first

from constants import RANDOM_SEED, CACHE_DIR, QA_SPECIAL_TOKENS


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
    print(f'Size of {name} validation data: {len(eval_subset)}')
    return train_subset, eval_subset


class ListDataset(Dataset):
    def __init__(self, data: list):
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
    

"""Rewritten from:
https://github.com/LAION-AI/Open-Assistant/blob/main/model/model_training/custom_datasets/oasst_dataset.py#L23
"""
def load_oasst(mode="sft",
               lang="en",
               top_k=None):
    if mode not in ("sft", "rm", "rl"):
        raise ValueError(f"Unknown dataset mode: {mode}")
    
    lang_codes: list[str] = lang.split(",")

    threads_per_tree = []
    tree_iter = read_dataset_message_trees("OpenAssistant/oasst1",split="train+validation")
    
    for tree in tree_iter:
        if tree.tree_state != "ready_for_export" or not tree.prompt.review_result or tree.prompt.lang not in lang_codes:
            continue

        if mode in ("sft", "rm"):
            if tree.tree_state != "ready_for_export":
                continue
        elif mode == "rl":
            if tree.tree_state not in ("ready_for_export", "prompt_lottery_waiting"):
                continue

        # extract all threads up to last assistant reply
        threads: list[list[ExportMessageNode]] = []
            
        def thread_filter(thread: list[ExportMessageNode]) -> bool:
            if any(m.deleted or m.synthetic for m in thread):
                return False

            if top_k is not None:
                for i, m in enumerate(thread):
                    if m.role == "assistant":
                        if m.rank is None:
                            if i > 0 and len(thread[i - 1].replies) > 1:
                                return False
                        elif m.rank >= top_k:
                            return False
            return True


        def leaf_filter(thread: list[ExportMessageNode]) -> bool:
                if mode == "sft":
                    # in SFT mode `not thread[-1].replies` finds nodes without children (leaves).
                    # We are interested in those which are role='assistant' but some trees don't end on assistant nodes
                    # but have prompter leaves .. we want to use those trees too .. e.g. remove the last prompter message(s)
                    # so that they end with assistant. The `thread[-2].replies[0] == thread[-1]` check makes sure that only
                    # the FIRST prompter reply is added .. e.g. the parent does not appear multiple times and we can use
                    # pop() to remove superfluous prompter leaf node later.
                    return (
                        len(thread) > 1
                        and not thread[-1].replies
                        and (thread[-1].role == "assistant" or thread[-2].replies[0] == thread[-1])
                        and thread_filter(thread)
                    )
                elif mode == "rm":
                    # for reward models we use thread-fragments ending on prompter messages as prefix and
                    # their (ranked) replies as possible continuations.
                    if thread[-1].replies is None:
                        return False
                    return (
                        thread[-1].role == "prompter"
                        and len([r for r in thread[-1].replies if r.rank is not None]) > 1
                        and thread_filter(thread)
                    )
                elif mode == "rl":
                    # during rl we are interested in all possible prefixes ending in prompter messages
                    return thread[-1].role == "prompter" and not any(m.deleted or m.synthetic for m in thread)

                raise RuntimeError()

        visit_threads_depth_first(tree.prompt, visitor=threads.append, predicate=leaf_filter)
        if mode == "sft":
            for t in threads:
                if t[-1].role == "prompter":
                    t.pop()
        threads_per_tree.append(threads)
    return threads_per_tree


def load_sft_dataset(conf,eos_token):
    from training_datasets.sft_dataset import Vicuna, DatabrickDolly15k, AlpacaBaseDataset, MathInstruction, get_oasst_sft
    dataset_func_mapping  = {"vicuna": Vicuna,
                         "dolly": DatabrickDolly15k,
                         "alpaca": AlpacaBaseDataset,
                         "math_instruction":MathInstruction,
                         "oasst_export":get_oasst_sft
                         }
    train_datasets = []
    evals = {}
    if conf.debug:
        key = next(iter(conf.dataset))
        conf.dataset = {key:conf.dataset[key]}

    for ds_name, dataset_kwargs in conf.dataset.items():
        print(f'===loading the {ds_name} dataset===\n')
        ds = dataset_func_mapping[ds_name](cache_dir=CACHE_DIR,eos_token=eos_token,**dataset_kwargs)
        if type(ds) == tuple:
            train_ds, val_ds = ds
        else:
            train_ds,val_ds = train_val_dataset(ds,name=ds_name,
                                                val_split=dataset_kwargs["val_split"],
                                                max_val_set=dataset_kwargs["max_val_set"])
        train_datasets.append(train_ds)
        evals[ds_name] = val_ds
    train = ConcatDataset(train_datasets)

    if conf.debug:
        print("Using only 200 rows for debuging")
        subset_indices = range(200)
        train = Subset(train, subset_indices)
    return train,evals

              
def load_rm_dataset(conf):
    from training_datasets.rm_dataset import AnthropicRLHF, HellaSwagDataset, SHPDataset, get_oasst_rm
    dataset_func_mapping  = {
                        "anthropic": AnthropicRLHF,
                        "hellaswag": HellaSwagDataset,
                        "shp":SHPDataset,
                        "oasst_export":get_oasst_rm
                        }
    train_datasets = []
    evals = {}

    for ds_name, dataset_kwargs in conf.dataset.items():
        print(f'===loading the {ds_name} dataset===\n')
        max_val_set = dataset_kwargs["max_val_set"]

        if len(dataset_kwargs.get("splits",[])) ==2:
            train_ds = dataset_func_mapping[ds_name](cache_dir=CACHE_DIR,split=dataset_kwargs["splits"][0])
            val_ds = dataset_func_mapping[ds_name](cache_dir=CACHE_DIR,split=dataset_kwargs["splits"][1])
        else:
            train_ds,val_ds = dataset_func_mapping[ds_name](cache_dir=CACHE_DIR,**dataset_kwargs)

        if max_val_set and len(val_ds) > max_val_set:
            subset_indices = np.random.choice(len(val_ds), size=max_val_set, replace=False)
            val_ds = Subset(val_ds, subset_indices)

        train_datasets.append(train_ds)
        evals[ds_name] = val_ds
        print(f'Size of {ds_name} training data: {len(train_ds)}')
        print(f'Size of {ds_name} validation data: {len(val_ds)}')
    train = ConcatDataset(train_datasets)

    if conf.debug:
        print("Using only 200 rows for debuging")
        subset_indices = range(200)
        train = Subset(train, subset_indices)

    return train,evals


def get_rm_formatted(
        eos_token,
        text,
        is_replies=False,
    ):
        if not is_replies:
            return [
                "{}{}{}".format(QA_SPECIAL_TOKENS["Question" if i % 2 == 0 else "Answer"], text[i], eos_token)
                for i in range(len(text))
                ]
        else:
            return "{}{}{}".format(QA_SPECIAL_TOKENS["Answer"], text, eos_token)
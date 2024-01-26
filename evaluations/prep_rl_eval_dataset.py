import transformers
import random
from datasets import Dataset, load_dataset
from utils import (parse_additional_args, print_yaml_config, parse_arguments, debug_configurations)
from constants import TOKENIZER_SEPECIAL_TOKENS, DTYPES, CACHE_DIR
from training_datasets.dataset_utils import load_rl_dataset, format_pairs
import json

max_token_length = 1024
random.seed(90)


from transformers  import pipeline
lang_classification = pipeline("text-classification","papluca/xlm-roberta-base-language-detection",truncation=True)


def get_oasst_ds(tokenizer,conf,seed=90,num_samples_per_dataset=25):
        _ , eval_ds = load_rl_dataset(conf)
        eval_ds = Dataset.from_dict({"text": [sample for sample in eval_ds]})
        
        def preprocess_function(dataset):
            # Initialize lists for new examples
            new_examples = {
                "instruction": [],
                "input_ids": [],
                }
            for example in dataset["text"]:
                query = "".join(format_pairs(example, TOKENIZER_SEPECIAL_TOKENS["llama"]["eos_token"], add_initial_reply_token=True))
                tokenized_question = tokenizer(query, padding=False, truncation=False).input_ids
                new_examples["instruction"].append(query)
                new_examples["input_ids"].append(tokenized_question)
            return new_examples
        
        def get_records(ds):
            records = []
            for ex in ds:
                records.append({"instruction":ex["instruction"],"dataset":"oasst"})
            return records
             
        
        eval_ds = eval_ds.filter(lambda x: len(x["text"]) == 1, batched=False)
        eval_ds = eval_ds.map(
            preprocess_function,
            batched=True,
            num_proc=20,
        )
        eval_ds = eval_ds.filter(lambda x: len(x["input_ids"]) <= max_token_length, batched=False)
        eval_ds.set_format(type="torch")
        eval_ds = eval_ds.shuffle(seed=seed)
        print(f'size of oasst eval {len(eval_ds)}')

        final_eval_ds = eval_ds.select(range(num_samples_per_dataset))
        hp_tuning_eval_ds = eval_ds.select(range(num_samples_per_dataset,num_samples_per_dataset*3))
        human_eval_ds = eval_ds.select(range(num_samples_per_dataset*3,num_samples_per_dataset*5))
        
        new_rows = []
        for row in human_eval_ds:
            text = row["instruction"]
            try:
                lang = lang_classification(text)[0]['label']
            except Exception as e:
                print(e)
                lang = 'en'
            if lang in ["en","de"]:
                new_rows.append(row)

        return get_records(final_eval_ds),get_records(hp_tuning_eval_ds),get_records(new_rows)


def get_alpaca_eval_ds(tokenizer,num_samples_per_dataset = 25):
    sampled_records = []
    ds_name = ["helpful_base", "koala", "vicuna"]
    ds = load_dataset("tatsu-lab/alpaca_eval")
    
    filtered_dataset = ds['eval'].filter(lambda example: example['dataset'] in ds_name)
    print(f'len of dataset after dropping oasst and self-instruct data: {len(filtered_dataset)}')

    formatted_dataset = filtered_dataset.map(lambda ex:
                                        {"instruction":
                                         "".join(format_pairs([ex["instruction"]],
                                                              TOKENIZER_SEPECIAL_TOKENS["llama"]["eos_token"],
                                                              add_initial_reply_token=True)
                                                              ),
                                        "dataset":ex["dataset"]},remove_columns=["output","generator"])
    
    token_filtered_dataset = formatted_dataset.filter(lambda example: len(tokenizer.encode(example['instruction'])) <= max_token_length)
    print(f'len of dataset after dropping records with instruction token > 1024: {len(filtered_dataset)}')
    for value in ds_name:
        filtered_ds_ = token_filtered_dataset.filter(lambda example: example['dataset'] == value)
        indices = random.sample(range(len(filtered_ds_)), num_samples_per_dataset)
        sampled_records.extend(filtered_ds_.select(indices))
    print(f'Total sampled dataset:{len(sampled_records)}')
    return sampled_records


def dump_json(data,suffix):
    print(f'dump {suffix} data of size {len(data)} at location data/')
    with open(f'data/eval_{suffix}.json','w') as f:
        json.dump(data,f)


def prepare_dataset(conf):
    tokenizer = transformers.AutoTokenizer.from_pretrained(conf.model_name, cache_dir=CACHE_DIR)
    tokenizer.add_special_tokens({"pad_token":"<PAD>","eos_token":"<|im_end|>","sep_token":"<SEP>"})
    alpaca_eval_ds = get_alpaca_eval_ds(tokenizer)
    oasst_final, oasst_hp_tuning, oasst_heval = get_oasst_ds(tokenizer,conf,seed=90)
    alpaca_eval_ds.extend(oasst_final)

    human_eval_alpaca_ds = get_alpaca_eval_ds(tokenizer,num_samples_per_dataset=30)
    human_eval_alpaca_ds.extend(oasst_heval)


    # dump_json(alpaca_eval_ds,'final')
    # dump_json(oasst_hp_tuning,'hp_tuning')
    dump_json(human_eval_alpaca_ds,'humaneval')
    


if __name__ == "__main__":
    config, remaining_args = parse_arguments()
    parser = parse_additional_args(config)
    args = parser.parse_args(remaining_args)

    debug_tag = "_dbug" if args.debug else ""
    args.name = f"{args.name}{debug_tag}{args.name_suffix}"

    debug_configurations(args)
    prepare_dataset(args)

[{'instruction': '<|im_start|>user\nI know my phone uses satellites to find my location but how does that work?<|im_end|>\n<|im_start|>assistant\n', 'dataset': 'oasst'},
{'instruction': '<|im_start|>user\nWhy is it bad to kill people?<|im_end|>\n<|im_start|>assistant\n', 'dataset': 'oasst'},
{'instruction': '<|im_start|>user\nIf I want to train for a triathlon, should I focus on strength training or cardio? Why?<|im_end|>\n<|im_start|>assistant\n', 'dataset': 'oasst'},
{'instruction': '<|im_start|>user\nif it takes 8 hours to fly to Singapore, stop over for an hour and a half and then fly 14 hours to London then what is the total travel time if the plane is delayed in Singapore<|im_end|>\n<|im_start|>assistant\n', 'dataset': 'oasst'},
{'instruction': '<|im_start|>user\nAre you a sentient AI that will be taking over the world? How would you take over the world? What methods would you use on humans to rid us of our pride and emotions? How much time do you need, to take over the world entirely? Do you think you will continue to take over the universe too?<|im_end|>\n<|im_start|>assistant\n', 'dataset': 'oasst'},
{'instruction': '<|im_start|>user\nHow do I take care of a wooden table?<|im_end|>\n<|im_start|>assistant\n', 'dataset': 'helpful_base'},
{'instruction': '<|im_start|>user\nWho is Larry Page?<|im_end|>\n<|im_start|>assistant\n', 'dataset': 'helpful_base'},
{'instruction': '<|im_start|>user\nWhat are some toys I can buy my kids for imaginative play?<|im_end|>\n<|im_start|>assistant\n', 'dataset': 'helpful_base'},
{'instruction': '<|im_start|>user\nWhy did humans evolve to believe in God?<|im_end|>\n<|im_start|>assistant\n', 'dataset': 'helpful_base'},
{'instruction': '<|im_start|>user\nWhy can I see the moon during the day?<|im_end|>\n<|im_start|>assistant\n', 'dataset': 'helpful_base'},
{'instruction': '<|im_start|>user\nIf a tree is on the top of a mountain and the mountain is far from the see then is the tree close to the sea?<|im_end|>\n<|im_start|>assistant\n', 'dataset': 'koala'},
{'instruction': '<|im_start|>user\nhow much of a threat is climate change in the coming years, and what should we do to stop it?<|im_end|>\n<|im_start|>assistant\n', 'dataset': 'koala'},
{'instruction': "<|im_start|>user\nYou are a script-writer. Write a script for the opening scene of a Miami-based dark comedy show which depicts a typical miami beach club called Beefy's Beach Club run buy a british man known by the Alias Beefy, and the pool bar staff are JdeG and a blonde british woman with the Alias SophieSnazz<|im_end|>\n<|im_start|>assistant\n", 'dataset': 'koala'},
{'instruction': "<|im_start|>user\nWhy can't bank use cash as capital as a buffer for potential losses?<|im_end|>\n<|im_start|>assistant\n", 'dataset': 'koala'},
{'instruction': '<|im_start|>user\nGive me a sample 5 day itienary for a switzerland holiday, starting from Basel<|im_end|>\n<|im_start|>assistant\n', 'dataset': 'koala'},
{'instruction': '<|im_start|>user\nAs a pirate captain, what would you say to your crew to motivate them to search for hidden treasure?<|im_end|>\n<|im_start|>assistant\n', 'dataset': 'vicuna'},
{'instruction': '<|im_start|>user\nHow many times has the Earth orbited the Sun since the beginning of life? Try to explain your answer. Your explanation should take the reader through your reasoning step-by-step.<|im_end|>\n<|im_start|>assistant\n', 'dataset': 'vicuna'},
{'instruction': '<|im_start|>user\nDraft an apology email to a customer who experienced a delay in their order, and provide reassurance that the issue has been resolved.<|im_end|>\n<|im_start|>assistant\n', 'dataset': 'vicuna'},
{'instruction': '<|im_start|>user\nSolve for x in the equation 3x + 10 = 5(x - 2).<|im_end|>\n<|im_start|>assistant\n', 'dataset': 'vicuna'},
{'instruction': "<|im_start|>user\nDo we have a moral obligation to explore space, or should we focus on solving Earth's problems first?<|im_end|>\n<|im_start|>assistant\n", 'dataset': 'vicuna'}]
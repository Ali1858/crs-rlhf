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
        human_eval_ds = eval_ds.select(range(num_samples_per_dataset*3,num_samples_per_dataset*4))
        
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

        return get_records(final_eval_ds),get_records(hp_tuning_eval_ds),get_records(new_rows[:5])


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

    human_eval_alpaca_ds = get_alpaca_eval_ds(tokenizer,num_samples_per_dataset=5)
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

@misc{koala_blogpost_2023,
  author = {Xinyang Geng and Arnav Gudibande and  and  and  and  and },
  title = {},
  url = {https://bair.berkeley.edu/blog/2023/04/03/koala/},
  urldate = {2023-04-03}
}
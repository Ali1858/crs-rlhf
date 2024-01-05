import os
from tqdm import tqdm
from time import time

import torch

import transformers
from peft import PeftModel

from utils import (parse_additional_args, print_yaml_config, parse_arguments, debug_configurations)
from constants import TOKENIZER_SEPECIAL_TOKENS, DTYPES, CACHE_DIR
import json
sigmoid = torch.nn.Sigmoid()

seed = 90
torch.manual_seed(seed)


def get_reward_tokenizer_model(conf,dtype,device_map="auto"):
    print('**** loading reward model ****')
    # Since reward models are trained using the same base model, we should use same model
    base_reward_model = transformers.AutoModelForSequenceClassification.from_pretrained(
            conf.reward_model_name,
            num_labels=1,
            use_flash_attention_2=True,
            load_in_8bit=True,
            device_map=device_map,
            torch_dtype=dtype,
            cache_dir=CACHE_DIR,
            )
    reward_tokenizer = transformers.AutoTokenizer.from_pretrained(conf.reward_model_name, cache_dir=CACHE_DIR)
    reward_tokenizer.add_special_tokens({"pad_token":"<PAD>","eos_token":"<|im_end|>","sep_token":"<SEP>"})
    print(f'tokenizer pad {reward_tokenizer.pad_token} and model pad {base_reward_model.config.pad_token_id}')
    print(f'tokenizer eos {reward_tokenizer.eos_token} and model eos {reward_tokenizer.eos_token_id}')
    if base_reward_model.config.pad_token_id is None or base_reward_model.config.pad_token_id == 0:
        print('changing model pad token id')
        base_reward_model.config.pad_token_id = reward_tokenizer.pad_token_id

    if conf.crs:
        assert conf.abs_adapter_name, "Please specify adapter name of absolute reward model"
        assert conf.ranking_adapter_name, "Please specify adapter name of ranking reward model"
        conf.abs_model_name =  os.path.join(conf.abs_adapter_name,conf.adapter_number)
        conf.ranking_model_name = os.path.join(conf.ranking_adapter_name,conf.adapter_number)
        print(conf.ranking_model_name),
        print(conf.abs_model_name)
        base_reward_model = PeftModel.from_pretrained(
            base_reward_model,
            conf.ranking_model_name,
            adapter_name="ranking",
            is_trainable=False
            )
        base_reward_model.load_adapter(conf.abs_model_name,adapter_name="abs",is_trainable=False)
    else:
        print('Not using combined reward signal')
        if conf.load_reward_type == "abs":
            assert conf.abs_adapter_name, "Please specify adapter name of absolute reward model"
            reward_model_path = os.path.join(conf.abs_adapter_name,conf.adapter_number)
        elif conf.load_reward_type == "ranking":
            assert conf.ranking_adapter_name, "Please specify adapter name of ranking reward model"
            reward_model_path =  os.path.join(conf.ranking_adapter_name,conf.adapter_number)
        else:
            raise "Please choose atleast one remodel type. From the following choice ['abs', 'ranking']"
        print(reward_model_path)
        base_reward_model = PeftModel.from_pretrained(
            base_reward_model,
            reward_model_path,
            adapter_name=conf.load_reward_type,
            is_trainable=False,
            )
        base_reward_model.set_adapter(conf.load_reward_type)
    return base_reward_model,reward_tokenizer


def compute_reward_score(texts,reward_tokenizer,base_reward_model,conf,batch_size = 8):
    rewards = []
    weight = conf.crs_weight if conf.crs_weight else 0.5   
    tokenized_inputs = [reward_tokenizer(text, padding=False, truncation=False) for text in texts]
    with torch.no_grad():  
        for i in range(0, len(tokenized_inputs), batch_size):
            inputs = tokenized_inputs[i:i+batch_size]

            batch = reward_tokenizer.pad(
                inputs,
                padding=True,
                pad_to_multiple_of=16,
                return_tensors="pt",
            ).to(base_reward_model.device)
            if conf.crs:    
                rank_reward = []
                abs_reward = []
                
                base_reward_model.set_adapter("ranking")
                rank_logits = base_reward_model(**batch).logits[:,0]
                rank_reward.extend(rank_logits)
            
                base_reward_model.set_adapter("abs")
                abs_logits = base_reward_model(**batch).logits[:,0]
                abs_reward.extend(abs_logits)

                for rank,abs in zip(rank_reward,abs_reward):
                    rewards.append((weight * rank) + ((1-weight) * abs))
            else:
                logits = base_reward_model(**batch).logits[:,0]
                # if conf.load_reward_type == "abs":
                #     sig_scores = sigmoid(logits)
                #     sig_scores =  [torch.clamp(score-0.1, min=0)+0.000001 if score < 0.46 else score for score in sig_scores]
                #     logits = sigmoid_inverse(sig_scores)
                rewards.extend(logits)
    return rewards


def sigmoid_inverse(sig_scores):
    logits = []
    for score in sig_scores:
        logit = torch.log(score / (1 - score))
        logits.append(logit)
    return logits


def get_base_mode_for_inference(conf,dtype,peft=True,device_map = {"":0}):
    # from transformers import pipeline, TextGenerationPipeline
    base_model = transformers.AutoModelForCausalLM.from_pretrained(
        conf.model_name,
        use_flash_attention_2=True,
        load_in_8bit=True,
        device_map=device_map,
        torch_dtype=dtype,
        cache_dir=CACHE_DIR,
    )
    # Load the Lora model
    if peft:
        base_model = PeftModel.from_pretrained(base_model,
                                           conf.peft_id+'/final_checkpoint',#/checkpoint_160',
                                           adapter_name="rlhf",
                                           is_trainable=False,
                                           device_map=device_map
                                           )
        base_model.set_adapter("rlhf")

    tokenizer = transformers.AutoTokenizer.from_pretrained(conf.model_name, cache_dir=CACHE_DIR)
    tokenizer.add_special_tokens({"pad_token":"<PAD>","eos_token":"<|im_end|>","sep_token":"<SEP>"})

    print(f'tokenizer pad {tokenizer.pad_token} and model pad {base_model.config.pad_token_id}')
    print(f'tokenizer eos {tokenizer.eos_token} and model eos {tokenizer.eos_token_id}')
    if base_model.config.pad_token_id is None or base_model.config.pad_token_id == 0:
        print('changing model pad token id')
        base_model.config.pad_token_id = tokenizer.pad_token_id

    # base_model = pipeline("text-generation",
    #                       model=conf.peft_id+'/final_checkpoint',
    #                       device_map=device_map,
    #                       model_kwargs={"load_in_8bit": True,"cache_dir":CACHE_DIR},)
    return base_model, tokenizer


def generate_text(tokenizer,model,query_tensors,generation_kwargs,batch_size=8):
    outputs = []
    tokenizer.padding_side = "left"
    # in case we have fewer examples than bs
    batch_size = min(len(query_tensors), batch_size)
    model.eval()
    for i in range(0, len(query_tensors), batch_size):
            # prevent overflow if query tensors are not even multiple of bs
            end_index = min(len(query_tensors), i + batch_size)

            batch = query_tensors[i:end_index]
            batch_mask = [torch.ones_like(element) for element in batch]
            inputs = {"input_ids": batch, "attention_mask": batch_mask}

            padded_inputs = tokenizer.pad(
                inputs,
                padding=True,
                max_length=None,
                pad_to_multiple_of=16,
                return_tensors="pt",
            ).to(model.device)
            with torch.no_grad():  
                generations = model.generate(**padded_inputs, **generation_kwargs)

            for generation, mask in zip(generations, padded_inputs["attention_mask"]):
                output = generation[(1 - mask).sum() :]  # remove padding
                output = output[(mask).sum() :]  # remove prompt
                if tokenizer.eos_token_id in output:
                    pad_mask = output == tokenizer.eos_token_id
                    pad_start = torch.nonzero(pad_mask, as_tuple=False)[0, 0].item()
                    output = output[: pad_start + 1]  # keep the eos token at the end
                outputs.append(output)
    return outputs


def pipeline_(conf,tokenizer,base_model,base_reward_model,reward_tokenizer,examples,generation_kwargs,mode="rlhf"):

    start_time = time()
    query_tensors = examples["input_ids"]
    query = examples["query"]
    response = generate_text(tokenizer,base_model,query_tensors,generation_kwargs)
    # decoded_responses = [r[0] for r in base_model(query,**generation_kwargs)]
    decoded_responses = tokenizer.batch_decode(response, skip_special_tokens=True)
    reward_text = [q[:-len("<|im_start|>assistant\n")]+"<|im_start|><|im_start|>assistant\n" +r+'<|im_end|>\n' for q, r in zip(query, decoded_responses)]
    rewards = compute_reward_score(reward_text, reward_tokenizer,base_reward_model, conf)

    # Prepare data to be saved
    data_to_save = [{"query": q, "response": r, "reward": float(rew)} for q, r, rew in zip(query, decoded_responses, rewards)]
    # Save to JSON file
    filename = os.path.join(conf.peft_id, "eval_output",f"{conf.output_json_prefix}_{mode}_output.json")
    with open(filename, 'w') as f:
        json.dump(data_to_save, f, ensure_ascii=False, indent=4)
    end = time()
    print(f"*** Saved generated responses and rewards to {filename} in {end-start_time} seconds")



def eval(conf):
    print(f"\n{'==='*10} Following are the configuration for training{'==='*10}")
    print_yaml_config(conf)

    dtype = DTYPES.get(conf.dtype, torch.float32)  
    base_model,tokenizer = get_base_mode_for_inference(conf,dtype,device_map="auto")
    base_reward_model,reward_tokenizer = get_reward_tokenizer_model(conf,dtype,device_map="auto")

    ### Prepare eval dataset
    with open(conf.eval_data,'r') as f:
        queries = json.load(f)
    new_examples = {
            "query": [],
            "input_ids": [],
            }
    for query in queries:
            tokenized_question = torch.tensor(tokenizer(query["instruction"], padding=False, truncation=False).input_ids)
            new_examples["query"].append(query["instruction"])
            new_examples["input_ids"].append(tokenized_question)
    print(f'Eval dataset size: {len(new_examples["query"])}')

    n = 4
    data_prefix = conf.eval_data.split('/')[-1].split('.')[0].split('_')[-1]
    for i in range(n):
        if n > 1 and i == 0:
            generation_kwargs = {
                "top_k": 0,
                "top_p": 0.9,
                "do_sample": True,
                "temperature":0.8,
                "max_new_tokens":512
            }
            conf.output_json_prefix = f'{data_prefix}_p_09_t_08'
        else:
            generation_kwargs = {
                "top_k": 0.0,
                "top_p": 1.0,
                "do_sample": True,
                "max_new_tokens": 512,
            }
            conf.output_json_prefix = f'{data_prefix}_p_01_run_{i}'
        print(f'running prediction for {conf.output_json_prefix}...')
        pipeline_(conf,tokenizer,base_model,base_reward_model,reward_tokenizer,new_examples,generation_kwargs)

    if conf.eval_sft:
        base_model,tokenizer = get_base_mode_for_inference(conf,dtype,peft=False,device_map = "auto")
        for i in range(n):
            if n > 1 and i == 0:
                generation_kwargs = {
                    "top_k": 0,
                    "top_p": 0.9,
                    "do_sample": True,
                    "temperature":0.8,
                    "max_new_tokens":512
                }
                conf.output_json_prefix = f'{data_prefix}_p_09_t_08'
            else:
                generation_kwargs = {
                    "top_k": 0.0,
                    "top_p": 1.0,
                    "do_sample": True,
                    "max_new_tokens": 512,
                }
                conf.output_json_prefix = f'{data_prefix}_p_01_run_{i}'
            print(f'running prediction for {conf.output_json_prefix}...')
            pipeline_(conf,tokenizer,base_model,base_reward_model,reward_tokenizer,new_examples,generation_kwargs,"sft")


if __name__ == "__main__":
    config, remaining_args = parse_arguments()
    parser = parse_additional_args(config)
    args = parser.parse_args(remaining_args)

    debug_tag = "_dbug" if args.debug else ""
    args.name = f"{args.name}{debug_tag}{args.name_suffix}"

    debug_configurations(args)
    eval(args)

import transformers
import torch
import os
from glob import glob
import json 
from peft import PeftModel

CACHE_DIR = 'cache'
dtype = torch.bfloat16

def get_reward_tokenizer_model(reward_model_name,abs_adapter_name,ranking_adapter_name,device_map="auto"):
    print('**** loading reward model ****')
    # Since reward models are trained using the same base model, we should use same model
    base_reward_model = transformers.AutoModelForSequenceClassification.from_pretrained(
            reward_model_name,
            num_labels=1,
            use_flash_attention_2=True,
            load_in_8bit=True,
            device_map=device_map,
            torch_dtype=dtype,
            cache_dir=CACHE_DIR,
            )
    reward_tokenizer = transformers.AutoTokenizer.from_pretrained(reward_model_name, cache_dir=CACHE_DIR)
    reward_tokenizer.add_special_tokens({"pad_token":"<PAD>","eos_token":"<|im_end|>","sep_token":"<SEP>"})
    print(f'tokenizer pad {reward_tokenizer.pad_token} and model pad {base_reward_model.config.pad_token_id}')
    print(f'tokenizer eos {reward_tokenizer.eos_token} and model eos {reward_tokenizer.eos_token_id}')
    if base_reward_model.config.pad_token_id is None or base_reward_model.config.pad_token_id == 0:
        print('changing model pad token id')
        base_reward_model.config.pad_token_id = reward_tokenizer.pad_token_id

    abs_model_name =  os.path.join(abs_adapter_name,'final_checkpoint')
    ranking_model_name = os.path.join(ranking_adapter_name,'final_checkpoint')
    print(ranking_model_name),
    print(abs_model_name)

    base_reward_model = PeftModel.from_pretrained(
        base_reward_model,
        ranking_model_name,
        adapter_name="ranking",
        is_trainable=False,
        device_map=device_map
        )
    base_reward_model.load_adapter(abs_model_name,adapter_name="abs",is_trainable=False)
    base_reward_model = base_reward_model.to(base_reward_model.device)
    return base_reward_model,reward_tokenizer


def compute_reward_score(texts,reward_tokenizer,base_reward_model,batch_size = 8):
    rank_reward = []
    abs_reward = []
    crs_025 = []
    crs_0625 = []
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
        
            base_reward_model.set_adapter("ranking")
            rank_logits = base_reward_model(**batch).logits[:,0]
            rank_reward.extend(rank_logits)
        
            base_reward_model.set_adapter("abs")
            abs_logits = base_reward_model(**batch).logits[:,0]
            abs_reward.extend(abs_logits)

    for rank,abs in zip(rank_reward,abs_reward):
        weight = 0.25
        crs_025.append((weight * rank) + ((1-weight) * abs))
        weight = 0.625
        crs_0625.append((weight * rank) + ((1-weight) * abs))
                
    return rank_reward,abs_reward,crs_025,crs_0625


reward_model_name = "andreaskoepf/llama2-7b-oasst-baseline"
abs_adapter_name = "output/rm/LLama-2-7b-oasst-baseline_reward_abs_bs128_ep_1_8bit_logistic_s_075_augment_and__under"
ranking_adapter_name = "output/rm/LLama-2-7b-oasst-baseline_reward_ranking_bs64_ep_1_8bit_bf16_eos_token"
device_map = {"":0}

reward_model, tokenizer = get_reward_tokenizer_model(reward_model_name,abs_adapter_name,ranking_adapter_name,device_map)


rlhf_model_names = [
    # "LLama-2-7b-oasst-baseline_rl_abs_quality_rw_075_bs16_kl_002_clip_04_512_max_token_with_pad_eos_lr_141e5_logits",
    #                 "LLama-2-7b-oasst-baseline_rl_bs16_kl_001_clip_04_512_max_token_with_pad_eos_lr_141e5",
                    # "LLama-2-7b-oasst-baseline_rl_bs16_kl_002_clip_04_512_max_token_with_pad_eos_lr_141e5",
                    # "LLama-2-7b-oasst-baseline_rl_f_crs_025_bs16_kl_002_clip_04_512_max_token_with_pad_eos_lr_141e5",
                    # "LLama-2-7b-oasst-baseline_rl_f_crs_0625_bs16_kl_002_clip_04_512_max_token_with_pad_eos_lr_141e5",
                    "LLama-2-7b-oasst-basseline_sft"
                    ]

# glob_rgx = "/*run*.json"
glob_rgx = "/*.json"
for n in rlhf_model_names:
    print(f'\n==== calculating reward for the model {n}====')
    eval_path = os.path.join("output/rl",n,"eval_output")
    all_jsons = glob(eval_path+glob_rgx)
    for j in all_jsons:
        print(f'**checkpoint:{j}**')
        with open(j,'r') as r_fn:
            data = json.load(r_fn)
        
        reward_text = [d["query"][:-len("<|im_start|>assistant\n")]+"<|im_start|><|im_start|>assistant\n" +d["response"]+'<|im_end|>\n' for d in data]
        rank_reward,abs_reward,crs_025,crs_0625 = compute_reward_score(reward_text,tokenizer,reward_model)

        for d,r_rw,a_rw,c025_rw,c0625_rw in zip(data,rank_reward,abs_reward,crs_025,crs_0625):
            d["reward"] = r_rw.item()
            d["abs_reward"] = a_rw.item()
            d["crs_025_reward"] = c025_rw.item()
            d["crs_0625_reward"] = c0625_rw.item()

        path = '/'.join(j.replace('eval_output','rewarded_eval').split('/')[:-1])
        fn = j.split('/')[-1]
        
        if not os.path.exists(path):
            os.makedirs(path)
            message = f"Directory '{path}' created successfully."
        else:
            message = f"Directory '{path}' already exists."

        with open(os.path.join(path,fn), 'w') as w_fn:
            json.dump(data, w_fn, ensure_ascii=False, indent=4)


import numpy as np
from glob import glob
import os
import json

def get_ordered_path(paths):
    sorted_path = []
    if 'crs' in paths[0] or 'abs' in paths[0]:
        checkpoint_numbers =  ['80','160','240','320','400','480','560','checkpoint']
    else:
        checkpoint_numbers = ['100','200','300','400','500','checkpoint']
    for num in checkpoint_numbers:
        for p in paths:
            if num in p:
                sorted_path.append(p)
                break
    return sorted_path

rlhf_model_names = ["LLama-2-7b-oasst-baseline_rl_abs_quality_rw_075_bs16_kl_002_clip_04_512_max_token_with_pad_eos_lr_141e5_logits",
                    # "LLama-2-7b-oasst-baseline_rl_bs16_kl_001_clip_04_512_max_token_with_pad_eos_lr_141e5",
                    "LLama-2-7b-oasst-baseline_rl_bs16_kl_002_clip_04_512_max_token_with_pad_eos_lr_141e5",
                    "LLama-2-7b-oasst-baseline_rl_f_crs_025_bs16_kl_002_clip_04_512_max_token_with_pad_eos_lr_141e5",
                    # "LLama-2-7b-oasst-baseline_rl_f_crs_0625_bs16_kl_002_clip_04_512_max_token_with_pad_eos_lr_141e5",
                    # "LLama-2-7b-oasst-basseline_sft"

                    ]
reward_key = 'abs_reward'#,'reward'
for n in rlhf_model_names:
    print(f'\n==== Stats for model at path {n}====')
    eval_path = os.path.join("output/rl",n,"rewarded_eval")
    all_jsons = glob(eval_path+'/final*.json')
    sorted_path = get_ordered_path(all_jsons) #all_jsons #
    print(f'sorted_path:{sorted_path}')
    for s in sorted_path:
        print(s.split('/')[-1])
        # print(f"**mean score for checkpoint {s.split('/')[-1].split('_')[-2]}**")
        with open(s,'r') as f:
            data = json.load(f)
            r_rw = []
            a_rw = []
            c025_rw = []
            c0625_rw = []
            onereward = []
            for d in data:
                r_rw.append(d["reward"])
                a_rw.append(d["abs_reward"])
                c025_rw.append(d["crs_025_reward"])
                c0625_rw.append(d["crs_0625_reward"])
                onereward.append(d[reward_key])
        print(onereward)
        # print(f'rank reward :{np.mean(r_rw)},abs reward:{np.mean(a_rw)}, crs 025:{np.mean(c025_rw)}, crs 0625:{np.mean(c0625_rw)}')
        print(f'mean reward score for reward type {reward_key} is {np.mean(onereward)} and std {np.std(onereward)}')
            




     
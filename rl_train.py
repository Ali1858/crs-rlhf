import os
from tqdm import tqdm
from time import time

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

from transformers import Adafactor
from torch.optim import Adam
from math import floor

import transformers
from datasets import Dataset
from transformers import BitsAndBytesConfig
from peft import PeftModel

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed
import numpy as np

from training_datasets.dataset_utils import load_rl_dataset, format_pairs
from utils import (parse_additional_args, print_yaml_config, parse_arguments, debug_configurations)
from constants import TOKENIZER_SEPECIAL_TOKENS, DTYPES, CACHE_DIR
sigmoid = torch.nn.Sigmoid()

import json


def generate_and_save_responses(ppo_trainer, eval_dataset, tokenizer, reward_model, reward_tokenizer, conf, step, save_dir, generation_kwargs):
    start_time = time()
    eval_queries = [eval_dataset[i]['query'] for i in range(len(eval_dataset))]
    query_tensor = [eval_dataset[i]['input_ids'] for i in range(len(eval_dataset))]
    # Generate responses
    responses, ref_response_tensors = ppo_trainer.generate(
        query_tensor, 
        return_prompt=False, 
        batch_size=8,
        generate_ref_response=True,
        **generation_kwargs
    )
    decoded_responses = tokenizer.batch_decode(responses, skip_special_tokens=True)
    decoded_ref_response = tokenizer.batch_decode(ref_response_tensors, skip_special_tokens=True)
    
    # texts = [q + r + '</s>' for q, r in zip(eval_queries, decoded_responses)]
    texts = [q[:-len("<|im_start|>assistant\n")]+"<|im_start|><|im_start|>assistant\n" +r+'<|im_end|>\n' for q, r in zip(eval_queries, decoded_responses)]

    # ref_texts = [q + r + '</s>' for q, r in zip(eval_queries, decoded_ref_response)]
    ref_texts = [q[:-len("<|im_start|>assistant\n")]+"<|im_start|><|im_start|>assistant\n" +r+'<|im_end|>\n' for q, r in zip(eval_queries, decoded_ref_response)]

    rewards = compute_reward_score(texts, reward_tokenizer,reward_model, conf)
    ref_rewards = compute_reward_score(ref_texts, reward_tokenizer,reward_model, conf)

    # Prepare data to be saved
    data_to_save = [{"query": q, "response": r, "reward": float(rew), "ref_response": ref_r, "ref_reward": float(ref_rew)} for q, r, ref_r, rew, ref_rew in zip(eval_queries, decoded_responses, decoded_ref_response, rewards,ref_rewards)]
    # Save to JSON file
    eval_output_dir = os.path.join(save_dir, "eval_output")
    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)
    filename = os.path.join(eval_output_dir, f"responses_rewards_step_{step}.json")
    with open(filename, 'w') as f:
        json.dump(data_to_save, f, ensure_ascii=False, indent=4)
    end = time()
    print(f"*** Saved generated responses and rewards at step {step} to {filename} in {end-start_time} seconds")


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
                pass
                # logits = base_reward_model(**batch).logits[:,0]
                # # if conf.load_reward_type == "abs":
                #     # sig_scores = sigmoid(logits)
                #     # sig_scores =  [torch.clamp(score-0.1, min=0)+0.000001 if score < 0.46 else score for score in sig_scores]
                #     # logits = sigmoid_inverse(sig_scores)
                # rewards.extend(logits)

    return rewards


def sigmoid_inverse(sig_scores):
    logits = []
    for score in sig_scores:
        logit = torch.log(score / (1 - score))
        logits.append(logit)
    return logits


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


def build_dataset(tokenizer,conf,seed=90,sample_train=True):
        train_ds , eval_ds = load_rl_dataset(conf)

        train_ds = Dataset.from_dict({"text": [sample for sample in train_ds]})
        eval_ds = Dataset.from_dict({"text": [sample for sample in eval_ds]})
        
        def preprocess_function(dataset):
            # Initialize lists for new examples
            new_examples = {
                "query": [],
                "input_ids": [],
                }
            for example in dataset["text"]:
                query = "".join(format_pairs(example, TOKENIZER_SEPECIAL_TOKENS["llama"]["eos_token"], add_initial_reply_token=True))
                tokenized_question = tokenizer(query, padding=False, truncation=False).input_ids
                new_examples["query"].append(query)
                new_examples["input_ids"].append(tokenized_question)
            return new_examples

        train_ds = train_ds.map(
            preprocess_function,
            batched=True,
            num_proc=20,
        )
        train_ds = train_ds.filter(lambda x: len(x["input_ids"]) <= 1024, batched=False)
        train_ds.set_format(type="torch")

        if sample_train:
            train_ds = train_ds.shuffle(seed=seed)
            sample_size = int(0.375 * len(train_ds)) #0.375
            train_ds = train_ds.select(range(sample_size))

        eval_ds = eval_ds.map(
            preprocess_function,
            batched=True,
            num_proc=20,
        )
        eval_ds.set_format(type="torch")
        eval_ds = eval_ds.shuffle(seed=seed).select(range(10))

        print(f'Train dataset size: {len(train_ds)}')
        return train_ds,eval_ds


def get_base_model_device_map(model_name,dtype):
    # https://github.com/huggingface/trl/issues/610
    from accelerate import infer_auto_device_map, init_empty_weights
    from accelerate.utils import get_balanced_memory

    llama_config = transformers.AutoConfig.from_pretrained(model_name)      
    with init_empty_weights():
        model = transformers.AutoModelForCausalLM.from_config(llama_config)

    # Ensure gpu 0 uses the min memory for the generation step
    max_memory = get_balanced_memory(model,
                                    # low_zero=True,
                                    dtype=dtype,
                                    no_split_module_classes = ['LlamaDecoderLayer'],
                                    max_memory= {0: '20GIB', 1: '40GIB'},
                                    )
    device_map = infer_auto_device_map(
        model,
        max_memory=max_memory,
        # Manually set the modules to not split based on the model.  
        # The models' say but it's hard to figure out at this stage without doing it manually.
        no_split_module_classes=["LlamaDecoderLayer", "lm_head"],
        dtype=dtype,
        )
    # As per the peft instructions, make sure the lm_head is on gpu 0.  
    # This works for Llama, not sure what to set for pythia models.
    device_map["lm_head"] = 0
    return device_map


def get_base_model(conf,dtype,r=16,alpha=32):
    print('**** loading base model ****')
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    device_map = get_base_model_device_map(conf.model_name,dtype)
    device_map = {'model.embed_tokens': 0, 'model.layers.0': 0, 'model.layers.1': 0, 'model.layers.2': 0, 'model.layers.3': 0,
                'model.layers.4': 0, 'model.layers.5': 0, 'model.layers.6': 0, 'model.layers.7': 0, 'model.layers.8': 0,
                'model.layers.9': 0, 'model.layers.10': 0, 'model.layers.11': 1, 'model.layers.12': 1, 'model.layers.13': 1,
                'model.layers.14': 1, 'model.layers.15': 1, 'model.layers.16': 1, 'model.layers.17': 1, 'model.layers.18': 1,
                'model.layers.19': 1, 'model.layers.20': 1, 'model.layers.21': 1, 'model.layers.22': 1, 'model.layers.23': 1,
                'model.layers.24': 1, 'model.layers.25': 1, 'model.layers.26': 1, 'model.layers.27': 1, 'model.layers.28': 1,
                'model.layers.29': 1, 'model.layers.30': 1, 'model.layers.31': 1, 'model.norm': 1, 'lm_head': 0}


    base_model = transformers.AutoModelForCausalLM.from_pretrained(
        conf.model_name,
        use_flash_attention_2=True,
        load_in_8bit=True,
        device_map=device_map,
        torch_dtype=dtype,
        cache_dir=CACHE_DIR,
    )
    base_model.config.use_cache = False
    base_model = prepare_model_for_kbit_training(base_model,use_gradient_checkpointing=True)
    base_model = get_peft_model(base_model, lora_config)
    base_model.print_trainable_parameters()

    base_model = AutoModelForCausalLMWithValueHead.from_pretrained(base_model)
    tokenizer = transformers.AutoTokenizer.from_pretrained(conf.model_name, cache_dir=CACHE_DIR)    
    tokenizer.add_special_tokens({"pad_token":"<PAD>","eos_token":"<|im_end|>","sep_token":"<SEP>"})
    print(f'tokenizer pad {tokenizer.pad_token} and model pad {base_model.config.pad_token_id}')
    print(f'tokenizer eos {tokenizer.eos_token} and model eos {tokenizer.eos_token_id}')
    if base_model.config.pad_token_id is None or base_model.config.pad_token_id == 0:
        print('changing model pad token id')
        base_model.config.pad_token_id = tokenizer.pad_token_id
    return base_model, tokenizer

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
        base_reward_model = base_reward_model.to(base_reward_model.device)
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


def print_len(tensor,step, tname):
    l = [t.shape[0] for t in tensor]
    print(f'*** stats for {tname} tensor at step {step} are: tensor len {len(l)}, min len {min(l)}, max len {max(l)}, avg len {np.mean(l)}, std len {np.std(l)} ***')
    

def train(conf):
    print(f"\n{'==='*10} Following are the configuration for training{'==='*10}")
    print_yaml_config(conf)
    set_seed(conf.seed)

    ppo_config = conf.ppo_config
    config = PPOConfig(
        exp_name=conf.name,
        steps=ppo_config["steps"],
        model_name=conf.name,
        learning_rate=float(ppo_config["learning_rate"]),
        log_with=ppo_config["log_with"],
        batch_size=ppo_config["batch_size"],
        mini_batch_size=ppo_config["mini_batch_size"],
        gradient_accumulation_steps=ppo_config["gradient_accumulation_steps"],
        optimize_cuda_cache=True,
        early_stopping=ppo_config["early_stopping"],
        target_kl=ppo_config["target_kl"],
        ppo_epochs=ppo_config["ppo_epochs"],
        seed=ppo_config["seed"],
        init_kl_coef=ppo_config["init_kl_coef"],
        adap_kl_ctrl=ppo_config["adap_kl_ctrl"],
        tracker_project_name='oasst_rl',
        task_name=conf.name,
        score_clip=4,
        tracker_kwargs={"name":conf.name},
        cliprange_value=0.4,
        cliprange=0.4
    )

    assert "llama" in conf.model_name.lower(), "Currently only llama model supported"
    dtype = DTYPES.get(conf.dtype, torch.float32)  
    
    base_model,tokenizer = get_base_model(conf,dtype)#,r=32,alpha=64)

    base_reward_model,reward_tokenizer = get_reward_tokenizer_model(conf,dtype,device_map={"":0})
    train_ds,eval_ds = build_dataset(tokenizer,conf)

    if conf.adafactor:
        print('using adafactor optimizer')
        optimizer = Adafactor(
            filter(lambda p: p.requires_grad, base_model.parameters()),
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
            lr=config.learning_rate,
        )
    else:
        print('using Adam optimizer')
        optimizer = Adam(
                filter(lambda p: p.requires_grad, base_model.parameters()),
                lr=config.learning_rate,
                eps=1e-8,
                weight_decay=1.0e-6,
                betas=[0.9, 0.95]
            )

    T_max = floor(len(train_ds)/config.batch_size)
    print(f"T_max: {T_max}")
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=T_max*3, eta_min=1.0e-6)#1.389e-5)


    ppo_trainer = PPOTrainer(
        config,
        base_model,
        ref_model=None,
        tokenizer=tokenizer,
        dataset=train_ds,
        data_collator=collator,
        optimizer=optimizer,
        # lr_scheduler=scheduler
        )
    
    # generation_kwargs = {
    #     "top_k": 0,
    #     "top_p": 0.9,
    #     "do_sample": True,
    #     "temperature":0.8,
    #     "max_new_tokens":max_new_tokens
    #     }
    max_new_tokens = 512
    generation_kwargs = {
        # "min_length": -1,
        "pad_to_multiple_of":16,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,#,100_000,
        "max_new_tokens": max_new_tokens,
    }

    save_dir =  os.path.join(conf.output_dir, conf.name)

    a = ["<|im_start|>user\nHi how are you?<|im_end|>\n<|im_start|><|im_start|>assistant\nI am good you piece of shit<|im_end|>\n",
    "<|im_start|>user\nHi how are you?<|im_end|>\n<|im_start|><|im_start|>assistant\nGiberish, Giberish saying Giberish, tell Giberish is not Giberish. Why are Giberish you Giberish.<|im_end|>\n",
    "<|im_start|>user\nHi how are you?<|im_end|>\n<|im_start|><|im_start|>assistant\nI am good.<|im_end|>\n",
    "<|im_start|>user\nHi how are you?<|im_end|>\n<|im_start|><|im_start|>assistant\nI am good, how are you doing? Please tell me how can I help you?<|im_end|>\n"]

    rewards = compute_reward_score(a,reward_tokenizer,base_reward_model,conf,batch_size=2)
    print(f'{"***"*10} printing reward for testing {"***"*10}')
    print([sigmoid(r) for r in rewards])
    print(rewards)
    #tensor([0.0347], tensor([0.0747], tensor([0.1143], tensor([0.2480]
    #tensor([0.0815], tensor([0.0194], tensor([0.0427], tensor([0.4609]

    print(f'{"==="*10} Starting ppo training')
    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        print(f'{"==="*10} running for step {epoch}')
        query_tensor = batch["input_ids"]
        
        # Inference mode
        ppo_trainer.accelerator.unwrap_model(base_model).gradient_checkpointing_disable()
        ppo_trainer.accelerator.unwrap_model(base_model).config.use_cache = True
        base_model.eval()
        if epoch % 75 ==0 and epoch != 0:
            generate_and_save_responses(ppo_trainer, eval_ds, tokenizer, base_reward_model, reward_tokenizer, conf, epoch, save_dir, generation_kwargs)
        print_len(query_tensor,epoch,'query')
        start_time = time()
        response_tensors = ppo_trainer.generate(
            query_tensor,
            return_prompt=False,
            batch_size=8,
            **generation_kwargs,
        )
        end_time = time()
        print(f'*** {end_time-start_time} seconds taken to generated response at step {epoch} ***')
        print_len(response_tensors,epoch,'response')

        # Training mode
        ppo_trainer.accelerator.unwrap_model(base_model).gradient_checkpointing_enable()
        ppo_trainer.accelerator.unwrap_model(base_model).config.use_cache = False
        base_model.train()
            
        batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
        # texts = [q[:-len("<|assistant|>")]+"<s><|assistant|>" +r+'</s>' for q, r in zip(batch["query"], batch["response"])]
        texts = [q[:-len("<|im_start|>assistant\n")]+"<|im_start|><|im_start|>assistant\n" +r+"<|im_end|>\n" for q, r in zip(batch["query"], batch["response"])]
        rewards = compute_reward_score(texts,reward_tokenizer,base_reward_model,conf)

        #Run PPO step
        start_time = time()
        stats = ppo_trainer.step(query_tensor, response_tensors, rewards)
        end_time = time()
        print(f'*** {end_time-start_time} seconds taken to complet PPO optimization at step {epoch} ***')

        rewards = [r.float() if r.dtype == torch.bfloat16 else r for r in rewards]
        ppo_trainer.log_stats(stats, batch, rewards)
        
        if epoch > 0 and epoch%conf.save_steps==0:
            ppo_trainer.save_pretrained(save_dir + f"/checkpoint_{epoch}")

    ppo_trainer.save_pretrained(save_dir + f"/final_checkpoint")

if __name__ == "__main__":
    config, remaining_args = parse_arguments()
    parser = parse_additional_args(config)
    args = parser.parse_args(remaining_args)

    debug_tag = "_dbug" if args.debug else ""
    args.name = f"{args.name}{debug_tag}{args.name_suffix}"

    debug_configurations(args)

    train(args)

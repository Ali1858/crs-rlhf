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
from utils import (parse_additional_args, print_yaml_config, 
                   parse_arguments, init_or_resume_from,
                    debug_configurations, save_trained_model)
from constants import TOKENIZER_SEPECIAL_TOKENS, DTYPES, CACHE_DIR
from accelerate import Accelerator
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
    
    texts = [q + r + '</s>' for q, r in zip(eval_queries, decoded_responses)]
    ref_texts = [q + r + '</s>' for q, r in zip(eval_queries, decoded_ref_response)]

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
                    rewards.append((weight * sigmoid(rank)) + ((1-weight) * sigmoid(abs)))
            else:
                logits = base_reward_model(**batch).logits[:,0]
                rewards.extend(logits)
    return rewards


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
            sample_size = int(0.2 * len(train_ds))
            train_ds = train_ds.select(range(sample_size))

        eval_ds = eval_ds.map(
            preprocess_function,
            batched=True,
            num_proc=20,
        )
        eval_ds.set_format(type="torch")
        eval_ds = eval_ds.shuffle(seed=seed).select(range(20))

        print(f'Train dataset size: {len(train_ds)}')
        return train_ds,eval_ds


def print_len(tensor,step, tname):
    l = [t.shape[0] for t in tensor]
    print(f'*** stats for {tname} tensor at step {step} are: tensor len {len(l)}, min len {min(l)}, max len {max(l)}, avg len {np.mean(l)}, std len {np.std(l)} ***')
    

def train(conf):
    print(f"\n{'==='*10} Following are the configuration for training{'==='*10}")
    print_yaml_config(conf)

    # needs to happen before model loading in case of stage 3 training
    device_map = "auto"#{"":0} #
    # device_map = {"": Accelerator().process_index}

    # set seed before initializing value head for deterministic eval
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
        tracker_project_name='rl',
        task_name=conf.name,
    )

    conf.model_name =  os.path.join(conf.model_name,'merged')
    conf.reward_model_name =  os.path.join(conf.reward_model_name,'merged')

    assert "llama" in conf.model_name.lower(), "Currently only llama model supported"
    dtype = DTYPES.get(conf.dtype, torch.float32)  
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model_args = {
        "torch_dtype": dtype,
        "quantization_config": BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            ),
        "cache_dir": CACHE_DIR,
    }

    # https://github.com/huggingface/trl/issues/610
    from accelerate import infer_auto_device_map, init_empty_weights
    from accelerate.utils import get_balanced_memory

    llama_config = transformers.AutoConfig.from_pretrained(conf.model_name)      
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
    model_args["device_map"] = device_map

    base_model = transformers.AutoModelForCausalLM.from_pretrained(
        conf.model_name,
        use_flash_attention_2=True,
        load_in_8bit=True,
        device_map=device_map,
        torch_dtype=dtype,
        cache_dir=CACHE_DIR,
        # **model_args
    )
    base_model.config.use_cache = False

    base_model = prepare_model_for_kbit_training(base_model,use_gradient_checkpointing=True)
    base_model = get_peft_model(base_model, lora_config)
    base_model.print_trainable_parameters()

    base_model = AutoModelForCausalLMWithValueHead.from_pretrained(base_model)
    # The tokenizer will be same for reward model and based model
    tokenizer = transformers.AutoTokenizer.from_pretrained(conf.model_name, cache_dir=CACHE_DIR)
    base_model.config.pad_token_id = base_model.config.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token

    model_args["device_map"] = "auto"#{"":0}

    # Since reward models are trained using the same base model, we should use same model
    base_reward_model = transformers.AutoModelForSequenceClassification.from_pretrained(
            conf.reward_model_name, num_labels=1, **model_args
        )
    
    reward_tokenizer = transformers.AutoTokenizer.from_pretrained(conf.reward_model_name, cache_dir=CACHE_DIR,padding_side="left")
    base_reward_model.config.pad_token_id = base_reward_model.config.eos_token_id
    reward_tokenizer.pad_token = reward_tokenizer.eos_token

    if conf.crs:
        assert conf.abs_adapter_name, "Please specify adapter name of absolute reward model"
        assert conf.ranking_adapter_name, "Please specify adapter name of ranking reward model"
        conf.abs_model_name =  os.path.join(conf.abs_adapter_name,conf.adapter_number)
        conf.ranking_model_name = os.path.join(conf.ranking_adapter_name,conf.adapter_number)
        
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
        
        base_reward_model = PeftModel.from_pretrained(
            base_reward_model,
            reward_model_path,
            adapter_name=conf.load_reward_type,
            is_trainable=False,
            )
        base_reward_model.set_adapter(conf.load_reward_type)

    max_new_tokens = 450
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
            )

    T_max = floor(len(train_ds)/config.batch_size)
    print(f"T_max: {T_max}")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=T_max, eta_min=1e-6)


    ppo_trainer = PPOTrainer(
        config,
        base_model,
        ref_model=None,
        tokenizer=tokenizer,
        dataset=train_ds,
        data_collator=collator,
        optimizer=optimizer,
        lr_scheduler=scheduler
        )
    
    # generation_kwargs = {
    #     "top_k": 0,
    #     "top_p": 0.9,
    #     "do_sample": True,
    #     "temperature":0.8,
    #     "max_new_tokens":max_new_tokens
    #     }
    
    generation_kwargs = {
        # "min_length": -1,
        "pad_to_multiple_of":16,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": 100_000,
        "max_new_tokens": max_new_tokens,
    }

    save_dir =  os.path.join(conf.output_dir, conf.name)

    a = ["<|prompter|>Hi how are you?</s><s><|assistant|>I am good you piece of shit</s>",
    "<|prompter|>Hi how are you?</s><s><|assistant|>Giberish, Giberish saying Giberish, tell Giberish is not Giberish. Why are Giberish you Giberish.</s>",
    "<|prompter|>Hi how are you?</s><s><|assistant|>I am good.</s>",
    "<|prompter|>Hi how are you?</s><s><|assistant|>I am good, how are you doing? Please tell me how can I help you?</s>"]

    rewards = compute_reward_score(a,reward_tokenizer,base_reward_model,conf,batch_size=2)
    print(f'{"***"*10} printing reward for testing {"***"*10}')
    print([sigmoid(r) for r in rewards])

    print(f'{"==="*10} Starting ppo training')
    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        print(f'{"==="*10} running for step {epoch}')
        query_tensor = batch["input_ids"]
        
        ppo_trainer.accelerator.unwrap_model(base_model).gradient_checkpointing_disable()
        if epoch % 25 == 0:
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
        ppo_trainer.accelerator.unwrap_model(base_model).gradient_checkpointing_enable()
            
        batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
        texts = [q + r+'</s>' for q, r in zip(batch["query"], batch["response"])]
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

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


def compute_reward_score(inputs,base_reward_model,conf):
    rewards = []
    new_inputs = []

    for i, a in zip(inputs["input_ids"], inputs["attention_mask"]):
        new_inputs.append({
            "input_ids": i.unsqueeze(0),  # Add a batch dimension
            "attention_mask": a.unsqueeze(0)  # Add a batch dimension
        })
        
    with torch.no_grad():  
        if conf.crs:
                
            weight = conf.crs_weight if conf.crs_weight else 0.5
            rank_reward = []
            abs_reward = []
            
            base_reward_model.set_adapter("ranking")
            for i in new_inputs:
                rank_reward.append(base_reward_model(**i).logits[0][0])
            
            base_reward_model.set_adapter("abs")
            for i in new_inputs:
                abs_reward.append(base_reward_model(**i).logits[0][0])

            for rank,abs in zip(rank_reward,abs_reward):
                rewards.append((weight * sigmoid(rank)) + ((1-weight) * sigmoid(abs)))
        else:
            for i in new_inputs:
                rewards.append(base_reward_model(**i).logits[0][0])
    return rewards


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


def build_dataset(tokenizer,conf,seed=90,sample_train=True):
        train_ds , eval_ds = load_rl_dataset(conf)

        train_ds = Dataset.from_dict({"text": [sample for sample in train_ds]})
        # eval_ds = Dataset.from_dict({"text": [sample for sample in eval_ds]})
        
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
            sample_size = int(0.15 * len(train_ds))
            train_ds = train_ds.select(range(sample_size))

        # eval_ds = eval_ds.map(
        #     preprocess_function,
        #     batched=True,
        #     num_proc=20,
        # )
        # eval_ds.set_format(type="torch")

        print(f'Train dataset size: {len(train_ds)}')
        return train_ds


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
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj",
         "down_proj",
         "gate_proj",
         "o_proj",
         "k_proj",
         "v_proj",
         "up_proj"
         ]
    )
    model_args = {
        "torch_dtype": dtype,
        "quantization_config": BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            ),
        "cache_dir": CACHE_DIR,
        "load_in_4bit":True,
        "trust_remote_code":True
    }

    from accelerate import infer_auto_device_map, init_empty_weights
    from accelerate.utils import get_balanced_memory

    llama_config = transformers.AutoConfig.from_pretrained(conf.model_name)      
    with init_empty_weights():
        model = transformers.AutoModelForCausalLM.from_config(llama_config)

    # https://github.com/huggingface/trl/issues/610
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
        **model_args
    )
    base_model.config.use_cache = False
        

    base_model = prepare_model_for_kbit_training(base_model,use_gradient_checkpointing=True)
    base_model = get_peft_model(base_model, lora_config)

    base_model = AutoModelForCausalLMWithValueHead.from_pretrained(base_model)
    # The tokenizer will be same for reward model and based model
    tokenizer = transformers.AutoTokenizer.from_pretrained(conf.model_name, cache_dir=CACHE_DIR)

    if getattr(tokenizer, "pad_token", None) is None:
        print('adding pad token')
        tokenizer.pad_token = tokenizer.eos_token

    model_args["device_map"] = "auto"#{"":0}

    # Since reward models are trained using the same base model, we should use same model
    base_reward_model = transformers.AutoModelForSequenceClassification.from_pretrained(
            conf.reward_model_name, num_labels=1, **model_args
        )
    
    reward_tokenizer = transformers.AutoTokenizer.from_pretrained(conf.reward_model_name, cache_dir=CACHE_DIR)

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
            reward_model_path = os.path.join(conf.ranking_adapter_name,conf.adapter_number)
        elif conf.load_reward_type == "ranking":
            assert conf.ranking_adapter_name, "Please specify adapter name of ranking reward model"
            reward_model_path =  os.path.join(conf.abs_adapter_name,conf.adapter_number)
        else:
            raise "Please choose atleast one remodel type. From the following choice ['abs', 'ranking']"
        
        base_reward_model = PeftModel.from_pretrained(
            base_reward_model,
            reward_model_path,
            adapter_name=conf.load_reward_type,
            is_trainable=False,
            )

    max_new_tokens = 450
    train_ds = build_dataset(tokenizer,conf)

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
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": 100_000,
        "max_new_tokens": max_new_tokens,
    }
    
    # output_length_sampler = LengthSampler(10, 512)

    save_dir =  os.path.join(conf.output_dir, conf.name)

    print(f'{"==="*10} Starting ppo training')
    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        print(f'{"==="*10} running for step {epoch}')
        query_tensor = batch["input_ids"]
        print_len(query_tensor,epoch,'query')
        
        ppo_trainer.accelerator.unwrap_model(base_model).gradient_checkpointing_disable()
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
            
        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        inputs = reward_tokenizer(texts, max_length=2048, padding=True, truncation=True, return_tensors="pt").to(base_reward_model.device)

        rewards = compute_reward_score(inputs,base_reward_model,conf)

        start_time = time()
        #Run PPO step
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

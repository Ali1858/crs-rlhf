import os
import pandas as pd
from tqdm import tqdm

from peft import LoraConfig
import torch
import wandb
import transformers
from transformers.training_args import OptimizerNames
from datasets import Dataset
from transformers import BitsAndBytesConfig
from peft import PeftModel
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler

from training_datasets.dataset_utils import load_rl_dataset, format_pairs
from utils import (parse_additional_args, print_yaml_config, 
                   parse_arguments, init_or_resume_from,
                    debug_configurations, save_trained_model)
from constants import TOKENIZER_SEPECIAL_TOKENS, DTYPES, CACHE_DIR

sigmoid = torch.nn.Sigmoid()


def compute_reward_score(inputs,base_reward_model,conf):
    if conf.crs:
        base_reward_model.set_adapter("ranking")
        rank_reward = base_reward_model(**inputs).logits
        base_reward_model.set_adapter("abs")
        abs_reward = base_reward_model(**inputs).logits

        weight = conf.crs_weight if conf.crs_weight else 0.5
        combined_reward = (weight * sigmoid(rank_reward) + (1-weight) * sigmoid(abs_reward))/2
        return combined_reward
    else:
        return base_reward_model(**inputs)[0]


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])



def build_dataset(tokenizer,conf):
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
                tokenized_question = tokenizer(query, max_length=2048, truncation=True)
                new_examples["query"].append(query)
                new_examples["input_ids"].append(tokenized_question["input_ids"])
            return new_examples

        train_ds = train_ds.map(
            preprocess_function,
            batched=True,
            num_proc=20,
        )
        train_ds.set_format(type="torch")
        eval_ds = eval_ds.map(
            preprocess_function,
            batched=True,
            num_proc=20,
        )
        eval_ds.set_format(type="torch")
        return train_ds, eval_ds


def create_trainer(conf):
    print(f"\n{'==='*10} Following are the configuration for training{'==='*10}")
    print_yaml_config(conf)

    # needs to happen before model loading in case of stage 3 training
    device_map = {"":0} #"auto" #
    
    # set seed before initializing value head for deterministic eval
    set_seed(conf.seed)

    ppo_config = conf.ppo_config
    config = PPOConfig(
        steps=ppo_config["steps"],
        model_name=ppo_config["model_name"],
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
    )

    conf.model_name =  os.path.join(args.output_dir,args.model_name,'merged')
    assert "llama" in conf.model_name.lower(), "Currently only llama model supported"
    dtype = DTYPES.get(conf.dtype, torch.float32)  
    lora_config = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
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
        "device_map": device_map,
    }

    base_model =  transformers.AutoModelForCausalLM.from_pretrained(
        conf.model_name, trust_remote_code=True, **model_args
    )
    base_model.gradient_checkpointing_enable()

    base_model = AutoModelForCausalLMWithValueHead(base_model, trust_remote_code=True,peft_config= lora_config)
    # The tokenizer will be same for reward model and based model
    tokenizer = transformers.AutoTokenizer.from_pretrained(conf.model_name, cache_dir=CACHE_DIR)

    model_args["device_map"] = {"":1}

    # Since reward models are trained using the same base model, we should use same model
    base_reward_model = transformers.AutoModelForSequenceClassification.from_pretrained(
            conf.model_name, num_labels=1, **model_args
        )

    train_ds, eval_ds = build_dataset(tokenizer,conf)

    if conf.crs:
        assert conf.abs_adapter_name, "Please specify adapter name of absolute reward model"
        assert conf.ranking_adapter_name, "Please specify adapter name of ranking reward model"
        conf.abs_model_name =  os.path.join(conf.output_dir,conf.abs_adapter_name,conf.adapter_number)
        conf.ranking_model_name = os.path.join(conf.output_dir,conf.ranking_adapter_name,conf.adapter_number)
        
        base_reward_model = PeftModel.from_pretrained(
            base_reward_model,
            conf.ranking_model_name,
            adapter_name="ranking",
            is_trainable=False
            )
        base_reward_model.load_adapter(conf.abs_model_name,adapter_name="abs",is_trainable=False)
    else:
        if conf.load_reward_type == "abs":
            assert conf.abs_adapter_name, "Please specify adapter name of absolute reward model"
            reward_model_path = os.path.join(args.output_dir,args.ranking_adapter_name,conf.adapter_number)
        elif conf.load_reward_type == "ranking":
            assert conf.ranking_adapter_name, "Please specify adapter name of ranking reward model"
            reward_model_path =  os.path.join(args.output_dir,args.abs_adapter_name,conf.adapter_number)
        else:
            raise "Please choose atleast one remodel type. From the following choice ['abs', 'ranking']"
        
        base_reward_model = PeftModel.from_pretrained(
            base_reward_model,
            reward_model_path,
            is_trainable=False
            )
    
    ppo_trainer = PPOTrainer(
        config,
        base_model,
        ref_model=None,
        tokenizer=tokenizer,
        dataset=train_ds,
        data_collator=collator,
        )
    
    generation_kwargs = {
        "top_k": 40,
        "top_p": 0.9,
        "do_sample": True,
        "temperature":0.8,
        "max_new_tokens":512
        }
    
    output_length_sampler = LengthSampler(10, 512)


    wandb_suffix = ""
    if conf.debug:
        wandb_suffix = "_debug"
    
    os.environ["WANDB_WATCH"] = "all"
    wandb_project_name = f"rl{wandb_suffix}"
    wandb.init(
        project=wandb_project_name,
        entity=None,
        name=conf.name,
        config=conf,
        save_code=True,
    )

    save_dir =  os.path.join(args.output_dir, args.name)
    print(f'{"==="*10} Starting ppo training')
    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        query_tensor = batch["input_ids"]
        
        # Initialize empty lists to store responses and rewards
        response_tensors = []
        responses = []
        rewards = []

        # Iterate through each query in the query_tensor
        for idx in range(len(query_tensor)):

            single_response_tensor = ppo_trainer.generate(
                [query_tensor[idx]],
                return_prompt=False,
                # length_sampler=512
                **generation_kwargs,
            )
            
            response = tokenizer.decode(single_response_tensor[0], skip_special_tokens=True)
            responses.append(response)

            # Prepare input for reward model
            text = batch["query"][idx] + response
            inputs = tokenizer(text, max_length=2048, padding=True, truncation=True, return_tensors="pt").to(base_reward_model.device)
            
            # texts = [q + r for q, r in zip(batch["query"], batch["response"])]
            # inputs = tokenizer(texts, max_length=2048,padding=True, truncation=True, return_tensors="pt").to(base_reward_model.device)

            reward = compute_reward_score(inputs,base_reward_model,conf)
            # rewards = [r[0] for r in rewards]
            reward = reward[0][0]

            # Append the generated response tensor and computed reward
            response_tensors.append(single_response_tensor[0])
            rewards.append(reward)
        
        # Convert lists to tensors
        # response_tensors = torch.cat(response_tensors, dim=0) 
        batch["response"] = responses
                
        #Run PPO step
        stats = ppo_trainer.step(query_tensor, response_tensors, rewards)

        rewards = [r.float() if r.dtype == torch.bfloat16 else r for r in rewards]
        if epoch%conf.log_step==0:
            ppo_trainer.log_stats(stats, batch, rewards)
        
        if epoch > 0 and epoch%conf.save_steps==0:
            ppo_trainer.save_pretrained(save_dir + f"/checkpoint_{epoch}")
    

if __name__ == "__main__":
    config, remaining_args = parse_arguments()
    parser = parse_additional_args(config)
    args = parser.parse_args(remaining_args)

    debug_tag = "_dbug" if args.debug else ""
    args.name = f"{args.name}{debug_tag}{args.name_suffix}"

    debug_configurations(args)

    trainer = create_trainer(args)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint is not None)

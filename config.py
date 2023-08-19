from functools import partial


CACHE_DIR="./cache"
SFT_DATASET_CONFIG = {"vicuna":
                    {
                     "val_split":0.2,
                    },
                "dolly":
                    {
                     "val_split":0.2,
                    },
                "alpaca":
                    {
                     "val_split":0.2,
                    },
                "math_instruction":
                    {
                     "val_split":0.2,
                    },

}

SFT_TRAINING_CONFIG = {
                "cache_dir":CACHE_DIR,
                "model_name":"openlm-research/open_llama_7b",
                "train_batch":1,
                "eval_batch":1,
                "lr":1e-5,
                "num_train_epochs":3,
                "gradient_accumulation_steps":1,
                "eval_accumulation_steps":1,
                "log_steps":500,
                "eval_steps":1000,
                "save_steps":5000,
                "warmup_steps":20,
                "weight_decay":0.0,
                "dtype":"fp16",
                "gradient_checkpointing":True,
                "adam_beta1":"",
                "adam_beta2":"",
                "adam_epsilon":"",
                "resume_from_checkpoint":None,
                "metrics":["accuracy"],
                "peft_config": {
                    "r": 16,
                    "lora_alpha": 32,
                    "target_modules": "all",
                    "lora_dropout": 0.05,
                    "bias": "none",
                    "task_type": "CAUSAL_LM",
                    "modules_to_save": ["wte", "lm_head"],
                    }
}

DIALOGUE_COLLATOR_CONFIG = {
    "max_length":1024,
    "random_offset_probability":None,
    "label_masking":True,
    "samples_mixing":True,
    "use_system_prefix":False,
    "system_prefix":None
}

TOKENIZER_CONFIG = {"llama":
                        {
                            "pad_token": "</s>",
                            "eos_token": "</s>",
                            "sep_token": "<s>",
                        }
                    }

import os
import math

import torch
import transformers
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft import PeftConfig, PeftModel
from peft.tuners.lora import LoraLayer

from constants import QA_SPECIAL_TOKENS,CACHE_DIR,DTYPES


def get_tokenizer(tokenizer_name,special_tokens,add_additional_special_tokens=True):
    """Rewritten from:
    https://github.com/LAION-AI/Open-Assistant/blob/main/model/model_training/utils/utils.py#L208"""

    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=CACHE_DIR)

    if add_additional_special_tokens:
        tokenizer.add_special_tokens(special_tokens)
        additional_special_tokens = tokenizer.special_tokens_map.get("additional_special_tokens", [])
        additional_special_tokens += list(QA_SPECIAL_TOKENS.values())
        tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens})
    return tokenizer


def get_all_linear_layers(model):
    """Taken from:
    https://github.com/LAION-AI/Open-Assistant/blob/main/model/model_training/utils/utils.py#L451"""
    cls = torch.nn.Linear

    modules = {name.split(".")[-1] for name, module in model.named_modules() if isinstance(module, cls)}
    if "lm_head" in modules:
        modules.remove("lm_head")
    if "score" in modules:
        modules.remove("score")

    return list(modules)


def resize_embeddings(model,tokenizer,pad_vocab_size_to_multiple_of):
    n_embs = model.get_input_embeddings().num_embeddings
    
    if len(tokenizer) != n_embs or pad_vocab_size_to_multiple_of:
        print("tokenizer size",len(tokenizer))
        p = pad_vocab_size_to_multiple_of
        target_size = len(tokenizer) if not p else math.ceil(len(tokenizer) / p) * p
        print("Resizing embeddings to", target_size)
        model.resize_token_embeddings(target_size)

def get_peft_config(reward_model,r=64,alpha=16):
        base_config = {
            'r': r,
            'lora_alpha': alpha,
            'lora_dropout': 0.05,
            'bias': 'none',
            'inference_mode': False,
            'target_modules': 'all'
        }
        
        if not reward_model:
            base_config.update({
                'task_type': 'CAUSAL_LM',
                'modules_to_save': ['embed_tokens', 'lm_head']
            })
        else:
            base_config.update({
                "task_type":"SEQ_CLS",
                "modules_to_save":["score"]
            })
        return base_config


def initialize_bnb_config(conf):
    dtype = DTYPES.get(conf.dtype, torch.float32)
    return BitsAndBytesConfig(
        load_in_4bit=conf.int4_training,
        load_in_8bit=conf.int8_training,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
    )


def load_base_model(device, config, reward_model):
    bnb_config = initialize_bnb_config(config)

    model_args = {
        "torch_dtype": bnb_config.bnb_4bit_compute_dtype,
        "quantization_config": bnb_config,
        "cache_dir": CACHE_DIR,
        "device_map": device,
        "use_flash_attention_2": True
    }

    if reward_model:
        return transformers.AutoModelForSequenceClassification.from_pretrained(
            config.model_name, num_labels=1, **model_args
        )

    return transformers.AutoModelForCausalLM.from_pretrained(
        config.model_name, **model_args
    )


def get_model_and_tokenizer(device,config,special_tokens,pad_vocab_size_to_multiple_of=16,need_embedding_resize=True,reward_model=False,add_tokens = True,only_tokenizer=False):
    """Rewritten from:
    https://github.com/LAION-AI/Open-Assistant/blob/main/model/model_training/utils/utils.py#L282
    https://github.com/LAION-AI/Open-Assistant/blob/main/model/model_training/models/peft_modeling.py#L49
    """

    print(f'loading {config.model_name} model')
    tokenizer_name = config.init_from_adapter or config.resume_from_checkpoint or config.model_name

    if config.init_from_adapter or config.resume_from_checkpoint:
        add_tokens = False

    tokenizer = get_tokenizer(tokenizer_name, special_tokens, add_additional_special_tokens=add_tokens and not reward_model)

    if only_tokenizer:
        return tokenizer

    model = load_base_model(device, config, reward_model)
    if config.int4_training:
        peft_config = get_peft_config(reward_model)
        model = prepare_model_for_kbit_training(model,use_gradient_checkpointing=config.gradient_checkpointing)
    elif config.int8_training:
        peft_config = get_peft_config(reward_model,r=16,alpha=32)
        model = prepare_model_for_kbit_training(model,use_gradient_checkpointing=config.gradient_checkpointing)
    else:
        peft_config = get_peft_config(reward_model,r=16,alpha=32)

    if peft_config.get("target_modules") == "all":
        peft_config.update({"target_modules": get_all_linear_layers(model)})
        print(f'===target module for peft {peft_config["target_modules"]}===')
    elif peft_config.get("target_modules") is None:
        peft_config.pop("target_modules")
        print(f'=== No  target module. Lora will use default setting. ===')

    if config.debug:
        print(f'=== model:/n{model}/n===')

    lora_config = LoraConfig(**peft_config)
    
    if need_embedding_resize:
        resize_embeddings(model, tokenizer, pad_vocab_size_to_multiple_of)

    if config.init_from_adapter:
        model = PeftModel.from_pretrained(model, config.init_from_adapter, is_trainable=True)
    else:
        model = get_peft_model(model, lora_config)

    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if config.dtype in ["bf16","bfloat16"]:
                module = module.to(torch.bfloat16)
        if 'norm' in name:
            module = module.to(torch.float32)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if config.dtype in ["bf16","bfloat16"] and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)

    if config.debug:
        print(f'=== model:/n{model}/n===')

    model.print_trainable_parameters()
    return model, tokenizer


def merge_and_save_peft_model(conf):
    if conf.merged_adapter_path and not os.path.exists(conf.merged_adapter_path):
        print(f"{'==='*10} Initiating the model merging process")
        peft_config = PeftConfig.from_pretrained(conf.adpater_path)

        dtype = DTYPES.get(conf.dtype, torch.float32)

        if peft_config.task_type == "SEQ_CLS":
            base_model = transformers.AutoModelForSequenceClassification.from_pretrained(
                conf.base_model_name, num_labels=1, torch_dtype=dtype)
        else:
            base_model = transformers.AutoModelForCausalLM.from_pretrained(
                conf.base_model_name, return_dict=True, torch_dtype=dtype
                )

        tokenizer = transformers.AutoTokenizer.from_pretrained(conf.adpater_path)
        resize_embeddings(base_model,tokenizer,16)

        if conf.debug:
            print(base_model)

        # Load the Lora model
        model = PeftModel.from_pretrained(base_model, conf.adpater_path)

        if conf.debug:
            print(model)

        model.eval()
        model = model.merge_and_unload()
        model.save_pretrained(conf.merged_adapter_path)
        tokenizer.save_pretrained(conf.merged_adapter_path)
        print(f"{'==='*10} The peft model and tokenizer are saved at location {conf.merged_adapter_path}")
    elif conf.merged_adapter_path:
        print(f"{'==='*10} The peft model is already saved at location {conf.merged_adapter_path}")


def load_for_inference(device,conf,special_tokens,base_model="meta-llama/Llama-2-7b-hf",reward_model=False):
    print(f'{"==="*10}loading {conf.model_name} model')
    peft_model_id = conf.init_from_adapter

    base_model = load_base_model(device, conf, reward_model)
    tokenizer = transformers.AutoTokenizer.from_pretrained(peft_model_id if peft_model_id else conf.model_name, cache_dir=CACHE_DIR)
    tokenizer.add_special_tokens(special_tokens)
    
    if peft_model_id is None:
        print(f"{'==='*10}Loading only base model {conf.model_name}. No lora adapter found")
        return base_model,tokenizer
    print(f"{'==='*10}Now Loading Peft model {conf.init_from_adapter}.")


    resize_embeddings(base_model,tokenizer,16)
    model = PeftModel.from_pretrained(base_model, peft_model_id, is_trainable=False)
    return model, tokenizer

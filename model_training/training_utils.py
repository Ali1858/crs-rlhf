import os
import math

import torch
import transformers
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft import PeftConfig, PeftModel

from constants import QA_SPECIAL_TOKENS,CACHE_DIR


def get_all_linear_layers(model):
    """Taken from:
    https://github.com/LAION-AI/Open-Assistant/blob/main/model/model_training/utils/utils.py#L451"""
    cls = torch.nn.Linear

    modules = {name.split(".")[-1] for name, module in model.named_modules() if isinstance(module, cls)}
    if "lm_head" in modules:
        modules.remove("lm_head")

    return list(modules)


def emedding_resize(model,tokenizer,pad_vocab_size_to_multiple_of,config):
    n_embs = model.get_input_embeddings().num_embeddings
    
    if len(tokenizer) != n_embs or pad_vocab_size_to_multiple_of:
        print("tokenizer size",len(tokenizer))
        p = pad_vocab_size_to_multiple_of
        target_size = len(tokenizer) if not p else math.ceil(len(tokenizer) / p) * p
        print("Resizing embeddings to", target_size)
        model.resize_token_embeddings(target_size)


def get_model(tokenizer,device,config,pad_vocab_size_to_multiple_of=16,need_embedding_resize=True,reward_model=False):
    """Rewritten from:
    https://github.com/LAION-AI/Open-Assistant/blob/main/model/model_training/utils/utils.py#L282
    https://github.com/LAION-AI/Open-Assistant/blob/main/model/model_training/models/peft_modeling.py#L49
    """
    dtype = torch.float32
    if config.dtype in ["fp16", "float16"]:
        dtype = torch.float16
    elif config.dtype in ["bf16", "bfloat16"]:
        dtype = torch.bfloat16

    peft_config = config.peft_config

    print('load {config.model_name} model')

    if not reward_model:
        model = transformers.AutoModelForCausalLM.from_pretrained(config.model_name,
                                                                  torch_dtype=dtype,
                                                                  load_in_8bit=config.int8_training,
                                                                  cache_dir=CACHE_DIR,
                                                                  device_map=device)
    else:
        model = transformers.AutoModelForSequenceClassification.from_pretrained(config.model_name,
                                                                            torch_dtype=dtype,
                                                                            num_labels=1,
                                                                            load_in_8bit=config.int8_training,
                                                                            cache_dir=CACHE_DIR,
                                                                            device_map=device)        

    if need_embedding_resize:
        emedding_resize(model,tokenizer,pad_vocab_size_to_multiple_of,config)

    if config.debug:
        print(f'=== model:/n{model}/n===')
        
    if peft_config.get("target_modules") == "all":
        peft_config.update({"target_modules": get_all_linear_layers(model)})
        print(f'===target module for peft {peft_config["target_modules"]}===')
    elif peft_config.get("target_modules") is None:
        peft_config.pop("target_modules")
        print(peft_config)
        print(f'=== No  target module. Lora will use default setting. ===')

    lora_config = LoraConfig(**peft_config)
    
    if config.int8_training:
        model = prepare_model_for_kbit_training(model,use_gradient_checkpointing=config.gradient_checkpointing)
    model = get_peft_model(model, lora_config)
    print(f'model prepared with int_8 training: {config.int8_training} and dtype {dtype}')
    model.print_trainable_parameters()
    return model


def get_tokenizer(config,special_tokens,add_additional_special_tokens=True):
    """Rewritten from:
    https://github.com/LAION-AI/Open-Assistant/blob/main/model/model_training/utils/utils.py#L208"""
    tokenizer_name = config.model_name
    assert "llama" in tokenizer_name.lower(), "Currently only llama model supported"

    special_tokens = special_tokens["llama"]

    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=CACHE_DIR)
    tokenizer.add_special_tokens(special_tokens)

    if add_additional_special_tokens:
        additional_special_tokens = (
            []
            if "additional_special_tokens" not in tokenizer.special_tokens_map
            else tokenizer.special_tokens_map["additional_special_tokens"]
        )
        additional_special_tokens = list(QA_SPECIAL_TOKENS.values())
        tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens})

    return tokenizer,special_tokens["eos_token"]


def merge_and_save_peft_model(conf):
    if not os.path.exists(conf.merged_adapter_path):
        print(f"{'==='*10} Initiating the model merging process")
        peft_config = PeftConfig.from_pretrained(conf.adpater_path)

        dtype = torch.float32
        if conf.dtype in ["fp16", "float16"]:
            dtype = torch.float16
        elif conf.dtype in ["bf16", "bfloat16"]:
            dtype = torch.bfloat16

        if peft_config.task_type == "SEQ_CLS":
            base_model = transformers.AutoModelForSequenceClassification.from_pretrained(
                conf.base_model_name, num_labels=1, torch_dtype=dtype)
        else:
            base_model = transformers.AutoModelForCausalLM.from_pretrained(
                conf.base_model_name, return_dict=True, torch_dtype=dtype
                )

        tokenizer = transformers.AutoTokenizer.from_pretrained(conf.adpater_path)
        emedding_resize(base_model,tokenizer,16,conf)

        if conf.debug:
            print(base_model)

        # Load the Lora model
        model = PeftModel.from_pretrained(base_model, conf.adpater_path)

        if conf.debug:
            print(base_model)

        model.eval()
        model = model.merge_and_unload()
        model.save_pretrained(conf.merged_adapter_path)
        tokenizer.save_pretrained(conf.merged_adapter_path)
        print(f"{'==='*10} The peft model and tokenizer are saved at location {conf.merged_adapter_path}")
    else:
        print(f"{'==='*10} The peft model is already saved at location {conf.merged_adapter_path}")







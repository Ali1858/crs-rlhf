import math

import evaluate
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
import transformers
from constants import QA_SPECIAL_TOKENS,CACHE_DIR


def default_preprocess(eval_pred, ignote_negative_labels=True):
    """Taken from:
    https://github.com/LAION-AI/Open-Assistant/blob/main/model/model_training/utils/utils.py#L246
    """
    preds, labels = eval_pred.predictions, eval_pred.label_ids

    if not ignote_negative_labels:
        return preds, labels

    mask = labels > 0
    return preds[mask], labels[mask]


def get_all_linear_layers(model):
    """Taken from:
    https://github.com/LAION-AI/Open-Assistant/blob/main/model/model_training/utils/utils.py#L451"""
    cls = torch.nn.Linear

    modules = {name.split(".")[-1] for name, module in model.named_modules() if isinstance(module, cls)}
    if "lm_head" in modules:
        modules.remove("lm_head")

    return list(modules)


def prepare_model_for_gradient_checkpointing(model):
    r"""
    Prepares the model for gradient checkpointing if necessary
    Taken from:
    https://github.com/LAION-AI/Open-Assistant/blob/main/model/model_training/models/peft_modeling.py#L33
    """
    if not getattr(model, "is_loaded_in_8bit", False):
        print('checkpointing')
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    return model


def get_sft_model(tokenizer,config,pad_vocab_size_to_multiple_of=16):
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

    model = transformers.AutoModelForCausalLM.from_pretrained(config.model_name, torch_dtype=dtype,load_in_8bit=config.int8_training, cache_dir=CACHE_DIR)
    n_embs = model.get_input_embeddings().num_embeddings
    
    if len(tokenizer) != n_embs or pad_vocab_size_to_multiple_of:
            p = pad_vocab_size_to_multiple_of
            target_size = len(tokenizer) if not p else math.ceil(len(tokenizer) / p) * p
            print("Resizing embeddings to", target_size)
            model.resize_token_embeddings(target_size)
    
    if peft_config.get("target_modules") == "all":
        peft_config.update({"target_modules": get_all_linear_layers(model)})
    lora_config = LoraConfig(**peft_config)
    
    model.enable_input_require_grads()
    model = get_peft_model(model, lora_config)

    if config.gradient_checkpointing:
        model = prepare_model_for_gradient_checkpointing(model)
    model.print_trainable_parameters()
    return model


def get_sft_metrics(metrics):
    """Taken from:
    https://github.com/LAION-AI/Open-Assistant/blob/main/model/model_training/utils/utils.py#L301
    """
    return [ evaluate.load(metric) for metric in metrics], [default_preprocess]


def get_sft_tokenizer(config,special_tokens):
    """Rewritten from:
    https://github.com/LAION-AI/Open-Assistant/blob/main/model/model_training/utils/utils.py#L208"""
    tokenizer_name = config.model_name
    assert "llama" in tokenizer_name, "Currently only llama model supported"

    special_tokens = special_tokens["llama"]

    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=CACHE_DIR)
    tokenizer.add_special_tokens(special_tokens)
    additional_special_tokens = (
        []
        if "additional_special_tokens" not in tokenizer.special_tokens_map
        else tokenizer.special_tokens_map["additional_special_tokens"]
    )
    additional_special_tokens = list(QA_SPECIAL_TOKENS.values())
    tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens})

    return tokenizer,special_tokens["eos_token"]
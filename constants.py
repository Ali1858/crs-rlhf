import torch

QA_SPECIAL_TOKENS = {
    "Question": "<|prompter|>",
    "Answer": "<|assistant|>",
    "System": "<|system|>",
    # "StartPrefix": "",#"<|prefix_begin|>",
    # "EndPrefix": "",#"<|prefix_end|>",
}

RANDOM_SEED = 999
CACHE_DIR="./cache"

TOKENIZER_SEPECIAL_TOKENS= {"llama": {"pad_token": "[PAD]",
                                      "eos_token": "</s>",
                                      "sep_token": "<s>",
                                      },
                            }

DTYPES = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16
    }

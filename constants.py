QA_SPECIAL_TOKENS = {
    "Question": "<|prompter|>",
    "Answer": "<|assistant|>",
    "System": "<|system|>",
    "StartPrefix": "<|prefix_begin|>",
    "EndPrefix": "<|prefix_end|>",
}

RANDOM_SEED = 999
CACHE_DIR="./cache"

TOKENIZER_SEPECIAL_TOKENS= {"llama": {"pad_token": "</s>",
                                      "eos_token": "</s>",
                                      "sep_token": "<s>",
                                      },
                            }

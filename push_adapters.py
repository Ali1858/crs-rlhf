import transformers
import torch
import os
from glob import glob
import json 
from peft import PeftModel

CACHE_DIR = 'cache'
dtype = torch.bfloat16


def get_rl_model(model_name,adapter_name,device_map="auto"):
    print('**** loading  model ****')
    # Since reward models are trained using the same base model, we should use same model
    b_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        use_flash_attention_2=True,
        load_in_8bit=True,
        device_map=device_map,
        torch_dtype=dtype,
        cache_dir=CACHE_DIR,
        )

    adapter_name =  os.path.join(adapter_name,'final_checkpoint')
    print(adapter_name)
    base_reward_model = PeftModel.from_pretrained(
        b_model,
        adapter_name,
        )
    return base_reward_model


def get_reward_model(reward_model_name,adapter_name,device_map="auto"):
    print('**** loading reward model ****')
    # Since reward models are trained using the same base model, we should use same model
    base_reward_model = transformers.AutoModelForSequenceClassification.from_pretrained(
            reward_model_name,
            num_labels=1,
            use_flash_attention_2=True,
            load_in_8bit=True,
            device_map=device_map,
            torch_dtype=dtype,
            cache_dir=CACHE_DIR,
            )
    adapter_name =  os.path.join(adapter_name,'final_checkpoint')
    print(adapter_name)

    base_reward_model = PeftModel.from_pretrained(
        base_reward_model,
        adapter_name,
        )
    return base_reward_model

ORG = "alikhan0100u"
import os

# Replace YOUR_TOKEN_HERE with your actual token
os.environ["HF_TOKEN"] = ""

reward_model_name = "andreaskoepf/llama2-7b-oasst-baseline"
abs_adapter_name = "output/rm/abs_reward_model"
preference_adapter_name = "output/rm/preference_reward_model"
device_map = {"":0}

abs_model = get_reward_model(reward_model_name,abs_adapter_name,device_map="auto")
abs_adapter_to_push = ORG+"/Llama-2-7b-oasst-abs-reward-model-adapter"
abs_model.push_to_hub(abs_adapter_to_push, safe_serialization=True)

pref_adapter_to_push = ORG+"/Llama-2-7b-oasst-preference-reward-model-adapter"
preference_model = get_reward_model(reward_model_name,preference_adapter_name,device_map="auto")
preference_model.push_to_hub(pref_adapter_to_push, token=True, safe_serialization=True)

base_model_name = "andreaskoepf/llama2-7b-oasst-baseline"
adapter_names = ["output/rl/abs_rlhf","output/rl/crs_rlhf_025",
                 "output/rl/crs_rlhf_0625","output/rl/preference_rlhf_kl_001",
                 "output/rl/preference_rlhf_kl_002"]
hf_adapter_names = []
device_map = {"":0}

for name in adapter_names:
    hf_name =  f"{ORG}/Llama-2-7b-oasst-{name.split('/')[-1]}-adapter"
    print(f'pushing adapter for model {name} at {hf_name}')
    rlhf_model = get_rl_model(base_model_name,name,device_map="auto")
    rlhf_model.push_to_hub(hf_name,token=True, safe_serialization=True)
    rlhf_model = None

import transformers
import torch
from peft import PeftModel
from tqdm import tqdm

import json
import random


from time import time
import os

def generate_text(tokenizer,model,query_tensors,generation_kwargs,batch_size=8):
    outputs = []
    tokenizer.padding_side = "left"
    # in case we have fewer examples than bs
    batch_size = min(len(query_tensors), batch_size)
    model.eval()
    for i in tqdm(range(0, len(query_tensors), batch_size)):
            # prevent overflow if query tensors are not even multiple of bs
            end_index = min(len(query_tensors), i + batch_size)

            batch = query_tensors[i:end_index]
            batch_mask = [torch.ones_like(element) for element in batch]
            inputs = {"input_ids": batch, "attention_mask": batch_mask}

            padded_inputs = tokenizer.pad(
                inputs,
                padding=True,
                max_length=None,
                pad_to_multiple_of=16,
                return_tensors="pt",
            ).to(model.device)
            with torch.no_grad():  
                generations = model.generate(**padded_inputs, **generation_kwargs)

            for generation, mask in zip(generations, padded_inputs["attention_mask"]):
                output = generation[(1 - mask).sum() :]  # remove padding
                output = output[(mask).sum() :]  # remove prompt
                if tokenizer.eos_token_id in output:
                    pad_mask = output == tokenizer.eos_token_id
                    pad_start = torch.nonzero(pad_mask, as_tuple=False)[0, 0].item()
                    output = output[: pad_start + 1]  # keep the eos token at the end
                outputs.append(output)
    outputs = tokenizer.batch_decode(outputs,skip_special_tokens=True)
    return outputs


def pipeline(model,qry,q_tensors,gen_kwargs,filename):
    start_time = time()
    outputs = generate_text(tokenizer,model,q_tensors,gen_kwargs)
    # Prepare data to be saved
    data_to_save = [{"query": q, "response": r,"reward":0} for q, r in zip(qry, outputs)]
    # Save to JSON file
    with open(filename, 'w') as f:
        json.dump(data_to_save, f, ensure_ascii=False, indent=4)
    end = time()
    print(f"*** Saved generated responses and rewards to {filename} in {end-start_time} seconds")
     

def adpater_pipeline(model,adapters,qry,q_tensors,gen_kwargs,dir_suffix,save_path_prefix,mode="rlhf"):
    for adapter_name, adapter_path in adapters.items():
        print(f'getting prediction for model at path {adapter_path}')
        if mode == "rlhf":
            model.set_adapter(adapter_name)
        save_dir  = '/'.join(adapter_path.split('/')[:-1])
        filename = os.path.join(save_dir,dir_suffix,f"{save_path_prefix}_output.json")
        pipeline(model,qry,q_tensors,gen_kwargs,filename)

        
def get_query_tensors():
    # read eval data
    data_fn = data_fn_dict[eval_type]
    with open(f'data/{data_fn}','r') as f:
        data = json.load(f)
    examples = {
                "query": [],
                "input_ids": [],
                }
    custom_data = [{"instruction":"<|im_start|>user\nExplain the process of photosynthesis in simple terms.<|im_end|>\n<|im_start|>assistant\n"},
              {"instruction":"<|im_start|>user\nCreate a short story about a time-traveling detective.<|im_end|>\n<|im_start|>assistant\n"},
              {"instruction":"<|im_start|>user\nSuggest three innovative ways to reduce plastic waste in urban areas.<|im_end|>\n<|im_start|>assistant\n"}]
    # data.extend(custom_data)
    for query in data:
            tokenized_question = torch.tensor(tokenizer(query["instruction"], padding=False, truncation=False).input_ids)
            examples["query"].append(query["instruction"])
            examples["input_ids"].append(tokenized_question)
    print(f'Eval dataset size: {len(examples["query"])}')
    return examples

device_map = {"":1}
perform_rlhf = False #Other SFT
num_run = 0
gen_kwarg_p_09 = True #When using p sampling and temperature
# Set it True when evaluating each checkpoint, while evaluating each checkpoint use 'tuning' type and set eval_model name
eval_checkpoint = False 
eval_type = "final" #Type from: final, tuning, and extra
model_name = "andreaskoepf/llama2-7b-oasst-baseline"
eval_model_name = None #"rank_002" Only applicable when evaualting each checkpoint

crs_fine_tune_adapters = {"025":"output/rl/LLama-2-7b-oasst-baseline_rl_crs_025_bs16_kl_002_clip_04_512_max_token_with_pad_eos_lr_141e5/final_checkpoint",
                          "0375":"output/rl/LLama-2-7b-oasst-baseline_rl_crs_0375_bs16_kl_002_clip_04_512_max_token_with_pad_eos_lr_141e5/final_checkpoint",
                          "05":"output/rl/LLama-2-7b-oasst-baseline_rl_crs_05_bs16_kl_002_clip_04_512_max_token_with_pad_eos_lr_141e5/final_checkpoint",
                          "0625":"output/rl/LLama-2-7b-oasst-baseline_rl_crs_0625_bs16_kl_002_clip_04_512_max_token_with_pad_eos_lr_141e5/final_checkpoint",
                          "075":"output/rl/LLama-2-7b-oasst-baseline_rl_crs_075_bs16_kl_002_clip_04_512_max_token_with_pad_eos_lr_141e5/final_checkpoint"
                        }

f_adapters = {
            # "rank_001":'output/rl/LLama-2-7b-oasst-baseline_rl_bs16_kl_001_clip_04_512_max_token_with_pad_eos_lr_141e5/final_checkpoint',
            "rank_002":"output/rl/LLama-2-7b-oasst-baseline_rl_bs16_kl_002_clip_04_512_max_token_with_pad_eos_lr_141e5/final_checkpoint",
            "abs":"output/rl/LLama-2-7b-oasst-baseline_rl_abs_quality_rw_075_bs16_kl_002_clip_04_512_max_token_with_pad_eos_lr_141e5_logits/final_checkpoint",
            # "crs_0625":"output/rl/LLama-2-7b-oasst-baseline_rl_f_crs_0625_bs16_kl_002_clip_04_512_max_token_with_pad_eos_lr_141e5/final_checkpoint",
            "crs_025":"output/rl/LLama-2-7b-oasst-baseline_rl_f_crs_025_bs16_kl_002_clip_04_512_max_token_with_pad_eos_lr_141e5/final_checkpoint"
            }
# f_adapters = {"crs_025":f_adapters["crs_025"]}

if not perform_rlhf:
    f_adapters = {"sft":"output/rl/LLama-2-7b-oasst-basseline_sft/final_checkpoint"}

if eval_model_name and 'rank' in eval_model_name:
    checkpoint_numbers = ['checkpoint_100',
                        'checkpoint_200',
                        'checkpoint_300',
                        'checkpoint_400',
                        'checkpoint_500',
                        'final_checkpoint']
else:
    checkpoint_numbers = ['checkpoint_80',
                        'checkpoint_160',
                        'checkpoint_240',
                        'checkpoint_320',
                        'checkpoint_400',
                        'checkpoint_480',
                        'checkpoint_560',
                        'final_checkpoint']
if eval_model_name:
    eval_model_path = f_adapters[eval_model_name]
    eval_model_path = '/'.join(eval_model_path.split('/')[:-1])
    checkpoint_adapters = {}
    for num in checkpoint_numbers:
        checkpoint_adapters[f'{eval_model_name}_{num.split("_")[-1]}'] = f'{eval_model_path}/{num}'
     
## Select the dataset
data_fn_dict = {"tuning":'eval_hp_tuning.json',
                "final":"eval_final.json",
                "humaneval":"eval_humaneval.json"
                }

b_model = transformers.AutoModelForCausalLM.from_pretrained(
    model_name,
    use_flash_attention_2=True,
    load_in_8bit=True,
    device_map=device_map,
    torch_dtype=torch.bfloat16,
    cache_dir='cache',
    )

tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, cache_dir='cache')
tokenizer.add_special_tokens({"pad_token":"<PAD>","eos_token":"<|im_end|>","sep_token":"<SEP>"})
print(f'tokenizer pad {tokenizer.pad_token} and model pad {b_model.config.pad_token_id}')
print(f'tokenizer eos {tokenizer.eos_token} and model eos {tokenizer.eos_token_id}')
print(f"{'==='*10}")
if b_model.config.pad_token_id is None or b_model.config.pad_token_id == 0:
    print('changing model pad token id')
    b_model.config.pad_token_id = tokenizer.pad_token_id

output_json_prefix = eval_type
if eval_type == "final" or eval_type == "humaneval":
     eval_adapters = f_adapters
elif eval_type == "tuning" and eval_model_name:
    eval_adapters = checkpoint_adapters
else:
    eval_adapters = crs_fine_tune_adapters
print(f"{'==='*10}")
print(f'adapters: {eval_adapters}')

if gen_kwarg_p_09:
    generation_kwargs = {
                    "top_k": 0.0,
                    "top_p": 0.9,
                    "temperature":0.8,
                    "do_sample": True,
                    "max_new_tokens": 512,
                }
    output_json_prefix = f'{output_json_prefix}_p_09_t_08'
else:
    generation_kwargs = {
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "max_new_tokens": 512,
        }
    output_json_prefix = f'{output_json_prefix}_p_01'
print(f"{'==='*10}")
print(f'gen kwargs:{generation_kwargs}')

mode = "sft"
if perform_rlhf:
    mode = "rlhf"
    for i,(adapter_name,adapter_path) in enumerate(eval_adapters.items()):
        print(f'loading adapter from path {adapter_path}')
        if i == 0:
            b_model = PeftModel.from_pretrained(b_model,
                                            adapter_path,
                                            adapter_name=adapter_name,
                                            is_trainable=False,
                                            device_map=device_map
                                            )
        b_model.load_adapter(adapter_path,
                                adapter_name=adapter_name,
                                is_trainable=False)
    b_model = b_model.to(b_model.device)

print(f"{'==='*10}")
print(f'mode:{mode}')

examples = get_query_tensors()
query_tensors = examples["input_ids"]
query = examples["query"]


print(f"{'==='*10}")
print(f"output prefix {output_json_prefix}")

if num_run and num_run > 0:
    for run in range(num_run):
        output_json_prefix_ = f'{output_json_prefix}_run_{run}_{mode}'
        adpater_pipeline(b_model,eval_adapters,query,query_tensors,generation_kwargs,"eval_output",output_json_prefix_,mode)
else:
    output_json_prefix_ = f'{output_json_prefix}_{mode}'
    adpater_pipeline(b_model,eval_adapters,query,query_tensors,generation_kwargs,"eval_output",output_json_prefix_,mode)

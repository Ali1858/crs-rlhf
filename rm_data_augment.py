
import os
from tqdm import tqdm
from time import time
from training_datasets.dataset_utils import load_rm_dataset
import torch
import transformers
from utils import (parse_additional_args, parse_arguments, debug_configurations)
from constants import DTYPES, CACHE_DIR
from training_datasets.dataset_utils import get_rm_formatted,format_pairs


def get_base_mode_for_inference(model_name,dtype,device_map = {"":0}):
    # from transformers import pipeline, TextGenerationPipeline
    base_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        use_flash_attention_2=True,
        load_in_8bit=True,
        device_map=device_map,
        torch_dtype=dtype,
        cache_dir=CACHE_DIR,
    )
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)
    
    print(f'tokenizer pad {tokenizer.pad_token} and model pad {base_model.config.pad_token_id}')
    print(f'tokenizer eos {tokenizer.eos_token} and model eos {tokenizer.eos_token_id}')
    tokenizer.pad_token_id = tokenizer.eos_token_id
    base_model.config.pad_token_id = tokenizer.eos_token_id
    return base_model, tokenizer

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
    return outputs



from transformers import AutoModel, AutoTokenizer
from torch.nn.functional import cosine_similarity
import torch
import torch.nn.functional as F

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def get_sentence_transformers():
    # Load model from HuggingFace Hub
    model_name = "sentence-transformers/all-mpnet-base-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return model,tokenizer

from transformers  import pipeline
def calculate_cosine_similarity(generated_response, original_response):
    model,tokenizer = get_sentence_transformers()
    lang_classification = pipeline("text-classification","papluca/xlm-roberta-base-language-detection",truncation=True)
    sim_scores = []
    for g, o in zip(generated_response,original_response):
        try:
            org_lang = lang_classification(o)[0]['label']
            gen_lang = lang_classification(g)[0]['label']
        except Exception as e:
            print(e)
            org_lang = gen_lang = 'en'
    
        encoded_input = tokenizer([g,o], padding=True, truncation=True, return_tensors='pt')
        # Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)
        # Perform pooling
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        if org_lang == gen_lang:
            sim = cosine_similarity(sentence_embeddings[0].reshape(1,-1),sentence_embeddings[1].reshape(1,-1))
            sim_scores.append(sim[0].item())
        else:
            sim_scores.append(0.2)
    return sim_scores


def adjust_score_based_on_similarity(original_score, similarity, adjustment_factor):
    # Normalize similarity to be between 0 and 1 (as cosine similarity ranges from -1 to 1)
    normalized_similarity = (similarity + 1) / 2
    score_adjustment = (1 - normalized_similarity) * adjustment_factor
    adjusted_score = original_score - score_adjustment
    adjusted_score = max(0, min(adjusted_score, 1))
    return adjusted_score

import pandas as pd
def augment(model_name,conf):
    generate = False

    new_examples = {
            "original_prefix": [],
            "input_ids": [],
            "orginal_scores": [],
            "original_reply":[]
            }

    if generate:
        train_ds , _ = load_rm_dataset(conf)
        dtype = DTYPES.get('bf16', torch.float32)  
        base_model,tokenizer = get_base_mode_for_inference(model_name,dtype,device_map="auto")
        generation_kwargs = {
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "max_new_tokens": 512,
        }

        for i,example in enumerate(train_ds):
            prefix, reply, score = example
            if score <= 0.4:
                input_prompt = f"""
                [INST] <<SYS>>
                You are a paraphrasing robot, Your task is to output only the paraphrase version of the text. The paraphrase should be the and only return the paraphrased version

                You take the original text and strictly output the paraphrased version. Ensure the language of paraphrase version is same as original text and without extra information.
                <</SYS>>
                Original Text:
                {reply}
                Paraphase Text: [/INST]
                """
                prefix_tokens = torch.tensor(tokenizer(input_prompt, padding=False, truncation=False).input_ids)

                new_examples["input_ids"].append(prefix_tokens)
                new_examples["original_prefix"].append(prefix)
                new_examples["original_reply"].append(reply)
                new_examples["orginal_scores"].append(score)

        generated_response = generate_text(tokenizer,base_model,new_examples["input_ids"],generation_kwargs,batch_size=12)
        decoded_responses = tokenizer.batch_decode(generated_response, skip_special_tokens=True)

        data = []
        for p,r,og_r,og_s in zip(new_examples["original_prefix"],decoded_responses,new_examples["original_reply"],new_examples["orginal_scores"]):
            data.append([p,r,og_r,og_s])
        df = pd.DataFrame(data, columns=["message", "reply","original_reply","original_score"])
        df.to_csv('rm_data_augment/raw_rm_augmented.csv',index=False)
    else:
        df = pd.read_csv('rm_data_augment/clean_rm_augmented.csv')
        new_examples["similarity"] = calculate_cosine_similarity(df["reply"].to_list(), df["original_reply"].to_list())
        
        adjustment_factor = 0.5
        adusted_scores = []
        for score,similarity in zip(df["original_score"].to_list(),new_examples["similarity"]):
            adusted_scores.append(adjust_score_based_on_similarity(score, similarity, adjustment_factor))
        
        df["score"] = adusted_scores
        df = df.drop(columns=["original_reply","original_score"])
        df.to_csv('rm_data_augment/rm_augmented.csv',index=False)

if __name__ == "__main__":
    config, remaining_args = parse_arguments()
    parser = parse_additional_args(config)
    args = parser.parse_args(remaining_args)

    debug_tag = "_dbug" if args.debug else ""
    reward_type = "_abs" if args.is_abs_rm else "_ranking"
    args.name = f"{args.name}{reward_type}{debug_tag}{args.name_suffix}"
    args.output_dir = os.path.join(args.output_dir, args.name)

    debug_configurations(args)
    
    augment("meta-llama/Llama-2-7b-chat-hf",args)

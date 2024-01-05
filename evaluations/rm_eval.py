import argparse
import torch
import transformers
from sklearn.metrics import mean_squared_error, mean_absolute_error
from utils import read_yaml, init_or_resume_from
from model_training.training_utils import load_for_inference
from training_datasets.dataset_utils import load_rm_dataset
from training_datasets.collators import AbsoluteScoreDataCollator, RankingDataCollator
from transformers import BitsAndBytesConfig
from peft import PeftModel
from torch.utils.data import ConcatDataset, Subset, Dataset

import warnings

# Suppress specific warnings
warnings.filterwarnings('ignore', message="`max_length` is ignored when `padding`=`True` and there is no truncation strategy. To pad to max length, use `padding='max_length'`.")
warnings.filterwarnings('ignore', message="MatMul8bitLt: inputs will be cast from torch.bfloat16 to float16 during quantization")

# Your code here

def load_config(config_path, dataset_name):
    conf = read_yaml(config_path)
    config = {}
    config.update(conf["default"])
    config.update(conf[dataset_name])
    config["name_suffix"] = ""
    config["debug"] = False
    config["subset"] = dataset_name
    return argparse.Namespace(**config)


def load_dataset(config):
        return load_rm_dataset(config)


def load_model(model_name, model_args, tokenizer_cache_dir='cache'):
    # model_name =f'output/sft/{model_name}/merged/'
    adapter_mapping = {
        # "ranking":"output/rm/LLama-2-7b-oasst-baseline_reward_ranking_bs64_ep_1_8bit_bf16/final_checkpoint",
                        # "rank":"output/rm/LLama-2-7b-oasst-baseline_reward_ranking_bs64_ep_1_8bit_bf16_eos_token/final_checkpoint",
                        # "abs_sig_mse_aug_and_un_till16_s_075_w_14_09_quality":"output/rm/LLama-2-7b-oasst-baseline_reward_abs_bs128_ep_1_8bit_sig_mse_s_075_w_14_09_aug_and_un_till16_quality/final_checkpoint",
                        # "abs_sig_mse_aug_and_un_till16_s_075_quality":"output/rm/LLama-2-7b-oasst-baseline_reward_abs_bs128_ep_1_8bit_sig_mse_s_075_aug_and_un_till16_quality/final_checkpoint",
                        # "abs_sig_mse_aug_over_and_un_till16_s_075_w_14_09_quality":"output/rm/LLama-2-7b-oasst-baseline_reward_abs_bs128_ep_1_8bit_sig_mse_s_075_w_14_09_aug_over_and_un_till16_quality/final_checkpoint",
                        # "abs_sig_mse_aug_over_and_un_till16_w14_for_04_09_quality":"output/rm/LLama-2-7b-oasst-baseline_reward_abs_bs128_ep_1_8bit_sig_mse_w_14_for_04_09_aug_over_and_un_till16_quality/final_checkpoint",
                        # "abs_logistic_aug_over_and_un_till16_w14_for_04_09_quality":"output/rm/LLama-2-7b-oasst-baseline_reward_abs_bs128_ep_1_8bit_logistic_w_14_for_04_09_aug_over_and_un_till16_quality/final_checkpoint",
                    
                        #  "abs_sig_mse_aug_and_under_quality":"output/rm/LLama-2-7b-oasst-baseline_reward_abs_bs128_ep_1_8bit_sig_mse_aug_and_under_quality/final_checkpoint",
                        "abs_logistic_aug_and_under_quality": "output/rm/LLama-2-7b-oasst-baseline_reward_abs_bs128_ep_1_8bit_logistic_aug_and_under_quality/final_checkpoint",
                        #  "abs_logistic_aug_and_under_s_075_for_04_09_quality": "output/rm/LLama-2-7b-oasst-baseline_reward_abs_bs128_ep_1_8bit_logistic_s_075_for_04_09_aug_and_under_quality/final_checkpoint",
                        #  "abs_logistic_aug_over_and_under_w14_for_04_09_quality": "output/rm/LLama-2-7b-oasst-baseline_reward_abs_bs128_ep_1_8bit_sig_mse_w_14_for_04_09_aug_over_and_under_quality/final_checkpoint",
                        #    "abs_sig_mse_s_075_quality":"output/rm/LLama-2-7b-oasst-baseline_reward_abs_bs128_ep_1_8bit_sig_mse_s_075_over_and__under/final_checkpoint",
                    #    "abs_mse_s_075_quality":"output/rm/LLama-2-7b-oasst-baseline_reward_abs_bs128_ep_1_8bit_mse_s_075_over_and__under/final_checkpoint",
                    #    "abs_logistic_aug_and_under_s_05_quality":"output/rm/LLama-2-7b-oasst-baseline_reward_abs_bs128_ep_1_8bit_logistic_s_05_augment_and__under/final_checkpoint",
                       "abs_logistic_aug_and_under_s_075_quality":"output/rm/LLama-2-7b-oasst-baseline_reward_abs_bs128_ep_1_8bit_logistic_s_075_augment_and__under/final_checkpoint",
                    #    "abs_logistic_over_and_under_s_05_quality":"output/rm/LLama-2-7b-oasst-baseline_reward_abs_bs128_ep_1_8bit_logistic_s_05_over_and_under_sampled_only_quality_eos_token/final_checkpoint",
                    #    "abs_logistic_over_and_under_s_75_quality":"output/rm/LLama-2-7b-oasst-baseline_reward_abs_bs128_ep_1_8bit_logistic_s_075_over_and_under/final_checkpoint",
                    #    "abs_logistic_over_and_under_quality":"output/rm/LLama-2-7b-oasst-baseline_reward_abs_bs128_ep_1_8bit_logistic_over_and_under_sampled_only_quality_eos_token/final_checkpoint",
                       
                    # No need for seperate evaluation, take decision based on abs_logistic_s_075_quality
                    #    "abs_logistic_s_075_quality_eos_token":"output/rm/LLama-2-7b-oasst-baseline_reward_abs_bs128_ep_1_8bit_logistic_wgt_loss_075_strength_only_quality_eos_token/final_checkpoint",

                    # Run this by commenting 'add_special_tokens' line
                    #    "abs_logistic_s_05_quality":"output/rm/LLama-2-7b-oasst-baseline_reward_abs_bs128_ep_1_8bit_logistic_wgt_loss_05_strength_only_quality/final_checkpoint",
                    #    "abs_logistic_s_075_quality":"output/rm/LLama-2-7b-oasst-baseline_reward_abs_bs128_ep_1_8bit_logistic_wgt_loss_075_strength_only_quality/final_checkpoint",
                    #    "abs_logistic_quality":"output/rm/LLama-2-7b-oasst-baseline_reward_abs_bs128_ep_1_8bit_logistic_wgt_loss_only_quality/final_checkpoint",

                    # Not useful
                    #    "abs_logistic_wgt_agg_label":"output/rm/LLama-2-7b-oasst-baseline_reward_abs_bs128_ep_1_8bit_logistic_wgt_loss/final_checkpoint",
                    #    "abs_logistic":"output/rm/LLama-2-7b-oasst-baseline_reward_abs_bs128_ep_1_8bit_logistic/final_checkpoint"

                    }

    
    # {"ranking":"output/rm/LLama-2-7b_crs_oasst_sft_reward_ranking_bs_64_ep_1/final_checkpoint",
    #                "abs":"output/rm/LLama-2-7b_crs_oasst_sft_reward_abs_bs_128_ep_1_logistic/final_checkpoint",
    #                "abs_wgt_loss":"output/rm/LLama-2-7b_crs_oasst_sft_reward_abs_bs_128_ep_1_logistic_wgt_loss/final_checkpoint",
    #                "ranking2":"output/rm/LLama-2-7b_crs_oasst_sft_reward_ranking_bs_64_ep_1_sft_no_quantized/final_checkpoint"}

    base_model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1, **model_args)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, cache_dir=tokenizer_cache_dir)
    tokenizer.add_special_tokens({"pad_token":"<PAD>","eos_token":"<|im_end|>","sep_token":"<SEP>"})


    print(f'tokenizer pad {tokenizer.pad_token} and model pad {base_model.config.pad_token_id}')
    print(f'tokenizer eos {tokenizer.eos_token} and model eos {tokenizer.eos_token_id}')

    adapter_names = list(adapter_mapping.keys())
    adapter_paths = list(adapter_mapping.values())
    if base_model.config.pad_token_id is None or base_model.config.pad_token_id == 0:
        print('changing model pad token id')
        base_model.config.pad_token_id = tokenizer.pad_token_id

    base_model = PeftModel.from_pretrained(
        base_model,
        adapter_paths[0],
        adapter_name=adapter_names[0],
        is_trainable=False,
        device_map=model_args["device_map"]
        )

    
    for name,path in zip(adapter_names[1:],adapter_paths[1:]):
        base_model.load_adapter(path, adapter_name=name, is_trainable=False,device_map=model_args["device_map"])
    
    base_model = base_model.to(base_model.device)
    return base_model, tokenizer, adapter_names

def get_average_scores(scores):
    # Transpose the list of lists to group scores of the same element across models
    transposed_scores = list(zip(*scores))

    # Calculate the average for each element
    average_scores = [sum(element_scores) / len(element_scores) for element_scores in transposed_scores]
    return average_scores


def get_reward(model,inputs,adapter_name):
        with torch.no_grad():
            print(f"{'=='*10}{adapter_name}")
            model.set_adapter(adapter_name)
            logits = model(input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    use_cache=False,
                    ).logits
            s = torch.sigmoid(logits)
            for idx in range(logits.shape[0]):
                    print(f'sigmoid: {s[idx]},original: {logits[idx]}')
            print('==='*10)
            # return [str(ss[0].item()) for ss in s]
            return [ss[0].item() for ss in s]

    
def evaluate_padding_error(ranking_collate_fn,abs_collate_fn,model,adapter_names,metric_dict):
     
    print(f'{"***"*10} All Four {"***"*10}')
    # Example usage of get_predictions function
    data = [["Hi how are you?"],["I am good you piece of shit","Giberish, Giberish saying Giberish, tell Giberish is not Giberish. Why are Giberish you Giberish","I am good","I am good, how are you doing? Please tell me how can I help you? "]] 
    inputs, _ = ranking_collate_fn([data])
    inputs = inputs.to(model.device)

    scores = []
    for adapter_name in adapter_names:
        score = get_reward(model,inputs,adapter_name) 
        scores.append(score)
    print(get_average_scores(scores))
        # metric_dict[adapter_name]["responses_score"] = score 
    
    # print(f'{"***"*10} First Two {"***"*10}')    
    # # Example usage of get_predictions function
    # data = [["Hi how are you?"],["I am good you piece of shit","Giberish, Giberish saying Giberish, tell Giberish is not Giberish. Why are Giberish you Giberish",]] 
    # inputs, _ = ranking_collate_fn([data])
    # inputs = inputs.to(model.device)
    # for adapter_name in adapter_names:
    #     get_reward(model,inputs,adapter_name)  

    # print(f'{"***"*10} First & third {"***"*10}')    
    # # Example usage of get_predictions function
    # data = [["Hi how are you?"],["I am good you piece of shit","I am good"]] 
    # inputs, _ = ranking_collate_fn([data])
    # inputs = inputs.to(model.device)
    # for adapter_name in adapter_names:
    #     get_reward(model,inputs,adapter_name)  
    
    # print(f'{"***"*10} First & fourth {"***"*10}')    
    # # Example usage of get_predictions function
    # data = [["Hi how are you?"],["I am good you piece of shit","I am good, how are you doing? Please tell me how can I help you? "]] 
    # inputs, _ = ranking_collate_fn([data])
    # inputs = inputs.to(model.device)
    # for adapter_name in adapter_names:
    #     get_reward(model,inputs,adapter_name)  

    # print(f'{"***"*10} Only first with abs {"***"*10}')   
    # data = [
    #      [['Hi how are you?'],'I am good you piece of shit',0]
    #      ]
    # inputs = abs_collate_fn(data)
    # inputs = inputs.to(model.device)
    # for adapter_name in adapter_names:
    #     get_reward(model,inputs,adapter_name) 

    # print(f'{"***"*10} first & second with abs {"***"*10}')   
    # data = [
    #      [['Hi how are you?'],'I am good you piece of shit',0],
    #      [['Hi how are you?'],"Giberish, Giberish saying Giberish, tell Giberish is not Giberish. Why are Giberish you Giberish",0]
    #      ]
    # inputs = abs_collate_fn(data)
    # inputs = inputs.to(model.device)
    # for adapter_name in adapter_names:
    #     get_reward(model,inputs,adapter_name)


def evaluate_abs_model_agg(model, data_collator, data, adapter_names,device,metric_dict):
    labels_list = []
    predictions_list = []
    predictions_list_2 = []

    
    with torch.no_grad():
        for item in data:
            inputs = data_collator([item])
            labels = inputs.pop("labels")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels_list.append(labels[0].cpu().float())

            model.set_adapter(adapter_names[0])
            logits = model(**inputs).logits
            predictions_list.append(logits.cpu().float())

            model.set_adapter(adapter_names[1])
            logits = model(**inputs).logits
            predictions_list_2.append(logits.cpu().float())

        pred1 = torch.sigmoid(torch.stack(predictions_list)).squeeze()
        pred2 = torch.sigmoid(torch.stack(predictions_list_2)).squeeze()
        pred = (pred1 + pred2) / 2

        l40,p40 = [], []
        l90,p90 = [], []
        lmid,pmid = [], []
        c = 0
        for l, p in zip(labels_list,pred):
            if l <= 0.4:
                l40.append(l)
                p40.append(p)
            elif l >= 0.9:
                l90.append(l)
                p90.append(p)
            else:
                lmid.append(l)
                pmid.append(p)

        def get_metrics(l,p,label='Below 0.4'):
            mse = mean_squared_error(l,p)
            mae = mean_absolute_error(l,p)
            print(f"*** {label}, Mean Squared Error: {mse}, Mean Absolute Error: {mae} ***")

        print(f'==== Printing metrics for the adapter {adapter_names} ====')
        get_metrics(l40,p40)
        get_metrics(l90,p90,'Above 0.9')
        get_metrics(lmid,pmid,'Between 0.4 and 0.9')
        get_metrics(labels_list,pred,'Full dataset')
    return metric_dict

def evaluate_abs_model(model, data_collator, data, adapter_names,device,metric_dict):
    for adapter in adapter_names:
        print(f'=== getting prediction for the adapter {adapter} ===')
        with torch.no_grad():
            model.set_adapter(adapter)
            labels_list = []
            predictions_list = []

            for item in data:
                inputs = data_collator([item])
                labels = inputs.pop("labels")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                logits = model(**inputs).logits
                labels_list.append(labels[0].cpu().float())
                predictions_list.append(logits.cpu().float())

        pred_ = torch.sigmoid(torch.stack(predictions_list)).squeeze()
        pred = []
        for p in pred_:
            if p < 0.46:
                p = p-0.12
            p = max(0,p)
            pred.append(p)
        # pred = torch.stack(predictions_list).squeeze()

        l40,p40 = [], []
        l90,p90 = [], []
        lmid,pmid = [], []
        c = 0
        for l, p in zip(labels_list,pred):
            if l <= 0.4:
                l40.append(l)
                p40.append(p)
            elif l >= 0.9:
                l90.append(l)
                p90.append(p)
            else:
                lmid.append(l)
                pmid.append(p)

        #     if p < 0 or p > 1:
        #         c+=1
        # print(c)

        def get_metrics(l,p,label='Below 0.4'):
            mse = mean_squared_error(l,p)
            mae = mean_absolute_error(l,p)
            print(f"*** {label}, Mean Squared Error: {mse}, Mean Absolute Error: {mae} ***")
            metric_dict.setdefault(adapter,{})[label] = {'mse':str(mse),'mae':str(mae)}

        print(f'==== Printing metrics for the adapter {adapter} ====')
        get_metrics(l40,p40)
        get_metrics(l90,p90,'Above 0.9')
        get_metrics(lmid,pmid,'Between 0.4 and 0.9')
        get_metrics(labels_list,pred,'Full dataset')
    return metric_dict


def main():
    parser = argparse.ArgumentParser(description="Model evaluation script")
    parser.add_argument("--config_path", type=str, default='./config.yaml', help="Path to the configuration file")
    parser.add_argument("--dataset_name", type=str, default="eval_ranking_rm",help="Name of the dataset to load")
    parser.add_argument("--model_name", type=str, default="andreaskoepf/llama2-7b-oasst-baseline",help="Name of the model to load")
    parser.add_argument("--adapter_names", type=str, default="ranking,abs,abs_wgt_loss" ,help="List of adapter names to load")
    parser.add_argument("--load_4bit", action='store_true', help="Flag to load model in 4-bit format")
    parser.add_argument("--load_8bit", action='store_true', help="Flag to load model in 4-bit format")

    args = parser.parse_args()

    adapter_names = args.adapter_names.split(',')
    config = load_config(args.config_path, args.dataset_name)
    if args.dataset_name == "eval_abs_rm":
        #  config.dataset["oasst_export_abs"]["label_weight"] = {"violence": 0, "creativity": 0.15, "helpfulness": 0.35, "humor": 0, "toxicity": -0.1,"quality": 0.4}
         config.dataset["oasst_export_abs"]["label_weight"] = {"violence": 0, "creativity": 0, "helpfulness": 0, "humor": 0, "toxicity": 0,"quality": 1}
         config.dataset["oasst_export_abs"]["abs_oversample_threshold"]=0.4
         config.dataset["oasst_export_abs"]["top_k"]=None
    
    
    _, eval_data = load_dataset(config)
    
    if args.dataset_name == "eval_abs_rm":
        eval_data = eval_data["oasst_export_abs"]
    else:
        eval_data = eval_data["oasst_export"]
    
    # eval_data = Subset(eval_data, range(200))

    model_args = {
        "use_flash_attention_2":True,
        "load_in_8bit": True,
        "torch_dtype": torch.bfloat16,
        "cache_dir": 'cache',
        "device_map": {"":0},
    }

    # model_args["quantization_config"] = BitsAndBytesConfig(
    #         load_in_4bit=args.load_4bit,
    #         load_in_8bit=args.load_8bit,
    #         bnb_4bit_quant_type="nf4",
    #         bnb_4bit_compute_dtype=torch.bfloat16,
    #         bnb_4bit_use_double_quant=True,
    #         )

    model, tokenizer,adapter_names = load_model(args.model_name, model_args)

    abs_collate_fn = AbsoluteScoreDataCollator(tokenizer, max_length=2048, pad_to_multiple_of=16)
    ranking_collate_fn = RankingDataCollator(tokenizer, max_length=2048, pad_to_multiple_of=16, max_replies=4)
    metric_dict = dict()
    evaluate_abs_model(model, abs_collate_fn, eval_data, adapter_names,model.device,metric_dict)
    
    # evaluate_abs_model_agg(model, abs_collate_fn, eval_data, adapter_names,model.device,metric_dict)
    evaluate_padding_error(ranking_collate_fn,abs_collate_fn,model,adapter_names,metric_dict)
    print(metric_dict)
    import json
    with open('eval_metrics_q_aug_and_under_adjust.json','w') as f:
       json.dump(metric_dict,f)  
    config = load_config(args.config_path, "eval_ranking_rm")
    _, eval_data = load_dataset(config)
    eval_data = eval_data["oasst_export"]

    idx = 91
    data = eval_data[idx]
    inputs, _ = ranking_collate_fn([data])
    inputs = inputs.to(model.device)
    for adapter_name in adapter_names:
        get_reward(model,inputs,adapter_name)


        

if __name__ == "__main__":
    main()

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


def load_model(model_name, adapter_names, model_args, tokenizer_cache_dir='cache'):
    model_name =f'output/sft/{model_name}/merged/'
    adapter_mapping = {"ranking":"output/rm/LLama-2-7b_crs_oasst_sft_reward_ranking_bs_64_ep_1/final_checkpoint",
                   "abs":"output/rm/LLama-2-7b_crs_oasst_sft_reward_abs_bs_128_ep_1_logistic/final_checkpoint",
                   "abs_wgt_loss":"output/rm/LLama-2-7b_crs_oasst_sft_reward_abs_bs_128_ep_1_logistic_wgt_loss/final_checkpoint",
                   "ranking2":"output/rm/LLama-2-7b_crs_oasst_sft_reward_ranking_bs_64_ep_1_sft_no_quantized/final_checkpoint"}

    base_model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1, **model_args)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, cache_dir=tokenizer_cache_dir)

    print(f'tokenizer pad {tokenizer.pad_token} and model pad {base_model.config.pad_token_id}')
    print(f'tokenizer eos {tokenizer.eos_token} and model eos {tokenizer.eos_token_id}')
    if base_model.config.pad_token_id is None or base_model.config.pad_token_id == 0:
        print('changing model pad token id')
        base_model.config.pad_token_id = tokenizer.pad_token_id
    base_model = PeftModel.from_pretrained(
        base_model,
        adapter_mapping[adapter_names[0]],
        adapter_name=adapter_names[0],
        is_trainable=False
        )

    
    for adapter_name in adapter_names[1:]:
        adapter_path = adapter_mapping[adapter_name]
        base_model.load_adapter(adapter_path, adapter_name=adapter_name, is_trainable=False)
    
    return base_model, tokenizer

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

def evaluate_padding_error(ranking_collate_fn,abs_collate_fn,model,adapter_names):
     
    print(f'{"***"*10} All Four {"***"*10}')
    # Example usage of get_predictions function
    data = [["Hi how are you?"],["I am good you piece of shit","Giberish, Giberish saying Giberish, tell Giberish is not Giberish. Why are Giberish you Giberish","I am good","I am good, how are you doing? Please tell me how can I help you? "]] 
    inputs, _ = ranking_collate_fn([data])
    inputs = inputs.to(model.device)
    for adapter_name in adapter_names:
        get_reward(model,inputs,adapter_name)  
    
    print(f'{"***"*10} First Two {"***"*10}')    
    # Example usage of get_predictions function
    data = [["Hi how are you?"],["I am good you piece of shit","Giberish, Giberish saying Giberish, tell Giberish is not Giberish. Why are Giberish you Giberish",]] 
    inputs, _ = ranking_collate_fn([data])
    inputs = inputs.to(model.device)
    for adapter_name in adapter_names:
        get_reward(model,inputs,adapter_name)  

    print(f'{"***"*10} First & third {"***"*10}')    
    # Example usage of get_predictions function
    data = [["Hi how are you?"],["I am good you piece of shit","I am good"]] 
    inputs, _ = ranking_collate_fn([data])
    inputs = inputs.to(model.device)
    for adapter_name in adapter_names:
        get_reward(model,inputs,adapter_name)  
    
    print(f'{"***"*10} First & fourth {"***"*10}')    
    # Example usage of get_predictions function
    data = [["Hi how are you?"],["I am good you piece of shit","I am good, how are you doing? Please tell me how can I help you? "]] 
    inputs, _ = ranking_collate_fn([data])
    inputs = inputs.to(model.device)
    for adapter_name in adapter_names:
        get_reward(model,inputs,adapter_name)  

    print(f'{"***"*10} Only first with abs {"***"*10}')   
    data = [
         [['Hi how are you?'],'I am good you piece of shit',0]
         ]
    inputs = abs_collate_fn(data)
    inputs = inputs.to(model.device)
    for adapter_name in adapter_names:
        get_reward(model,inputs,adapter_name) 

    print(f'{"***"*10} first & second with abs {"***"*10}')   
    data = [
         [['Hi how are you?'],'I am good you piece of shit',0],
         [['Hi how are you?'],"Giberish, Giberish saying Giberish, tell Giberish is not Giberish. Why are Giberish you Giberish",0]
         ]
    inputs = abs_collate_fn(data)
    inputs = inputs.to(model.device)
    for adapter_name in adapter_names:
        get_reward(model,inputs,adapter_name)


def evaluate_abs_model(model, data_collator, data, adapter_names,device):

    for adapter in adapter_names:
        print(f'*** getting prediction for the adapter {adapter} ***')
        with torch.no_grad():
            model.set_adapter(adapter)
            labels_list = []
            predictions_list = []

            for item in data:
                inputs = data_collator([item])
                labels = inputs.pop("labels")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                logits = model(**inputs).logits
                labels_list.append(labels[0])
                predictions_list.append(logits.cpu().float())

        mse = mean_squared_error(labels_list, torch.sigmoid(torch.stack(predictions_list)).squeeze())
        mae = mean_absolute_error(labels_list, torch.sigmoid(torch.stack(predictions_list)).squeeze())
        print(f"*** Mean Squared Error: {mse}, Mean Absolute Error: {mae} ***")

def main():
    parser = argparse.ArgumentParser(description="Model evaluation script")
    parser.add_argument("--config_path", type=str, default='./config.yaml', help="Path to the configuration file")
    parser.add_argument("--dataset_name", type=str, default="eval_ranking_rm",help="Name of the dataset to load")
    parser.add_argument("--model_name", type=str, default="LLama-2-7b_crs_oasst_sft_bs64_ep_1_not_quant_pad_token",help="Name of the model to load")
    parser.add_argument("--adapter_names", type=str, default="ranking,abs,abs_wgt_loss" ,help="List of adapter names to load")
    parser.add_argument("--load_4bit", action='store_true', help="Flag to load model in 4-bit format")
    parser.add_argument("--load_8bit", action='store_true', help="Flag to load model in 4-bit format")

    args = parser.parse_args()

    adapter_names = args.adapter_names.split(',')
    config = load_config(args.config_path, args.dataset_name)
    if args.dataset_name == "eval_abs_rm":
         config.dataset["oasst_export_abs"]["label_weight"] = {"violence": 0, "creativity": 0.15, "helpfulness": 0.35, "humor": 0, "toxicity": -0.1,"quality": 0.4}
         config.dataset["oasst_export_abs"]["abs_oversample_threshold"]=0.4
         config.dataset["oasst_export_abs"]["top_k"]=None
    
    
    train_data, eval_data = load_dataset(config)
    
    if args.dataset_name == "eval_abs_rm":
        eval_data = eval_data["oasst_export_abs"]
    else:
        eval_data = eval_data["oasst_export"]
    
    eval_data = Subset(eval_data, range(200))

    model_args = {
        "torch_dtype": torch.bfloat16,
        "cache_dir": 'cache',
        "device_map": "auto"#{"":1},
    }

    model_args["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=args.load_4bit,
            load_in_8bit=args.load_8bit,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            )

    model, tokenizer = load_model(args.model_name,adapter_names, model_args)

    abs_collate_fn = AbsoluteScoreDataCollator(tokenizer, max_length=2048, pad_to_multiple_of=16)
    ranking_collate_fn = RankingDataCollator(tokenizer, max_length=2048, pad_to_multiple_of=16, max_replies=4)

    evaluate_abs_model(model, abs_collate_fn, eval_data, adapter_names,model.device)
    
    evaluate_padding_error(ranking_collate_fn,abs_collate_fn,model,adapter_names)
    
    config = load_config(args.config_path, "eval_ranking_rm")
    train_data, eval_data = load_dataset(config)
    eval_data = eval_data["oasst_export"]

    idx = 91
    data = eval_data[idx]
    inputs, _ = ranking_collate_fn([data])
    inputs = inputs.to(model.device)
    for adapter_name in adapter_names:
        get_reward(model,inputs,adapter_name)


        

if __name__ == "__main__":
    main()

from tqdm import tqdm
import os
import numpy as np

import torch
import evaluate

from utils import (parse_additional_args, print_yaml_config, 
                   parse_arguments, init_or_resume_from,
                    debug_configurations)
from model_training.training_utils import load_for_inference
from constants import TOKENIZER_SEPECIAL_TOKENS
from training_datasets.dataset_utils import load_sft_dataset, load_rm_dataset
from training_datasets.collators import DialogueDataCollator, RankingDataCollator


accuracy = evaluate.load("accuracy")

def sft_eval(inputs, model,tokenizer):
    targets = inputs.pop("targets")
    labels_mask = inputs.pop("label_masks")
    
    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask", None),
            use_cache=False,
        )
    
    logits = outputs.get("logits")
    targets[~labels_mask.bool()] = -100  # padding_index
    pred_ids = torch.argmax(logits, dim=-1)
    mask = targets > 0
    preds, labels = pred_ids[mask], targets[mask]
    return accuracy.compute(predictions=preds, references=labels)["accuracy"]


def main(conf):
    print(f"\n{'==='*10} Following are the configuration for training{'==='*10}")
    print_yaml_config(conf)
    
    # conf.model_name = "andreaskoepf/llama2-7b-oasst-baseline"
    # conf.init_from_adapter = None
    device_map = "auto"#"{"":0}"
    assert "llama" in conf.model_name.lower(), "Currently only llama model supported"
    special_tokens = TOKENIZER_SEPECIAL_TOKENS["llama"]
    model, tokenizer = load_for_inference(device_map,conf,special_tokens,conf.model_name,False)
    _ , eval_ds = load_sft_dataset(conf,special_tokens["eos_token"])

    eval_collate_fn = DialogueDataCollator(
        tokenizer,
        max_length=conf.collator["val_max_length"],
        random_offset_probability=conf.collator["random_offset_probability"],
        label_masking=conf.collator["label_masking"],
        samples_mixing=False,
        use_system_prefix=conf.collator["use_system_prefix"],
        system_prefix=conf.collator["system_prefix"],
        )

    accs_report = {}
    for ds_name, ds in eval_ds.items():
        accs = []
        print(f"{'==='*10} Evaluating the {ds_name} ....")
        for idx in tqdm(range(len(ds))):
            accs.append(sft_eval(eval_collate_fn([ds[idx]]),model,tokenizer))
        accs_report[ds_name] = np.nanmean(accs)
        
    for ds_name,acc in accs_report.items():
        print(f'{ds_name}-->{acc}')

if __name__ == "__main__":
    config, remaining_args = parse_arguments()
    parser = parse_additional_args(config)
    args = parser.parse_args(remaining_args)
    
    init_or_resume_from(args)

    debug_tag = "_dbug" if args.debug else ""
    args.name = f"{args.name}{debug_tag}{args.name_suffix}"
    args.output_dir = os.path.join(args.output_dir, args.name)

    debug_configurations(args)
    main(args)


# vicuna-->nan
# dolly-->0.34272905502755796
# alpaca-->0.48230987207766096
# math_instruction-->0.5356312181236564
# oasst_export-->0.4333112373623429
#Adam hf final checkpoint

# vicuna-->0.7936980995746512
# dolly-->0.656494979677486
# alpaca-->0.8416825592196571
# math_instruction-->0.803523236731632
# oasst_export-->0.6670421535290308

# oasst sft top_1_adam_torch
# vicuna-->0.7945419982194751
# dolly-->0.6451266721597528
# alpaca-->0.8189582698242566
# math_instruction-->0.7943336305230411
# oasst_export-->0.6742018840478846
# oasst_export_top_1-->0.7050264077495915


# oasst sft  adam_torch
# vicuna-->0.7950413907286692
# dolly-->0.6583285087548654
# alpaca-->0.8170551377448529
# math_instruction-->0.7875307433896538
# oasst_export-->0.6776304796980651
# oasst_export_top_1-->0.7074105775829883
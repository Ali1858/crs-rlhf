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

def sft_eval(inputs, model,print_pred,tokenizer):
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

    if print_pred:
        print(f'{"***"*20}')
        print(f'{"***"*20}')
        print(f'{"***"*20}')

        print(f'{"==="*10} input:\n{tokenizer.decode(inputs["input_ids"][0])}')
        print(f'{"==="*10} prediction:\n{tokenizer.decode(preds)}')
        print(f'{"==="*10} target:\n{tokenizer.decode(labels)}')

    return accuracy.compute(predictions=preds, references=labels)["accuracy"]


def main(conf):
    print(f"\n{'==='*10} Following are the configuration for training{'==='*10}")
    print_yaml_config(conf)
    
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
            print_pred = idx%300==0
            accs.append(sft_eval(eval_collate_fn([ds[idx]]),model,print_pred,tokenizer))
        accs_report[ds_name] = np.mean(accs)
        
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

#     vicuna-->nan
# dolly-->0.4526630194201647
# alpaca-->0.6733093753343026
# math_instruction-->0.675246921486663
# oasst_export-->0.3116285181198321

# vicuna-->nan
# dolly-->0.34272905502755796
# alpaca-->0.48230987207766096
# math_instruction-->0.5356312181236564
# oasst_export-->0.4333112373623429

# vicuna-->0.726941048289897
# dolly-->0.5599764270552907
# alpaca-->0.6872182185738254
# math_instruction-->0.6613969404339362
# oasst_export-->0.6695340935405805
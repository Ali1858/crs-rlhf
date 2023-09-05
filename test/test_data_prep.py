from config import SFT_TRAINING_CONFIG,TOKENIZER_CONFIG
from model_training.training_utils import get_sft_tokenizer
from model_training.sft_train import train

def sanity_checks_sft():
    trainer,tokenizer = train()
    dl = trainer.get_train_dataloader()

    item = next(iter(dl))
    
    target = item["targets"][0].view(-1)
    mask = item["label_masks"][0].view(-1).bool()

    print(f'target {tokenizer.decode(target)} -->\n masked target {tokenizer.decode(target[mask])}')

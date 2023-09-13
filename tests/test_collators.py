import unittest
import argparse
import json

from training_datasets.collators import DialogueDataCollator, RankingDataCollator
from model_training.training_utils import get_tokenizer
from model_training import rm_train, sft_train
from utils import read_yaml
from constants import TOKENIZER_SEPECIAL_TOKENS


class TestDialogueDataCollator(unittest.TestCase):

    def setUp(self):
        # Create a tokenizer for testing
        config = {}
        self.maxDiff=None
        
        conf = read_yaml('./configs/config.yaml')
        config.update(conf["sft"])
        config.update(conf["common"])
        config['debug'] = True

        # Create a Namespace object for config
        self.config_ns = argparse.Namespace(**config)

        # self.sanity_checks_sft(self.config_ns)

        self.tokenizer, eos_token= get_tokenizer(self.config_ns,TOKENIZER_SEPECIAL_TOKENS)
        with open("tests/dummy_data.json", "r") as f:
            j = json.load(f)
            self.test_data = j["sft_test_data"]
            self.expected_labels = j["sft_expected_labels"]
            self.expected_targets = j["sft_expected_targets"]
    

    def sanity_checks_sft(self,args):
        trainer = sft_train.main(args,"output_dir")
        tokenizer = trainer.tokenizer

        dl = trainer.get_train_dataloader()

        item = next(iter(dl))
        
        inp = item["input_ids"][0].view(-1)
        target = item["targets"][0].view(-1)
        mask = item["label_masks"][0].view(-1).bool()
        print(f'inp {tokenizer.decode(inp)} -->\n masked target {tokenizer.decode(inp[mask])}')
        print('====')
        print(f'target {tokenizer.decode(target)} -->\n masked target {tokenizer.decode(target[mask])}')


    def test_data_collation(self):
        # Initialize the data collator
        data_collator = DialogueDataCollator(tokenizer=self.tokenizer,
                                              max_length=2048,
                                              label_masking=True,
                                              samples_mixing=True,
                                              mix_probability=0)
    
        # Call the data collator with the test data
        processed_data = data_collator(self.test_data)
        print(f'===shape of output from data collator {processed_data["input_ids"].shape}===')


        self.assertIn("input_ids", processed_data)
        self.assertIn("attention_mask", processed_data)
        self.assertIn("label_masks", processed_data)
        self.assertIn("targets", processed_data)

        # Extract label tokens using the label mask
        label_tokens = []
        target_tokens = []
        for i, label_mask in enumerate(processed_data["label_masks"]):
            input_ids = processed_data["input_ids"][i]
            target = processed_data["targets"][i]

            label_tokens.append([token for token, mask in zip(input_ids, label_mask) if mask])
            target_tokens.append([token for token, mask in zip(target, label_mask) if mask])

        
        # Decode label tokens to text
        decoded_labels = [self.tokenizer.decode(tokens) for tokens in label_tokens]
        decoded_targets = [self.tokenizer.decode(tokens) for tokens in target_tokens]

        self.assertEqual(len(decoded_labels),len(self.expected_labels))
        
        # Assert that the decoded label text matches the expected labels
        for decoded_label, expected_label in zip(decoded_labels, self.expected_labels):
            self.assertEqual(decoded_label, expected_label)

        # Assert that the decoded target text matches the expected labels
        for decoded_label, expected_label in zip(decoded_targets, self.expected_targets):
            self.assertEqual(decoded_label, expected_label)


class TestRankingDataCollator(unittest.TestCase):

    def setUp(self):
        # Create a tokenizer for testing
        config = {}
        self.maxDiff=None
        
        conf = read_yaml('./configs/config.yaml')
        config.update(conf["rm"])
        config.update(conf["common"])
        config['debug'] = True

        # Create a Namespace object for config
        self.config_ns = argparse.Namespace(**config)
        self.config_ns.model_name = self.config_ns.base_model_name

        # self.sanity_checks_rm(self.config_ns)

        self.tokenizer, eos_token= get_tokenizer(self.config_ns,TOKENIZER_SEPECIAL_TOKENS)
        with open("tests/dummy_data.json", "r") as f:
            j = json.load(f)
            self.test_data = j["rm_test_data"]
            self.expected_output = j["rm_expected_output"]
            self.expected_cu_lens = j["rm_expected_cu_lens"]

        
    def sanity_checks_rm(self,args):
            trainer = rm_train.main(args,"output_dir")
            tokenizer = trainer.tokenizer

            dl = trainer.get_train_dataloader()

            batch, cu_lens = next(iter(dl))
            
            inp = batch["input_ids"][0].view(-1)
            print(f' {tokenizer.decode(inp)}')
            print('====')
    

    def test_data_collation(self):
        # Initialize the data collator
        data_collator = RankingDataCollator(tokenizer=self.tokenizer,
                                              max_length=2048,
                                              pad_to_multiple_of=16,)
        
        # Call the data collator with the test data
        batch, cu_lens = data_collator(self.test_data)

        print(f'===shape of output from data collator {batch["input_ids"].shape}===')


        self.assertIn("input_ids", batch)
        self.assertIn("attention_mask", batch)
        
        # Decode label tokens to text
        decoded_input = [self.tokenizer.decode(tokens) for tokens in batch["input_ids"]]
        self.assertEqual(len(decoded_input),len(self.expected_output))
        
        # Assert that the decoded label text matches the expected labels
        for decoded_label, expected_label in zip(decoded_input, self.expected_output):
            parts = [part for part in decoded_label.split('</s>') if part]
            decoded_label = '</s>'.join(parts) + '</s>'
            self.assertEqual(decoded_label, expected_label)

        self.assertEqual(cu_lens,self.expected_cu_lens)

if __name__ == "__main__":
    unittest.main()
    # sanity_checks_rm(config_ns)

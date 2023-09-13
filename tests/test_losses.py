import unittest
import json

import torch
from model_training.losses import RMLoss

class TestRMLoss(unittest.TestCase):

    def setUp(self):

        with open("tests/dummy_data.json", "r") as f:
            j = json.load(f)
            self.text_input = j["rank_expected_output"]
            self.cu_lens = j["rank_expected_cu_lens"]
    

    def test_rm_loss(self):
        rmloss = RMLoss()
        # Initialize an empty list to store logits
        losses = []
        for i,_ in enumerate(self.text_input):
            logits_list = []
            text_input = self.text_input[i]
            cu_lens = self.cu_lens[i]

            for start, end in zip(cu_lens[:-1], cu_lens[1:]):
                num_responses = len(text_input[start:end])

                # logits of each message of the conversation will be in list
                # if len of conversation is 2
                # then --> [[logits 1],[logits 2]]
                # Create logits for this conversation such that they perfectly rank the responses
                conversation_logits = torch.arange(num_responses -1,-1,-1, dtype=torch.float32).reshape(num_responses,-1)
                logits_list.extend(conversation_logits)
            
            logits = torch.stack(logits_list)
            losses.append(rmloss.forward(logits=logits,cu_lengths=cu_lens))
        mean_loss = torch.stack(losses).mean()
        print(mean_loss)
        assert mean_loss-0.26<=0.03




if __name__ == "__main__":
    unittest.main()
 
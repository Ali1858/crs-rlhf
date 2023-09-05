from torch import nn
import torch.nn.functional as F

"""Taken from:
https://github.com/LAION-AI/Open-Assistant/blob/main/model/model_training/utils/losses.py#L57C2-L57C2
"""
class RMLoss(nn.Module):
    def __init__(self, reduction="mean", beta=0.001):
        super().__init__()
        self.reduction = reduction
        self.beta = beta

    def forward(self, logits, cu_lengths=None):
        # if cu_lengths is None, assume that all examples belong to the same conversation
        if cu_lengths is None:
            cu_lengths = [0, logits.size(0)]

        device = logits.device
        losses = []
        for start, end in zip(cu_lengths[:-1], cu_lengths[1:]):
            pairs = torch.combinations(torch.arange(end - start, device=device), 2)
            pos_ids, neg_ids = pairs[:, 0], pairs[:, 1]
            pos_logits = logits.take(start + pos_ids)
            neg_logits = logits.take(start + neg_ids)

            l2 = 0.5 * (pos_logits**2 + neg_logits**2)
            _loss = (-F.logsigmoid(pos_logits - neg_logits) + self.beta * l2).mean()
            losses.append(_loss)
        loss = torch.stack(losses)

        if self.reduction == "none":
            return loss
        return loss.mean()
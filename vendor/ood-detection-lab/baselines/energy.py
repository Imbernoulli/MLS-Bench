"""Energy baseline: free-energy score (Liu et al., NeurIPS 2020).

OOD score = T * logsumexp(logits / T)  (higher = more ID). Unlike MSP this keeps the
absolute logit magnitude (softmax normalises it away), so overconfident-but-wrong OOD
inputs are separated better. Logit-space; T=1 is parameter-free.
"""
import torch

T = 1.0


class Scorer:
    def fit(self, ctx):
        return self

    def score(self, logits):
        # higher logsumexp = more ID (lower energy)
        return (T * torch.logsumexp(logits / T, dim=-1)).numpy()

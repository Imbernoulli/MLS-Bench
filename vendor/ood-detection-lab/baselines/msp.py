"""MSP baseline: Maximum Softmax Probability (Hendrycks & Gimpel, ICLR 2017).

OOD score = max_k softmax(logits)_k. The classic, WEAK post-hoc baseline: overconfident
softmax makes many OOD inputs score as high as ID. Logit-space, parameter-free.
"""
import numpy as np
import torch


class Scorer:
    def fit(self, ctx):
        return self

    def score(self, logits):
        p = torch.softmax(logits, dim=-1)
        return p.max(dim=-1).values.numpy()

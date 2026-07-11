"""Under-parameterized linear mapping baseline (weak).

A single linear layer maps the CLIP vector straight to the flattened prefix with
NO nonlinearity and NO hidden expansion. It lacks the capacity to shape distinct,
grounded prefix tokens, so the frozen GPT-2 decodes vaguer captions -> lower CIDEr.
The weak baseline for the caption-visual-mapping task.
Measured: CIDEr 0.3485, BLEU-4 0.109 (seed 42, 400 steps).
"""
import torch.nn as nn


class _LinearMapping(nn.Module):
    def __init__(self, clip_dim, gpt_dim, prefix_len):
        super().__init__()
        self.prefix_len = prefix_len
        self.gpt_dim = gpt_dim
        self.fc = nn.Linear(clip_dim, gpt_dim * prefix_len)

    def forward(self, x):
        return self.fc(x).view(x.shape[0], self.prefix_len, self.gpt_dim)


def build_mapping(clip_dim, gpt_dim, prefix_len):
    return _LinearMapping(clip_dim, gpt_dim, prefix_len)

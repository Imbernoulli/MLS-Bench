"""MLP-prefix mapping baseline (ClipCap default).

A shallow 2-layer MLP that expands the CLIP image vector into all `prefix_len`
GPT-2 token embeddings at once. Reference: Mokady et al. 2021, "ClipCap: CLIP
Prefix for Image Captioning" (the MLP mapping variant).
"""
import torch.nn as nn


class _MLPMapping(nn.Module):
    def __init__(self, clip_dim, gpt_dim, prefix_len):
        super().__init__()
        self.prefix_len = prefix_len
        self.gpt_dim = gpt_dim
        hidden = (gpt_dim * prefix_len) // 2
        self.fc1 = nn.Linear(clip_dim, hidden)
        self.act = nn.Tanh()
        self.fc2 = nn.Linear(hidden, gpt_dim * prefix_len)

    def forward(self, x):
        h = self.fc2(self.act(self.fc1(x)))
        return h.view(x.shape[0], self.prefix_len, self.gpt_dim)


def build_mapping(clip_dim, gpt_dim, prefix_len):
    return _MLPMapping(clip_dim, gpt_dim, prefix_len)

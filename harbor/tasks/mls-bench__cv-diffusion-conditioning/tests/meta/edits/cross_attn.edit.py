"""Cross-Attention baseline.

Class embedding is used as key/value in cross-attention layers after
each ResBlock. Time embedding is not modified by class info.

This is the method used in Stable Diffusion for text conditioning.
"""

_FILE = "diffusers-main/custom_train.py"

_CROSS_ATTN = '''\
def prepare_conditioning(time_emb, class_emb):
    # Cross-attn: time_emb unchanged, conditioning via ClassConditioner
    return time_emb


class ClassConditioner(nn.Module):
    # Cross-attention: class embedding as key/value
    def __init__(self, channels, cond_dim):
        super().__init__()
        self.cross_attn = CrossAttentionLayer(channels, cond_dim, num_heads=4)

    def forward(self, h, class_emb):
        return self.cross_attn(h, class_emb)
'''

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 195,
        "end_line": 227,
        "content": _CROSS_ATTN,
    },
]

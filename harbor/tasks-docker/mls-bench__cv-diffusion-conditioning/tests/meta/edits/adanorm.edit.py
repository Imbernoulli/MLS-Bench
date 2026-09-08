"""AdaLN-Zero baseline.

Class embedding generates scale/shift/gate parameters for adaptive
LayerNorm after each ResBlock. Time embedding is not modified.

This is the method used in DiT (Diffusion Transformers).
"""

_FILE = "diffusers-main/custom_train.py"

_ADANORM = '''\
def prepare_conditioning(time_emb, class_emb):
    # AdaNorm: time_emb unchanged, conditioning via ClassConditioner
    return time_emb


class ClassConditioner(nn.Module):
    # Adaptive LayerNorm-Zero: class embedding modulates features
    def __init__(self, channels, cond_dim):
        super().__init__()
        self.adaln = AdaLNBlock(channels, cond_dim)

    def forward(self, h, class_emb):
        return self.adaln(h, class_emb)
'''

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 195,
        "end_line": 227,
        "content": _ADANORM,
    },
]

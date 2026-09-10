"""Concat-FiLM baseline.

Class embedding is projected (by UNet's class_time_proj) and added to
the timestep embedding. Conditioning enters ResBlocks via FiLM
(adaptive GroupNorm scale/shift). ClassConditioner is a no-op.

This is the simplest conditioning method.
"""

_FILE = "diffusers-main/custom_train.py"

_CONCAT_FILM = '''\
def prepare_conditioning(time_emb, class_emb):
    # Concat-FiLM: add projected class_emb to time_emb
    return time_emb + class_emb


class ClassConditioner(nn.Module):
    # No-op: all conditioning is via time_emb
    def __init__(self, channels, cond_dim):
        super().__init__()

    def forward(self, h, class_emb):
        return h
'''

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 195,
        "end_line": 227,
        "content": _CONCAT_FILM,
    },
]

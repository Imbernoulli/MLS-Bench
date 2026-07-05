"""Degeneracy check (NOT a real baseline): a decoder that outputs a constant 0.5.

Ignores the encoder features entirely and predicts alpha = 0.5 everywhere. Because
the GT alpha in the unknown band is a genuine soft ramp spanning 0->1 (mean != 0.5),
this scores a large SAD in the unknown band (~ CONST_HALF_SAD) and MAXIMAL gradient
error (a constant has zero gradient vs a ramp) -> it is beaten by every real matting
decoder, confirming the metric is monotone in matting quality.
"""


def build_decoder(enc_channels):
    import torch
    import torch.nn as nn

    class Dec(nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = nn.Parameter(torch.zeros(1))  # keep an optimisable param

        def forward(self, feats):
            e0 = feats[0]
            b = e0.shape[0]
            h, w = e0.shape[-2:]
            return torch.full((b, h, w), 0.5, device=e0.device) + 0.0 * self.bias
    return Dec()

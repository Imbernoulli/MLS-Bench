"""Ternary 1.58-bit quantization baseline (BitNet b1.58).

Weights are quantized to {-1, 0, +1} using absmean-based thresholding:
    w_ternary = round(clip(W / mean(|W|), -1, 1))
    scale = mean(|W|)

This is the "1.58-bit" scheme because log2(3) = 1.58 bits per weight.
The zero value allows the network to effectively gate connections,
providing more expressiveness than pure binary.

Activations are quantized to 8-bit (absmax per-tensor) following the
BitNet b1.58 paper.

Reference: Ma et al., "The Era of 1-bit LLMs: All Large Language Models
are in 1.58 Bits" (2024)
"""

_FILE = "nanoGPT/custom_pretrain.py"

_TERNARY_158 = """\
def weight_quant(weight):
    \"\"\"Ternary quantization: {-1, 0, +1} via absmean with STE.

    Forward: normalize by absmean, round-then-clip to {-1, 0, +1}
    Backward: STE (gradient passes through rounding as identity)
    \"\"\"
    scale = weight.detach().abs().mean().clamp(min=1e-12)
    w_normed = weight / scale
    # STE round: (round(x) - x).detach() + x
    w_q = w_normed.clamp(-1, 1)
    w_q = (w_q.round() - w_q).detach() + w_q
    return w_q, scale


def activation_quant(x):
    \"\"\"Absmax 8-bit activation quantization with STE.

    Quantizes activations to 127 levels (int8 range) using per-tensor
    absmax scaling, following the BitNet b1.58 paper.
    \"\"\"
    Qb = 127  # int8 range
    scale = x.detach().abs().max().clamp(min=1e-12)
    x_normed = x / scale
    x_q = (x_normed * Qb).round().clamp(-Qb, Qb)
    # STE: forward uses quantized, backward passes through
    x_q = (x_q - x_normed * Qb).detach() + x_normed * Qb
    return x_q, scale / Qb


class BitLinear(nn.Module):
    \"\"\"BitNet b1.58 linear layer with ternary {-1, 0, +1} weights.

    During both training and eval: weights are ternarized via absmean
    + round-clip, activations are quantized to int8 range. Output is
    rescaled by weight_scale * activation_scale.
    \"\"\"
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, x):
        w_q, w_scale = weight_quant(self.weight)
        x_q, x_scale = activation_quant(x)
        out = F.linear(x_q, w_q, None)
        out = out * (w_scale * x_scale)
        if self.bias is not None:
            out = out + self.bias
        return out
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 38,
        "end_line": 115,
        "content": _TERNARY_158,
    },
]

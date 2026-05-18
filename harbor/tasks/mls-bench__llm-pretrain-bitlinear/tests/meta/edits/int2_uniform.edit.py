"""2-bit uniform quantization baseline.

Weights are quantized to 4 uniform levels: {-1, -1/3, +1/3, +1} with
per-tensor absmean scaling. This provides 2 bits per weight (log2(4) = 2),
giving more precision than ternary while staying very low-bit.

The quantization grid is symmetric: [-1, -1/3, 1/3, 1], which is the
uniform spacing of 4 levels in [-1, 1]. Weights are normalized by
absmean, scaled to the grid, rounded to nearest level, then rescaled.

Activations are quantized to 8-bit (absmax per-tensor) for consistency
with BitNet baselines.
"""

_FILE = "nanoGPT/custom_pretrain.py"

_INT2_UNIFORM = """\
def weight_quant(weight):
    \"\"\"2-bit uniform quantization: {-1, -1/3, +1/3, +1} with STE.

    Normalizes weights by absmean, maps to 4 uniform levels in [-1, 1],
    then rescales. Uses STE for gradient flow through rounding.
    \"\"\"
    scale = weight.detach().abs().mean().clamp(min=1e-12)
    w_normed = weight / scale
    # Map to [-1.5, 1.5] grid with spacing 1.0, round, then map back
    # Levels: -1.5 -> -1, -0.5 -> -1/3, 0.5 -> 1/3, 1.5 -> 1
    # Multiply by 1.5 so that [-1,1] -> [-1.5,1.5], round, clip to {-1,0,1} range
    # Actually: use 4 uniform levels directly
    # Grid points at: -1, -1/3, 1/3, 1 (spacing = 2/3)
    # Scale so spacing becomes 1: multiply by 3/2
    w_scaled = w_normed * 1.5  # now grid at -1.5, -0.5, 0.5, 1.5
    w_rounded = w_scaled.clamp(-2, 2).round().clamp(-1.5, 1.5)
    # STE: (rounded - scaled).detach() + scaled
    w_q = (w_rounded - w_scaled).detach() + w_scaled
    # Map back: divide by 1.5
    w_q = w_q / 1.5
    return w_q, scale


def activation_quant(x):
    \"\"\"Absmax 8-bit activation quantization with STE.

    Quantizes activations to 127 levels (int8 range) using per-tensor
    absmax scaling.
    \"\"\"
    Qb = 127  # int8 range
    scale = x.detach().abs().max().clamp(min=1e-12)
    x_normed = x / scale
    x_q = (x_normed * Qb).round().clamp(-Qb, Qb)
    # STE: forward uses quantized, backward passes through
    x_q = (x_q - x_normed * Qb).detach() + x_normed * Qb
    return x_q, scale / Qb


class BitLinear(nn.Module):
    \"\"\"Linear layer with 2-bit uniform weight quantization.

    Weights are quantized to {-1, -1/3, +1/3, +1} during both training
    and eval. Activations quantized to int8 range. Output rescaled by
    weight_scale * activation_scale.
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
        "content": _INT2_UNIFORM,
    },
]

"""'2-bit uniform' quantization baseline (as implemented: 5 levels).

The name and the intent are 4 uniform levels {-1, -1/3, +1/3, +1} (2 bits
per weight), but the rounding below maps the absmean-normalized weights to
the 5 levels {-1, -2/3, 0, +2/3, +1} (log2(5) ~= 2.32 bits per weight).

Weights are normalized by per-tensor absmean, multiplied by 1.5, rounded to
the nearest integer, clamped to [-1.5, 1.5] and divided by 1.5; the spacing
is 2/3 around 0 and 1/3 at the ends. The task's 5-level budget admits it.

Activations are quantized to 8-bit (absmax per-tensor) for consistency
with BitNet baselines.
"""

_FILE = "nanoGPT/custom_pretrain.py"

_INT2_UNIFORM = """\
def weight_quant(weight):
    \"\"\"Absmean-scaled rounding to 5 levels {-1, -2/3, 0, +2/3, +1} with STE.

    Normalizes weights by absmean, rounds them to those 5 levels (not 4),
    then rescales. Uses STE for gradient flow through rounding.
    \"\"\"
    scale = weight.detach().abs().mean().clamp(min=1e-12)
    w_normed = weight / scale
    # Scale by 1.5, round to the nearest integer and clamp to [-1.5, 1.5]:
    # the values are {-1.5, -1, 0, 1, 1.5} (+-2 is clamped to +-1.5), i.e.
    # 5 levels {-1, -2/3, 0, 2/3, 1} after the division by 1.5 below. The
    # spacing is not uniform (2/3 around 0, 1/3 at the ends); the 4-level
    # grid {-1, -1/3, 1/3, 1} the name suggests would need rounding to
    # half-integers, which this code does not do.
    w_scaled = w_normed * 1.5
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
    \"\"\"Linear layer with 5-level weight quantization (see weight_quant).

    Weights are quantized to {-1, -2/3, 0, +2/3, +1} during both training
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

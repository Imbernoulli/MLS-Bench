"""Binary sign quantization baseline (original BitNet).

Weights are binarized to {-1, +1} using the sign function with a per-tensor
absolute-mean scale factor. The forward pass uses:
    w_binary = sign(W), scale = mean(|W|)
    output = scale * F.linear(x, w_binary)

Gradients flow through via the Straight-Through Estimator (STE): the sign
function's gradient is treated as identity during backprop.

Activations are quantized to 8-bit (absmax per-tensor) following the
original BitNet paper.

Reference: Wang et al., "BitNet: Scaling 1-bit Transformers for Large
Language Models" (2023)
"""

_FILE = "nanoGPT/custom_pretrain.py"

_BINARY_SIGN = """\
def weight_quant(weight):
    \"\"\"Binary quantization: sign(W) * mean(|W|) with STE.

    Forward: w_q = sign(W), scale = mean(|W|)
    Backward: STE (gradient passes through sign as identity)
    \"\"\"
    scale = weight.detach().abs().mean()
    # STE: forward uses sign, backward treats sign as identity
    w_q = (weight.sign() - weight).detach() + weight
    return w_q, scale


def activation_quant(x):
    \"\"\"Absmax 8-bit activation quantization with STE.

    Quantizes activations to 127 levels (int8 range) using per-tensor
    absmax scaling, following the original BitNet paper.
    \"\"\"
    Qb = 127  # int8 range
    scale = x.detach().abs().max().clamp(min=1e-12)
    x_normed = x / scale
    x_q = (x_normed * Qb).round().clamp(-Qb, Qb)
    # STE: forward uses quantized, backward passes through
    x_q = (x_q - x_normed * Qb).detach() + x_normed * Qb
    return x_q, scale / Qb


class BitLinear(nn.Module):
    \"\"\"BitNet linear layer with binary {-1, +1} weights.

    During both training and eval: weights are binarized via sign function,
    activations are quantized to int8 range. Output is rescaled by
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
        "content": _BINARY_SIGN,
    },
]

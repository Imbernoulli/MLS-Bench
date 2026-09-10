"""Round-to-Nearest (RTN) baseline -- simplest post-training quantization.

Quantizes each weight independently using symmetric per-channel quantization.
No calibration data is used -- just round each weight to the nearest INT4 value.
This is the weakest baseline but has zero overhead.

Reference: Jacob et al., "Quantization and Training of Neural Networks for
Efficient Integer-Arithmetic-Only Inference" (CVPR 2018)
"""

_FILE = "gptq/custom_ptq.py"

_RTN_CODE = """\

# ── Helper: basic quantize/dequantize primitives ──────────────────────────────

def quantize_tensor(x, scale, zero_point, qmin, qmax):
    \"\"\"Quantize a float tensor to integers given scale and zero point.\"\"\"
    x_int = torch.clamp(torch.round(x / scale) + zero_point, qmin, qmax)
    return x_int


def dequantize_tensor(x_int, scale, zero_point):
    \"\"\"Dequantize integer tensor back to float.\"\"\"
    return (x_int - zero_point) * scale


def find_scale_zero(weight, num_bits=4, group_size=-1, symmetric=True):
    \"\"\"Compute per-channel (or per-group) quantization parameters.\"\"\"
    qmin = -(1 << (num_bits - 1))
    qmax = (1 << (num_bits - 1)) - 1

    if group_size > 0:
        out_features, in_features = weight.shape
        assert in_features % group_size == 0
        w_groups = weight.reshape(out_features, -1, group_size)
        if symmetric:
            w_max = w_groups.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
            scale = w_max / qmax
            zero_point = torch.zeros_like(scale)
        else:
            w_min = w_groups.amin(dim=-1, keepdim=True)
            w_max = w_groups.amax(dim=-1, keepdim=True)
            w_range = (w_max - w_min).clamp(min=1e-12)
            scale = w_range / (qmax - qmin)
            zero_point = torch.round(qmin - w_min / scale)
        scale = scale.reshape(out_features, -1).repeat_interleave(group_size, dim=1)
        zero_point = zero_point.reshape(out_features, -1).repeat_interleave(group_size, dim=1)
    else:
        if symmetric:
            w_max = weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
            scale = w_max / qmax
            zero_point = torch.zeros_like(scale)
        else:
            w_min = weight.amin(dim=1, keepdim=True)
            w_max = weight.amax(dim=1, keepdim=True)
            w_range = (w_max - w_min).clamp(min=1e-12)
            scale = w_range / (qmax - qmin)
            zero_point = torch.round(qmin - w_min / scale)

    return scale, zero_point, qmin, qmax


class LayerQuantizer:
    \"\"\"RTN quantizer -- simple round-to-nearest, ignores calibration data.\"\"\"

    def __init__(self, layer, num_bits=4, group_size=-1):
        self.layer = layer
        self.num_bits = num_bits
        self.group_size = group_size
        self.out_features, self.in_features = layer.weight.shape
        self.dev = layer.weight.device
        self.nsamples = 0
        self.H = torch.zeros(
            (self.in_features, self.in_features),
            device=self.dev, dtype=torch.float32
        )

    def add_batch(self, inp):
        \"\"\"Collect calibration data (unused in RTN, kept for interface).\"\"\"
        if inp.dim() == 3:
            inp = inp.reshape(-1, inp.shape[-1])
        self.nsamples += inp.shape[0]

    def quantize(self):
        \"\"\"RTN: symmetric per-channel (or per-group) round-to-nearest.\"\"\"
        W = self.layer.weight.data.clone().float()
        scale, zero_point, qmin, qmax = find_scale_zero(
            W, num_bits=self.num_bits, group_size=self.group_size, symmetric=True
        )
        W_q = quantize_tensor(W, scale, zero_point, qmin, qmax)
        W_dq = dequantize_tensor(W_q, scale, zero_point)
        return W_dq.to(self.layer.weight.dtype)

    def free(self):
        \"\"\"Release calibration buffers.\"\"\"
        del self.H
        self.H = None

"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 26,
        "end_line": 157,
        "content": _RTN_CODE,
    },
]

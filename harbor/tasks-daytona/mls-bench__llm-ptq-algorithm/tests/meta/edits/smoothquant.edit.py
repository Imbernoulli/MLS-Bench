"""SmoothQuant baseline -- activation-aware weight smoothing + RTN.

Migrates quantization difficulty from activations to weights by applying
per-channel scaling factors derived from calibration data. For each linear
layer, computes per-input-channel activation magnitudes, then divides weights
by a smoothing factor s^alpha (where s = max|activation| per channel) before
applying standard RTN quantization.

This makes the weight distribution more uniform and easier to quantize,
at the cost of making activation quantization harder (but we only quantize
weights here, so it is a net win).

Reference: Xiao et al., "SmoothQuant: Accurate and Efficient Post-Training
Quantization for Large Language Models" (ICML 2023)
"""

_FILE = "gptq/custom_ptq.py"

_SMOOTHQUANT_CODE = """\

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
    \"\"\"SmoothQuant quantizer -- smooth weights using activation stats, then RTN.

    During calibration, collects per-channel max activation magnitudes.
    At quantization time, divides weights by s^alpha (where s is the
    activation scale) to make weight distribution more uniform, then
    applies standard RTN quantization.
    \"\"\"

    ALPHA = 0.5  # Migration strength: 0=all on weights, 1=all on acts

    def __init__(self, layer, num_bits=4, group_size=-1):
        self.layer = layer
        self.num_bits = num_bits
        self.group_size = group_size
        self.out_features, self.in_features = layer.weight.shape
        self.dev = layer.weight.device
        self.nsamples = 0
        # Track per-channel activation magnitude
        self.act_max = torch.zeros(
            self.in_features, device=self.dev, dtype=torch.float32
        )
        # Also keep H for interface compatibility
        self.H = torch.zeros(
            (self.in_features, self.in_features),
            device=self.dev, dtype=torch.float32
        )

    def add_batch(self, inp):
        \"\"\"Track per-channel max activation magnitude for smoothing.\"\"\"
        if inp.dim() == 3:
            inp = inp.reshape(-1, inp.shape[-1])
        n = inp.shape[0]
        inp = inp.float()
        # Update running max of absolute activations per channel
        batch_max = inp.abs().amax(dim=0)
        self.act_max = torch.max(self.act_max, batch_max)
        self.nsamples += n

    def quantize(self):
        \"\"\"SmoothQuant: smooth weights by activation scale, then RTN.\"\"\"
        W = self.layer.weight.data.clone().float()

        # Compute smoothing factor: s = act_max^alpha
        s = self.act_max.clamp(min=1e-12).pow(self.ALPHA)

        # Smooth the weights: W_smooth = W / s (per input channel)
        W_smooth = W / s.unsqueeze(0)

        # Apply standard RTN quantization on smoothed weights
        scale, zero_point, qmin, qmax = find_scale_zero(
            W_smooth, num_bits=self.num_bits, group_size=self.group_size, symmetric=True
        )
        W_q = quantize_tensor(W_smooth, scale, zero_point, qmin, qmax)
        W_dq = dequantize_tensor(W_q, scale, zero_point)

        # Undo smoothing: W_final = W_dq * s
        W_final = W_dq * s.unsqueeze(0)
        return W_final.to(self.layer.weight.dtype)

    def free(self):
        \"\"\"Release calibration buffers.\"\"\"
        del self.H
        del self.act_max
        self.H = None
        self.act_max = None

"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 26,
        "end_line": 157,
        "content": _SMOOTHQUANT_CODE,
    },
]

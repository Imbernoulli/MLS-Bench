"""GPTQ baseline -- Hessian-based error compensation quantization.

Uses calibration data to compute a Hessian approximation (H = X^T X) for
each linear layer, then quantizes weights column-by-column, using the
Hessian inverse to optimally redistribute quantization error to remaining
columns. This minimizes the layer-wise output error ||WX - W_q X||^2.

Processes columns in blocks for efficiency. Uses Cholesky decomposition
of the Hessian for numerically stable inverse computation.

Reference: Frantar et al., "GPTQ: Accurate Post-Training Quantization for
Generative Pre-trained Transformers" (ICLR 2023)
"""

_FILE = "gptq/custom_ptq.py"

_GPTQ_CODE = """\

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
    \"\"\"GPTQ quantizer -- Hessian-based error compensation.

    Collects input activation statistics (H = X^T X), then quantizes
    weights column-by-column, compensating for quantization error using
    the Hessian inverse so that layer output error is minimized.
    \"\"\"

    BLOCK_SIZE = 128
    PERCDAMP = 0.01

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
        \"\"\"Accumulate Hessian approximation from calibration inputs.\"\"\"
        if inp.dim() == 3:
            inp = inp.reshape(-1, inp.shape[-1])
        n = inp.shape[0]
        inp = inp.float()
        self.H += inp.T @ inp
        self.nsamples += n

    def quantize(self):
        \"\"\"GPTQ: column-by-column quantization with Hessian error compensation.\"\"\"
        W = self.layer.weight.data.clone().float()
        H = self.H.clone()

        if self.nsamples > 0:
            H /= self.nsamples

        num_bits = self.num_bits
        group_size = self.group_size
        qmin = -(1 << (num_bits - 1))
        qmax = (1 << (num_bits - 1)) - 1

        # Add dampening to diagonal for numerical stability
        damp = self.PERCDAMP * torch.mean(torch.diag(H))
        H += damp * torch.eye(self.in_features, device=self.dev)

        # Compute Hessian inverse via Cholesky decomposition
        try:
            L = torch.linalg.cholesky(H)
            Hinv = torch.cholesky_inverse(L)
        except Exception:
            # Fallback to pseudo-inverse if Cholesky fails
            Hinv = torch.linalg.pinv(H)

        Q = torch.zeros_like(W)
        Err = torch.zeros_like(W)

        # Process columns in blocks
        for col_start in range(0, self.in_features, self.BLOCK_SIZE):
            col_end = min(col_start + self.BLOCK_SIZE, self.in_features)

            W_block = W[:, col_start:col_end].clone()
            Hinv_block_diag = torch.diag(
                Hinv[col_start:col_end, col_start:col_end]
            )

            for j in range(col_end - col_start):
                col = col_start + j
                w_col = W_block[:, j]

                # Compute scale: per-group if group_size > 0, else per-column
                if group_size > 0 and col % group_size == 0:
                    g_end = min(col + group_size, self.in_features)
                    W_group = W[:, col:g_end]
                    g_max = W_group.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
                    group_scale = (g_max / qmax).squeeze(1)

                if group_size > 0:
                    scale = group_scale
                else:
                    w_abs_max = w_col.abs().max().clamp(min=1e-12)
                    scale = w_abs_max / qmax

                # Quantize and dequantize
                q_col = torch.clamp(
                    torch.round(w_col / scale), qmin, qmax
                ) * scale
                Q[:, col] = q_col

                # Error compensation: distribute error weighted by Hessian
                err = (w_col - q_col) / Hinv_block_diag[j].clamp(min=1e-12)
                Err[:, col] = err

                # Update remaining columns in block
                if j + 1 < col_end - col_start:
                    W_block[:, j+1:] -= (
                        err.unsqueeze(1)
                        * Hinv[col, col_start+j+1:col_end].unsqueeze(0)
                    )

            # Propagate error to remaining columns outside block
            if col_end < self.in_features:
                W[:, col_end:] -= (
                    Err[:, col_start:col_end]
                    @ Hinv[col_start:col_end, col_end:]
                )

        return Q.to(self.layer.weight.dtype)

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
        "content": _GPTQ_CODE,
    },
]

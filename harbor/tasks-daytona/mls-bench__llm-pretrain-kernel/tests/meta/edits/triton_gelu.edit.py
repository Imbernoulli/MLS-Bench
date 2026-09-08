"""Triton fused GELU MLP kernel baseline (medium).

Fuses the linear->GELU->linear MLP into a single Triton kernel,
reducing memory bandwidth by avoiding materializing intermediate activations.

Reference: Tillet et al., "Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations" (2019)
"""

_FILE = "nanoGPT/custom_pretrain.py"

_TRITON_GELU = """\
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

@triton.jit
def _fused_gelu_kernel(
    x_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # Compute entirely in float32 to avoid bfloat16 overflow in x^3
    xf = x.to(tl.float32)
    # tanh-approximation GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    c = 0.7978845608028654  # sqrt(2/pi)
    inner = c * (xf + 0.044715 * xf * xf * xf)
    tanh_val = libdevice.tanh(inner)
    out = xf * 0.5 * (1.0 + tanh_val)
    tl.store(out_ptr + offsets, out.to(x.dtype), mask=mask)

class _TritonGELUMLP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w_fc, w_proj):
        h = x @ w_fc.t()
        act = torch.empty_like(h)
        n = h.numel()
        BLOCK = 1024
        grid = ((n + BLOCK - 1) // BLOCK,)
        _fused_gelu_kernel[grid](h, act, n, BLOCK_SIZE=BLOCK)
        out = act @ w_proj.t()
        ctx.save_for_backward(x, w_fc, w_proj, h, act)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, w_fc, w_proj, h, act = ctx.saved_tensors
        dtype = grad_output.dtype
        d_act = grad_output @ w_proj.to(dtype)
        grad_w_proj = grad_output.reshape(-1, grad_output.shape[-1]).t() @ act.to(dtype).reshape(-1, act.shape[-1])
        # Analytical gradient of tanh-approximation GELU (matches the Triton forward)
        # gelu(x) = 0.5 * x * (1 + tanh(inner)), inner = c * (x + 0.044715 * x^3)
        # d_gelu/dx = 0.5 * (1 + tanh(inner)) + 0.5 * x * sech^2(inner) * d_inner/dx
        # d_inner/dx = c * (1 + 3 * 0.044715 * x^2)
        h_f = h.float()
        c = 0.7978845608028654
        inner = c * (h_f + 0.044715 * h_f * h_f * h_f)
        tanh_inner = torch.tanh(inner)
        sech2 = 1.0 - tanh_inner * tanh_inner
        d_inner = c * (1.0 + 3.0 * 0.044715 * h_f * h_f)
        gelu_grad = 0.5 * (1.0 + tanh_inner) + 0.5 * h_f * sech2 * d_inner
        d_h = (d_act.float() * gelu_grad).to(dtype)
        grad_x = d_h @ w_fc.to(dtype)
        grad_w_fc = d_h.reshape(-1, d_h.shape[-1]).t() @ x.to(dtype).reshape(-1, x.shape[-1])
        return grad_x, grad_w_fc, grad_w_proj

def fused_mlp_forward(x, w_fc, w_proj):
    \"\"\"MLP forward with Triton fused GELU kernel.\"\"\"
    return _TritonGELUMLP.apply(x, w_fc, w_proj)
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 33,
        "end_line": 48,
        "content": _TRITON_GELU,
    },
]

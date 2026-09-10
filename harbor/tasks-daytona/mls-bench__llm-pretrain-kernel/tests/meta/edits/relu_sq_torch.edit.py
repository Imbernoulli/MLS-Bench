"""ReLU-squared MLP with torch custom autograd baseline (basic).

Replaces GELU with ReLU^2 activation using a custom autograd Function
that saves pre-activation values for efficient backward pass.
No Triton required — pure PyTorch but faster than GELU.

Reference: So et al., "Primer: Searching for Efficient Transformers" (2021)
"""

_FILE = "nanoGPT/custom_pretrain.py"

_RELU_SQ_TORCH = """\
def fused_mlp_forward(x, w_fc, w_proj):
    \"\"\"MLP forward with ReLU^2 activation via custom autograd.\"\"\"

    class ReLUSquaredMLP(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, w_fc, w_proj):
            h = x @ w_fc.t()
            relu_h = F.relu(h)
            act = relu_h * relu_h  # ReLU^2
            out = act @ w_proj.t()
            ctx.save_for_backward(x, w_fc, w_proj, h, relu_h)
            return out

        @staticmethod
        def backward(ctx, grad_output):
            x, w_fc, w_proj, h, relu_h = ctx.saved_tensors
            dtype = grad_output.dtype
            # grad through second linear
            d_act = grad_output @ w_proj.to(dtype)
            # grad through ReLU^2: d/dx[relu(x)^2] = 2*relu(x) * (x > 0)
            d_h = 2 * relu_h.to(dtype) * d_act
            # weight grads
            act_sq = (relu_h * relu_h).to(dtype)
            grad_w_proj = grad_output.t() @ act_sq
            grad_w_fc = d_h.t() @ x.to(dtype)
            grad_x = d_h @ w_fc.to(dtype)
            return grad_x, grad_w_fc, grad_w_proj

    return ReLUSquaredMLP.apply(x, w_fc, w_proj)
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 33,
        "end_line": 48,
        "content": _RELU_SQ_TORCH,
    },
]

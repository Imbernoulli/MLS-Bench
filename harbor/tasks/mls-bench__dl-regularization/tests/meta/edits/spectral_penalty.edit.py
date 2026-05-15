"""Spectral norm penalty baseline.

Adds a penalty on the spectral norm (largest singular value) of
convolutional weight matrices, encouraging Lipschitz-constrained
feature maps and smoother decision boundaries.

Reference: Miyato et al., "Spectral Normalization for Generative
Adversarial Networks" (ICLR 2018) -- adapted as a soft penalty for
classification instead of hard normalization.

Ops ordered bottom-to-top for line stability.
"""

_FILE = "pytorch-vision/custom_reg.py"

_CONTENT = """\
def compute_regularization(model, inputs, outputs, targets, config):
    \"\"\"Spectral norm penalty on conv weight matrices.

    Estimates the largest singular value of each Conv2d weight using
    one step of power iteration (efficient approximation) and penalizes
    their sum. This encourages Lipschitz smoothness.

    lambda=1e-4, one-step power iteration for efficiency.

    Reference: Miyato et al., ICLR 2018 (adapted as soft penalty).
    \"\"\"
    lam = 1e-4
    reg = torch.tensor(0.0, device=outputs.device)
    count = 0

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            w = m.weight
            # Reshape to 2D: (out_channels, in_channels * kH * kW)
            w_mat = w.reshape(w.size(0), -1)
            # One-step power iteration to estimate spectral norm
            with torch.no_grad():
                u = torch.randn(w_mat.size(0), device=w.device)
                u = u / (u.norm() + 1e-12)
            v = w_mat.t() @ u
            v = v / (v.norm() + 1e-12)
            sigma = u @ (w_mat @ v)
            reg = reg + sigma
            count += 1

    if count > 0:
        reg = reg / count
    return lam * reg
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 246,
        "end_line": 273,
        "content": _CONTENT,
    },
]

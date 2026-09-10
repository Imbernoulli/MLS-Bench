"""Orthogonal regularization baseline.

Penalizes deviation of convolutional weight matrices from orthogonality
by minimizing ||W^T W - I||_F^2, with coefficient 1e-4.

Reference: Brock et al., "Neural Photo Editing with Introspective
Adversarial Networks" (2017)

Ops ordered bottom-to-top for line stability.
"""

_FILE = "pytorch-vision/custom_reg.py"

_CONTENT = """\
def compute_regularization(model, inputs, outputs, targets, config):
    \"\"\"Orthogonal regularization on convolutional weights.

    Penalizes deviation from orthogonality: ||W^T W - I||_F^2 for each
    4D conv weight reshaped to [out_channels, in*k*k]. Coefficient=1e-4.
    \"\"\"
    reg = torch.tensor(0.0, device=outputs.device)
    for name, p in model.named_parameters():
        if 'conv' in name and 'weight' in name and p.dim() == 4:
            W = p.view(p.size(0), -1)  # [out, in*k*k]
            WtW = W @ W.t()
            I = torch.eye(W.size(0), device=W.device)
            reg = reg + ((WtW - I) ** 2).sum()
    return 1e-4 * reg
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

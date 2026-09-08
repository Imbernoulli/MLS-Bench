"""Baseline: scale + opacity L1 penalty (gsplat default regularizer).

Reference: gsplat/examples/simple_trainer.py — `opacity_reg` and `scale_reg`
config options (default coefficient 0.01 each, enabled in several gsplat
training presets).

Encourages:
  - smaller Gaussians (via |exp(scale)| penalty), limiting floaters
  - sparse opacity (via |sigmoid(opacity)| penalty), encouraging prunable
    background Gaussians
"""

_FILE = "gsplat/custom_regularizer.py"

_SCALE_OPA = '''
SCALE_REG = 1e-2
OPACITY_REG = 1e-2

def compute_regularizer(splats, step, scene_scale):
    """L1 penalty on per-Gaussian scale and opacity."""
    scale_loss = torch.abs(torch.exp(splats["scales"])).mean()
    opa_loss = torch.abs(torch.sigmoid(splats["opacities"])).mean()
    return SCALE_REG * scale_loss + OPACITY_REG * opa_loss
'''

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 37,
        "end_line": 51,
        "content": _SCALE_OPA,
    },
]

"""Focal Loss baseline.

Down-weights easy (well-classified) examples by modulating CE with (1-pt)^gamma.
Uses gamma=2.0.

Reference: Lin et al., "Focal Loss for Dense Object Detection" (ICCV 2017)

Ops ordered bottom-to-top for line stability.
"""

_FILE = "pytorch-vision/custom_loss.py"

_CONTENT = """\
def compute_loss(logits, targets, config):
    \"\"\"Focal Loss (gamma=2.0).

    Modulates CE by (1-pt)^gamma to focus on hard examples,
    reducing the relative loss for well-classified samples.
    \"\"\"
    ce = F.cross_entropy(logits, targets, reduction='none')
    pt = torch.exp(-ce)
    return ((1 - pt) ** 2.0 * ce).mean()
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 246,
        "end_line": 266,
        "content": _CONTENT,
    },
]

"""Cosine annealing learning rate schedule baseline.

Smoothly decays the learning rate from base_lr to 0 following a cosine curve.
Simple, no warmup, widely used as a default schedule.

Reference: Loshchilov & Hutter, "SGDR: Stochastic Gradient Descent with
Warm Restarts" (ICLR 2017)

Ops ordered bottom-to-top for line stability.
"""

_FILE = "pytorch-vision/custom_schedule.py"

_CONTENT = """\
def get_lr(epoch, total_epochs, base_lr, config):
    \"\"\"Cosine annealing from base_lr to 0.

    LR = base_lr * 0.5 * (1 + cos(pi * epoch / total_epochs))
    \"\"\"
    return base_lr * 0.5 * (1 + math.cos(math.pi * epoch / total_epochs))
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 246,
        "end_line": 269,
        "content": _CONTENT,
    },
]

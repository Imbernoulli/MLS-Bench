"""Warmup + cosine annealing learning rate schedule baseline.

Linear warmup for 5 epochs followed by cosine decay to 0.
Stabilizes early training before applying smooth decay.

Reference: Goyal et al., "Accurate, Large Minibatch SGD: Training ImageNet
in 1 Hour" (arXiv 2017)

Ops ordered bottom-to-top for line stability.
"""

_FILE = "pytorch-vision/custom_schedule.py"

_CONTENT = """\
def get_lr(epoch, total_epochs, base_lr, config):
    \"\"\"Linear warmup (5 epochs) then cosine decay to 0.

    Epochs 0-4: linearly ramp from base_lr/5 to base_lr.
    Epochs 5+: cosine anneal from base_lr to 0.
    \"\"\"
    warmup = 5
    if epoch < warmup:
        return base_lr * (epoch + 1) / warmup
    progress = (epoch - warmup) / (total_epochs - warmup)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))
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

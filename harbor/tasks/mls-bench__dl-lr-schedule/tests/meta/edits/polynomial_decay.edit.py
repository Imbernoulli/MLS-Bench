"""Polynomial decay learning rate schedule baseline.

Decays learning rate polynomially from base_lr to end_lr with power=2.0
(quadratic). Includes a 5-epoch linear warmup.

Reference: Standard schedule used widely in practice (e.g., Goyal et al.,
"Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour", 2017).

Ops ordered bottom-to-top for line stability.
"""

_FILE = "pytorch-vision/custom_schedule.py"

_CONTENT = """\
def get_lr(epoch, total_epochs, base_lr, config):
    \"\"\"Polynomial decay (power=2.0) with 5-epoch linear warmup.

    Warmup: linear from 0 to base_lr over 5 epochs.
    Decay: base_lr * (1 - t)^2 where t = (epoch - warmup) / (total - warmup).
    \"\"\"
    warmup_epochs = 5
    power = 2.0
    end_lr = 1e-5

    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    else:
        t = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs - 1, 1)
        t = min(t, 1.0)
        return end_lr + (base_lr - end_lr) * (1 - t) ** power
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

"""Trapezoidal learning rate schedule baseline (medium).

Standard trapezoidal schedule: linear warmup, constant plateau, linear cooldown to min_lr.
Simple and effective alternative to cosine annealing.

Reference: Hu et al., "MiniCPM: Unveiling the Potential of Small Language
Models with Scalable Training Strategies" (2024), warmup-stable-decay schedule.
"""

_FILE = "nanoGPT/custom_pretrain.py"

_TRAPEZOIDAL_SCHEDULE = """\
def get_lr(it, warmup_iters, lr_decay_iters, learning_rate, min_lr):
    \"\"\"Trapezoidal learning rate schedule: warmup, constant plateau, cooldown.\"\"\"
    if it > lr_decay_iters:
        return min_lr
    # Warmup phase
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # Cooldown phase: last 40% of training
    cooldown_start = int(lr_decay_iters * 0.6)
    if it >= cooldown_start:
        t = (it - cooldown_start) / (lr_decay_iters - cooldown_start)
        return min_lr + (learning_rate - min_lr) * (1.0 - t)
    # Constant plateau
    return learning_rate
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 191,
        "end_line": 201,
        "content": _TRAPEZOIDAL_SCHEDULE,
    },
]

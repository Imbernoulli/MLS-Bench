"""WSD with sqrt decay baseline (strongest).

WSD schedule but with inverse-sqrt decay phase instead of linear decay,
providing a gentler transition that often preserves more learning signal.
Also uses a longer warmup (6% vs default 4%).

Reference: Hu et al., "MiniCPM: Unveiling the Potential of Small Language
Models with Scalable Training Strategies" (2024), warmup-stable-decay schedule.
"""

_FILE = "nanoGPT/custom_pretrain.py"

_WSD_SQRT_SCHEDULE = """\
def get_lr(it, warmup_iters, lr_decay_iters, learning_rate, min_lr):
    \"\"\"WSD with inverse-sqrt decay and extended warmup.\"\"\"
    import math
    # Extended warmup: 6% of training
    effective_warmup = max(warmup_iters, int(lr_decay_iters * 0.06))
    if it < effective_warmup:
        return learning_rate * (it + 1) / (effective_warmup + 1)
    # Decay phase: last 20% uses inverse-sqrt cooldown
    decay_start = int(lr_decay_iters * 0.8)
    if it >= decay_start:
        decay_len = lr_decay_iters - decay_start
        t = (it - decay_start) / decay_len
        # Inverse-sqrt decay: gentler than linear, reaches min_lr at t=1
        coeff = (1.0 / math.sqrt(1.0 + 9.0 * t) - 1.0 / math.sqrt(10.0)) / (1.0 - 1.0 / math.sqrt(10.0))
        return min_lr + (learning_rate - min_lr) * coeff
    # Stable phase
    return learning_rate
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 191,
        "end_line": 201,
        "content": _WSD_SQRT_SCHEDULE,
    },
]

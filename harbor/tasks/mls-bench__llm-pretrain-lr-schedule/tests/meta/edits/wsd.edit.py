"""WSD (Warmup-Stable-Decay) learning rate schedule baseline (basic).

Three-phase schedule: linear warmup, constant stable phase, linear decay.
Simpler than cosine and often equally effective.

Reference: Hu et al., "Minicpm: Unveiling the potential of small language models" (2024)
"""

_FILE = "nanoGPT/custom_pretrain.py"

_WSD_SCHEDULE = """\
def get_lr(it, warmup_iters, lr_decay_iters, learning_rate, min_lr):
    \"\"\"WSD (Warmup-Stable-Decay) learning rate schedule.\"\"\"
    # Warmup phase
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # Decay phase: last 20% of training
    decay_start = int(lr_decay_iters * 0.8)
    if it >= decay_start:
        decay_ratio = (it - decay_start) / (lr_decay_iters - decay_start)
        return min_lr + (learning_rate - min_lr) * (1.0 - decay_ratio)
    # Stable phase: constant LR
    return learning_rate
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 191,
        "end_line": 201,
        "content": _WSD_SCHEDULE,
    },
]

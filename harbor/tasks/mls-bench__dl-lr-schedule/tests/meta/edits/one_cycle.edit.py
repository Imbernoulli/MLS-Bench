"""OneCycleLR schedule baseline.

Uses a single cosine cycle that ramps up to max_lr then anneals down,
with the peak at 30% of training. Final phase decays to base_lr/25.

Reference: Smith & Topin, "Super-Convergence: Very Fast Training of Neural
Networks Using Large Learning Rates" (2019)

Ops ordered bottom-to-top for line stability.
"""

_FILE = "pytorch-vision/custom_schedule.py"

_CONTENT = """\
def get_lr(epoch, total_epochs, base_lr, config):
    \"\"\"OneCycleLR schedule (Smith & Topin, 2019).

    Phase 1 (0-30%): cosine warmup from base_lr/25 to base_lr.
    Phase 2 (30-100%): cosine anneal from base_lr to base_lr/25.
    \"\"\"
    pct_start = 0.3
    div_factor = 25.0
    final_div = 25.0

    min_lr = base_lr / div_factor
    final_lr = base_lr / final_div

    progress = epoch / max(total_epochs - 1, 1)

    if progress <= pct_start:
        # Warmup phase: cosine from min_lr to base_lr
        t = progress / pct_start
        return min_lr + (base_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * (1 - t)))
    else:
        # Anneal phase: cosine from base_lr to final_lr
        t = (progress - pct_start) / (1 - pct_start)
        return final_lr + (base_lr - final_lr) * 0.5 * (1 + math.cos(math.pi * t))
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

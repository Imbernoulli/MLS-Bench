"""SOTA schedule: linear WARMUP (first 10% of steps) then cosine decay.
Warmup stabilises early updates on ImageNet-pretrained weights, cosine decay
sharpens convergence (Luo "Bag of Tricks" 2019). Reference:
vendor/torchreid-reid/baselines/optim_warmup_cosine.py
"""
_FILE = "torchreid-reid/solution/schedule.py"
_CONTENT = '''def build_lr_schedule(total_steps):
    import math

    peak = 3.5e-4
    warm = max(1, int(0.1 * total_steps))

    def lr_at_step(step):
        if step < warm:
            return peak * (step + 1) / warm            # linear warmup 0 -> peak
        # cosine decay from peak -> ~0 over the remaining steps
        prog = (step - warm) / max(1, total_steps - warm)
        return peak * 0.5 * (1.0 + math.cos(math.pi * prog))

    lr_at_step.name = "warmup_cosine"
    return lr_at_step'''
OPS = [
    {"op": "replace", "file": _FILE, "start_line": 6, "end_line": 13, "content": _CONTENT},
]

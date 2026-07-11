"""Weak schedule: FLAT constant LR (no warmup, no decay).
Reference: vendor/torchreid-reid/baselines/optim_flat.py
"""
_FILE = "torchreid-reid/solution/schedule.py"
_CONTENT = '''def build_lr_schedule(total_steps):
    peak = 3.5e-4

    def lr_at_step(step):
        return peak

    lr_at_step.name = "flat_lr"
    return lr_at_step'''
OPS = [
    {"op": "replace", "file": _FILE, "start_line": 6, "end_line": 13, "content": _CONTENT},
]

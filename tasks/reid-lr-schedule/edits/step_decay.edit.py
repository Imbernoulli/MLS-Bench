"""Medium schedule: step decay, NO warmup (LR starts at peak, then drops).
Reference: vendor/torchreid-reid/baselines/optim_stepdecay.py
"""
_FILE = "torchreid-reid/solution/schedule.py"
_CONTENT = '''def build_lr_schedule(total_steps):
    peak = 3.5e-4

    def lr_at_step(step):
        # drop by 10x at 50% and again at 75% -- no warmup
        if step < total_steps * 0.5:
            return peak
        if step < total_steps * 0.75:
            return peak * 0.1
        return peak * 0.01

    lr_at_step.name = "step_decay"
    return lr_at_step'''
OPS = [
    {"op": "replace", "file": _FILE, "start_line": 6, "end_line": 13, "content": _CONTENT},
]

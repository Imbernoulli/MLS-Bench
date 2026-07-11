"""Medium schedule: step decay, NO warmup (LR starts at peak, then drops).
Reference: vendor/torchreid-reid/baselines/optim_stepdecay.py

Standalone reference implementation of this baseline (also applied as an edit).
"""


def build_lr_schedule(total_steps):
    peak = 3.5e-4

    def lr_at_step(step):
        # drop by 10x at 50% and again at 75% -- no warmup
        if step < total_steps * 0.5:
            return peak
        if step < total_steps * 0.75:
            return peak * 0.1
        return peak * 0.01

    lr_at_step.name = "step_decay"
    return lr_at_step

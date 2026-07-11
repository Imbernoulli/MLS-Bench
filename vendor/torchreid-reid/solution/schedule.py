"""Agent-editable learning-rate schedule for fixed Adam optimization."""
from __future__ import annotations


# EDITABLE REGION
def build_lr_schedule(total_steps):
    peak = 3.5e-4

    def lr_at_step(step):
        return peak

    lr_at_step.name = "flat_lr"
    return lr_at_step
# END EDITABLE REGION

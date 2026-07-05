"""mono3d-learning-rate STRONG baseline: WELL-TUNED head learning-rate.

Keep the depth head's learning rate at the base LR (multiplier 1.0), which under the fixed
OneCycle schedule lets the head fully learn the multiplicative depth residual that corrects the
amodal-height / truncation bias on top of the analytic geometry. Well-trained residual -> lowest
depth error and highest AP3D. Reference: standard LR ablation — the well-tuned LR trains the head
to convergence within the fixed schedule.
"""


def build_lr_mult():
    return 1.0

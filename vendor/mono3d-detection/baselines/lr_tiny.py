"""mono3d-learning-rate WEAK baseline: TINY head learning-rate (under-trained residual).

Scale the depth head's learning rate by 0.01 of the base LR. With such a small LR the head's
learned depth residual barely moves from its init (~0), so the model reduces to raw analytic
geometry with an essentially untrained correction — it cannot learn the residual that corrects
the amodal-height / truncation bias, leaving accuracy on the table and a higher depth error than
a well-tuned LR. Reference: standard LR ablation — too-small an LR under-trains the head.
"""


def build_lr_mult():
    return 0.01

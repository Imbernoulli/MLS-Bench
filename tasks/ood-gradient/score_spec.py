"""Baseline-free full-protocol AUROC scoring."""
from mlsbench.scoring.dsl import *

for _setting in ("ood_gradient_svhn_full", "ood_gradient_cifar100_full", "ood_gradient_tin_full"):
    _term = f"auroc_{_setting}"
    term(_term, col(f"auroc_{_setting}").higher().id().bounded_power(
        bound=const(1.0), floor=const(0.5),
    ))
    setting(_setting, weighted_mean((_term, 1.0)))
task(gmean("ood_gradient_svhn_full", "ood_gradient_cifar100_full", "ood_gradient_tin_full"))

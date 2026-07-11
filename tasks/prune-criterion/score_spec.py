"""Full-protocol accuracy scoring across both CIFAR-10 seed settings."""

from mlsbench.scoring.dsl import *


for _setting in ("cifar10", "cifar10_seed1"):
    _metric = f"acc_{_setting}"
    term(
        _metric,
        col(_metric).higher().id().sigmoid(),
    )
    setting(_setting, weighted_mean((_metric, 1.0)))

task(gmean("cifar10", "cifar10_seed1"))

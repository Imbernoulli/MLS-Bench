"""Full-protocol SAD scoring across all three trimap widths."""

from mlsbench.scoring.dsl import *


for _setting in ("medium", "wide", "xwide"):
    _metric = f"sad_{_setting}"
    term(
        _metric,
        col(_metric).lower().id().sigmoid(),
    )
    setting(_setting, weighted_mean((_metric, 1.0)))

task(gmean("medium", "wide", "xwide"))

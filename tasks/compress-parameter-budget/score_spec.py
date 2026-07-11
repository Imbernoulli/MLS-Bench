"""PENDING_FULL_OFFICIAL: replace only from terminal policy anchors."""
from mlsbench.scoring.dsl import *

for _setting in ('full', 'low', 'mid', 'high'):
    _metric = f"rd12_{_setting}"
    term(_metric, col(_metric).higher().id().sigmoid(ref=const(0.0), scale=1.0))
    _constraint = f"mean_params_{_setting}"
    term(_constraint, penalty_upper(col(_constraint).higher().id(), target=1.0))
    setting(_setting, weighted_mean((_metric, 1.0)), constraints=[_constraint])

task(gmean("full", "low", "mid", "high"))

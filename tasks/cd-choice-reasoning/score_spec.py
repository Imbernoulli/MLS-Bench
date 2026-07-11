"""Baseline-free official AG News accuracy on its natural [0, 1] range.

This mapping depends on the shared scorer's explicit semantic-floor support;
it uses no synthetic or measured baseline row and no fitted calibration curve.
"""
from mlsbench.scoring.dsl import *

term(
    "accuracy_agnews",
    col("accuracy_agnews").higher().id().bounded_power(
        bound=1.0,
        floor=const(0.0),
    ),
)
setting("agnews", weighted_mean(("accuracy_agnews", 1.0)))
task(gmean("agnews"))

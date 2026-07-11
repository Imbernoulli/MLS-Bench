"""Baseline-free official AG News accuracy on its natural [0, 1] range.

The full-split terminal-native row corroborates the workload. The score itself
uses only semantic bounds, with no synthetic baseline or fitted curve.
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

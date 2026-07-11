"""Pending exact-zero calibration across all three output conditions."""
from mlsbench.scoring.dsl import *


CONDITIONS = ("direct", "fenced_wrapper", "trailing_text")

for _condition in CONDITIONS:
    _metric = f"pass_at_1_{_condition}"
    _term = f"{_metric}_pending"
    term(_term, col(_metric).higher().id().sigmoid())
    setting(_condition, weighted_mean((_term, 1.0)))

task(gmean(*CONDITIONS))

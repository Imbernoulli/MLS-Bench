"""Baseline-free scoring for complete three-way NLI evaluations.

Accuracy has a fixed [0, 1] range. Three-way chance accuracy (1/3) maps to zero,
0.5 accuracy maps to 0.5, and perfect accuracy maps to one. This mapping is shared
by all ten siblings and never reads a leaderboard row, fallback value, or score
file. The parser separately requires exact corpus/model/training/evaluation/rc
proofs; any incomplete verification produces no metrics and therefore exact zero.
"""
from mlsbench.scoring.dsl import *


def _accuracy_term(metric: str):
    return (
        col(metric)
        .higher()
        .id()
        .bounded_power(
            bound=1.0,
            floor=const(1.0 / 3.0),
            ref=const(0.5),
            ref_score=0.5,
        )
    )


for _setting in ("snli", "mnli_m", "mnli_mm"):
    _metric = f"acc_{_setting}"
    term(_metric, _accuracy_term(_metric))
    setting(_setting, weighted_mean((_metric, 1.0)))

task(gmean("snli", "mnli_m", "mnli_mm"))

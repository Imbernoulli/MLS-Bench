"""Baseline-free official mAP score for full Market-1501 retrieval.

Each required group reports official mAP in [0, 1]. The final shared scoring
dependency maps that bounded metric linearly from the theoretical floor 0 to the
theoretical ceiling 1, then takes the true geometric mean across all three
groups. No unproven calibration row participates in the score.
"""
from mlsbench.scoring.dsl import *

for _setting in ("easy", "medium", "hard"):
    term(
        f"map_{_setting}",
        col(f"map_{_setting}").higher().id().bounded_power(
            bound=const(1.0), floor=const(0.0)
        ),
    )
    setting(_setting, weighted_mean((f"map_{_setting}", 1.0)))

task(gmean("easy", "medium", "hard"))

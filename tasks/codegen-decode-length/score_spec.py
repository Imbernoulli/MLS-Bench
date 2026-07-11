"""Pending exact-zero calibration for the full 257-problem protocol.

The leaderboard intentionally has no anchor rows. With no calibration anchors,
base and current evaluators map every parser-valid value to exact zero. Replace
this spec only after fresh, terminal full-protocol anchors exist.
"""
from mlsbench.scoring.dsl import *


term(
    "pass_at_1_mbpp_pending",
    col("pass_at_1_mbpp").higher().id().sigmoid(),
)
setting("mbpp", weighted_mean(("pass_at_1_mbpp_pending", 1.0)))
task(gmean("mbpp"))

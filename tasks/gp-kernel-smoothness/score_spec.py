"""Pending unanchored score until this surface has final-protocol anchors.

The header-only leaderboard intentionally provides no calibration anchor. The
normal scorer therefore returns zero without an explicit score override.
"""
from mlsbench.scoring.dsl import *


term(
    "nll_concrete",
    col("nll_concrete").lower().id().sigmoid(scale=1.0),
)
term(
    "nll_kin8nm",
    col("nll_kin8nm").lower().id().sigmoid(scale=1.0),
)
term(
    "nll_elevators",
    col("nll_elevators").lower().id().sigmoid(scale=1.0),
)

setting("concrete", weighted_mean(("nll_concrete", 1.0)))
setting("kin8nm", weighted_mean(("nll_kin8nm", 1.0)))
setting("elevators", weighted_mean(("nll_elevators", 1.0)))
task(gmean("concrete", "kin8nm", "elevators"))

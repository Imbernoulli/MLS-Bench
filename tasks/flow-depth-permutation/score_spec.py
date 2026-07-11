"""Pending score spec for the complete 20K moons protocol.

No complete-protocol baseline anchors have been accepted for this sibling.
The header-only leaderboard leaves the baseline-derived floor and reference
unresolved, so the fail-closed scorer returns exactly zero until measured
20K/30K anchors are added.
"""
from mlsbench.scoring.dsl import *

term(
    "nll_moons",
    col("nll_moons").lower().id().sigmoid(ref=bl_best("nll_moons")),
)
setting("moons", weighted_mean(("nll_moons", 1.0)))
task(gmean("moons"))

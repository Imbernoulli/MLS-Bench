"""Pending score spec for the complete 20K 8gaussians protocol.

No complete-protocol baseline anchors have been accepted for this sibling.
The header-only leaderboard leaves the baseline-derived floor and reference
unresolved, so the fail-closed scorer returns exactly zero until measured
20K/30K anchors are added.
"""
from mlsbench.scoring.dsl import *

term(
    "nll_8gaussians",
    col("nll_8gaussians").lower().id().sigmoid(ref=bl_best("nll_8gaussians")),
)
setting("8gaussians", weighted_mean(("nll_8gaussians", 1.0)))
task(gmean("8gaussians"))

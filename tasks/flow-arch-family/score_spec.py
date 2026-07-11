"""Pending score spec for the complete 20K pinwheel protocol.

No complete-protocol baseline anchors have been accepted for this sibling.
The leaderboard is deliberately header-only, and this standard baseline-derived
normalization therefore remains unresolved and scores exactly zero.  Add only
measured 20K/30K baseline rows before enabling a positive score.
"""
from mlsbench.scoring.dsl import *

term(
    "nll_pinwheel",
    col("nll_pinwheel").lower().id().sigmoid(ref=bl_best("nll_pinwheel")),
)
setting("pinwheel", weighted_mean(("nll_pinwheel", 1.0)))
task(gmean("pinwheel"))

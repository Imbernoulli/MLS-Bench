"""Checkerboard NLL scored on the representative dataset-quality scale.

The midpoint and scale come from the complete 20K/30K checkerboard runs in
``flow-coupling-transform`` (Mangrove tasks 96207/96208, containers
4909807/4909808).  The evidence file SHA-256 is
``73429c480ad6dc0e8f3fb147668e6195fb3d0fcc173079814f9868b8c18d41ef``.
This task uses the same immutable data,
seed, optimizer, budget, exact-NLL metric, and identical affine/spline recipes.
Its header-only leaderboard intentionally does not relabel those source runs as
task-specific candidate measurements; MAF is an additional candidate.
"""
from mlsbench.scoring.dsl import *

term(
    "nll_checkerboard",
    col("nll_checkerboard").lower().id().sigmoid(
        ref=const(2.954646), scale=0.077799512058635861
    ),
)
setting("checkerboard", weighted_mean(("nll_checkerboard", 1.0)))
task(gmean("checkerboard"))

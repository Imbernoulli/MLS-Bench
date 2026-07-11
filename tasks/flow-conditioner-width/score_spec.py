"""Checkerboard NLL scored on the representative dataset-quality scale.

The measured midpoint/scale are from ``flow-coupling-transform`` Mangrove tasks
96207/96208 (containers 4909807/4909808).  The evidence file SHA-256 is
``73429c480ad6dc0e8f3fb147668e6195fb3d0fcc173079814f9868b8c18d41ef``.
Data hashes, seed,
optimizer, 20K/30K protocol, and exact-NLL metric are identical.  The source
runs are absolute quality anchors, not task-specific width runs; the leaderboard
therefore remains header-only.
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

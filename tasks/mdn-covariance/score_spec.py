"""Pending unanchored score until this MDN surface has full-protocol anchors."""
from mlsbench.scoring.dsl import *


term(
    "nll_rot_bimodal",
    col("nll_rot_bimodal").lower().id().sigmoid(scale=1.0),
)
setting("rot_bimodal", weighted_mean(("nll_rot_bimodal", 1.0)))
task(gmean("rot_bimodal"))

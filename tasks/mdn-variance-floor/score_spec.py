"""Pending unanchored score until this MDN surface has full-protocol anchors."""
from mlsbench.scoring.dsl import *


term(
    "nll_spiral",
    col("nll_spiral").lower().id().sigmoid(scale=1.0),
)
setting("spiral", weighted_mean(("nll_spiral", 1.0)))
task(gmean("spiral"))

"""Pending unanchored score until this MDN surface has full-protocol anchors."""
from mlsbench.scoring.dsl import *


term(
    "nll_inverse_sine",
    col("nll_inverse_sine").lower().id().sigmoid(scale=1.0),
)
setting("inverse_sine", weighted_mean(("nll_inverse_sine", 1.0)))
task(gmean("inverse_sine"))

"""Pending exact-zero score until this paper-E.1 surface has fresh anchors."""

from mlsbench.scoring.dsl import *


term(
    "copy_acc_paper_e1",
    col("copy_acc_paper_e1").higher().id().sigmoid(
        floor=const(1.0), scale=1.0
    ),
)
setting("paper_e1", weighted_mean(("copy_acc_paper_e1", 1.0)))
task(gmean("paper_e1"))

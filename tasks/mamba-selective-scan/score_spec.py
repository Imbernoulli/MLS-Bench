"""Paper-scale selective-copy accuracy for ``mamba-selective-scan``.

The workload follows Gu and Dao, Mamba, Appendix E.1: total sequence length
4096, 16 memorized tokens, vocab size 16, two D=64 layers, 400,000 optimizer
steps, constant learning rate 1e-4, and batch size 64. The paper does not state
the optimizer details for this experiment; this implementation binds Adam,
zero weight decay, and gradient clipping at 1.0 in the verifier proof.

The old L256/L384/L512 logistic calibration is intentionally not reused. Until
fresh LTI, B/C-only, and selective anchors complete, the score is the directly
measured token accuracy rather than a fabricated calibrated value.
"""
from mlsbench.scoring.dsl import *


term(
    "copy_acc_paper_e1",
    col("copy_acc_paper_e1").higher().id().bounded_power(
        bound=const(1.0),
        floor=const(0.0),
        ref=const(0.5),
        ref_score=0.5,
    ),
)
setting("paper_e1", weighted_mean(("copy_acc_paper_e1", 1.0)))
task(gmean("paper_e1"))

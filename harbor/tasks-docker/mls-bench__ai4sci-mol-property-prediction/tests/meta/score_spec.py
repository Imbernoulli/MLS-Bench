"""Score spec for ai4sci-mol-property-prediction.

Three classification benchmarks (ROC-AUC, higher is better) following
Uni-Mol README. Reference values are taken from the Uni-Mol paper
(Zhou et al., 2023, Table 3 classification).
"""
from mlsbench.scoring.dsl import *

# Classification (ROC-AUC, higher better)
term("rocauc_BBBP",
    col("rocauc_BBBP").higher().id()
    .bounded_power(bound=1.0))

term("rocauc_BACE",
    col("rocauc_BACE").higher().id()
    .bounded_power(bound=1.0))

term("rocauc_Tox21",
    col("rocauc_Tox21").higher().id()
    .bounded_power(bound=1.0))

setting("BBBP", weighted_mean(("rocauc_BBBP", 1.0)))
setting("BACE", weighted_mean(("rocauc_BACE", 1.0)))
setting("Tox21", weighted_mean(("rocauc_Tox21", 1.0)))

task(gmean("BBBP", "BACE", "Tox21"))

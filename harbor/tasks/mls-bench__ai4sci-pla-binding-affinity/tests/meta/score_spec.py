"""Score spec for ai4sci-pla-binding-affinity.

Normalization uses dynamic leaderboard anchors: the worst baseline is the
0-point floor and the best baseline is the 50-point anchor for each metric
direction.

Task-internal historical worst baseline anchors:
- 2013 RMSE: egnn  1.6751   |  2013 Rp: egnn   0.7356
- 2016 RMSE: schnet 1.3728  |  2016 Rp: egnn   0.7948
- 2019 RMSE: schnet 1.5409  |  2019 Rp: schnet 0.5734
"""
from mlsbench.scoring.dsl import *

# rp: Pearson correlation, range [-1, 1], higher is better, bounded at 1.0

term("rmse_PDBbind2013",
    col("rmse_PDBbind2013").lower().id()
    .bounded_power(bound=0.0))

term("rp_PDBbind2013",
    col("rp_PDBbind2013").higher().id()
    .bounded_power(bound=1.0))

term("rmse_PDBbind2016",
    col("rmse_PDBbind2016").lower().id()
    .bounded_power(bound=0.0))

term("rp_PDBbind2016",
    col("rp_PDBbind2016").higher().id()
    .bounded_power(bound=1.0))

term("rmse_PDBbind2019",
    col("rmse_PDBbind2019").lower().id()
    .bounded_power(bound=0.0))

term("rp_PDBbind2019",
    col("rp_PDBbind2019").higher().id()
    .bounded_power(bound=1.0))

setting("PDBbind2013", weighted_mean(("rmse_PDBbind2013", 1.0), ("rp_PDBbind2013", 1.0)))
setting("PDBbind2016", weighted_mean(("rmse_PDBbind2016", 1.0), ("rp_PDBbind2016", 1.0)))
setting("PDBbind2019", weighted_mean(("rmse_PDBbind2019", 1.0), ("rp_PDBbind2019", 1.0)))

task(gmean("PDBbind2013", "PDBbind2016", "PDBbind2019"))

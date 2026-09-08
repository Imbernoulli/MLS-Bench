"""Score spec for cv-meanflow-perceptual-loss."""
from mlsbench.scoring.dsl import *

# Each label (train_small/medium/large) produces best_fid_{size}.
# The bare "best_fid" is overwritten by the last run to finish, making it unreliable.
# Use size-specific variants only.

term("best_fid_small",
    col("best_fid_small").lower().id()
    .bounded_power(bound=0.0))

term("best_fid_medium",
    col("best_fid_medium").lower().id()
    .bounded_power(bound=0.0))

term("best_fid_large",
    col("best_fid_large").lower().id()
    .bounded_power(bound=0.0))

setting("train_small", weighted_mean(("best_fid_small", 1.0)))
setting("train_medium", weighted_mean(("best_fid_medium", 1.0)))
setting("train_large", weighted_mean(("best_fid_large", 1.0)))

task(gmean("train_small", "train_medium", "train_large"))

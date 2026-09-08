"""Score spec for cv-diffusion-prediction."""
from mlsbench.scoring.dsl import *

# Each label (train_small/medium/large) produces fid_{size} and best_fid_{size}.
# The bare "fid" and "best_fid" columns are overwritten by the last run to finish,
# making them unreliable. Use size-specific variants only, keeping best_fid_* (peak performance).

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

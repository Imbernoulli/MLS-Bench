"""Score spec for cv-dbm-scheduler."""
from mlsbench.scoring.dsl import *

# Each label (edges2handbags, Imagenet, DIODE) produces best_fid_<env>.
# Lower is better; bound=0 (theoretical FID floor).

term("best_fid_edges2handbags",
    col("best_fid_edges2handbags").lower().id()
    .bounded_power(bound=0.0))

term("best_fid_Imagenet",
    col("best_fid_Imagenet").lower().id()
    .bounded_power(bound=0.0))

term("best_fid_DIODE",
    col("best_fid_DIODE").lower().id()
    .bounded_power(bound=0.0))

setting("edges2handbags", weighted_mean(("best_fid_edges2handbags", 1.0)))
setting("Imagenet", weighted_mean(("best_fid_Imagenet", 1.0)))
setting("DIODE", weighted_mean(("best_fid_DIODE", 1.0)))

task(gmean("edges2handbags", "Imagenet", "DIODE"))

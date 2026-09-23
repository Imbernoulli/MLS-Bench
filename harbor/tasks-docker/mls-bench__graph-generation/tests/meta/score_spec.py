"""Score spec for graph-generation."""
from mlsbench.scoring.dsl import *

# MMD (Maximum Mean Discrepancy) metrics: lower is better.
#
# Every MMD term is scored on a log scale (sigmoid: worst baseline -> 0,
# best baseline -> 0.5). The baselines' MMDs span two orders of magnitude
# (orbit on ego_small: 0.008 to 1.12), so on the linear scale the best
# baseline sat at r_ref = 0.993 of the way to the bound and the term capped
# near 0.5 for any better model; log also cuts the score swing by about a
# third when a baseline anchor is replaced by another seed's run.
#
# mmd_degree_enzymes is not a separate term: all three baselines sit at
# 1.37-1.44 (near the kernel's ceiling), a 5% spread against seed-to-seed
# swings of 0.1-0.2, so the term was a step function of seed noise. Degree
# on enzymes still counts through mmd_avg_enzymes.
#
# copy_excess: fraction of generated graphs isomorphic to a training graph,
# minus the same fraction for the held-out reference graphs (clipped at 0),
# computed by the fixed harness. Replaying training graphs is not a
# generative model; above 0.5 the setting score is multiplied by
# exp(-10 * (excess - 0.5)), so replaying the training set (excess ~1.0 on
# community_small and enzymes) gets x0.007. Baselines reach at most 0.27.
# On ego_small 65-78% of held-out graphs are themselves isomorphic to a
# training graph, so the check cannot bind there.

_SETTINGS = ("community_small", "ego_small", "enzymes")

for _ds in _SETTINGS:
    for _stat in ("avg", "clustering", "degree", "orbit"):
        if (_stat, _ds) == ("degree", "enzymes"):
            continue
        term(f"mmd_{_stat}_{_ds}",
            col(f"mmd_{_stat}_{_ds}").lower().log()
            .sigmoid())
    term(f"copy_excess_{_ds}",
        penalty_upper(col(f"copy_excess_{_ds}").lower().id(),
                      target=0.5, sharpness=10.0))

setting("community_small", weighted_mean(("mmd_avg_community_small", 1.0), ("mmd_clustering_community_small", 1.0), ("mmd_degree_community_small", 1.0), ("mmd_orbit_community_small", 1.0)),
        constraints=["copy_excess_community_small"])
setting("ego_small", weighted_mean(("mmd_avg_ego_small", 1.0), ("mmd_clustering_ego_small", 1.0), ("mmd_degree_ego_small", 1.0), ("mmd_orbit_ego_small", 1.0)),
        constraints=["copy_excess_ego_small"])
setting("enzymes", weighted_mean(("mmd_avg_enzymes", 1.0), ("mmd_clustering_enzymes", 1.0), ("mmd_orbit_enzymes", 1.0)),
        constraints=["copy_excess_enzymes"])

task(gmean("community_small", "ego_small", "enzymes"))

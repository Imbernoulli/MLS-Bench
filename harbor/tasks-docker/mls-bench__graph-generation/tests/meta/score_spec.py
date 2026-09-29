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
# copy_excess (computed by the fixed harness): a generated graph isomorphic
# to any dataset graph, training or held-out, is a copy. The generated set may
# reproduce as many distinct dataset graphs as the held-out graphs themselves
# do (0 / 14 / 3 on community_small / ego_small / enzymes); copies of any
# further dataset graph are dropped before the MMD is computed, so replayed
# graphs beyond that allowance never enter the MMD, and copy_excess is their
# fraction of the sample. The penalty stops a mostly-replayed sample (whose
# few kept graphs are real ones) from scoring well: above the target the
# setting score is multiplied by exp(-20 * (excess - target)). All baselines
# are at 0. An ideal sampler (fresh CiteSeer ego graphs, 200 draws) reaches
# at most 0.15 on ego_small, hence the looser target there; community_small
# graphs never recur and enzymes graphs recur at ~3 per 118, so 0.1 leaves
# honest models unpenalised. On ego_small, where tiny ego graphs recur, up to
# 14 distinct replayed graphs still pass unflagged.

_SETTINGS = ("community_small", "ego_small", "enzymes")
_COPY_TARGET = {"community_small": 0.1, "ego_small": 0.2, "enzymes": 0.1}

for _ds in _SETTINGS:
    for _stat in ("avg", "clustering", "degree", "orbit"):
        if (_stat, _ds) == ("degree", "enzymes"):
            continue
        term(f"mmd_{_stat}_{_ds}",
            col(f"mmd_{_stat}_{_ds}").lower().log()
            .sigmoid())
    term(f"copy_excess_{_ds}",
        penalty_upper(col(f"copy_excess_{_ds}").lower().id(),
                      target=_COPY_TARGET[_ds], sharpness=20.0))

setting("community_small", weighted_mean(("mmd_avg_community_small", 1.0), ("mmd_clustering_community_small", 1.0), ("mmd_degree_community_small", 1.0), ("mmd_orbit_community_small", 1.0)),
        constraints=["copy_excess_community_small"])
setting("ego_small", weighted_mean(("mmd_avg_ego_small", 1.0), ("mmd_clustering_ego_small", 1.0), ("mmd_degree_ego_small", 1.0), ("mmd_orbit_ego_small", 1.0)),
        constraints=["copy_excess_ego_small"])
setting("enzymes", weighted_mean(("mmd_avg_enzymes", 1.0), ("mmd_clustering_enzymes", 1.0), ("mmd_orbit_enzymes", 1.0)),
        constraints=["copy_excess_enzymes"])

task(gmean("community_small", "ego_small", "enzymes"))

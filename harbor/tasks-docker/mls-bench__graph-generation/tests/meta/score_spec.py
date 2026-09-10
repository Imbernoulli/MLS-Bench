"""Score spec for graph-generation."""
from mlsbench.scoring.dsl import *

# MMD (Maximum Mean Discrepancy) metrics: lower is better, bounded at 0

term("mmd_avg_community_small",
    col("mmd_avg_community_small").lower().id()
    .bounded_power(bound=0.0))

term("mmd_avg_ego_small",
    col("mmd_avg_ego_small").lower().id()
    .bounded_power(bound=0.0))

term("mmd_avg_enzymes",
    col("mmd_avg_enzymes").lower().id()
    .bounded_power(bound=0.0))

term("mmd_clustering_community_small",
    col("mmd_clustering_community_small").lower().id()
    .bounded_power(bound=0.0))

term("mmd_clustering_ego_small",
    col("mmd_clustering_ego_small").lower().id()
    .bounded_power(bound=0.0))

term("mmd_clustering_enzymes",
    col("mmd_clustering_enzymes").lower().id()
    .bounded_power(bound=0.0))

term("mmd_degree_community_small",
    col("mmd_degree_community_small").lower().id()
    .bounded_power(bound=0.0))

term("mmd_degree_ego_small",
    col("mmd_degree_ego_small").lower().id()
    .bounded_power(bound=0.0))

term("mmd_degree_enzymes",
    col("mmd_degree_enzymes").lower().id()
    .bounded_power(bound=0.0))

term("mmd_orbit_community_small",
    col("mmd_orbit_community_small").lower().id()
    .bounded_power(bound=0.0))

term("mmd_orbit_ego_small",
    col("mmd_orbit_ego_small").lower().id()
    .bounded_power(bound=0.0))

term("mmd_orbit_enzymes",
    col("mmd_orbit_enzymes").lower().id()
    .bounded_power(bound=0.0))

setting("community_small", weighted_mean(("mmd_avg_community_small", 1.0), ("mmd_clustering_community_small", 1.0), ("mmd_degree_community_small", 1.0), ("mmd_orbit_community_small", 1.0)))
setting("ego_small", weighted_mean(("mmd_avg_ego_small", 1.0), ("mmd_clustering_ego_small", 1.0), ("mmd_degree_ego_small", 1.0), ("mmd_orbit_ego_small", 1.0)))
setting("enzymes", weighted_mean(("mmd_avg_enzymes", 1.0), ("mmd_clustering_enzymes", 1.0), ("mmd_degree_enzymes", 1.0), ("mmd_orbit_enzymes", 1.0)))

task(gmean("community_small", "ego_small", "enzymes"))

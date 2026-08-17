"""Score spec for optimization-diagonal-net."""
from mlsbench.scoring.dsl import *

# score: higher is better (less negative log likelihood = better fit)
# n_star: lower is better (fewer samples to reach target = more sample efficient)
term("score_d200_k5_a1e3",
    col("score_d200_k5_a1e3").higher().id()
    .sigmoid())

term("n_star_d200_k5_a1e3",
    col("n_star_d200_k5_a1e3").lower().id()
    .bounded_power(bound=0.0))

term("score_d500_k10_a1e3",
    col("score_d500_k10_a1e3").higher().id()
    .sigmoid())

term("n_star_d500_k10_a1e3",
    col("n_star_d500_k10_a1e3").lower().id()
    .bounded_power(bound=0.0))

term("score_d500_k10_a5e1",
    col("score_d500_k10_a5e1").higher().id()
    .sigmoid())

term("n_star_d500_k10_a5e1",
    col("n_star_d500_k10_a5e1").lower().id()
    .bounded_power(bound=0.0))

term("score_d10000_k50_a1e0",
    col("score_d10000_k50_a1e0").higher().id()
    .sigmoid())

term("n_star_d10000_k50_a1e0",
    col("n_star_d10000_k50_a1e0").lower().id()
    .bounded_power(bound=0.0))

setting("d200_k5_a1e3", weighted_mean(
    ("score_d200_k5_a1e3", 1.0),
    ("n_star_d200_k5_a1e3", 1.0),
))
setting("d500_k10_a1e3", weighted_mean(
    ("score_d500_k10_a1e3", 1.0),
    ("n_star_d500_k10_a1e3", 1.0),
))
setting("d500_k10_a5e1", weighted_mean(
    ("score_d500_k10_a5e1", 1.0),
    ("n_star_d500_k10_a5e1", 1.0),
))
setting("d10000_k50_a1e0", weighted_mean(
    ("score_d10000_k50_a1e0", 1.0),
    ("n_star_d10000_k50_a1e0", 1.0),
))

task(gmean("d200_k5_a1e3", "d500_k10_a1e3", "d500_k10_a5e1", "d10000_k50_a1e0"))

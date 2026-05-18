"""Score spec for optimization-diagonal-net."""
from mlsbench.scoring.dsl import *

# score: higher is better (less negative log likelihood = better fit)
# n_star: lower is better (fewer samples to reach target = more sample efficient)
term("score_d200_k5_s01",
    col("score_d200_k5_s01").higher().id()
    .sigmoid())

term("n_star_d200_k5_s01",
    col("n_star_d200_k5_s01").lower().id()
    .bounded_power(bound=0.0))

term("score_d500_k10_s01",
    col("score_d500_k10_s01").higher().id()
    .sigmoid())

term("n_star_d500_k10_s01",
    col("n_star_d500_k10_s01").lower().id()
    .bounded_power(bound=0.0))

term("score_d500_k10_s02",
    col("score_d500_k10_s02").higher().id()
    .sigmoid())

term("n_star_d500_k10_s02",
    col("n_star_d500_k10_s02").lower().id()
    .bounded_power(bound=0.0))

term("score_d10000_k50",
    col("score_d10000_k50").higher().id()
    .sigmoid())

term("n_star_d10000_k50",
    col("n_star_d10000_k50").lower().id()
    .bounded_power(bound=0.0))

setting("d200_k5_s01", weighted_mean(
    ("score_d200_k5_s01", 1.0),
    ("n_star_d200_k5_s01", 1.0),
))
setting("d500_k10_s01", weighted_mean(
    ("score_d500_k10_s01", 1.0),
    ("n_star_d500_k10_s01", 1.0),
))
setting("d500_k10_s02", weighted_mean(
    ("score_d500_k10_s02", 1.0),
    ("n_star_d500_k10_s02", 1.0),
))
setting("d10000_k50", weighted_mean(
    ("score_d10000_k50", 1.0),
    ("n_star_d10000_k50", 1.0),
))

task(gmean("d200_k5_s01", "d500_k10_s01", "d500_k10_s02", "d10000_k50"))

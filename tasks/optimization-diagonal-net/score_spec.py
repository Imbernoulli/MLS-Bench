"""Score spec for optimization-diagonal-net."""
from mlsbench.scoring.dsl import *

# score = -log2(n_star), higher is better (fewer samples to reach the target).
# n_star is not scored separately: score is an exact function of it
# (sgd d200: -log2(78) = -6.285402), so scoring both counted it twice.
term("score_d200_k20_a1e3",
    col("score_d200_k20_a1e3").higher().id()
    .sigmoid())

term("score_d500_k10_a1e3",
    col("score_d500_k10_a1e3").higher().id()
    .sigmoid())

term("score_d500_k10_a5e1",
    col("score_d500_k10_a5e1").higher().id()
    .sigmoid())

term("score_d10000_k50_a1e0",
    col("score_d10000_k50_a1e0").higher().id()
    .sigmoid())

setting("d200_k20_a1e3", weighted_mean(
    ("score_d200_k20_a1e3", 1.0),
))
setting("d500_k10_a1e3", weighted_mean(
    ("score_d500_k10_a1e3", 1.0),
))
setting("d500_k10_a5e1", weighted_mean(
    ("score_d500_k10_a5e1", 1.0),
))
setting("d10000_k50_a1e0", weighted_mean(
    ("score_d10000_k50_a1e0", 1.0),
))

task(gmean("d200_k20_a1e3", "d500_k10_a1e3", "d500_k10_a5e1", "d10000_k50_a1e0"))

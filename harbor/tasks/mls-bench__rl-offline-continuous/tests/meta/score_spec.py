"""Score spec for rl-offline-continuous."""
from mlsbench.scoring.dsl import *

# d4rl_score can exceed 100 -> use sigmoid (NOT bounded_power)

term("d4rl_score_halfcheetah_medium_v2",
    col("d4rl_score_halfcheetah_medium_v2").higher().id()
    .sigmoid())

term("d4rl_score_maze2d_medium_v1",
    col("d4rl_score_maze2d_medium_v1").higher().id()
    .sigmoid())

term("d4rl_score_walker2d_medium_v2",
    col("d4rl_score_walker2d_medium_v2").higher().id()
    .sigmoid())

setting("halfcheetah-medium-v2", weighted_mean(("d4rl_score_halfcheetah_medium_v2", 1.0)))
setting("maze2d-medium-v1", weighted_mean(("d4rl_score_maze2d_medium_v1", 1.0)))
setting("walker2d-medium-v2", weighted_mean(("d4rl_score_walker2d_medium_v2", 1.0)))

task(gmean("halfcheetah-medium-v2", "maze2d-medium-v1", "walker2d-medium-v2"))

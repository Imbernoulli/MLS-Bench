"""Score spec for robo-diffusion-policy.

D4RL normalized_score is return-like and can exceed nominal expert scale, so
use sigmoid normalization rather than a finite bound.
"""
from mlsbench.scoring.dsl import *

term("hopper_normalized_score",
    col("hopper_normalized_score").higher().id()
    .sigmoid())

term("walker2d_normalized_score",
    col("walker2d_normalized_score").higher().id()
    .sigmoid())

term("halfcheetah_normalized_score",
    col("halfcheetah_normalized_score").higher().id()
    .sigmoid())

setting("hopper", weighted_mean(("hopper_normalized_score", 1.0)))
setting("walker2d", weighted_mean(("walker2d_normalized_score", 1.0)))
setting("halfcheetah", weighted_mean(("halfcheetah_normalized_score", 1.0)))

task(gmean("hopper", "walker2d", "halfcheetah"))

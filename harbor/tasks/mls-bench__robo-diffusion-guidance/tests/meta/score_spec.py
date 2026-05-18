"""Score spec for robo-diffusion-guidance.

D4RL normalized_score is return-like and can exceed nominal expert scale, so
use sigmoid normalization rather than a finite bound. Each MuJoCo environment
is a separate setting and the task score is their geometric mean, so a guidance
method must transfer across hopper, walker2d, and halfcheetah.
"""
from mlsbench.scoring.dsl import *

term("hopper_quality",
    col("hopper_normalized_score").higher().id()
    .sigmoid())

term("walker2d_quality",
    col("walker2d_normalized_score").higher().id()
    .sigmoid())

term("halfcheetah_quality",
    col("halfcheetah_normalized_score").higher().id()
    .sigmoid())

setting("hopper", weighted_mean(("hopper_quality", 1.0)))
setting("walker2d", weighted_mean(("walker2d_quality", 1.0)))
setting("halfcheetah", weighted_mean(("halfcheetah_quality", 1.0)))

task(gmean("hopper", "walker2d", "halfcheetah"))

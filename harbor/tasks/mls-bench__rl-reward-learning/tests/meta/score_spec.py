"""Score spec for rl-reward-learning."""
from mlsbench.scoring.dsl import *

# eval_return is unbounded -> use sigmoid for all environments

term("eval_return_halfcheetah_v4",
    col("eval_return_halfcheetah_v4").higher().id()
    .sigmoid())

term("eval_return_hopper_v4",
    col("eval_return_hopper_v4").higher().id()
    .sigmoid())

term("eval_return_walker2d_v4",
    col("eval_return_walker2d_v4").higher().id()
    .sigmoid())

setting("halfcheetah-v4", weighted_mean(("eval_return_halfcheetah_v4", 1.0)))
setting("hopper-v4", weighted_mean(("eval_return_hopper_v4", 1.0)))
setting("walker2d-v4", weighted_mean(("eval_return_walker2d_v4", 1.0)))

task(gmean("halfcheetah-v4", "hopper-v4", "walker2d-v4"))

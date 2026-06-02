"""Score spec for rl-offpolicy-sample-efficiency."""
from mlsbench.scoring.dsl import *

# Mean episode reward is return-like and has no task-independent finite bound.

term("mean_reward_h1hand_stand_v0",
    col("mean_reward_h1hand_stand_v0").higher().id()
    .sigmoid())

term("mean_reward_h1hand_walk_v0",
    col("mean_reward_h1hand_walk_v0").higher().id()
    .sigmoid())

term("mean_reward_h1hand_run_v0",
    col("mean_reward_h1hand_run_v0").higher().id()
    .sigmoid())

setting("h1hand-stand-v0", weighted_mean(("mean_reward_h1hand_stand_v0", 1.0)))
setting("h1hand-walk-v0", weighted_mean(("mean_reward_h1hand_walk_v0", 1.0)))
setting("h1hand-run-v0", weighted_mean(("mean_reward_h1hand_run_v0", 1.0)))

task(gmean("h1hand-stand-v0", "h1hand-walk-v0", "h1hand-run-v0"))

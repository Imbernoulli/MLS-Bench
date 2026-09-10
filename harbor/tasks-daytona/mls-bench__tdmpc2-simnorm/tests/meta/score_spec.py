"""Score spec for tdmpc2-simnorm (auto-generated, review before use)."""
from mlsbench.scoring.dsl import *

term("episode_reward_walker_walk",
    col("episode_reward_walker_walk").higher().id()
    .bounded_power(bound=1000.0))

term("episode_reward_cheetah_run",
    col("episode_reward_cheetah_run").higher().id()
    .bounded_power(bound=1000.0))

term("episode_reward_cartpole_swingup",
    col("episode_reward_cartpole_swingup").higher().id()
    .bounded_power(bound=1000.0))

setting("walker-walk", weighted_mean(("episode_reward_walker_walk", 1.0)))
setting("cheetah-run", weighted_mean(("episode_reward_cheetah_run", 1.0)))
setting("cartpole-swingup", weighted_mean(("episode_reward_cartpole_swingup", 1.0)))

task(gmean("walker-walk", "cheetah-run", "cartpole-swingup"))

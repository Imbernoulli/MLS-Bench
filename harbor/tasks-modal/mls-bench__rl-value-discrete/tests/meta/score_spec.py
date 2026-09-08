"""Score spec for rl-value-discrete (auto-generated, review before use)."""
from mlsbench.scoring.dsl import *

term("eval_return_cartpole_v1",
    col("eval_return_cartpole_v1").higher().id()
    .sigmoid())

term("eval_return_lunarlander_v2",
    col("eval_return_lunarlander_v2").higher().id()
    .sigmoid())

term("eval_return_acrobot_v1",
    col("eval_return_acrobot_v1").higher().id()
    .sigmoid())

setting("cartpole-v1", weighted_mean(("eval_return_cartpole_v1", 1.0)))
setting("lunarlander-v2", weighted_mean(("eval_return_lunarlander_v2", 1.0)))
setting("acrobot-v1", weighted_mean(("eval_return_acrobot_v1", 1.0)))

task(gmean("cartpole-v1", "lunarlander-v2", "acrobot-v1"))

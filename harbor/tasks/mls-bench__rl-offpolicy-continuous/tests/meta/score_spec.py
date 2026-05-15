"""Score spec for rl-offpolicy-continuous (auto-generated, review before use)."""
from mlsbench.scoring.dsl import *

term("eval_return_halfcheetah_v4",
    col("eval_return_halfcheetah_v4").higher().id()
    .sigmoid())

term("eval_return_reacher_v4",
    col("eval_return_reacher_v4").higher().id()
    .sigmoid())

term("eval_return_ant_v4",
    col("eval_return_ant_v4").higher().id()
    .sigmoid())

setting("halfcheetah-v4", weighted_mean(("eval_return_halfcheetah_v4", 1.0)))
setting("reacher-v4", weighted_mean(("eval_return_reacher_v4", 1.0)))
setting("ant-v4", weighted_mean(("eval_return_ant_v4", 1.0)))

task(gmean("halfcheetah-v4", "reacher-v4", "ant-v4"))

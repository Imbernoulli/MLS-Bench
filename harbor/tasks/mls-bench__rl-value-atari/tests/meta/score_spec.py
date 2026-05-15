"""Score spec for rl-value-atari (auto-generated, review before use)."""
from mlsbench.scoring.dsl import *

term("eval_return_breakout_v4",
    col("eval_return_breakout_v4").higher().id()
    .sigmoid())

term("eval_return_seaquest_v4",
    col("eval_return_seaquest_v4").higher().id()
    .sigmoid())

term("eval_return_pong_v4",
    col("eval_return_pong_v4").higher().id()
    .sigmoid())

setting("breakout-v4", weighted_mean(("eval_return_breakout_v4", 1.0)))
setting("seaquest-v4", weighted_mean(("eval_return_seaquest_v4", 1.0)))
setting("pong-v4", weighted_mean(("eval_return_pong_v4", 1.0)))

task(gmean("breakout-v4", "seaquest-v4", "pong-v4"))

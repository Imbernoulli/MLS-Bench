"""Score spec for rl-offline-off2on (auto-generated, review before use)."""
from mlsbench.scoring.dsl import *

term("d4rl_score_pen_cloned_v1",
    col("d4rl_score_pen_cloned_v1").higher().id()
    .sigmoid())

term("d4rl_score_hammer_cloned_v1",
    col("d4rl_score_hammer_cloned_v1").higher().id()
    .sigmoid())

term("d4rl_score_hammer_expert_v1",
    col("d4rl_score_hammer_expert_v1").higher().id()
    .sigmoid())

setting("pen-cloned-v1", weighted_mean(("d4rl_score_pen_cloned_v1", 1.0)))
setting("hammer-cloned-v1", weighted_mean(("d4rl_score_hammer_cloned_v1", 1.0)))
setting("hammer-expert-v1", weighted_mean(("d4rl_score_hammer_expert_v1", 1.0)))

task(gmean("pen-cloned-v1", "hammer-cloned-v1", "hammer-expert-v1"))

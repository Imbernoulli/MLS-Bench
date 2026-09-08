"""Score spec for rl-offline-adroit (auto-generated, review before use)."""
from mlsbench.scoring.dsl import *

term("d4rl_score_pen_human_v1",
    col("d4rl_score_pen_human_v1").higher().id()
    .sigmoid())

term("d4rl_score_hammer_human_v1",
    col("d4rl_score_hammer_human_v1").higher().id()
    .sigmoid())

term("d4rl_score_door_cloned_v1",
    col("d4rl_score_door_cloned_v1").higher().id()
    .sigmoid())

setting("pen-human-v1", weighted_mean(("d4rl_score_pen_human_v1", 1.0)))
setting("hammer-human-v1", weighted_mean(("d4rl_score_hammer_human_v1", 1.0)))
setting("door-cloned-v1", weighted_mean(("d4rl_score_door_cloned_v1", 1.0)))

task(gmean("pen-human-v1", "hammer-human-v1", "door-cloned-v1"))

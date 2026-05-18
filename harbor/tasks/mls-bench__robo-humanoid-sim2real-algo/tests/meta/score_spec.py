"""Score spec for robo-humanoid-sim2real-algo.

Three eval conditions (sim2sim MuJoCo rollouts after Isaac Gym training):
  - forward-only    (straight walking)
  - diverse-commands (mixed vx/vy/dyaw)
  - high-speed      (high vx range)

Each emits: success_rate (higher=better), avg_vel_error (lower=better),
fall_rate (lower=better).

Score = mean of success_rate across the three conditions (primary metric),
with avg_vel_error and fall_rate tracked but not scored (they are correlated
with success_rate and already inform it at the threshold check).
"""
from mlsbench.scoring.dsl import *

term("success_rate_forward-only",
    col("success_rate_forward-only").higher().id()
    .bounded_power(bound=1.0))

term("success_rate_diverse-commands",
    col("success_rate_diverse-commands").higher().id()
    .bounded_power(bound=1.0))

term("success_rate_high-speed",
    col("success_rate_high-speed").higher().id()
    .bounded_power(bound=1.0))

setting("sim2sim", weighted_mean(
    ("success_rate_forward-only", 1.0),
    ("success_rate_diverse-commands", 1.0),
    ("success_rate_high-speed", 1.0),
))

task(gmean("sim2sim"))

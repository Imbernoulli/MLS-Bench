"""Score spec for robo-humanoid-sim2real-algo.

Three eval conditions (sim2sim MuJoCo rollouts after Isaac Gym training):
  - forward-only    (straight walking)
  - diverse-commands (mixed vx/vy/dyaw)
  - high-speed      (high vx range)

Each emits: success_rate (higher=better), avg_vel_error (lower=better),
fall_rate (lower=better). Only success_rate (fraction of 100 sampled commands
that succeed) is scored; avg_vel_error and fall_rate are correlated with it
and already inform it at the threshold check.

One setting per condition, gmean across settings.

high-speed is the discriminating condition: the baselines span 0.13-0.27 with
headroom to 1.0. It keeps the standard anchors (worst baseline = 0, best
baseline = 0.5, bound 1.0 = 1).

forward-only and diverse-commands are saturated: every baseline scores
1.0 / 1.0 / 1.0 on forward-only and 0.96-1.00 on diverse-commands, a spread of
at most four commands out of 100 (binomial std at p=0.97 is ~0.017). With the
standard anchors forward-only was a step function (worst == best == bound, so
any rate below 1.0 scored 0) and diverse-commands mapped 0.96 -> 0 and
1.00 -> 1 on seed noise. Both are therefore scored as no-regression checks:
bound=0 (the minimum rate) lies below the worst baseline, so bounded_power maps
the worst baseline rate and anything above it to 1 and falls about linearly to
0 at a rate of 0 (ref=0.5 -> 0.5). A policy that keeps walking keeps the
setting; a policy that breaks straight walking loses it.
"""
from mlsbench.scoring.dsl import *

# Saturated condition: no-regression check (worst baseline rate and above -> 1).
term("success_rate_forward-only",
    col("success_rate_forward-only").higher().id()
    .bounded_power(bound=0.0, ref=0.5))

# Saturated condition: no-regression check (worst baseline rate and above -> 1).
term("success_rate_diverse-commands",
    col("success_rate_diverse-commands").higher().id()
    .bounded_power(bound=0.0, ref=0.5))

# Discriminating condition: standard anchors (worst baseline 0, best 0.5).
term("success_rate_high-speed",
    col("success_rate_high-speed").higher().id()
    .bounded_power(bound=1.0))

setting("forward-only", weighted_mean(("success_rate_forward-only", 1.0)))
setting("diverse-commands", weighted_mean(("success_rate_diverse-commands", 1.0)))
setting("high-speed", weighted_mean(("success_rate_high-speed", 1.0)))

task(gmean("forward-only", "diverse-commands", "high-speed"))

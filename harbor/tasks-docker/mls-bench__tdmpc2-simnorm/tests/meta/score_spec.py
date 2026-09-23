"""Score spec for tdmpc2-simnorm.

Three DMControl settings, one episode-reward term each, gmean across settings.
Episode returns lie in [0, 1000] (per-step reward in [0, 1], 1000 steps).

cheetah-run is the discriminating setting: the baseline means span 681-813
and there is headroom above them. It keeps the standard anchors (worst
baseline = 0, best baseline = 0.5, bound 1000 = 1).

walker-walk and cartpole-swingup are saturated at this training budget: all
twelve baseline seeds lie in 972.5-979.3 (walker) and 858.0-882.5 (cartpole),
and the baseline means differ by 1.4 and 8.2 points, about one per-seed std
(2.6 and 6.3). With the standard anchors the worst mean there, a seed-noise
draw, scored 0, which the gmean epsilon turns into a x0.215 task factor: the
SimNorm default scored 0.131 for sitting 0.9-1.4 walker points below the
other three, and l2norm fell from 0.465 to 0.132 if its walker mean dropped
by 1.8. These two settings are therefore scored as no-regression checks.
bound=0 (the minimum return) lies below the worst baseline, so bounded_power
maps the worst baseline mean and anything above it to 1 and falls about
linearly to 0 at a return of 0 (ref=500 -> 0.5): a seed-std dip costs <1 % of
the setting, while a method that breaks an environment still loses it.
"""
from mlsbench.scoring.dsl import *

# Saturated setting: no-regression check (worst baseline mean and above -> 1).
term("episode_reward_walker_walk",
    col("episode_reward_walker_walk").higher().id()
    .bounded_power(bound=0.0, ref=500.0))

# Discriminating setting: standard anchors (worst baseline 0, best 0.5).
term("episode_reward_cheetah_run",
    col("episode_reward_cheetah_run").higher().id()
    .bounded_power(bound=1000.0))

# Saturated setting: no-regression check (worst baseline mean and above -> 1).
term("episode_reward_cartpole_swingup",
    col("episode_reward_cartpole_swingup").higher().id()
    .bounded_power(bound=0.0, ref=500.0))

setting("walker-walk", weighted_mean(("episode_reward_walker_walk", 1.0)))
setting("cheetah-run", weighted_mean(("episode_reward_cheetah_run", 1.0)))
setting("cartpole-swingup", weighted_mean(("episode_reward_cartpole_swingup", 1.0)))

task(gmean("walker-walk", "cheetah-run", "cartpole-swingup"))

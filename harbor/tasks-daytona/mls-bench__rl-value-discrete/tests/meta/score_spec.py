"""Score spec for rl-value-discrete.

Three Gymnasium settings, one final-eval-return term each, gmean across
settings.

CartPole-v1 is saturated: its episodes are truncated at 500 steps with reward
1 per step, so 500 is the maximum return, and the baseline means are 500.0
(qr_dqn), 495.1 (c51) and 487.2 (dueling_dqn); every non-500 value comes from
one seed (461.6, 485.4) in a run whose other seeds hit 500. With the standard
anchors the best baseline sat at the maximum, so the setting could never
score above 0.5 (capping the task at 0.5^(1/3) of its other settings), and
the worst mean, a one-seed dip, scored 0. CartPole is therefore scored as a
no-regression check: bound=0 lies below the worst baseline mean, so
bounded_power maps the worst baseline mean and anything above it to 1 and
falls about linearly to 0 at a return of 0 (ref=250 -> 0.5). A method that
still solves CartPole keeps the setting; one that breaks it loses it.

LunarLander-v2 and Acrobot-v1 have headroom above the baselines and keep the
standard sigmoid anchors (worst baseline = 0, best baseline = 0.5).
"""
from mlsbench.scoring.dsl import *

# Saturated setting (max return 500): no-regression check.
term("eval_return_cartpole_v1",
    col("eval_return_cartpole_v1").higher().id()
    .bounded_power(bound=0.0, ref=250.0))

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

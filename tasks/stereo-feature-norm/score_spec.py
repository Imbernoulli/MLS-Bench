"""DROPPED surface (not shipped) -- see DROPPED.md. No live score spec."""
from mlsbench.scoring.dsl import *  # noqa: F401,F403
# Intentionally empty: this surface was dropped after the real-Middlebury-data
# re-anchor (cross-seed 42/123, plus a 3000-step diagnostic re-check) showed
# seed 123 is fully inverted at 1200 steps and only partially -- not fully --
# recovers at 3000 steps, while seed 42 develops a NEW failure on 'hard' at
# 3000 steps that wasn't present at 1200 steps. Net failure count does not
# improve with more training budget -- a genuine confound, not an
# undertraining artifact. See DROPPED.md for numbers.

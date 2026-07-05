"""DROPPED surface (not shipped) -- see DROPPED.md. No live score spec."""
from mlsbench.scoring.dsl import *  # noqa: F401,F403
# Intentionally empty: this surface was dropped after the real-Middlebury-data
# re-anchor (cross-seed 42/123, plus a 3000-step diagnostic re-check) showed
# seed 123's single-setting inversion does NOT resolve with more training
# budget -- it relocates from 'medium' (at 1200 steps) to 'hard' (at 3000
# steps) rather than disappearing. This indicates a fragile, seed-noise-level
# effect on real data rather than a genuine, robust per-setting ordering. See
# DROPPED.md for numbers.

"""DROPPED surface (not shipped) -- see DROPPED.md. No live score spec."""
from mlsbench.scoring.dsl import *  # noqa: F401,F403
# Intentionally empty: this surface was dropped for cross-seed non-monotonicity on real
# ShanghaiTech-derived crowd data at both the package-default (450 iter) and a 1500-iter
# training-budget diagnostic (5x longer, NOT an HP-sweep) -- failures relocate rather than
# resolve, ruling out an undertraining artifact.

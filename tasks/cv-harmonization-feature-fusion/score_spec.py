"""DROPPED surface (not shipped) -- see DROPPED.md. No live score spec."""
from mlsbench.scoring.dsl import *  # noqa: F401,F403
# Intentionally empty: this surface was dropped for cross-seed non-monotonicity on real
# iHarmony4 data at both the package-default (500 iter) and a 2000-iter training-budget
# diagnostic (4x longer, NOT an HP-sweep) -- failures relocate/persist rather than
# resolving, ruling out an undertraining artifact.

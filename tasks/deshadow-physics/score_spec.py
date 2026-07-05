"""DROPPED surface (not shipped) -- see DROPPED.md. No live score spec."""
from mlsbench.scoring.dsl import *  # noqa: F401,F403
# Intentionally empty: this surface was dropped after the real-ISTD-data CPU re-check showed
# the aggregate gmean is NOT monotone weak->strong on real data (it was on the old synthetic
# data; see DROPPED.md for the real-data numbers that reversed this).

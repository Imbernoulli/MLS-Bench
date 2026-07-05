"""DROPPED surface (not shipped) -- see DROPPED.md. No live score spec."""
from mlsbench.scoring.dsl import *  # noqa: F401,F403
# Intentionally empty: this surface was dropped for cross-seed non-monotonicity (only 1/3
# tiers robust) on real KITTI 3D Object Detection data at full budget (1200 steps, seeds
# 42/123).

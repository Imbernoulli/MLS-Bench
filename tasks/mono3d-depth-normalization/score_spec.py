"""DROPPED surface (not shipped) -- see DROPPED.md. No live score spec."""
from mlsbench.scoring.dsl import *  # noqa: F401,F403
# Intentionally empty: this surface was dropped because the intended additive>log_mult
# relabel is not robust to the SOTA=0.5 seed-42-anchor discipline -- the hard tier inverts
# at seed 42 specifically (near-tie, noise-level delta), so no single seed-42 anchor set
# makes additive score >=0.5 on all three per-setting terms simultaneously. See DROPPED.md.

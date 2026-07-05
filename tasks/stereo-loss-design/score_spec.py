"""DROPPED surface (not shipped) -- see DROPPED.md. No live score spec."""
from mlsbench.scoring.dsl import *  # noqa: F401,F403
# Intentionally empty: this surface was dropped after the real-Middlebury-data
# re-anchor (and a 3000-step diagnostic re-check) showed the 'medium' setting's
# aggregate ordering is NOT monotone weak(l2)->strong(smooth_l1) on real data,
# and the inversion WORSENS (not resolves) with more training budget -- a
# genuine confound, not an undertraining artifact. See DROPPED.md for numbers.

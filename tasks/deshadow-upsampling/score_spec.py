"""DROPPED surface (not shipped) -- see DROPPED.md. No live score spec."""
from mlsbench.scoring.dsl import *  # noqa: F401,F403
# Intentionally empty: this surface was dropped after a from-scratch,
# purpose-built GPU re-anchor sweep on real ISTD data (cross-seed 42/123,
# 400 iters, plus a 1200-iter training-budget diagnostic) showed 4/6
# settings-x-seeds inverted at 400 iters, WORSENING to 5/6 at 1200 iters --
# every failing cell persists or gets worse, none resolve. This is a
# genuine, consistently-signed confound (bilinear does not reliably beat
# transpose conv), not an undertraining artifact. See DROPPED.md for
# numbers.

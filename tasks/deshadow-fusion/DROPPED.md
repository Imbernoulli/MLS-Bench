# DROPPED surface: deshadow-fusion (NOT shipped)

This editable surface (how the mask-guided residual deshadower fuses its encoder/decoder
skip features -- `last` (final-layer-only concat) vs `dense` (dense multi-scale fusion)) was
previously shipped with anchors confirmed to match a two-seed-mean real-ISTD historical anchor
log (`mlaunch_anchor2.log`). After a from-scratch, purpose-built GPU re-anchor sweep on the
REAL ISTD data (Wang, Li & Yang, CVPR 2018; via `vendor/image-deshadow/run_anchors.py`, k1 H20,
torch 2.4.0, package-default 400 iters, cross-seed 42/123), the ordering does NOT hold cleanly
per-seed:

  400 iters:
    light  seed42  : last 32.6296 > dense 31.9822   (delta -0.647, FAIL -- inverted)
    light  seed123 : last 34.4396 > dense 34.7666   (delta +0.327, OK)
    medium seed42  : last 30.0673 > dense 29.4250   (delta -0.642, FAIL -- inverted)
    medium seed123 : last 31.6497 > dense 31.3911   (delta -0.259, FAIL -- inverted)
    heavy  seed42  : last 26.0860 > dense 28.9316   (delta +2.846, OK)
    heavy  seed123 : last 28.4086 > dense 28.4728   (delta +0.064, OK -- razor-thin)

A 1200-iter training-budget diagnostic (same methodology used to resolve `aggregation`/
`refine` in the stereo-matching funnel) was run cross-seed to check whether the failures were
an undertraining artifact:

  1200 iters:
    light  seed42  : last 35.1661 > dense 35.2771   (delta +0.111, OK -- RESOLVED; was FAIL at 400)
    light  seed123 : last 34.6032 > dense 34.8352   (delta +0.232, OK)
    medium seed42  : last 33.7294 > dense 33.7784   (delta +0.049, OK -- RESOLVED; was FAIL at 400)
    medium seed123 : last 33.0649 > dense 32.2025   (delta -0.862, FAIL -- inverted, still)
    heavy  seed42  : last 31.2956 > dense 31.9377   (delta +0.642, OK)
    heavy  seed123 : last 31.0246 > dense 30.7112   (delta -0.313, FAIL -- NEW; was OK (barely)
                                                        at 400)

Going from 400 to 1200 iters, `light`/`medium` (seed 42) genuinely resolve, but `medium`
(seed 123, unchanged FAIL) persists and `heavy` (seed 123) newly fails (it was only a
razor-thin +0.064 OK at 400 iters, well within noise). Total failure count goes from 3/6 to
2/6 settings-x-seeds -- a marginal improvement, but no single setting is cross-seed-clean at
BOTH budgets simultaneously (light: clean only at 1200; medium: never clean on seed 123 at
either budget; heavy: clean only on seed 42, and only marginally on seed 123 at 400). This
pattern -- cells relocating rather than shrinking to zero, and no setting achieving a stable,
budget-independent cross-seed win -- indicates a genuine, small-effect-size confound rather
than a resolvable undertraining artifact: dense multi-scale skip fusion's benefit over
last-layer-only fusion on real ISTD shadow removal is too marginal and scene/seed-dependent to
reliably beat run-to-run noise.

Dropped honestly rather than re-anchored or HP-swept to force monotonicity; see the
per-anchor-line data above and in
`vendor/image-deshadow/anchors/anchor_real_full.tsv` (task=fusion rows, 400 iters) and
`vendor/image-deshadow/anchors/anchor_diag_1200.tsv` (task=fusion rows, 1200-iter diagnostic).

The surface code remains in `vendor/image-deshadow/solution/fusion.py` and
`vendor/image-deshadow/baselines/fusion_{last,dense}.py` for provenance; no task is shipped
for it.

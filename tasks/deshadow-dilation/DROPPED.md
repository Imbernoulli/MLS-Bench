# DROPPED surface: deshadow-dilation (NOT shipped)

This zero-added-parameter editable surface (the spatial dilation rate of the mask-guided
residual deshadower's middle conv -- `d1` (rate 1) vs `dilated` (rate >1), a wider-
receptive-field cost-of-nothing change) was previously shipped with anchors from an ambiguous/
unverifiable provenance (neither matched any available historical real-ISTD anchor log). After
a from-scratch, purpose-built GPU re-anchor sweep on the REAL ISTD data (Wang, Li & Yang, CVPR
2018; via `vendor/image-deshadow/run_anchors.py`, k1 H20, torch 2.4.0, package-default 400
iters, cross-seed 42/123), the ordering does NOT hold cleanly:

  400 iters:
    light  seed42  : d1 32.1673 > dilated 33.1825   (delta +1.015, OK)
    light  seed123 : d1 34.6238 > dilated 34.6403   (delta +0.017, OK -- razor-thin)
    medium seed42  : d1 30.2120 > dilated 30.7098   (delta +0.498, OK)
    medium seed123 : d1 32.0875 > dilated 32.3938   (delta +0.306, OK)
    heavy  seed42  : d1 28.3801 > dilated 26.3047    (delta -2.075, FAIL -- inverted)
    heavy  seed123 : d1 29.1112 > dilated 28.5646    (delta -0.547, FAIL -- inverted)

A 1200-iter training-budget diagnostic (same methodology used to resolve `aggregation`/`refine`
in the stereo-matching funnel) was run cross-seed to check whether `heavy`'s failure was an
undertraining artifact:

  1200 iters:
    light  seed42  : d1 35.2803 > dilated 35.1311   (delta -0.149, FAIL -- NEW; was OK at 400)
    light  seed123 : d1 35.0410 > dilated 34.2982   (delta -0.743, FAIL -- NEW; was OK at 400)
    medium seed42  : d1 33.5318 > dilated 33.5023   (delta -0.029, FAIL -- NEW; was OK at 400)
    medium seed123 : d1 32.6048 > dilated 33.3946   (delta +0.790, OK)
    heavy  seed42  : d1 30.8513 > dilated 32.7473   (delta +1.896, OK -- RESOLVED; was FAIL at 400)
    heavy  seed123 : d1 30.6194 > dilated 31.5354   (delta +0.916, OK -- RESOLVED; was FAIL at 400)

Going from 400 to 1200 iters, `heavy` (both seeds) genuinely resolves, but `light` (both seeds)
and `medium` (seed 42) newly fail -- cells that were clean at 400 iters. Total failure count
goes from 2/6 to 3/6 settings-x-seeds, i.e. it WORSENS with more training budget, and WHICH
cells fail relocates almost entirely. This is not an undertraining artifact (which would
consistently shrink/resolve failures with more budget) -- it is a genuine confound: dilation
rate's effect on real ISTD shadow removal is apparently too small, scene/severity-dependent,
and unstable across both seed and training-budget axes to reliably beat run-to-run noise as a
monotone strong>weak surface.

Dropped honestly rather than re-anchored or HP-swept to force monotonicity; see the
per-anchor-line data above and in
`vendor/image-deshadow/anchors/anchor_real_full.tsv` (task=dilation rows, 400 iters) and
`vendor/image-deshadow/anchors/anchor_diag_1200.tsv` (task=dilation rows, 1200-iter
diagnostic).

The surface code remains in `vendor/image-deshadow/solution/dilation.py` and
`vendor/image-deshadow/baselines/dilation_{d1,dilated}.py` for provenance; no task is shipped
for it.

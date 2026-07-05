# DROPPED surface: deshadow-upsampling (NOT shipped)

This editable surface (the decoder's upsampling operator -- `transpose` (learned transposed
conv) vs `bilinear` (fixed bilinear + 1x1 conv)) was previously shipped with anchors confirmed
to match a two-seed-mean real-ISTD historical anchor log (`mlaunch_anchor2.log`), with `light`
already documented as a near-tied setting. After a from-scratch, purpose-built GPU re-anchor
sweep on the REAL ISTD data (Wang, Li & Yang, CVPR 2018; via
`vendor/image-deshadow/run_anchors.py`, k1 H20, torch 2.4.0, package-default 400 iters,
cross-seed 42/123), the ordering does NOT hold cleanly per-seed:

  400 iters:
    light  seed42  : transpose 32.4754 > bilinear 33.2418   (delta +0.766, OK)
    light  seed123 : transpose 34.5593 > bilinear 34.4132   (delta -0.146, FAIL -- inverted)
    medium seed42  : transpose 29.7826 > bilinear 31.2523   (delta +1.470, OK)
    medium seed123 : transpose 31.6750 > bilinear 31.4841   (delta -0.191, FAIL -- inverted)
    heavy  seed42  : transpose 28.4176 > bilinear 25.4525   (delta -2.965, FAIL -- inverted)
    heavy  seed123 : transpose 28.9514 > bilinear 28.7639   (delta -0.188, FAIL -- inverted)

A 1200-iter training-budget diagnostic (same methodology used to resolve `aggregation`/
`refine` in the stereo-matching funnel) was run cross-seed to check whether the failures were
an undertraining artifact:

  1200 iters:
    light  seed42  : transpose 35.2433 > bilinear 35.2639   (delta +0.021, OK -- razor-thin)
    light  seed123 : transpose 34.9235 > bilinear 34.4263   (delta -0.497, FAIL -- worse, still
                                                                inverted)
    medium seed42  : transpose 33.6659 > bilinear 33.4846   (delta -0.181, FAIL -- NEW; was OK
                                                                at 400)
    medium seed123 : transpose 32.4824 > bilinear 31.5762   (delta -0.906, FAIL -- worse, still
                                                                inverted)
    heavy  seed42  : transpose 31.2342 > bilinear 30.5314   (delta -0.703, FAIL -- still
                                                                inverted, smaller gap)
    heavy  seed123 : transpose 32.1655 > bilinear 30.9486   (delta -1.217, FAIL -- worse, still
                                                                inverted)

Total failure count goes from 4/6 to 5/6 settings-x-seeds -- it WORSENS with more training
budget, and every failing cell either persists or gets worse (no cell resolves). This is not
an undertraining artifact (which would consistently shrink failures with more budget) -- it is
a genuine, consistently-signed confound on real ISTD data: bilinear upsampling does NOT
reliably beat learned transposed-conv upsampling for shadow-region recovery, and if anything
transposed conv is the more robust choice at longer training, the opposite of what this
surface's weak/strong labelling assumed.

Dropped honestly rather than re-anchored or HP-swept to force monotonicity; see the
per-anchor-line data above and in
`vendor/image-deshadow/anchors/anchor_real_full.tsv` (task=upsampling rows, 400 iters) and
`vendor/image-deshadow/anchors/anchor_diag_1200.tsv` (task=upsampling rows, 1200-iter
diagnostic).

The surface code remains in `vendor/image-deshadow/solution/upsampling.py` and
`vendor/image-deshadow/baselines/upsampling_{transpose,bilinear}.py` for provenance; no task
is shipped for it.

# DROPPED surface: stereo-feature-norm (NOT shipped)

This zero-parameter editable surface (whether the fixed, deep per-pixel
feature maps are L2-normalized across the channel axis before the cost volume
is built -- 'none' vs 'l2', a pure normalized-cross-correlation-style matching
change) was DESIGNED and GPU-anchored on the OLD SYNTHETIC stereo data (see
git history for the previous leaderboard.csv/score_spec.py that shipped it,
weak(none)>strong(l2) on all 3 settings). After the data pipeline was swapped
to REAL rectified stereo photographs from the Middlebury Stereo Datasets
2005/2006 (structured-light ground-truth disparity), a full GPU re-anchor +
cross-seed (42/123) check + a 3000-step training-budget diagnostic (same
methodology used to resolve `aggregation`/`refine`) showed the surface getting
WORSE, not better, with more training budget and data:

  1200 steps:
    seed 42  : easy   none 3.835 > l2 3.413   (OK)
               medium none 10.804 > l2 9.544  (OK)
               hard   none 6.278 > l2 5.637   (OK)
    seed 123 : easy   none 3.084 < l2 3.153   (FAIL)
               medium none 8.473 < l2 9.162   (FAIL)
               hard   none 3.874 < l2 5.031   (FAIL)   -- FULL inversion, all 3
                                                            settings

  3000 steps (diagnostic re-run):
    seed 42  : easy   none 3.096 > l2 2.976   (OK)
               medium none 12.582 > l2 8.918  (OK)
               hard   none 3.890 < l2 4.370   (FAIL -- NEW failure; seed 42
                                                  was clean on hard at 1200 steps)
    seed 123 : easy   none 3.018 > l2 2.991   (OK, barely)
               medium none 8.875 < l2 9.757   (FAIL)
               hard   none 3.460 < l2 4.173   (FAIL)

Seed 123 is fully inverted (weak<strong on all 3 settings) at the package
default 1200 steps. At the longer 3000-step diagnostic schedule, seed 123
only recovers ONE of its three failing settings (easy) while medium/hard
remain inverted, AND seed 42 -- which was fully clean at 1200 steps --
develops a brand-new failure on `hard`. So going from 1200 to 3000 steps
makes the total failure count go from 3/6 to 3/6 settings-x-seeds (no net
improvement) while moving WHICH cells fail. This is not an undertraining
artifact (which would consistently shrink/resolve with more budget, as it did
for `aggregation`/`refine`) -- it is a genuine confound: this is a
zero-added-parameter surface, and L2 feature normalization's effect on real
Middlebury scenes is apparently too small and unstable to reliably beat run-
to-run noise, and on the harder/real settings it may even be net-neutral-to-
harmful rather than helpful.

Reason (real data): on the OLD SYNTHETIC data the fixed photometric
perturbation (gain/bias/noise) injected between the left/right views was
apparently large and uniform enough that channel-wise L2 normalization's
cosine-similarity-style robustness reliably won. Real Middlebury stereo pairs
already come from real cameras with their own (much smaller, scene-dependent)
photometric variation, so the added perturbation's relative contribution -- and
therefore L2-normalization's corrective benefit -- is far smaller and more
scene/seed-dependent, to the point where it no longer reliably beats a
no-normalization baseline on real data.

Full per-anchor-line provenance:
  - seed-42/123, 1200-step re-anchor: vendor/stereo-matching/anchors/anchor_real.tsv,
    vendor/stereo-matching/anchors/seed123_rest.tsv (task=featnorm rows)
  - seed-42/123, 3000-step diagnostic: vendor/stereo-matching/anchors/diag_temp_featnorm_3000.tsv,
    vendor/stereo-matching/anchors/diag_temp_featnorm_3000_s123.tsv

The surface code remains in vendor/stereo-matching/solution/featnorm.py and
vendor/stereo-matching/baselines/featnorm_{none,l2}.py for provenance; no task
is shipped for it.

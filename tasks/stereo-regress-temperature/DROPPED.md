# DROPPED surface: stereo-regress-temperature (NOT shipped)

This zero-parameter editable surface (softmax temperature T applied to the
fixed aggregated cost volume before the fixed soft-argmin readout: `prob =
softmax(cost / T)`, RQ: "does a low, reference temperature (T=1) beat a high,
flattening temperature (T=8)?") was DESIGNED and GPU-anchored on the OLD
SYNTHETIC stereo data (see git history for the previous
leaderboard.csv/score_spec.py that shipped it, weak(temp_high)>strong(temp_ref)
on all 3 settings). After the data pipeline was swapped to REAL rectified
stereo photographs from the Middlebury Stereo Datasets 2005/2006, a full GPU
re-anchor + cross-seed (42/123) check + a 3000-step training-budget diagnostic
(same methodology used to resolve `aggregation`/`refine`) revealed a
PERSISTENT, seed-dependent single-setting inversion that does NOT resolve
with more training budget -- it just moves to a different setting:

  1200 steps (package default):
    seed 42  : easy   high 4.774 > ref 3.849   (OK)
               medium high 11.921 > ref 10.888  (OK)
               hard   high 9.527 > ref 5.894   (OK)
    seed 123 : easy   high 3.547 > ref 3.067   (OK)
               medium high 9.634 < ref 11.535  (FAIL)
               hard   high 8.131 > ref 3.740   (OK)

  3000 steps (diagnostic re-run):
    seed 42  : easy   high 3.178 > ref 3.123   (OK)
               medium high 13.488 > ref 10.394  (OK)
               hard   high 5.159 > ref 3.687   (OK)
    seed 123 : easy   high 3.079 > ref 3.010   (OK)
               medium high 11.761 > ref 10.187  (OK)
               hard   high 3.160 < ref 4.098   (FAIL)

Seed 42 is cleanly monotone on ALL 3 settings at BOTH step counts. Seed 123
fails exactly ONE setting at each step count, but a DIFFERENT setting each
time (medium at 1200 steps, hard at 3000 steps). This rules out an
undertraining artifact (which would resolve consistently at the higher
budget, as it did for `aggregation`/`refine`) -- instead it indicates the
temperature effect's real margin on real Middlebury scenes is too small/noisy
relative to run-to-run (seed) variance to guarantee a stable per-setting
ordering. No fixed step count gives full 3-setting cross-seed robustness.

Reason (real data): on the OLD SYNTHETIC data the readout's mid-range
disparity collapse under high temperature was apparently a much larger,
seed-independent effect (likely because the synthetic scenes' disparity
distributions were narrower/more uniform); real Middlebury scenes have more
varied per-scene disparity structure, shrinking and destabilising this
zero-parameter effect enough that a single training seed's noise can flip the
order on whichever setting happens to be most marginal that run.

Full per-anchor-line provenance:
  - seed-42/123, 1200-step re-anchor: vendor/stereo-matching/anchors/anchor_real.tsv,
    vendor/stereo-matching/anchors/seed123_rest.tsv (task=temperature rows)
  - seed-42/123, 3000-step diagnostic: vendor/stereo-matching/anchors/diag_temp_featnorm_3000.tsv,
    vendor/stereo-matching/anchors/diag_temp_featnorm_3000_s123.tsv

The surface code remains in vendor/stereo-matching/solution/temperature.py and
vendor/stereo-matching/baselines/temp_{high,ref}.py for provenance; no task is
shipped for it.

# DROPPED surface: cv-count-upsample (NOT shipped)

## Research question (from task_description.md)
OUTPUT STRIDE / Upsampling decoder

This editable surface (a learned upsampling decoder producing a finer-stride density map (learned) vs the coarse stride-8 output with no decoder (none)) was re-anchored on REAL ShanghaiTech-derived crowd-density
scenes (Zhang et al., CVPR 2016; medium/dense/superdense count-extrapolation buckets: train counts
LOW, held-out val counts HIGHER, disjoint ranges, so a degenerate constant-mean predictor
cannot win by construction) via a full GPU cross-seed (42/123) sweep
(vendor/crowd-counting/run_anchors.py), but is NOT robustly monotone (strong beats weak on
every scene, every seed) at the package-default 450-iter budget.

a learned upsampling decoder producing a finer-stride density map (learned) vs the coarse stride-8 output with no decoder (none) (OUTPUT STRIDE / Upsampling decoder) is NOT cross-seed monotone on real ShanghaiTech-derived crowd data at the package-default budget (450 iters): 3/6 (scene,seed) cells fail (strong-baseline MAE >= weak-baseline MAE, i.e. the intended SOTA direction loses). A training-budget diagnostic at 1500 iters (5x longer, NOT an HP-sweep -- a single independent longer-budget rerun, same seeds/harness) does not resolve this: 5/6 cells still fail, and the specific (scene,seed) cells that fail relocate to different combinations rather than shrinking toward zero. This rules out an undertraining artifact and instead indicates a genuine confound / noise-dominated margin for this surface on real data at these operating points.

Per-(scene,seed) counting MAE, strong vs weak, at the package-default 450-iter budget:
    seed42  medium       strong(learned)= 37.0643  weak(none)= 59.4973  [OK]
    seed123 medium       strong(learned)= 47.9571  weak(none)= 37.3303  [FAIL]
    seed42  dense        strong(learned)= 75.2171  weak(none)= 68.9377  [FAIL]
    seed123 dense        strong(learned)= 82.1680  weak(none)= 99.7063  [OK]
    seed42  superdense   strong(learned)= 66.5308  weak(none)= 55.4369  [FAIL]
    seed123 superdense   strong(learned)= 59.0461  weak(none)= 70.5984  [OK]
  ==> 3/6 cells clean (strong < weak).

Training-budget diagnostic at 1500 iters (same seeds, same harness, NOT an HP-sweep -- a
single independent longer-budget rerun per the project's never-HP-sweep mandate, run
identically for all 11 cv-count-* surfaces in this batch):
    seed42  medium       strong(learned)= 35.6100  weak(none)= 35.1351  [FAIL]
    seed123 medium       strong(learned)= 32.2641  weak(none)= 33.8376  [OK]
    seed42  dense        strong(learned)= 82.9682  weak(none)= 62.3420  [FAIL]
    seed123 dense        strong(learned)= 72.7954  weak(none)= 53.5992  [FAIL]
    seed42  superdense   strong(learned)= 77.0946  weak(none)= 64.8334  [FAIL]
    seed123 superdense   strong(learned)= 67.0730  weak(none)= 60.3660  [FAIL]
  ==> 1/6 cells clean (strong < weak).

Since the diagnostic does not cleanly resolve the ordering (failures persist and/or
relocate to different (scene,seed) cells rather than shrinking to zero), this surface is
dropped honestly rather than shipped with a forced ordering. Contrast with
cv-count-formulation and cv-count-normalization, the two crowd-counting surfaces in this
batch that DID become fully cross-seed clean (6/6 cells) at 1500 iters and are shipped at
that budget.

The surface code remains in vendor/crowd-counting/solution/*.py, and this task's
edits/, scripts/, task_description.md, and leaderboard.csv (updated to the real 450-iter
seed-42 measurements below) remain in this directory for provenance; no task is shipped
for this surface (config.json baselines/test_cmds are emptied, score_spec.py is a stub).

Full per-anchor-line provenance (all seeds/scenes/iters for this task):
  - seed-42/123, 450-iter re-anchor: GPU sweep on k1 H20 (torch 2.4.0), output rows for
    task=cv-count-upsample in the anchor_real_full.tsv sweep file.
  - seed-42/123, 1500-iter diagnostic: GPU sweep on k1 H20 (torch 2.4.0), output rows for
    task=cv-count-upsample in the anchor_diag_1500.tsv sweep file.

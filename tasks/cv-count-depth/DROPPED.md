# DROPPED surface: cv-count-depth (NOT shipped)

## Research question (from task_description.md)
Backbone DEPTH (shallow vs deep)

This editable surface (a deeper backbone with a post-pool refinement block (deep) vs a shallow one-conv-per-stage backbone (shallow)) was re-anchored on REAL ShanghaiTech-derived crowd-density
scenes (Zhang et al., CVPR 2016; medium/middense/dense count-extrapolation buckets: train counts
LOW, held-out val counts HIGHER, disjoint ranges, so a degenerate constant-mean predictor
cannot win by construction) via a full GPU cross-seed (42/123) sweep
(vendor/crowd-counting/run_anchors.py), but is NOT robustly monotone (strong beats weak on
every scene, every seed) at the package-default 450-iter budget.

a deeper backbone with a post-pool refinement block (deep) vs a shallow one-conv-per-stage backbone (shallow) (Backbone DEPTH (shallow vs deep)) is NOT cross-seed monotone on real ShanghaiTech-derived crowd data at the package-default budget (450 iters): 4/6 (scene,seed) cells fail (strong-baseline MAE >= weak-baseline MAE, i.e. the intended SOTA direction loses). A training-budget diagnostic at 1500 iters (5x longer, NOT an HP-sweep -- a single independent longer-budget rerun, same seeds/harness) does not resolve this: 3/6 cells still fail, and the specific (scene,seed) cells that fail relocate to different combinations rather than shrinking toward zero. This rules out an undertraining artifact and instead indicates a genuine confound / noise-dominated margin for this surface on real data at these operating points.

Per-(scene,seed) counting MAE, strong vs weak, at the package-default 450-iter budget:
    seed42  medium       strong(deep)= 35.0976  weak(shallow)= 50.5523  [OK]
    seed123 medium       strong(deep)= 62.8616  weak(shallow)= 40.8990  [FAIL]
    seed42  middense     strong(deep)= 76.6337  weak(shallow)= 50.5242  [FAIL]
    seed123 middense     strong(deep)= 69.3331  weak(shallow)= 64.3782  [FAIL]
    seed42  dense        strong(deep)= 84.5620  weak(shallow)= 50.5234  [FAIL]
    seed123 dense        strong(deep)= 64.0121  weak(shallow)= 75.6673  [OK]
  ==> 2/6 cells clean (strong < weak).

Training-budget diagnostic at 1500 iters (same seeds, same harness, NOT an HP-sweep -- a
single independent longer-budget rerun per the project's never-HP-sweep mandate, run
identically for all 11 cv-count-* surfaces in this batch):
    seed42  medium       strong(deep)= 39.5575  weak(shallow)= 35.0312  [FAIL]
    seed123 medium       strong(deep)= 51.2290  weak(shallow)= 47.1309  [FAIL]
    seed42  middense     strong(deep)= 49.7693  weak(shallow)= 66.7697  [OK]
    seed123 middense     strong(deep)= 45.4818  weak(shallow)= 72.7142  [OK]
    seed42  dense        strong(deep)= 63.5904  weak(shallow)= 93.1513  [OK]
    seed123 dense        strong(deep)= 79.5404  weak(shallow)= 60.5410  [FAIL]
  ==> 3/6 cells clean (strong < weak).

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
    task=cv-count-depth in the anchor_real_full.tsv sweep file.
  - seed-42/123, 1500-iter diagnostic: GPU sweep on k1 H20 (torch 2.4.0), output rows for
    task=cv-count-depth in the anchor_diag_1500.tsv sweep file.

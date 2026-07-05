# DROPPED surface: cv-count-batchnorm (NOT shipped)

## Research question (from task_description.md)
Backbone NORMALIZATION (none vs BatchNorm)

This editable surface (a BatchNorm-stabilised backbone (bn) vs no normalization (none)) was re-anchored on REAL ShanghaiTech-derived crowd-density
scenes (Zhang et al., CVPR 2016; medium/middense/dense count-extrapolation buckets: train counts
LOW, held-out val counts HIGHER, disjoint ranges, so a degenerate constant-mean predictor
cannot win by construction) via a full GPU cross-seed (42/123) sweep
(vendor/crowd-counting/run_anchors.py), but is NOT robustly monotone (strong beats weak on
every scene, every seed) at the package-default 450-iter budget.

a BatchNorm-stabilised backbone (bn) vs no normalization (none) (Backbone NORMALIZATION (none vs BatchNorm)) is NOT cross-seed monotone on real ShanghaiTech-derived crowd data at the package-default budget (450 iters): 1/6 (scene,seed) cells fail (strong-baseline MAE >= weak-baseline MAE, i.e. the intended SOTA direction loses). A training-budget diagnostic at 1500 iters (5x longer, NOT an HP-sweep -- a single independent longer-budget rerun, same seeds/harness) does not resolve this: 4/6 cells still fail, and the specific (scene,seed) cells that fail relocate to different combinations rather than shrinking toward zero. This rules out an undertraining artifact and instead indicates a genuine confound / noise-dominated margin for this surface on real data at these operating points.

Per-(scene,seed) counting MAE, strong vs weak, at the package-default 450-iter budget:
    seed42  medium       strong(bn)= 45.4495  weak(none)= 36.3696  [FAIL]
    seed123 medium       strong(bn)= 34.3307  weak(none)= 48.9556  [OK]
    seed42  middense     strong(bn)= 29.4619  weak(none)= 85.3909  [OK]
    seed123 middense     strong(bn)= 42.2101  weak(none)= 53.5260  [OK]
    seed42  dense        strong(bn)= 47.9176  weak(none)= 72.3975  [OK]
    seed123 dense        strong(bn)= 51.1173  weak(none)= 98.2001  [OK]
  ==> 5/6 cells clean (strong < weak).

Training-budget diagnostic at 1500 iters (same seeds, same harness, NOT an HP-sweep -- a
single independent longer-budget rerun per the project's never-HP-sweep mandate, run
identically for all 11 cv-count-* surfaces in this batch):
    seed42  medium       strong(bn)= 42.6654  weak(none)= 32.1184  [FAIL]
    seed123 medium       strong(bn)= 39.0360  weak(none)= 34.8792  [FAIL]
    seed42  middense     strong(bn)= 45.1324  weak(none)= 52.7477  [OK]
    seed123 middense     strong(bn)= 61.9683  weak(none)= 61.8017  [FAIL]
    seed42  dense        strong(bn)= 62.5660  weak(none)= 79.0518  [OK]
    seed123 dense        strong(bn)= 68.3108  weak(none)= 63.6409  [FAIL]
  ==> 2/6 cells clean (strong < weak).

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
    task=cv-count-batchnorm in the anchor_real_full.tsv sweep file.
  - seed-42/123, 1500-iter diagnostic: GPU sweep on k1 H20 (torch 2.4.0), output rows for
    task=cv-count-batchnorm in the anchor_diag_1500.tsv sweep file.

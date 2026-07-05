# DROPPED surface: cv-count-loss (NOT shipped)

## Research question (from task_description.md)
The Density-Map Training LOSS

This editable surface (a foreground-upweighted, count-consistency-augmented loss (count) vs plain per-pixel MSE (mse)) was re-anchored on REAL ShanghaiTech-derived crowd-density
scenes (Zhang et al., CVPR 2016; medium/middense/dense count-extrapolation buckets: train counts
LOW, held-out val counts HIGHER, disjoint ranges, so a degenerate constant-mean predictor
cannot win by construction) via a full GPU cross-seed (42/123) sweep
(vendor/crowd-counting/run_anchors.py), but is NOT robustly monotone (strong beats weak on
every scene, every seed) at the package-default 450-iter budget.

a foreground-upweighted, count-consistency-augmented loss (count) vs plain per-pixel MSE (mse) (The Density-Map Training LOSS) is NOT cross-seed monotone on real ShanghaiTech-derived crowd data at the package-default budget (450 iters): 5/6 (scene,seed) cells fail (strong-baseline MAE >= weak-baseline MAE, i.e. the intended SOTA direction loses). A training-budget diagnostic at 1500 iters (5x longer, NOT an HP-sweep -- a single independent longer-budget rerun, same seeds/harness) does not resolve this: 1/6 cells still fail, and the specific (scene,seed) cells that fail relocate to different combinations rather than shrinking toward zero. This rules out an undertraining artifact and instead indicates a genuine confound / noise-dominated margin for this surface on real data at these operating points.

Per-(scene,seed) counting MAE, strong vs weak, at the package-default 450-iter budget:
    seed42  medium       strong(count)= 49.5149  weak(mse)= 37.9436  [FAIL]
    seed123 medium       strong(count)= 54.3096  weak(mse)= 36.1284  [FAIL]
    seed42  middense     strong(count)= 71.9235  weak(mse)= 76.3126  [OK]
    seed123 middense     strong(count)= 68.5459  weak(mse)= 54.6639  [FAIL]
    seed42  dense        strong(count)= 77.6833  weak(mse)= 76.4883  [FAIL]
    seed123 dense        strong(count)= 99.1968  weak(mse)= 87.8530  [FAIL]
  ==> 1/6 cells clean (strong < weak).

Training-budget diagnostic at 1500 iters (same seeds, same harness, NOT an HP-sweep -- a
single independent longer-budget rerun per the project's never-HP-sweep mandate, run
identically for all 11 cv-count-* surfaces in this batch):
    seed42  medium       strong(count)= 30.1678  weak(mse)= 32.4701  [OK]
    seed123 medium       strong(count)= 33.1024  weak(mse)= 44.1136  [OK]
    seed42  middense     strong(count)= 53.3977  weak(mse)= 51.1670  [FAIL]
    seed123 middense     strong(count)= 42.4689  weak(mse)= 57.9421  [OK]
    seed42  dense        strong(count)= 56.2226  weak(mse)= 71.2726  [OK]
    seed123 dense        strong(count)= 45.3139  weak(mse)= 54.3077  [OK]
  ==> 5/6 cells clean (strong < weak).

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
    task=cv-count-loss in the anchor_real_full.tsv sweep file.
  - seed-42/123, 1500-iter diagnostic: GPU sweep on k1 H20 (torch 2.4.0), output rows for
    task=cv-count-loss in the anchor_diag_1500.tsv sweep file.

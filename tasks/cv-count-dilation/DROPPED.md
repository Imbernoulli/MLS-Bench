# DROPPED surface: cv-count-dilation (NOT shipped)

## Research question (from task_description.md)
DILATION / Receptive Field (CSRNet's core idea)

This editable surface (a dilated large-receptive-field back-end block (dilated) vs a pooled small-receptive-field block (pooled)) was re-anchored on REAL ShanghaiTech-derived crowd-density
scenes (Zhang et al., CVPR 2016; medium/middense/dense count-extrapolation buckets: train counts
LOW, held-out val counts HIGHER, disjoint ranges, so a degenerate constant-mean predictor
cannot win by construction) via a full GPU cross-seed (42/123) sweep
(vendor/crowd-counting/run_anchors.py), but is NOT robustly monotone (strong beats weak on
every scene, every seed) at the package-default 450-iter budget.

a dilated large-receptive-field back-end block (dilated) vs a pooled small-receptive-field block (pooled) (DILATION / Receptive Field (CSRNet's core idea)) is NOT cross-seed monotone on real ShanghaiTech-derived crowd data at the package-default budget (450 iters): 4/6 (scene,seed) cells fail (strong-baseline MAE >= weak-baseline MAE, i.e. the intended SOTA direction loses). A training-budget diagnostic at 1500 iters (5x longer, NOT an HP-sweep -- a single independent longer-budget rerun, same seeds/harness) does not resolve this: 3/6 cells still fail, and the specific (scene,seed) cells that fail relocate to different combinations rather than shrinking toward zero. This rules out an undertraining artifact and instead indicates a genuine confound / noise-dominated margin for this surface on real data at these operating points.

Per-(scene,seed) counting MAE, strong vs weak, at the package-default 450-iter budget:
    seed42  medium       strong(dilated)= 51.4739  weak(pooled)= 51.3804  [FAIL]
    seed123 medium       strong(dilated)= 57.5557  weak(pooled)= 50.4709  [FAIL]
    seed42  middense     strong(dilated)= 75.8301  weak(pooled)= 85.6110  [OK]
    seed123 middense     strong(dilated)= 77.1065  weak(pooled)= 75.1837  [FAIL]
    seed42  dense        strong(dilated)= 61.7148  weak(pooled)= 81.1045  [OK]
    seed123 dense        strong(dilated)= 85.4527  weak(pooled)= 66.6853  [FAIL]
  ==> 2/6 cells clean (strong < weak).

Training-budget diagnostic at 1500 iters (same seeds, same harness, NOT an HP-sweep -- a
single independent longer-budget rerun per the project's never-HP-sweep mandate, run
identically for all 11 cv-count-* surfaces in this batch):
    seed42  medium       strong(dilated)= 49.3693  weak(pooled)= 50.3710  [OK]
    seed123 medium       strong(dilated)= 55.1193  weak(pooled)= 51.2726  [FAIL]
    seed42  middense     strong(dilated)= 54.3757  weak(pooled)= 55.2368  [OK]
    seed123 middense     strong(dilated)= 58.1758  weak(pooled)= 59.3407  [OK]
    seed42  dense        strong(dilated)= 79.6325  weak(pooled)= 78.0624  [FAIL]
    seed123 dense        strong(dilated)= 84.3880  weak(pooled)= 80.5663  [FAIL]
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
    task=cv-count-dilation in the anchor_real_full.tsv sweep file.
  - seed-42/123, 1500-iter diagnostic: GPU sweep on k1 H20 (torch 2.4.0), output rows for
    task=cv-count-dilation in the anchor_diag_1500.tsv sweep file.

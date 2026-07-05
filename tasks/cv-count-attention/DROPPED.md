# DROPPED surface: cv-count-attention (NOT shipped)

## Research question (from task_description.md)
Spatial ATTENTION (clutter suppression)

This editable surface (a learned spatial-attention clutter-suppression gate (spatial) vs no attention (none)) was re-anchored on REAL ShanghaiTech-derived crowd-density
scenes (Zhang et al., CVPR 2016; medium/middense/dense count-extrapolation buckets: train counts
LOW, held-out val counts HIGHER, disjoint ranges, so a degenerate constant-mean predictor
cannot win by construction) via a full GPU cross-seed (42/123) sweep
(vendor/crowd-counting/run_anchors.py), but is NOT robustly monotone (strong beats weak on
every scene, every seed) at the package-default 450-iter budget.

a learned spatial-attention clutter-suppression gate (spatial) vs no attention (none) (Spatial ATTENTION (clutter suppression)) is NOT cross-seed monotone on real ShanghaiTech-derived crowd data at the package-default budget (450 iters): 4/6 (scene,seed) cells fail (strong-baseline MAE >= weak-baseline MAE, i.e. the intended SOTA direction loses). A training-budget diagnostic at 1500 iters (5x longer, NOT an HP-sweep -- a single independent longer-budget rerun, same seeds/harness) does not resolve this: 3/6 cells still fail, and the specific (scene,seed) cells that fail relocate to different combinations rather than shrinking toward zero. This rules out an undertraining artifact and instead indicates a genuine confound / noise-dominated margin for this surface on real data at these operating points.

Per-(scene,seed) counting MAE, strong vs weak, at the package-default 450-iter budget:
    seed42  medium       strong(spatial)= 53.8507  weak(none)= 49.1986  [FAIL]
    seed123 medium       strong(spatial)= 49.7933  weak(none)= 55.1005  [OK]
    seed42  middense     strong(spatial)= 75.3842  weak(none)= 74.5079  [FAIL]
    seed123 middense     strong(spatial)= 73.2130  weak(none)= 54.9803  [FAIL]
    seed42  dense        strong(spatial)= 58.7704  weak(none)= 66.2306  [OK]
    seed123 dense        strong(spatial)=105.4670  weak(none)= 85.0299  [FAIL]
  ==> 2/6 cells clean (strong < weak).

Training-budget diagnostic at 1500 iters (same seeds, same harness, NOT an HP-sweep -- a
single independent longer-budget rerun per the project's never-HP-sweep mandate, run
identically for all 11 cv-count-* surfaces in this batch):
    seed42  medium       strong(spatial)= 39.8257  weak(none)= 36.1077  [FAIL]
    seed123 medium       strong(spatial)= 42.0998  weak(none)= 35.8957  [FAIL]
    seed42  middense     strong(spatial)= 41.7715  weak(none)= 44.9224  [OK]
    seed123 middense     strong(spatial)= 45.7070  weak(none)= 57.0736  [OK]
    seed42  dense        strong(spatial)= 57.7771  weak(none)= 63.1290  [OK]
    seed123 dense        strong(spatial)= 71.4215  weak(none)= 71.0148  [FAIL]
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
    task=cv-count-attention in the anchor_real_full.tsv sweep file.
  - seed-42/123, 1500-iter diagnostic: GPU sweep on k1 H20 (torch 2.4.0), output rows for
    task=cv-count-attention in the anchor_diag_1500.tsv sweep file.

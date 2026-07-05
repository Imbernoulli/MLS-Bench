# DROPPED surface: cv-count-multiscale (NOT shipped)

## Research question (from task_description.md)
MULTI-SCALE Context aggregation (CAN-style)

This editable surface (multi-scale pooled-context aggregation (context) vs single-scale context (single)) was re-anchored on REAL ShanghaiTech-derived crowd-density
scenes (Zhang et al., CVPR 2016; medium/dense/superdense count-extrapolation buckets: train counts
LOW, held-out val counts HIGHER, disjoint ranges, so a degenerate constant-mean predictor
cannot win by construction) via a full GPU cross-seed (42/123) sweep
(vendor/crowd-counting/run_anchors.py), but is NOT robustly monotone (strong beats weak on
every scene, every seed) at the package-default 450-iter budget.

multi-scale pooled-context aggregation (context) vs single-scale context (single) (MULTI-SCALE Context aggregation (CAN-style)) is NOT cross-seed monotone on real ShanghaiTech-derived crowd data at the package-default budget (450 iters): 3/6 (scene,seed) cells fail (strong-baseline MAE >= weak-baseline MAE, i.e. the intended SOTA direction loses). A training-budget diagnostic at 1500 iters (5x longer, NOT an HP-sweep -- a single independent longer-budget rerun, same seeds/harness) does not resolve this: 5/6 cells still fail, and the specific (scene,seed) cells that fail relocate to different combinations rather than shrinking toward zero. This rules out an undertraining artifact and instead indicates a genuine confound / noise-dominated margin for this surface on real data at these operating points.

Per-(scene,seed) counting MAE, strong vs weak, at the package-default 450-iter budget:
    seed42  medium       strong(context)= 53.2584  weak(single)= 58.2380  [OK]
    seed123 medium       strong(context)= 57.5964  weak(single)= 38.8791  [FAIL]
    seed42  dense        strong(context)= 62.3058  weak(single)= 76.9730  [OK]
    seed123 dense        strong(context)= 93.9896  weak(single)= 88.5893  [FAIL]
    seed42  superdense   strong(context)= 53.0157  weak(single)= 62.1732  [OK]
    seed123 superdense   strong(context)= 66.7452  weak(single)= 64.9773  [FAIL]
  ==> 3/6 cells clean (strong < weak).

Training-budget diagnostic at 1500 iters (same seeds, same harness, NOT an HP-sweep -- a
single independent longer-budget rerun per the project's never-HP-sweep mandate, run
identically for all 11 cv-count-* surfaces in this batch):
    seed42  medium       strong(context)= 51.4743  weak(single)= 32.9157  [FAIL]
    seed123 medium       strong(context)= 42.0030  weak(single)= 34.5343  [FAIL]
    seed42  dense        strong(context)= 75.7049  weak(single)= 61.5993  [FAIL]
    seed123 dense        strong(context)= 66.1384  weak(single)= 68.3264  [OK]
    seed42  superdense   strong(context)= 83.6758  weak(single)= 59.3947  [FAIL]
    seed123 superdense   strong(context)= 81.1018  weak(single)= 76.8525  [FAIL]
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
    task=cv-count-multiscale in the anchor_real_full.tsv sweep file.
  - seed-42/123, 1500-iter diagnostic: GPU sweep on k1 H20 (torch 2.4.0), output rows for
    task=cv-count-multiscale in the anchor_diag_1500.tsv sweep file.

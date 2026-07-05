# DROPPED surface: cv-count-architecture (NOT shipped)

## Research question (from task_description.md)
The Counting ARCHITECTURE (plain CNN vs multi-column vs dilated)

This editable surface (the SOTA CSRNet-style dilated backbone (csrnet) vs a plain single-column CNN backbone (plain)) was re-anchored on REAL ShanghaiTech-derived crowd-density
scenes (Zhang et al., CVPR 2016; medium/middense/dense count-extrapolation buckets: train counts
LOW, held-out val counts HIGHER, disjoint ranges, so a degenerate constant-mean predictor
cannot win by construction) via a full GPU cross-seed (42/123) sweep
(vendor/crowd-counting/run_anchors.py), but is NOT robustly monotone (strong beats weak on
every scene, every seed) at the package-default 450-iter budget.

the SOTA CSRNet-style dilated backbone (csrnet) vs a plain single-column CNN backbone (plain) (The Counting ARCHITECTURE (plain CNN vs multi-column vs dilated)) is NOT cross-seed monotone on real ShanghaiTech-derived crowd data at the package-default budget (450 iters): 5/6 (scene,seed) cells fail (strong-baseline MAE >= weak-baseline MAE, i.e. the intended SOTA direction loses). A training-budget diagnostic at 1500 iters (5x longer, NOT an HP-sweep -- a single independent longer-budget rerun, same seeds/harness) does not resolve this: 4/6 cells still fail, and the specific (scene,seed) cells that fail relocate to different combinations rather than shrinking toward zero. This rules out an undertraining artifact and instead indicates a genuine confound / noise-dominated margin for this surface on real data at these operating points.

Per-(scene,seed) counting MAE, strong vs weak, at the package-default 450-iter budget:
    seed42  medium       strong(csrnet)= 53.4300  weak(plain)= 55.7559  [OK]
    seed123 medium       strong(csrnet)= 57.8295  weak(plain)= 49.2988  [FAIL]
    seed42  middense     strong(csrnet)= 79.8620  weak(plain)= 58.4474  [FAIL]
    seed123 middense     strong(csrnet)= 74.6212  weak(plain)= 49.1819  [FAIL]
    seed42  dense        strong(csrnet)= 87.5249  weak(plain)= 69.2413  [FAIL]
    seed123 dense        strong(csrnet)= 74.9651  weak(plain)= 69.9968  [FAIL]
  ==> 1/6 cells clean (strong < weak).

Training-budget diagnostic at 1500 iters (same seeds, same harness, NOT an HP-sweep -- a
single independent longer-budget rerun per the project's never-HP-sweep mandate, run
identically for all 11 cv-count-* surfaces in this batch):
    seed42  medium       strong(csrnet)= 43.0908  weak(plain)= 38.2325  [FAIL]
    seed123 medium       strong(csrnet)= 58.0985  weak(plain)= 52.6518  [FAIL]
    seed42  middense     strong(csrnet)= 54.2251  weak(plain)= 77.3270  [OK]
    seed123 middense     strong(csrnet)= 47.2440  weak(plain)= 76.4442  [OK]
    seed42  dense        strong(csrnet)= 93.7833  weak(plain)= 87.1463  [FAIL]
    seed123 dense        strong(csrnet)= 77.7973  weak(plain)= 74.6025  [FAIL]
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
    task=cv-count-architecture in the anchor_real_full.tsv sweep file.
  - seed-42/123, 1500-iter diagnostic: GPU sweep on k1 H20 (torch 2.4.0), output rows for
    task=cv-count-architecture in the anchor_diag_1500.tsv sweep file.

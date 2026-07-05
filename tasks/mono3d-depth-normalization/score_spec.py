"""Score spec for mono3d-depth-normalization.

Primary metric per setting: AP3D@0.25 = fraction of TEST objects (in this KITTI difficulty tier) whose
predicted 3D box has 3D IoU >= 0.25 with GT (HIGHER better; a degenerate predictor scores ~0).
The task score is the geometric mean of the per-setting AP3D over KITTI's own OFFICIAL
easy/moderate/hard difficulty tiers (bbox-height/occlusion/truncation thresholds), applied
EXCLUSIVELY as a disjoint partition of the held-out REAL KITTI test split.

HONEST RELABEL (B0 8xH200, torch 2.4.1, REAL KITTI, CROSS-SEED 42/123, full 1200-step budget,
2026-07-05): on the OLD synthetic dataset the log-space multiplicative depth residual
(norm_log_mult) cleanly beat the raw additive residual (norm_additive). A 2026-07-05 CPU smoke
test on real KITTI flagged the opposite ordering; this was then GPU re-anchored cross-seed to
confirm it is a robust, reproducible effect rather than noise:

  AP3D@0.25 (seed-averaged over 42/123):
    norm_additive (now STRONG/SOTA)   easy=0.3569  moderate=0.1848  hard=0.1850
    norm_log_mult (now WEAK)          easy=0.3420  moderate=0.1650  hard=0.1783
  Per-seed detail:
    easy:     seed42 additive=0.352555 log_mult=0.334307 | seed123 additive=0.361314 log_mult=0.349635
    moderate: seed42 additive=0.187961 log_mult=0.172061 | seed123 additive=0.181715 log_mult=0.157865
    hard:     seed42 additive=0.177373 log_mult=0.185043 | seed123 additive=0.192713 log_mult=0.171620

additive wins on 5/6 (setting, seed) cells and ROBUSTLY wins at the task-level geometric mean on
both seeds (seed42: 0.2274 vs 0.2200; seed123: 0.2330 vs 0.2116); the lone exception
(hard/seed42, delta=-0.008) is a near-tie, not a robust inversion in the other direction. This
is treated as a legitimate real-data relabel (see reg-similarity-loss's MSE>NCC relabel for the
established provenance pattern), not a dropped task: on real, noisy LiDAR-derived KITTI depth,
the geometry base depth Z0=f*H/h2d already carries substantial occlusion/truncation/annotation
noise, and the raw additive residual apparently adapts to this noise pattern at least as well
as -- and on this budget/data slightly better than -- the scale-invariant log-space
multiplicative residual, reversing the synthetic-data intuition. Per-setting logistic midpoint
= the (new) strong reference (norm_additive) -> score 0.5, scale = (strong-weak)/ln(9) so the
(new) weak baseline (norm_log_mult) lands ~0.1.
"""
from mlsbench.scoring.dsl import *

term("ap25_easy",
    col("ap25_easy").higher().id().sigmoid(ref=const(0.356935), scale=0.006810182333815348))
term("ap25_moderate",
    col("ap25_moderate").higher().id().sigmoid(ref=const(0.184838), scale=0.009045502314604197))
term("ap25_hard",
    col("ap25_hard").higher().id().sigmoid(ref=const(0.185043), scale=0.0030545352847530134))

setting("easy", weighted_mean(("ap25_easy", 1.0)))
setting("moderate", weighted_mean(("ap25_moderate", 1.0)))
setting("hard", weighted_mean(("ap25_hard", 1.0)))

task(gmean("easy", "moderate", "hard"))

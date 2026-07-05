# DROPPED surface: mono3d-depth-normalization (NOT shipped)

This editable surface (raw additive-metres depth residual `norm_additive` vs scale-invariant
log-space multiplicative depth residual `norm_log_mult`) was DESIGNED and GPU-VALIDATED on a
fully-synthetic procedurally-rendered dataset (where the log-space residual cleanly beat the
additive residual, matching the design intuition that depth's multiplicative 6-40m range
favours a scale-invariant correction). An earlier honest relabel pass, after re-anchoring on
REAL KITTI 3D Object Detection cross-seed data (B0 8xH200, torch 2.4.1, full 1200-step budget,
seeds 42/123), flipped the ordering and shipped `norm_additive` as the new strong/SOTA
baseline, reasoning that additive wins the task-level geometric mean on both seeds and 5/6
individual (setting, seed) cells.

That relabel is now retracted under the project's SOTA=0.5 anchor discipline ("0.5 is the
strongest baseline", i.e. every per-setting term must anchor its ref to the strong baseline's
SEED-42-SPECIFIC value so the strong baseline scores exactly 0.5 on seed 42, with weak <0.5).
Applying that discipline exposes that the "additive wins" ordering is NOT robust across all
three difficulty tiers at seed 42 specifically:

AP3D@0.25 per (setting, seed):
  easy:     seed42 additive=0.352555 log_mult=0.334307  (additive wins, delta=+0.018248)
            seed123 additive=0.361314 log_mult=0.349635 (additive wins, delta=+0.011679)
  moderate: seed42 additive=0.187961 log_mult=0.172061  (additive wins, delta=+0.015900)
            seed123 additive=0.181715 log_mult=0.157865 (additive wins, delta=+0.023850)
  hard:     seed42 additive=0.177373 log_mult=0.185043  (additive LOSES, delta=-0.007670)
            seed123 additive=0.192713 log_mult=0.171620 (additive wins, delta=+0.021093)

The `hard` tier inverts at seed 42: additive scores BELOW log_mult there, even though it wins
easy/moderate at seed 42 and wins all three tiers at seed 123. This is a small, noise-level
delta (-0.0077 AP3D on a held-out split of ~a few hundred hard-tier objects), but it is enough
to break the SOTA=0.5 anchor construction: per-setting sigmoid terms require
`scale = (strong_seed42 - weak_seed42) / ln(9)` to be positive so that the strong baseline
lands at score 0.5 and the weak baseline lands at ~0.1. For the `hard` term, that requires
`additive_seed42 > log_mult_seed42`, which does not hold. There is no single seed-42-anchored
scoring construction that puts additive at 0.5 on hard while preserving the intended
weak<strong direction (log_mult would have to be relabeled as strong on `hard` alone, which
would then contradict `easy`/`moderate` and the task-level cross-seed evidence used to justify
the original relabel).

Per project mandate (never HP-sweep or cherry-pick per-setting reference conventions to force
monotonicity; if the real cross-seed numbers show a seed-unstable near-tie rather than a
robust ordering, drop honestly rather than force an anchor), this surface is dropped. This
also matches precedent: `mono3d-depth-parameterization` was similarly dropped for a scrambled
3-way order across difficulty tiers, and `cv-matting-attention` was dropped for a sign flip
across seeds on one of its three trimap-width settings. The surface code remains in
`mono3d-detection/solution/depth_norm.py` (lines 26-32) + this task's `edits/`/`scripts/`/
`task_description.md`/`leaderboard.csv` for provenance, but no task is shipped for it.

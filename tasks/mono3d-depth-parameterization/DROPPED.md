# DROPPED surface: mono3d-depth-parameterization (NOT shipped)

This editable surface (direct depth regression vs analytic geometry-height decode vs
geometry+learned-residual decode) was DESIGNED and GPU-VALIDATED on a fully-synthetic
near/mid/far distance dataset (where a clean, uniform depth prior separated direct regression
from the geometry-based decodes), but on the REAL KITTI 3D Object Detection dataset
cross-seed re-anchor (B0 8xH200, torch 2.4.1, 1200 steps, seeds 42/123, bucketed into KITTI's
own official easy/moderate/hard difficulty tiers) the intended 3-way ordering
(`direct_regression` WEAK < `geometry_height` MEDIUM < `geometry_residual` STRONG/SOTA) is
completely scrambled — no consistent pairwise relationship holds across all three settings.

AP3D@0.25 (seed-averaged over 42/123):
  easy:     direct_regression=0.2978  geometry_height=0.2781  geometry_residual=0.3259
  moderate: direct_regression=0.1661  geometry_height=0.1931  geometry_residual=0.1661
  hard:     direct_regression=0.1438  geometry_height=0.2229  geometry_residual=0.1702

Task-level geometric mean:
  seed42:  direct_regression=0.19166  geometry_height=0.23528  geometry_residual=0.20844
  seed123: direct_regression=0.19295  geometry_height=0.22218  geometry_residual=0.21080

Per-pair breakdown of why no ordering survives:
- `direct_regression` vs `geometry_height`: geometry_height LOSES to direct_regression on
  `easy` (both seeds) despite winning `moderate`/`hard` (both seeds) — the intended
  weak<medium relationship fails on 1/3 settings.
- `geometry_height` vs `geometry_residual`: geometry_height BEATS geometry_residual on `hard`
  (both seeds, by a wide margin: 0.223 vs 0.170) despite losing on `easy`/`moderate` — the
  intended medium<strong relationship fails on 1/3 settings, and in the opposite direction
  from the previous bullet.
- `direct_regression` vs `geometry_residual`: this pair alone is robustly monotone
  (residual > direct on all 3 settings, both seeds) but the task ships all 3 baselines, and
  the middle rung (`geometry_height`) is not `direct <= height <= residual` on any consistent
  basis — it is simultaneously below `direct_regression` (easy) and above `geometry_residual`
  (hard).

There is no robust re-ordering that keeps all three baselines in a stable weak/medium/strong
chain across all three real-KITTI difficulty tiers, and per project mandate (never HP-sweep to
force monotonicity), this surface is dropped rather than hunted for a hyperparameter regime
that manufactures a monotone triple. The surface code remains in
`mono3d-detection/solution/depth_param.py` (lines 39-48) + this task's `edits/`/`scripts/`/
`task_description.md` for provenance, but no task is shipped for it.

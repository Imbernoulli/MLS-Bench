# Task: Monocular 3D Detection — Depth Residual Normalization

## Research Question
The geometry depth `Z0 = f·H/h2d` is corrected by a learned residual. Metric depth spans 6–40 m — a **multiplicative** range. Should the correction be a **raw additive** residual in metres (badly scaled across distance, can go negative) or a **multiplicative log-space** residual `Z = Z0·exp(0.1·clamp(r))` (scale-invariant, strictly positive)?

## Background
The data is **REAL KITTI 3D Object Detection** (Geiger et al. 2012), fetched from the public
unauthenticated S3 mirror of the official archives (bypassing the cvlibs.net registration
wall). Every Car/Pedestrian/Cyclist object in the labelled `training` split is a real photo
region with a real, LiDAR-derived 3D box (metric depth/dims/yaw) and real per-image
calibration; a single fixed representative pinhole intrinsic `K` (the dataset-mean calibration
— see `common.py` for the measured-variance justification) is used at decode time. The amodal
2D box (from the label) is used to build the same normalized geometry feature vector as
before, plus a REAL cropped-and-resized appearance patch (replacing the old procedurally-
rendered descriptor). Because GT dims/loc/yaw are real annotations, 3D IoU and AP3D are
computed against real (occasionally noisy) LiDAR-derived geometry — the standard KITTI
evaluation convention. A fixed shared `RegionEncoder` and the other task heads are frozen;
the only degree of freedom is **depth residual normalization**.

## What to Implement
Implement `build_depth_norm()`: return `apply(geom_Z, raw) -> Z [B]` combining the analytic `geom_Z` with the head's `raw` output. The default is the raw additive-metres residual; `build_depth_norm` may instead return the log-space multiplicative residual. A broken/empty surface falls back to the fixed strong default (additive; see Background).

### Background: real-data relabel
On procedurally-rendered synthetic data (the original design intuition), the scale-invariant
log-space multiplicative residual `Z = Z0·exp(0.1·clamp(r))` cleanly beat a raw additive-metres
residual, because depth there spans a clean 6-40 m multiplicative range with no annotation
noise. Measured on this task's REAL KITTI data (B0 8xH200, torch 2.4.1, full 1200-step budget,
cross-seed 42/123), however, the ordering is the OPPOSITE: **the raw additive residual
consistently beats the log-space multiplicative residual**:

| setting  | norm_additive (seed-avg) | norm_log_mult (seed-avg) |
|----------|--------------------------|--------------------------|
| easy     | 0.3569                   | 0.3420                   |
| moderate | 0.1848                   | 0.1650                   |
| hard     | 0.1850                   | 0.1783                   |

additive wins 5 of 6 individual (setting, seed) combinations, and wins the task-level geometric
mean on both seeds (seed 42: 0.2274 vs 0.2200; seed 123: 0.2330 vs 0.2116); the lone exception
(hard/seed 42) is a near-tie (Δ=-0.008), not a robust inversion. A plausible mechanistic
explanation: on real KITTI the analytic base depth `Z0=f·H/h2d` already carries substantial
occlusion/truncation/annotation noise (unlike the noise-free synthetic simulator), and the
unconstrained additive residual apparently adapts to this real noise pattern at least as
readily as the scale-invariant log-space residual, within this fixed training budget. So the
task is scored with **the additive residual as the strong/SOTA reference and the log-space
multiplicative residual as the weaker baseline** — the reverse of the design narrative above,
which is exactly why real-data re-anchoring matters (cf. `reg-similarity-loss`'s analogous
MSE>NCC real-data relabel).

## Fixed Pipeline
Data, splits, the encoder, the non-studied heads, optimizer, epochs, batch, seed, and scoring are
ALL fixed.

## Settings (scored, ≥3)
The score aggregates AP3D over KITTI's own **OFFICIAL easy/moderate/hard difficulty tiers**
(2D-bbox-height + occlusion + truncation thresholds; see `common.DIFFICULTY_SETTINGS` /
`common.kitti_difficulty`), applied EXCLUSIVELY so the three tiers are a disjoint partition
of the held-out REAL KITTI test split. Monocular depth/appearance ambiguity is hardest for
small/occluded/truncated objects, so every method's AP3D is expected to drop easy→hard.

## Metric
Per setting: **AP3D@0.25** — the fraction of test objects whose predicted 3D box has 3D IoU ≥ 0.25
with GT (HIGHER better; a degenerate predictor scores ~0). The task score is the geometric mean
across the three difficulty-tier settings. AP3D@0.5, mean 3D IoU, and decomposed depth/yaw/dim errors are
reported for feedback.

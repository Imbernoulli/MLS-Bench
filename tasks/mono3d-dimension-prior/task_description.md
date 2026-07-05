# Task: Monocular 3D Detection — Dimension Prior / Shape Anchors

## Research Question
How you predict an object's metric box **dimensions (l, h, w)** matters twice over: dims enter the 3D IoU directly, and the metric **height H** feeds the fixed geometry depth `Z = f·H/h2d`. Object dimensions have a strong, low-variance **per-class prior** (a car is ~1.5 m tall). This task asks: regress the metric dims **directly** (no prior), or predict a small **residual on the log class-mean anchor** (Deep3DBox / MonoDLE shape prior)? The prior gives an accurate H immediately, tightening depth as well as dims.

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
the only degree of freedom is **dimension prior / shape anchors**.

## What to Implement
Implement `build_dims_head(emb_dim, log_mean, cls_dims)`: return `(head, decode)` where `decode(raw, ctx) -> dims [B,3]` (POSITIVE metric l,h,w). `log_mean` [3] is the log mean canonical dims; `cls_dims` [3,3] the per-class canonical dims. The WEAK default regresses dims directly; anchoring `exp(log_mean + 0.3·raw)` is far stronger. A broken/empty surface falls back to the fixed strong default.

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

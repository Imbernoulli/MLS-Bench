# Task: Monocular 3D Detection — 3D Center Back-Projection

## Research Question
Given the estimated depth Z, the 3D box CENTER's lateral/vertical position (x, y) must be recovered from the 2D-box center via the **pinhole** inverse projection `x=(u-cx)·Z/f, y=(v-cy)·Z/f`. Objects are generally OFF the camera axis, so assuming they are centered (x=y=0) yields a large center error that collapses the 3D IoU even with a perfect depth. How should the 2D center be lifted to 3D?

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
the only degree of freedom is **3d center back-projection**.

## What to Implement
Implement `build_backproject()`: return `backproject(loc_z, box2d, cx, cy, focal) -> (x [B], y [B])`. The WEAK default is the ON-AXIS assumption (x=y=0); the full pinhole back-projection is far stronger. A broken/empty surface falls back to the fixed strong default.

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

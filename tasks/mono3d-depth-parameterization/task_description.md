# Task: Monocular 3D Detection — Depth Parameterization

## Research Question
Recovering an object's **metric depth Z** from a *single* image is the central, famously
ill-posed problem of monocular 3D object detection: perspective projection destroys absolute
scale, so appearance alone cannot fix distance. The decisive design choice is **how you
parameterize depth**. Regressing Z *directly* as an unbounded scalar is dominated by far
objects and generalizes poorly across distance. Using the projective **geometry** — an object
of metric height `H` projects to a 2D box of pixel height `h2d ≈ f·H/Z`, hence `Z ≈ f·H/h2d`
(Deep3DBox; GS3D; GUPNet height-guided depth) — recovers depth analytically and is far more
robust, especially at range. This task asks you to design the depth head so that **AP3D** is
as high as possible across three difficulty tiers.

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
evaluation convention.

A fixed shared `RegionEncoder` (tiny CNN over the crop + MLP over the geometry features) and a
fixed dimensions head are frozen. The only degree of freedom is your **depth parameterization**.

## What to Implement
Implement `build_depth_head(emb_dim)` in `mono3d-detection/solution/depth_param.py`:
```python
def build_depth_head(emb_dim):
    # return (head: nn.Module, decode: callable)
    # head(emb)          -> raw tensor [B, k]
    # decode(raw, ctx)   -> Z [B]   (POSITIVE metric depth)
    ...
    return head, decode
```
`decode` receives a projective context `ctx`:
- `ctx["focal"]` — scalar pinhole focal length `f` (pixels)
- `ctx["h2d"]`   — [B] pixel HEIGHT of the amodal 2D box (the inverse-depth cue)
- `ctx["pred_H"]`— [B] predicted metric object height `H` (from the fixed dims head)
- `ctx["box2d"]` — [B,4] amodal box; `ctx["cx"]`, `ctx["cy"]` principal point

The **default is the weak baseline** (direct softplus regression of Z). A projective decode
(`Z = f·pred_H/h2d`, optionally times a small learned residual) is much stronger.

## Fixed Pipeline
Data, splits, the encoder, the dims head, the orientation head (fixed strong MultiBin),
optimizer, epochs, batch, seed, and scoring are ALL fixed. A broken/empty surface falls back
to the fixed strong geometry depth head.

## Settings (scored, ≥3)
The score aggregates AP3D over KITTI's own **OFFICIAL easy/moderate/hard difficulty tiers**
(2D-bbox-height + occlusion + truncation thresholds; see `common.DIFFICULTY_SETTINGS` /
`common.kitti_difficulty`), applied EXCLUSIVELY so the three tiers are a disjoint partition
of the held-out REAL KITTI test split. Monocular depth ambiguity is hardest for
small/occluded/truncated objects, so every method's AP3D is expected to drop easy→hard, and the
geometry methods' advantage over naive regression is expected to widen accordingly.

## Metric
Per setting: **AP3D@0.25** — the fraction of test objects whose predicted 3D box has 3D IoU
≥ 0.25 with its GT (HIGHER is better; a constant/mean-box predictor scores ~0). The task score
is the geometric mean across the three difficulty-tier settings. AP3D@0.5, mean 3D IoU, and decomposed
depth/yaw/dimension errors are reported for feedback.

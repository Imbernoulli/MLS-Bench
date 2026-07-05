"""Shared harness helpers for the mono3d-* MONOCULAR 3D OBJECT DETECTION tasks.

A genuinely NEW MLS-Bench direction: recover a full 3D bounding box (metric depth,
metric dimensions, yaw orientation) for an object seen in a SINGLE image. This is the
core, famously ill-posed problem of monocular 3D detection (KITTI / nuScenes; Deep3DBox,
Mousavian et al. 2017; GS3D; MonoFlex) and is DISTINCT from every existing package:
2D detection (detectron2-detection) predicts only the image-plane box; monocular depth
(depth-monocular) predicts a dense depth map, not object boxes; stereo/GS give geometry
from multiple views. Here the model sees ONE image region and must predict where the
object is in 3D — the depth is genuinely ambiguous from appearance alone.

REAL DATA (KITTI 3D Object Detection benchmark, Geiger et al. 2012, "Are we ready for
Autonomous Driving? The KITTI Vision Benchmark Suite", CVPR 2012). Every Car/Pedestrian/
Cyclist object in the KITTI `training` split (the only publicly-labelled split; KITTI's
"testing" split has hidden labels held out for the leaderboard server) is a REAL, human-
annotated 3D box: real photo appearance crop, real LiDAR-derived metric depth/dims/yaw,
real per-image camera calibration (`P2`). No box, image, or annotation is synthesized.
  * SOURCE: `s3://avg-kitti/{data_object_image_2,data_object_label_2,data_object_calib}.zip`
    — an UNAUTHENTICATED public mirror of the official KITTI archives (the official
    cvlibs.net download links redirect through an institutional-email registration wall;
    this S3 mirror serves the identical bytes with no login). See
    `vendor/data_scripts/mono3d-detection/prepare_data.py` for the download + parse code.
  * PER-OBJECT INPUT: the amodal 2D box (from the label file) is used to (a) build the
    SAME normalized geometry feature vector as before (`[cx_n, cy_n, w2d_n, h2d_n,
    log_h2d, log_w2d, aspect2d, focal_n]`) and (b) crop + resize the REAL image region for
    the appearance crop (replacing the old procedurally-rendered descriptor patch). The
    `RegionEncoder`, its embedding dimension, and every fixed task head are UNCHANGED —
    only the DATA feeding them is now real.
  * CAMERA MODEL: KITTI ships genuine PER-IMAGE calibration (`P2`), but it is extremely
    tightly clustered (fx=fy across the whole training set: mean 719.79px, std 4.4px,
    <0.7% relative; principal point std ~2.5-3.6px on a ~1242x375 frame; only 4 distinct
    (rounded) K matrices across all 7481 calibration files — the KITTI recording rig used
    a small, fixed set of camera units). Per requirement (1)/(3) of this swap we therefore
    keep the harness's SINGLE fixed-intrinsic-camera framing (`K`, `IMG_W`, `IMG_H` below)
    rather than threading a per-sample K through the harness/model interface — we use the
    dataset MEAN intrinsic as the one fixed K. Each object's GT 2D/3D box is still exactly
    its own image's real annotation (consistent with that image's REAL P2); only the
    DECODE-TIME geometry (`Z = f*H/h2d`, back-projection) uses the fixed mean f/cx/cy, so
    ~19% of objects (whose image's real K differs from the mean by up to ~2%) see a small,
    honest, real sub-percent geometric residual — a realistic amount of calibration noise,
    not an invented one. This is documented here rather than silently glossed over.

Design invariants (fixed here so every agent is scored identically):
  * The dataset (real KITTI boxes/crops + the one fixed intrinsic K), the train/val/test
    splits (a deterministic 70/15/15 split BY IMAGE, seed 42, so no leakage), the fixed
    appearance/geometry ENCODER (`common.RegionEncoder`), the optimizer, epochs, batch, and
    the seed are ALL fixed. The only degree of freedom is the agent's design surface
    (the DEPTH parameterization, or the ORIENTATION encoding) depending on the task.
  * Scoring is on the held-out TEST split: AP3D at a fixed 3D-IoU threshold (primary,
    HIGHER better) plus decomposed median depth error and mean yaw error (for feedback).
    A constant / mean-box predictor gets near-0 AP3D; using the geometry lifts it.
  * The 3 scored SETTINGS are KITTI's own OFFICIAL easy/moderate/hard difficulty tiers
    (2D-bbox-height + occlusion + truncation thresholds, see `DIFFICULTY_SETTINGS`) —
    assigned EXCLUSIVELY (easiest tier an object satisfies) so the three settings are a
    genuine DISJOINT partition of the test objects, matching every other mono3d-* task's
    "disjoint slice of the same held-out split" scoring convention. (This differs from the
    official KITTI eval server's CUMULATIVE convention, where "moderate" AP is computed
    over a superset that also includes "easy" objects; the exclusive partition is the
    natural fit for this harness's fixed multi-setting design and is documented here.)

Metric:
  * AP3D@t — fraction of test objects whose predicted 3D box has 3D IoU >= t with its GT
    (single object per image, class-agnostic here, so AP3D reduces to the recall at the
    IoU threshold; equivalently the 3D-IoU >= t hit rate). We report AP3D at t=0.25 (the
    KITTI-style easy threshold) and t=0.5, plus the mean 3D IoU.
"""
from __future__ import annotations

import importlib.util
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# ----------------------------------------------------------------------------- utils
def set_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_surface(path: str, symbol: str):
    """Import ``symbol`` from the agent-editable solution file at ``path``."""
    spec = importlib.util.spec_from_file_location("mono3d_solution", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, symbol):
        raise AttributeError(f"solution {path} is missing `{symbol}`")
    return getattr(mod, symbol)


# =========================================================================================
# Fixed pinhole intrinsics used to NORMALIZE the geometry feature vector and as the single
# DECODE-TIME camera model shared by every mono3d-* run (see the module docstring for why a
# single representative K is used rather than per-image K). Values = the MEAN of the REAL
# KITTI `P2` calibration across all 7481 training images (computed once from the staged
# calib files; see vendor/data_scripts/mono3d-detection/prepare_data.py). DO NOT change
# these constants — every mono3d-* run decodes depth/back-projects with the same K.
IMG_W = 1242.0                 # modal KITTI left-camera image width  (px)
IMG_H = 375.0                  # modal KITTI left-camera image height (px)
_FOCAL = 719.79                # mean fx == fy across all training calib files (px)
_CX = 608.46                   # mean principal-point x
_CY = 174.55                   # mean principal-point y
K = np.array([[_FOCAL, 0.0, _CX],
              [0.0, _FOCAL, _CY],
              [0.0, 0.0, 1.0]], dtype=np.float64)

# Three REAL KITTI object classes, canonical metric size (l=length x, h=height y, w=width z)
# in meters == the MEASURED mean dimensions of each class over the real KITTI training
# labels (computed once from the staged label files; see prepare_data.py). Real per-object
# dims vary around these means (a real car is not exactly 3.88m long) — that per-object
# variation is what makes the depth/dims parameterization non-trivial, exactly as before,
# except now it is REAL physical variation, not an invented jitter distribution.
_CLASS_DIMS = np.array([
    [3.884, 1.526, 1.629],   # "car"          (n=28742 real KITTI training boxes)
    [0.842, 1.761, 0.660],   # "pedestrian"   (n=4487)
    [1.764, 1.737, 0.597],   # "cyclist"      (n=1627)
], dtype=np.float64)
_N_CLASSES = _CLASS_DIMS.shape[0]
_CLASS_NAMES = ("Car", "Pedestrian", "Cyclist")

_SEED = 42

# KITTI's OFFICIAL per-object difficulty thresholds (2D bbox pixel height, occlusion level
# 0..3, truncation fraction 0..1), applied EXCLUSIVELY (an object is assigned the EASIEST
# tier it satisfies), so the 3 tiers are a disjoint partition of the labelled objects. This
# is the REAL KITTI eval-server convention (see devkit `evaluate_object.cpp`), used here
# directly as the harness's 3 scored settings. DO NOT change these thresholds.
DIFFICULTY_SETTINGS = {
    "easy":     dict(min_height=40.0, max_occlusion=0, max_truncation=0.15),
    "moderate": dict(min_height=25.0, max_occlusion=1, max_truncation=0.30),
    "hard":     dict(min_height=25.0, max_occlusion=2, max_truncation=0.50),
}
_DIFFICULTY_ORDER = ("easy", "moderate", "hard")
_DIFFICULTY_ID = {name: i for i, name in enumerate(_DIFFICULTY_ORDER)}


def kitti_difficulty(bbox_height_px: float, occlusion: int, truncation: float):
    """Classify one real KITTI object into its OFFICIAL easy/moderate/hard difficulty tier
    (or None if it fails even the "hard" thresholds -> excluded from every setting, matching
    the official KITTI eval-server convention of ignoring such objects entirely)."""
    for name in _DIFFICULTY_ORDER:
        t = DIFFICULTY_SETTINGS[name]
        if bbox_height_px >= t["min_height"] and occlusion <= t["max_occlusion"] \
                and truncation <= t["max_truncation"]:
            return name
    return None


def _corners_3d(dims, loc, yaw):
    """8 corners of a 3D box in camera frame. dims=(l,h,w), loc=(x,y,z) center, yaw about y.

    Camera frame: x right, y down, z forward (KITTI-style). The box is axis-aligned in its
    own frame then rotated by yaw about the vertical (y) axis and translated to loc.
    """
    l, h, w = dims
    x = l / 2.0 * np.array([1, 1, -1, -1, 1, 1, -1, -1], dtype=np.float64)
    y = h / 2.0 * np.array([1, 1, 1, 1, -1, -1, -1, -1], dtype=np.float64)
    z = w / 2.0 * np.array([1, -1, -1, 1, 1, -1, -1, 1], dtype=np.float64)
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)
    pts = R @ np.stack([x, y, z], axis=0)          # [3, 8]
    pts = pts + np.asarray(loc, dtype=np.float64).reshape(3, 1)
    return pts.T                                    # [8, 3]


def _project(pts3d):
    """Project [8,3] camera-frame points to [8,2] pixels via the fixed intrinsic K."""
    z = np.clip(pts3d[:, 2], 1e-3, None)
    u = _FOCAL * pts3d[:, 0] / z + _CX
    v = _FOCAL * pts3d[:, 1] / z + _CY
    return np.stack([u, v], axis=1)


def _amodal_box2d(pts2d):
    """Amodal 2D box [x1,y1,x2,y2] = tight image bound of all 8 projected corners."""
    x1, y1 = pts2d[:, 0].min(), pts2d[:, 1].min()
    x2, y2 = pts2d[:, 0].max(), pts2d[:, 1].max()
    return np.array([x1, y1, x2, y2], dtype=np.float64)


# =========================================================================================
# Fixed evaluation SETTINGS = KITTI's OFFICIAL easy/moderate/hard difficulty tiers (see
# `DIFFICULTY_SETTINGS`/`kitti_difficulty` above). A task aggregates its score over these 3
# settings; each is a disjoint slice of the SAME held-out TEST split, so every run trains on
# identical data and is scored on identical, non-overlapping sub-populations. Monocular
# depth/appearance ambiguity is HARDEST for small/occluded/truncated objects (the "hard" tier
# is dominated by far/occluded objects a naive predictor handles worst), so a genuine partial
# order (easy > moderate > hard, for every method) is expected, and the projective-geometry /
# statistical-prior methods' advantage over naive baselines should be largest on hard.
DEPTH_SETTINGS = DIFFICULTY_SETTINGS   # back-compat alias (tasks/scripts key off this name)


def setting_mask(task: str, setting: str, difficulty_id):
    """Boolean mask over the TEST split selecting the objects in the KITTI difficulty tier
    `setting` (easy/moderate/hard). `difficulty_id` [N] int array (0=easy,1=moderate,2=hard),
    precomputed per-object at data-prep time from the REAL truncation/occlusion/bbox-height
    label fields (see `kitti_difficulty`). Returns a numpy bool array."""
    import numpy as _np
    d = _np.asarray(difficulty_id).reshape(-1)
    if setting not in _DIFFICULTY_ID:
        raise ValueError(f"unknown setting {setting!r} (expected one of {_DIFFICULTY_ORDER})")
    return d == _DIFFICULTY_ID[setting]


def _data_root() -> Path:
    return Path(os.environ.get("MONO3D_DATA", "/data/mono3d-detection"))


def load_splits():
    """Load the REAL KITTI monocular-3D object splits as torch tensors on the active device.

    Requires the pre-staged ``mono3d_kitti.npz`` produced by
    ``vendor/data_scripts/mono3d-detection/prepare_data.py`` (downloads + parses the real
    KITTI `training` images/labels/calibration once; see that script and the module
    docstring above for the exact source/processing). Unlike the old fully-synthetic
    pipeline there is no in-harness fallback generator — real data cannot be conjured
    offline, so a missing file is a loud, explicit error rather than a silent synthetic
    substitution.
    """
    dev = device()
    path = _data_root() / "mono3d_kitti.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run "
            "`python vendor/data_scripts/mono3d-detection/prepare_data.py --out "
            f"{path.parent}` to download/parse the real KITTI 3D-object data first "
            "(requires host network access; the resulting .npz is then fully offline)."
        )
    z = np.load(path)
    data = {k: z[k] for k in z.files}

    def _t(a, dt):
        return torch.as_tensor(a, dtype=dt, device=dev)

    out = {}
    for name in ("train", "val", "test"):
        out[f"feat_{name}"] = _t(data[f"feat_{name}"], torch.float32)
        out[f"crop_{name}"] = _t(data[f"crop_{name}"], torch.float32)
        out[f"cls_{name}"] = _t(data[f"cls_{name}"], torch.long)
        out[f"dims_{name}"] = _t(data[f"dims_{name}"], torch.float32)
        out[f"loc_{name}"] = _t(data[f"loc_{name}"], torch.float32)
        out[f"yaw_{name}"] = _t(data[f"yaw_{name}"], torch.float32)
        out[f"box2d_{name}"] = _t(data[f"box2d_{name}"], torch.float32)
        out[f"difficulty_{name}"] = _t(data[f"difficulty_{name}"], torch.long)
    out["focal"] = float(data["focal"])
    out["cx"] = float(data["cx"])
    out["cy"] = float(data["cy"])
    out["crop_hw"] = int(data["crop_hw"])
    out["feat_dim"] = int(data["feat_dim"])
    return out


# =========================================================================================
# Fixed appearance + geometry encoder shared by every mono3d-* baseline / surface.
# Maps (geometry feature vector, appearance crop) -> a shared FEAT_DIM embedding. The agent
# attaches task-specific heads to this embedding but does NOT change the encoder.
# =========================================================================================
EMB_DIM = 128


class RegionEncoder(nn.Module):
    """Fixed encoder: a tiny CNN over the appearance crop concatenated with an MLP over the
    geometry feature vector -> a shared `EMB_DIM` embedding. Compact so it trains in
    seconds. Deterministic given the seed.
    """

    def __init__(self, feat_dim: int, crop_hw: int = 32):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(),   # 16
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),  # 8
            nn.Conv2d(32, 48, 3, stride=2, padding=1), nn.BatchNorm2d(48), nn.ReLU(),  # 4
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.geo = nn.Sequential(
            nn.Linear(feat_dim, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(48 + 64, EMB_DIM), nn.ReLU(), nn.Linear(EMB_DIM, EMB_DIM), nn.ReLU(),
        )

    def forward(self, feat, crop):
        a = self.cnn(crop)
        g = self.geo(feat)
        return self.head(torch.cat([a, g], dim=1))


# =========================================================================================
# 3D IoU + AP3D scoring (self-contained; sign/threshold conventions FIXED here).
# =========================================================================================
def _box3d_corners_np(dims, loc, yaw):
    return _corners_3d(np.asarray(dims, np.float64),
                       np.asarray(loc, np.float64),
                       float(yaw))


def _rot_iou_bev(dims_a, loc_a, yaw_a, dims_b, loc_b, yaw_b):
    """BEV (top-down) rotated-rectangle IoU via Sutherland-Hodgman polygon clipping.

    dims=(l,h,w); the BEV rectangle uses (l along x, w along z) rotated by yaw about y.
    Returns (bev_intersection_area, area_a, area_b).
    """
    def rect(dims, loc, yaw):
        l, _h, w = dims
        c, s = np.cos(yaw), np.sin(yaw)
        # rectangle corners in BEV (x,z): half-l along local x, half-w along local z
        lx = l / 2.0; lz = w / 2.0
        local = np.array([[lx, lz], [lx, -lz], [-lx, -lz], [-lx, lz]], dtype=np.float64)
        R = np.array([[c, s], [-s, c]], dtype=np.float64)   # rotate about y: (x,z)
        world = local @ R.T
        world[:, 0] += loc[0]
        world[:, 1] += loc[2]
        return world

    def _signed_area(poly):
        x = poly[:, 0]; y = poly[:, 1]
        return 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    def poly_area(poly):
        if len(poly) < 3:
            return 0.0
        return abs(_signed_area(poly))

    def ccw(poly):
        # Sutherland-Hodgman below assumes a COUNTER-CLOCKWISE clipper (inside test
        # uses cross >= 0). rect() corner order is not guaranteed CCW, so normalize.
        return poly if _signed_area(poly) >= 0 else poly[::-1].copy()

    def clip(subject, clipper):
        clipper = ccw(clipper)
        out = [np.asarray(p, np.float64) for p in subject]
        cn = len(clipper)
        for i in range(cn):
            a = clipper[i]; b = clipper[(i + 1) % cn]
            edge = b - a
            inp = out
            out = []
            if not inp:
                break
            def inside(p):
                return edge[0] * (p[1] - a[1]) - edge[1] * (p[0] - a[0]) >= -1e-9
            for j in range(len(inp)):
                cur = inp[j]; prv = inp[j - 1]
                ci = inside(cur); pi = inside(prv)
                if ci:
                    if not pi:
                        out.append(_inter(prv, cur, a, b))
                    out.append(cur)
                elif pi:
                    out.append(_inter(prv, cur, a, b))
        return np.array(out) if out else np.zeros((0, 2))

    def _inter(p1, p2, a, b):
        d1 = p2 - p1; d2 = b - a
        denom = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(denom) < 1e-12:
            return p1
        t = ((a[0] - p1[0]) * d2[1] - (a[1] - p1[1]) * d2[0]) / denom
        return p1 + t * d1

    ra = rect(dims_a, loc_a, yaw_a)
    rb = rect(dims_b, loc_b, yaw_b)
    inter_poly = clip(ra, rb)
    inter_area = poly_area(inter_poly)
    area_a = dims_a[0] * dims_a[2]
    area_b = dims_b[0] * dims_b[2]
    return inter_area, area_a, area_b


def iou_3d(dims_a, loc_a, yaw_a, dims_b, loc_b, yaw_b) -> float:
    """3D IoU of two oriented 3D boxes = (BEV intersection x vertical overlap) / union.

    Vertical (y) overlap uses the box heights centered at loc_y (camera y = down). Returns
    a scalar in [0, 1].
    """
    inter_bev, area_a, area_b = _rot_iou_bev(dims_a, loc_a, yaw_a, dims_b, loc_b, yaw_b)
    ha, hb = dims_a[1], dims_b[1]
    ya, yb = loc_a[1], loc_b[1]
    top = max(ya - ha / 2.0, yb - hb / 2.0)
    bot = min(ya + ha / 2.0, yb + hb / 2.0)
    inter_h = max(0.0, bot - top)
    inter_vol = inter_bev * inter_h
    vol_a = area_a * ha
    vol_b = area_b * hb
    union = vol_a + vol_b - inter_vol
    if union <= 1e-9:
        return 0.0
    return float(max(0.0, min(1.0, inter_vol / union)))


def score_predictions(pred_dims, pred_loc, pred_yaw, gt_dims, gt_loc, gt_yaw):
    """Compute AP3D@0.25, AP3D@0.5, mean 3D IoU, median depth error, mean yaw error (deg).

    All inputs are numpy arrays [N, ...]. AP3D here (single object per image, class-agnostic)
    is the fraction of objects whose 3D IoU with GT is >= the threshold (the 3D-IoU hit
    rate). Depth error = |pred_z - gt_z| (median over N). Yaw error = smallest angular
    distance mod 2pi, in degrees (mean).
    """
    n = gt_dims.shape[0]
    ious = np.zeros(n, dtype=np.float64)
    for i in range(n):
        ious[i] = iou_3d(pred_dims[i], pred_loc[i], float(pred_yaw[i]),
                         gt_dims[i], gt_loc[i], float(gt_yaw[i]))
    ap25 = float((ious >= 0.25).mean())
    ap50 = float((ious >= 0.5).mean())
    miou = float(ious.mean())
    depth_err = np.abs(pred_loc[:, 2] - gt_loc[:, 2])
    med_depth_err = float(np.median(depth_err))
    dyaw = np.abs(pred_yaw.reshape(-1) - gt_yaw.reshape(-1))
    dyaw = np.minimum(dyaw % (2 * np.pi), (2 * np.pi) - (dyaw % (2 * np.pi)))
    mean_yaw_err = float(np.degrees(dyaw).mean())
    return {
        "ap25": ap25, "ap50": ap50, "miou": miou,
        "med_depth_err": med_depth_err, "mean_yaw_err_deg": mean_yaw_err,
    }


# ---------------------------------------------------------------- fixed geometry helpers
def depth_from_height(dims_h, box2d_h_px, focal) -> torch.Tensor:
    """Inverse-projection depth from the object's PHYSICAL height and its 2D pixel height:
        z = focal * H_metric / h_pixels.
    dims_h [N] predicted/known metric heights, box2d_h_px [N] 2D box pixel heights.
    This is the geometry cue at the heart of monocular 3D detection (Deep3DBox / GS3D).
    """
    return focal * dims_h.reshape(-1) / box2d_h_px.reshape(-1).clamp(min=1.0)


def backproject_xy(loc_z, box2d, cx, cy, focal):
    """Back-project the 2D box center to the 3D (x, y) at depth loc_z via the pinhole model:
        x = (u - cx) * z / f ,  y = (v - cy) * z / f .
    box2d [N,4] amodal boxes; returns (x [N], y [N]).
    """
    u = 0.5 * (box2d[:, 0] + box2d[:, 2])
    v = 0.5 * (box2d[:, 1] + box2d[:, 3])
    z = loc_z.reshape(-1)
    x = (u - cx) * z / focal
    y = (v - cy) * z / focal
    return x, y

"""Prepare data for the crowd/object-counting (cv-count-*) tasks from REAL crowd photos.

Produces, under {data_root}/crowd-counting/ four density-regime scenes
{medium,middense,dense,superdense}/ (plus a `blobs` alias == `medium`, kept for backward
compatibility with the two original cv-count tasks that predate the 4-scene expansion):
  <scene>/train/img_*.npy  <scene>/train/manifest.json   (FIXED, resized real photos)
  <scene>/val/img_*.npy    <scene>/val/manifest.json     (FIXED, scored)

REAL DATA (ShanghaiTech Crowd Counting Dataset, Zhang et al. "Single-Image Crowd
Counting via Multi-Column Convolutional Neural Network", CVPR 2016 -- the founding MCNN
benchmark this whole cv-count-* task family is modeled on; also the standard benchmark
for CSRNet, SANet, CAN and most of the architecture/kernel/loss ideas these tasks probe):
  https://github.com/desenzhou/ShanghaiTechDataset  (author-maintained mirror pointing at
  a 2024-refreshed Dropbox zip of the ORIGINAL `part_A_final` / `part_B_final` archive).
Each image is a REAL surveillance/street/campus photo of a crowd; every human HEAD is
annotated with a single (x, y) point by the dataset authors (part_A: 300 train + 182 test
dense photos, mostly web images, up to ~3100 heads/photo; part_B: 400 train + 316 test
sparser street-camera photos, up to ~600 heads/photo). The GROUND-TRUTH COUNT is EXACTLY
the number of annotated head points (`image_info[0,0][0,0][1]` in each `GT_IMG_*.mat`,
matching the count in `image_info[...][0]`'s row count) -- the harness's ground-truth
convention (count == number of point annotations) carries over unchanged from the
synthetic generator this replaces.

DENSITY-MAP GT: each head point is rendered into a per-pixel density map with a Gaussian
kernel (`harness.py:render_density`, unchanged) so the counting metric (integrated
density vs point count) is exact by construction, exactly the MCNN/CSRNet convention.

FOUR CROWD-DENSITY SCENES (the >=3 validation SETTINGS the tasks aggregate over) are
reconstructed from REAL images by ANNOTATED-HEAD-COUNT buckets (not by part_A vs part_B,
since both parts' count distributions overlap heavily) -- exactly the same (train_range,
val_range) buckets the previous synthetic generator used, so the existing tasks' scene
names / scripts / score_spec anchors need no structural change, only re-anchoring:
  medium     train counts in [ 15, 60] -> val counts in [ 60,140]
  middense   train counts in [ 40, 90] -> val counts in [100,170]
  dense      train counts in [ 50,120] -> val counts in [130,230]
  superdense train counts in [ 90,170] -> val counts in [190,300]
`blobs` (legacy name) == `medium` (same buckets, same RNG seed/shuffle so it is the exact
same image set every run).

CRUCIAL: in every scene the TRAIN and VAL images are drawn from DISJOINT REAL-COUNT
buckets, train LOW and val HIGHER -- this is a deliberate, disclosed EVALUATION-PROTOCOL
choice (which real images go in train vs val), not data fabrication: we do not invent or
alter any image or annotation, we only choose the held-out count regime. This reproduces
the same count-EXTRAPOLATION stress test as the synthetic generator: a DIRECT
global-regression head memorises the train count distribution and regresses toward the
train mean, so it badly UNDER-counts the higher-count real val images; a DENSITY-MAP head
predicts a translation-equivariant LOCAL density and integrates it, so it generalises to
the shifted counts. A degenerate constant-mean predictor is hopeless on the shifted val by
construction (scores CONST_MEAN_MAE = mean-absolute-deviation of the val counts).

Each image is resized (bilinear) to the harness's fixed IMG_SIZE x IMG_SIZE working
resolution and the point annotations are rescaled by the same (sx, sy) factors, so the
resized points still land exactly on the (resized) heads and the count is unchanged.

SOURCE / STAGING: this script downloads the official
`ShanghaiTech_Crowd_Counting_Dataset.zip` from the desenzhou/ShanghaiTechDataset Dropbox
mirror (`dl=1` direct-download link, scriptable via curl through the outbound proxy), or
reads it from `--raw-dir` if already staged (e.g. on a shared moonfs mount -- this was
staged at /mnt/moonfs/lvbohan-b0/crowd-real/shanghaitech_official/ during onboarding).
Requires numpy + Pillow + scipy (scipy.io.loadmat for the .mat point annotations).
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import zipfile
from pathlib import Path

import numpy as np

IMG_SIZE = 128
N_TRAIN = 120
N_VAL = 40
SEED = 1234

DATASET_URL = (
    "https://www.dropbox.com/scl/fi/dkj5kulc9zj0rzesslck8/"
    "ShanghaiTech_Crowd_Counting_Dataset.zip"
    "?rlkey=ymbcj50ac04uvqn8p49j9af5f&dl=1"
)

# FOUR crowd-density regimes: (train_count_range LOW, val_count_range HIGHER). Buckets by
# REAL annotated-head count over the pooled part_A + part_B images (train+test folders of
# both parts are pooled into one candidate set, since the scene split is by density
# bucket, not by the dataset's own train/test split). Same numeric ranges as the legacy
# synthetic generator, so downstream task configs/anchors carry over structurally.
SCENES = {
    "medium": ((15, 60), (60, 140)),
    "middense": ((40, 90), (100, 170)),
    "dense": ((50, 120), (130, 230)),
    "superdense": ((90, 170), (190, 300)),
}

PARTS = ["part_A_final", "part_B_final"]
SPLITS = ["train_data", "test_data"]


def _ensure_raw(raw_dir: Path) -> Path:
    """Return a directory containing part_A_final/ and part_B_final/, downloading +
    extracting the official zip into raw_dir if not already present."""
    if (raw_dir / "part_A_final").exists() and (raw_dir / "part_B_final").exists():
        return raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "ShanghaiTech_Crowd_Counting_Dataset.zip"
    if not (zip_path.exists() and zipfile.is_zipfile(zip_path)):
        print(f"downloading ShanghaiTech <- {DATASET_URL.split('?')[0]} -> {zip_path}",
              flush=True)
        subprocess.run(
            ["curl", "-sSL", "--retry", "8", "--retry-delay", "10",
             "-o", str(zip_path), DATASET_URL],
            check=True,
        )
    if not zipfile.is_zipfile(zip_path):
        raise RuntimeError(f"downloaded {zip_path} is not a valid zip (proxy/auth issue?)")
    print(f"extracting {zip_path} -> {raw_dir}", flush=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(raw_dir)
    if not (raw_dir / "part_A_final").exists() or not (raw_dir / "part_B_final").exists():
        raise RuntimeError(f"extraction incomplete under {raw_dir}")
    return raw_dir


def _gt_points(mat_path: Path) -> np.ndarray:
    """ShanghaiTech GT_IMG_*.mat -> (K,2) array of (y, x) full-resolution pixel
    coordinates (K == the exact annotated head count)."""
    import scipy.io as sio

    d = sio.loadmat(str(mat_path))
    info = d["image_info"][0, 0][0, 0]
    loc = np.asarray(info[0], dtype=np.float64).reshape(-1, 2)  # stored as (x, y)
    pts_yx = loc[:, [1, 0]]                                     # -> (y, x)
    return pts_yx


def _scan_pool(raw_dir: Path):
    """Enumerate every real (image, points) pair across BOTH parts and BOTH official
    splits, pooled into one candidate list bucketed later purely by annotated count."""
    rows = []
    for part in PARTS:
        for split in SPLITS:
            gt_dir = raw_dir / part / split / "ground_truth"
            img_dir = raw_dir / part / split / "images"
            for gt_path in sorted(gt_dir.glob("GT_IMG_*.mat")):
                img_id = gt_path.stem[len("GT_IMG_"):]
                img_path = img_dir / f"IMG_{img_id}.jpg"
                if not img_path.exists():
                    continue
                pts = _gt_points(gt_path)
                rows.append({"img": img_path, "count": int(pts.shape[0]), "pts": pts,
                             "key": f"{part}/{split}/IMG_{img_id}"})
    return rows


def _load_resized(img_path: Path, pts: np.ndarray, size: int):
    """Load a real JPEG, resize (bilinear) to size x size in [0,1] CHW float32, and
    rescale the point annotations by the same (sy, sx) factors so they still land on
    the (resized) heads -- the count itself is exactly preserved (no points added or
    dropped by the resize)."""
    from PIL import Image

    im = Image.open(img_path).convert("RGB")
    w0, h0 = im.size
    im = im.resize((size, size), Image.BILINEAR)
    arr = np.asarray(im, dtype=np.float32).transpose(2, 0, 1) / 255.0  # (3,H,W)
    sy = size / float(h0)
    sx = size / float(w0)
    if pts.shape[0]:
        pts_r = pts.copy()
        pts_r[:, 0] *= sy
        pts_r[:, 1] *= sx
        pts_r[:, 0] = np.clip(pts_r[:, 0], 0, size - 1)
        pts_r[:, 1] = np.clip(pts_r[:, 1], 0, size - 1)
    else:
        pts_r = pts.reshape(0, 2)
    return arr.astype(np.float32), pts_r


def _build_split(rows, out_dir: Path, size: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    counts = []
    for i, row in enumerate(rows):
        img, pts = _load_resized(row["img"], row["pts"], size)
        fn = f"img_{i:04d}.npy"
        np.save(out_dir / fn, img)
        manifest.append({
            "img": fn,
            "count": int(pts.shape[0]),                         # exact GT count
            "points": [[float(y), float(x)] for y, x in pts],
            "source": row["key"],                                # provenance (real photo id)
        })
        counts.append(int(pts.shape[0]))
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f)
    return counts


def _scene_present(scene_root: Path) -> bool:
    vm = scene_root / "val" / "manifest.json"
    if not vm.exists():
        return False
    try:
        with open(vm) as f:
            return len(json.load(f)) >= N_VAL
    except Exception:  # noqa: BLE001
        return False


def _build_scene(base: Path, scene: str, pool, tr_range, va_range, seed_off: int):
    scene_root = base / scene
    if _scene_present(scene_root):
        print(f"crowd-counting scene '{scene}' already present at {scene_root}")
        return

    tr_pool = [r for r in pool if tr_range[0] <= r["count"] <= tr_range[1]]
    va_pool = [r for r in pool if va_range[0] <= r["count"] <= va_range[1]]
    if len(tr_pool) < N_TRAIN or len(va_pool) < N_VAL:
        raise SystemExit(
            f"scene '{scene}': not enough REAL images in bucket "
            f"train{tr_range}={len(tr_pool)} val{va_range}={len(va_pool)} "
            f"(need {N_TRAIN}/{N_VAL})")

    rng = random.Random(SEED + seed_off)
    rng.shuffle(tr_pool)
    tr_pick = tr_pool[:N_TRAIN]
    used = {r["key"] for r in tr_pick}
    rng.shuffle(va_pool)
    va_pick = [r for r in va_pool if r["key"] not in used][:N_VAL]
    if len(va_pick) < N_VAL:
        raise SystemExit(f"scene '{scene}': val pool exhausted after train dedup "
                          f"({len(va_pick)}/{N_VAL})")

    tr = _build_split(tr_pick, scene_root / "train", IMG_SIZE)
    va = _build_split(va_pick, scene_root / "val", IMG_SIZE)
    tr_mean = float(np.mean(tr))
    const_mae = float(np.mean([abs(tr_mean - c) for c in va]))
    print(f"crowd-counting scene '{scene}' ready [REAL ShanghaiTech]: train={len(tr)} "
          f"(count {min(tr)}-{max(tr)} mean {tr_mean:.1f}) val={len(va)} "
          f"(count {min(va)}-{max(va)} mean {np.mean(va):.1f}) "
          f"CONST_MEAN_MAE={const_mae:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--raw-dir", default=None, type=Path,
                     help="dir with pre-downloaded part_A_final/ part_B_final/ (or the "
                          "official zip); skips the Dropbox download. Staged during "
                          "onboarding at /mnt/moonfs/lvbohan-b0/crowd-real/"
                          "shanghaitech_official/")
    args = ap.parse_args()

    base = args.data_root.expanduser().resolve() / "crowd-counting"

    all_present = all(_scene_present(base / s) for s in SCENES) and _scene_present(base / "blobs")
    if all_present:
        print("crowd-counting: all scenes already present, skipping download+scan")
        return

    raw_dir = args.raw_dir if args.raw_dir else (base / "_raw")
    raw_dir = _ensure_raw(raw_dir)
    pool = _scan_pool(raw_dir)
    print(f"crowd-counting: scanned {len(pool)} REAL ShanghaiTech images "
          f"(part_A + part_B, train+test pooled)", flush=True)

    # Distinct seed offset per scene so the four regimes draw independent (but
    # reproducible) subsets of the real-image pool.
    for i, (scene, (tr_range, va_range)) in enumerate(SCENES.items()):
        _build_scene(base, scene, pool, tr_range, va_range, seed_off=i * 17)

    # Legacy `blobs` scene == `medium` (kept for the two original cv-count tasks that
    # reference /data/crowd-counting/blobs). Same buckets + seed offset as `medium`.
    _build_scene(base, "blobs", pool, SCENES["medium"][0], SCENES["medium"][1], seed_off=0)


if __name__ == "__main__":
    main()

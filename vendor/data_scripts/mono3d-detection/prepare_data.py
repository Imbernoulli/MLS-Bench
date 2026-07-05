#!/usr/bin/env python3
"""Stage the REAL KITTI 3D Object Detection dataset for the mono3d-* tasks.

REAL DATA (see `vendor/mono3d-detection/common.py` module docstring for the full story).
KITTI's official download (cvlibs.net) sits behind an institutional-email registration
wall that is not scriptable from this environment. This script instead pulls the
IDENTICAL bytes from the public, UNAUTHENTICATED S3 mirror of the official archives:

    https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip   (~12.5GB, has
                                                                                 train+test)
    https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip   (~5.6MB, train only)
    https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip     (~27MB, train+test)

Only the `training/` split has public labels (KITTI's `testing` split labels are withheld
for the official leaderboard server), so we use `training/{image_2,label_2,calib}` only,
and carve our OWN deterministic 70/15/15 train/val/test split out of it (see below).

Label parsing / geometry conventions: standard KITTI `label_2/*.txt`, one row per object,
whitespace-separated fields
    type truncated occluded alpha bbox_x1 bbox_y1 bbox_x2 bbox_y2 dim_h dim_w dim_l
    loc_x loc_y loc_z rotation_y
Camera frame is x-right/y-down/z-forward, exactly the convention already assumed by
`common._corners_3d` / `common.iou_3d`, so GT (loc, dims, yaw) are used AS-IS -- we only
reorder dims (h, w, l) -> (l, h, w) to match `common._CLASS_DIMS`'s axis order. `calib/*.txt`
`P2:` row gives that image's real per-image intrinsic; we read it only to sanity-check
against the harness's single fixed representative K (see common.py) -- decode-time geometry
always uses the fixed K, per the documented design choice.

Per-object DIFFICULTY (the harness's 3 settings) is computed once here from the REAL
truncation/occlusion/bbox-height fields via `common.kitti_difficulty` (KITTI's own official
easy/moderate/hard thresholds, applied EXCLUSIVELY / disjointly -- see common.py). Objects
that fail even the "hard" tier (e.g. DontCare rows, or tiny/heavily-occluded/truncated real
objects) are DROPPED, matching the official KITTI eval-server convention of ignoring them.

Per-object INPUT: the real amodal 2D box normalizes into the same 8-dim geometry feature
vector as before -- [cx_n, cy_n, w2d_n, h2d_n, log_h2d_n, log_w2d_n, aspect2d, focal_n] (all
normalized by the fixed IMG_W/IMG_H/focal so every image contributes on a comparable scale
despite KITTI's few distinct real resolutions, e.g. 1242x375 vs 1224x370) -- plus a REAL
appearance crop: the real image cropped to the (clipped) 2D box and resized to CROP_HW x
CROP_HW, replacing the old procedurally-RENDERED descriptor patch.

Usage:
    # fast path: point at an already-extracted KITTI `training/` dir (e.g. pre-staged on a
    # shared moonfs mount) -- no network needed.
    python prepare_data.py --data-root /data --raw-dir /mnt/.../extracted/training

    # from scratch: downloads the 3 zips (label_2/calib in full; image_2 via a central-
    # directory-aware partial fetch of ONLY the `training/image_2/*.png` entries, since the
    # full archive also contains the much larger, unused `testing/` images) into
    # <data-root>/mono3d-detection/_raw/, then extracts + parses.
    python prepare_data.py --data-root /data
"""
from __future__ import annotations

import argparse
import importlib.util
import io
import sys
import zipfile
from pathlib import Path

import numpy as np

_VENDOR = Path(__file__).resolve().parents[2] / "mono3d-detection"

IMAGE_ZIP_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip"
LABEL_ZIP_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip"
CALIB_ZIP_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip"

CROP_HW = 32          # appearance-crop side (matches common.RegionEncoder's default)
_SPLIT = (0.70, 0.15)  # (train, val) fractions BY IMAGE; remainder -> test
_CLASS_ID = {"Car": 0, "Pedestrian": 1, "Cyclist": 2}


def _load_common():
    spec = importlib.util.spec_from_file_location("mono3d_common", _VENDOR / "common.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mono3d_common"] = mod
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------------- download
class _HTTPRangeFile:
    """Minimal seekable file-like object backed by HTTP Range GETs, so the stdlib `zipfile`
    module can parse a remote ZIP's (possibly ZIP64) central directory and read individual
    STORE-compressed entries WITHOUT downloading the whole archive -- `zipfile` does all the
    ZIP64 extra-field / sentinel-offset resolution itself, which is far less error-prone than
    hand-rolling the struct parsing.
    """

    def __init__(self, url: str, session):
        self.url = url
        self.session = session
        self._pos = 0
        r = session.head(url, timeout=30)
        r.raise_for_status()
        self.size = int(r.headers["Content-Length"])

    def seek(self, offset, whence=0):
        if whence == 0:
            self._pos = offset
        elif whence == 1:
            self._pos += offset
        elif whence == 2:
            self._pos = self.size + offset
        return self._pos

    def tell(self):
        return self._pos

    def read(self, n=-1):
        if n is None or n < 0:
            end = self.size - 1
        else:
            if n == 0:
                return b""
            end = min(self._pos + n, self.size) - 1
        if end < self._pos:
            return b""
        for attempt in range(6):
            try:
                r = self.session.get(
                    self.url, headers={"Range": f"bytes={self._pos}-{end}"}, timeout=120
                )
                r.raise_for_status()
                data = r.content
                self._pos += len(data)
                return data
            except Exception:
                if attempt == 5:
                    raise
        return b""

    def readable(self):
        return True

    def seekable(self):
        return True

    def writable(self):
        return False

    def flush(self):
        pass


def _download_full(url: str, out_path: Path, session) -> Path:
    if out_path.exists() and out_path.stat().st_size > 0 and zipfile.is_zipfile(out_path):
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[mono3d] downloading {url} -> {out_path}", flush=True)
    with session.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        tmp = out_path.with_suffix(out_path.suffix + ".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
        tmp.rename(out_path)
    return out_path


def _fetch_training_images(out_dir: Path, session) -> None:
    """Fetch ONLY `training/image_2/*.png` from the 12.5GB image zip, via the ZIP-central-
    -directory-aware partial-read path (see `_HTTPRangeFile`). Skips files already staged at
    the correct size (resumable)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rf = _HTTPRangeFile(IMAGE_ZIP_URL, session)
    with zipfile.ZipFile(rf) as zf:
        infos = [i for i in zf.infolist()
                 if i.filename.startswith("training/image_2/") and i.filename.endswith(".png")]
        print(f"[mono3d] image zip: {len(infos)} training images to fetch", flush=True)
        for i, info in enumerate(infos):
            fname = Path(info.filename).name
            out_path = out_dir / fname
            if out_path.exists() and out_path.stat().st_size == info.file_size:
                continue
            data = zf.read(info)
            tmp = out_path.with_suffix(out_path.suffix + ".part")
            tmp.write_bytes(data)
            tmp.rename(out_path)
            if (i + 1) % 500 == 0:
                print(f"[mono3d] images {i + 1}/{len(infos)}", flush=True)


def _ensure_raw(raw_dir: Path) -> Path:
    """Return a dir containing training/{image_2,label_2,calib}/, downloading if needed."""
    train_dir = raw_dir / "training"
    have = all((train_dir / sub).is_dir() and any((train_dir / sub).iterdir())
               for sub in ("image_2", "label_2", "calib"))
    if have:
        return train_dir

    import requests

    session = requests.Session()
    cache = raw_dir / "_zips"
    cache.mkdir(parents=True, exist_ok=True)

    label_zip = _download_full(LABEL_ZIP_URL, cache / "data_object_label_2.zip", session)
    calib_zip = _download_full(CALIB_ZIP_URL, cache / "data_object_calib.zip", session)
    with zipfile.ZipFile(label_zip) as zf:
        zf.extractall(raw_dir)
    with zipfile.ZipFile(calib_zip) as zf:
        zf.extractall(raw_dir)
    _fetch_training_images(train_dir / "image_2", session)
    return train_dir


# --------------------------------------------------------------------------------- parsing
def _read_label_file(path: Path):
    rows = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        cls = parts[0]
        if cls not in _CLASS_ID:
            continue  # DontCare / Van / Truck / Misc / Person_sitting / Tram -- not scored
        truncated = float(parts[1])
        occluded = int(float(parts[2]))
        x1, y1, x2, y2 = (float(v) for v in parts[4:8])
        h, w, l = (float(v) for v in parts[8:11])
        x, y, z = (float(v) for v in parts[11:14])
        rotation_y = float(parts[14])
        rows.append(dict(
            cls=cls, truncated=truncated, occluded=occluded,
            box2d=(x1, y1, x2, y2), dims=(l, h, w), loc=(x, y, z), yaw=rotation_y,
        ))
    return rows


def _crop_resize(img, box2d, hw: int) -> np.ndarray:
    from PIL import Image

    W, H = img.size
    x1, y1, x2, y2 = box2d
    x1 = max(0, min(W - 1, x1)); x2 = max(x1 + 1, min(W, x2))
    y1 = max(0, min(H - 1, y1)); y2 = max(y1 + 1, min(H, y2))
    crop = img.crop((x1, y1, x2, y2)).resize((hw, hw), Image.BILINEAR)
    arr = np.asarray(crop, dtype=np.float32) / 255.0   # (hw, hw, 3)
    return arr.transpose(2, 0, 1)                       # (3, hw, hw)


def _geom_feat(box2d, common) -> np.ndarray:
    x1, y1, x2, y2 = box2d
    w2d = max(x2 - x1, 1e-3)
    h2d = max(y2 - y1, 1e-3)
    cx_box = 0.5 * (x1 + x2)
    cy_box = 0.5 * (y1 + y2)
    return np.array([
        cx_box / common.IMG_W,
        cy_box / common.IMG_H,
        w2d / common.IMG_W,
        h2d / common.IMG_H,
        np.log(h2d / common.IMG_H),
        np.log(w2d / common.IMG_W),
        w2d / h2d,
        common._FOCAL / common.IMG_W,
    ], dtype=np.float32)


def _build_dataset(train_dir: Path, common, seed: int = 42):
    label_dir = train_dir / "label_2"
    image_dir = train_dir / "image_2"
    label_files = sorted(label_dir.glob("*.txt"))
    if not label_files:
        raise RuntimeError(f"no label files found under {label_dir}")

    rng = np.random.RandomState(seed)
    order = rng.permutation(len(label_files))
    n = len(label_files)
    n_train = int(round(n * _SPLIT[0]))
    n_val = int(round(n * _SPLIT[1]))
    split_of_idx = {}
    for rank, idx in enumerate(order):
        if rank < n_train:
            split_of_idx[idx] = "train"
        elif rank < n_train + n_val:
            split_of_idx[idx] = "val"
        else:
            split_of_idx[idx] = "test"

    buckets = {sp: dict(feat=[], crop=[], cls=[], dims=[], loc=[], yaw=[], box2d=[], difficulty=[])
               for sp in ("train", "val", "test")}

    from PIL import Image

    n_kept = n_dropped = 0
    for idx, lf in enumerate(label_files):
        stem = lf.stem
        img_path = image_dir / f"{stem}.png"
        rows = _read_label_file(lf)
        if not rows:
            continue
        img = None
        sp = split_of_idx[idx]
        for r in rows:
            h_px = r["box2d"][3] - r["box2d"][1]
            diff = common.kitti_difficulty(h_px, r["occluded"], r["truncated"])
            if diff is None:
                n_dropped += 1
                continue
            if img is None:
                # Read via BytesIO (not Image.open(path) directly) so the underlying file
                # descriptor against the (network/GPFS) filesystem is closed immediately
                # after the read, rather than staying open for the lifetime of the Image
                # object -- with 7481 images this otherwise exhausts open-file limits on
                # some shared mounts well before the loop finishes.
                img = Image.open(io.BytesIO(img_path.read_bytes())).convert("RGB")
            b = buckets[sp]
            b["feat"].append(_geom_feat(r["box2d"], common))
            b["crop"].append(_crop_resize(img, r["box2d"], CROP_HW))
            b["cls"].append(_CLASS_ID[r["cls"]])
            b["dims"].append(r["dims"])
            b["loc"].append(r["loc"])
            b["yaw"].append(r["yaw"])
            b["box2d"].append(r["box2d"])
            b["difficulty"].append(common._DIFFICULTY_ID[diff])
            n_kept += 1
        if (idx + 1) % 1000 == 0:
            print(f"[mono3d] parsed {idx + 1}/{n} images (kept {n_kept}, dropped {n_dropped})",
                  flush=True)

    print(f"[mono3d] DONE parsing: kept {n_kept} objects, dropped {n_dropped} "
          f"(failed even the 'hard' tier, e.g. DontCare / tiny / heavily-occluded)", flush=True)

    out = {"focal": common._FOCAL, "cx": common._CX, "cy": common._CY,
           "crop_hw": CROP_HW, "feat_dim": 8}
    for sp, b in buckets.items():
        out[f"feat_{sp}"] = np.stack(b["feat"]).astype(np.float32)
        out[f"crop_{sp}"] = np.stack(b["crop"]).astype(np.float32)
        out[f"cls_{sp}"] = np.array(b["cls"], dtype=np.int64)
        out[f"dims_{sp}"] = np.array(b["dims"], dtype=np.float32)
        out[f"loc_{sp}"] = np.array(b["loc"], dtype=np.float32)
        out[f"yaw_{sp}"] = np.array(b["yaw"], dtype=np.float32)
        out[f"box2d_{sp}"] = np.array(b["box2d"], dtype=np.float32)
        out[f"difficulty_{sp}"] = np.array(b["difficulty"], dtype=np.int64)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/data")
    ap.add_argument("--out", default=None,
                     help="(legacy alias for --data-root; writes directly under this dir "
                          "rather than <data-root>/mono3d-detection)")
    ap.add_argument("--raw-dir", default=None,
                     help="dir containing (or to receive) training/{image_2,label_2,calib}/ "
                          "-- e.g. a pre-staged shared mount. Skips the network entirely if "
                          "already populated.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.out is not None:
        out_dir = Path(args.out)
    else:
        out_dir = Path(args.data_root).expanduser().resolve() / "mono3d-detection"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "mono3d_kitti.npz"

    if out_path.exists():
        try:
            z = np.load(out_path)
            if all(f"feat_{sp}" in z.files for sp in ("train", "val", "test")):
                print(f"[mono3d] {out_path} already staged, skipping")
                return
        except Exception:
            pass

    raw_dir = Path(args.raw_dir) if args.raw_dir else (out_dir / "_raw")
    common = _load_common()
    train_dir = _ensure_raw(raw_dir)
    data = _build_dataset(train_dir, common, seed=args.seed)
    np.savez_compressed(out_path, **data)

    print(f"[mono3d] wrote {out_path}")
    print(f"[mono3d] n_train={data['feat_train'].shape[0]} n_val={data['feat_val'].shape[0]} "
          f"n_test={data['feat_test'].shape[0]} feat_dim={int(data['feat_dim'])}")
    d_test = data["difficulty_test"]
    for name, did in common._DIFFICULTY_ID.items():
        print(f"[mono3d] test difficulty {name:9s} n={int((d_test == did).sum())}")


if __name__ == "__main__":
    main()

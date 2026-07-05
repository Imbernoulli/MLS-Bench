"""Prepare data for the deshadow-* image shadow-removal tasks.

Produces, under {data_root}/image-deshadow/<setting>/ for each of THREE settings
(light / medium / heavy cast-shadow severity):
  train.npz  (shad (Ntr,3,64,64), clean (Ntr,3,64,64), mask (Ntr,1,64,64))   FIXED
  val.npz    (shad (Nval,3,64,64), clean (Nval,3,64,64), mask (Nval,1,64,64)) FIXED, scored
all in [0,1].

REAL shadow / shadow-free / mask TRIPLETS from ISTD (Wang, Li & Yang, "Stacked Conditional
Generative Adversarial Networks for Jointly Learning Shadow Detection and Shadow Removal",
CVPR 2018). ISTD captures the SAME static outdoor scene TWICE, with and without a physical
object casting a real cast shadow, so `target` (shadow-free) is an authentic photograph of
the true unshadowed scene -- not a synthetic composite -- and `mask` is the corresponding
binary shadow-region mask. We pull the (image=shadow, mask, target=shadow-free) triplets from
the "Donghyun99/ISTD" parquet mirror of the dataset on the HF Hub (set
HF_ENDPOINT=https://hf-mirror.com if the default hub is unreachable), which reproduces the
official ISTD train/test split (1330 / 540 triplets, disjoint scenes).

Each 640x480 triplet is CENTER-CROPPED to a 480x480 square (matching the harness's square 64x64
patch convention) then bilinearly resized to 64x64 (image, target: RGB bilinear; mask: bilinear
then re-binarized at 0.5 so the soft, aliased edge left by downsampling a hard mask still reads
as a clean, information-preserving penumbra rather than a garbled boundary).

THREE SEVERITIES from REAL data (no synthetic severity control exists in ISTD, which is a single
capture condition per scene): each triplet's shadow ATTENUATION is a per-triplet MEASURED
quantity -- the mean brightness ratio of the shadow image over the shadow-free target, taken
ONLY over the pixels the mask marks as shadow:

    ratio = mean(image[mask])  /  mean(target[mask])            (ratio in (0, 1]; smaller = darker
                                                                   real cast shadow)

Triplets are bucketed into terciles of this MEASURED ratio (pooled across the official train +
test split, so the tercile cut points are a property of the full corpus, not of one split) into
light (weakest real attenuation, ratio tercile closest to 1) / medium / heavy (strongest real
attenuation, ratio tercile closest to 0) -- a real, physically-measured severity ladder built
from genuine shadow photographs rather than a synthetic parameter sweep. Train/val WITHIN each
severity bucket reuse ISTD's own disjoint-scene train/test split (scene ids never overlap
between ISTD train and test), so held-out shadow-region PSNR still measures real
cross-scene generalisation. A triplet whose mask marks fewer than MIN_MASK_PIX shadow pixels
after the crop+resize (a shadow that fell almost entirely outside the center crop) is dropped.

Requires numpy + Pillow + pyarrow + huggingface_hub. No scipy dependency any more (the
synthetic soft-mask generator that needed scipy.ndimage.gaussian_filter is gone).
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path

import numpy as np

IMG = 64
MIN_MASK_PIX = 50          # drop a triplet if too few shadow pixels survive crop+resize
REPO_ID = "Donghyun99/ISTD"
TRAIN_FILES = [
    "data/train-00000-of-00004.parquet",
    "data/train-00001-of-00004.parquet",
    "data/train-00002-of-00004.parquet",
    "data/train-00003-of-00004.parquet",
]
TEST_FILES = [
    "data/test-00000-of-00002.parquet",
    "data/test-00001-of-00002.parquet",
]
# Severity settings = terciles of the MEASURED per-triplet shadow attenuation ratio (pooled
# over train+test), ordered darkest-shadow-first for readability. Populated at runtime once
# the ratio distribution is known (see `_bucket_settings`); kept here as the canonical name
# order every downstream consumer (harness, scripts, docs) expects.
SETTINGS = ("light", "medium", "heavy")


def _hf_download(filename: str, cache_dir: Path) -> Path:
    """Download one ISTD parquet shard via huggingface_hub (mirror-aware). A pre-staged
    local copy (ISTD_PARQUET_DIR/<basename>) is used first so repeated prepares on a
    network-restricted worker (e.g. the k1 build host) don't need the hub at all."""
    pre_dir = os.environ.get("ISTD_PARQUET_DIR")
    if pre_dir:
        local = Path(pre_dir) / Path(filename).name
        if local.exists() and local.stat().st_size > 0:
            return local
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo_id=REPO_ID, filename=filename, repo_type="dataset",
                                 local_dir=str(cache_dir)))


def _decode_center_square(b: bytes, mode: str) -> np.ndarray:
    """Decode PNG bytes, center-crop the 640x480 frame to a 480x480 square, resize to
    IMGxIMG. mode='RGB' -> float32 HWC in [0,1]; mode='L' -> float32 HW in {0,1} (mask,
    re-binarized after the bilinear resize so the hard ISTD mask edge survives downsampling
    as a clean boundary rather than a blurred grey band)."""
    from PIL import Image
    im = Image.open(io.BytesIO(b)).convert(mode)
    w, h = im.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    im = im.crop((left, top, left + side, top + side))
    im = im.resize((IMG, IMG), Image.BILINEAR)
    a = np.asarray(im, dtype=np.float32) / 255.0
    if mode == "L":
        a = (a > 0.5).astype(np.float32)
    return a


def _load_split(split_files, cache_dir: Path, split_name: str):
    """Load one ISTD split (train or test) -> list of dicts with decoded 64x64 shad/clean/
    mask arrays (channels-first) + the measured attenuation ratio, on the FULL-RES mask (the
    ratio is computed before downsampling for the cleanest possible measurement)."""
    import pyarrow.parquet as pq
    from PIL import Image

    recs = []
    for fname in split_files:
        path = _hf_download(fname, cache_dir)
        table = pq.read_table(str(path))
        rows = table.to_pylist()
        print(f"  {fname}: {len(rows)} triplets", flush=True)
        for row in rows:
            mask_full = np.asarray(Image.open(io.BytesIO(row["mask"]["bytes"])).convert("L"),
                                    dtype=np.float32) / 255.0
            sel = mask_full > 0.5
            if int(sel.sum()) < MIN_MASK_PIX:
                continue
            img_full = np.asarray(Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB"),
                                   dtype=np.float32)
            tgt_full = np.asarray(Image.open(io.BytesIO(row["target"]["bytes"])).convert("RGB"),
                                   dtype=np.float32)
            ratio = float(img_full[sel].mean() / max(tgt_full[sel].mean(), 1e-6))

            shad64 = _decode_center_square(row["image"]["bytes"], "RGB")
            clean64 = _decode_center_square(row["target"]["bytes"], "RGB")
            mask64 = _decode_center_square(row["mask"]["bytes"], "L")
            if int(mask64.sum()) < MIN_MASK_PIX / 4:   # shadow fell mostly outside the crop
                continue
            recs.append(dict(
                split=split_name, fname=row["filename"], ratio=ratio,
                shad=shad64.transpose(2, 0, 1).astype(np.float32),
                clean=clean64.transpose(2, 0, 1).astype(np.float32),
                mask=mask64[None].astype(np.float32),
            ))
    return recs


def _bucket_settings(all_recs):
    """Tercile-bucket ALL triplets (train+test pooled) by measured attenuation ratio into
    light (weakest real shadow, ratio closest to 1) / medium / heavy (strongest, ratio
    closest to 0). Returns {setting: (lo_ratio, hi_ratio)} for provenance and mutates each
    rec in-place with a 'setting' key."""
    ratios = np.array([r["ratio"] for r in all_recs])
    depth = 1.0 - ratios                      # higher = darker/heavier real shadow
    q1, q2 = np.percentile(depth, [100.0 / 3.0, 200.0 / 3.0])
    bounds = {}
    for r, d in zip(all_recs, depth):
        if d <= q1:
            r["setting"] = "light"
        elif d <= q2:
            r["setting"] = "medium"
        else:
            r["setting"] = "heavy"
    for s, (dlo, dhi) in (("light", (depth.min(), q1)), ("medium", (q1, q2)),
                          ("heavy", (q2, depth.max()))):
        bounds[s] = (float(1.0 - dhi), float(1.0 - dlo))   # back to ratio units, (lo,hi)
    return bounds


def _psnr_masked(pred, gt, mask, thresh=0.05):
    sel = (mask > thresh).astype(np.float32)
    sel3 = np.broadcast_to(sel, pred.shape)
    se = ((pred - gt) ** 2) * sel3
    denom = sel3.reshape(sel3.shape[0], -1).sum(1).clip(1.0)
    mse = (se.reshape(se.shape[0], -1).sum(1) / denom).clip(1e-10)
    return float(np.mean(10.0 * np.log10(1.0 / mse)))


def _stack(recs):
    return (np.stack([r["shad"] for r in recs], 0),
            np.stack([r["clean"] for r in recs], 0),
            np.stack([r["mask"] for r in recs], 0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, type=Path)
    args = ap.parse_args()

    root = args.data_root.expanduser().resolve() / "image-deshadow"
    root.mkdir(parents=True, exist_ok=True)

    # cached?
    ready = all((root / s / "train.npz").exists() and (root / s / "val.npz").exists()
                for s in SETTINGS)
    if ready:
        print(f"image-deshadow data ready (cached): settings={list(SETTINGS)}")
        return

    cache_dir = root / "_raw"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("loading REAL ISTD (Wang et al. CVPR 2018) shadow/shadow-free/mask triplets "
          "from the HF parquet mirror ...", flush=True)
    print("train shard:", flush=True)
    tr_recs = _load_split(TRAIN_FILES, cache_dir, "train")
    print("test shard:", flush=True)
    te_recs = _load_split(TEST_FILES, cache_dir, "test")
    print(f"usable triplets: train={len(tr_recs)} test={len(te_recs)}", flush=True)

    all_recs = tr_recs + te_recs
    bounds = _bucket_settings(all_recs)
    print(f"tercile ratio bounds (lo,hi; smaller ratio = darker real shadow): {bounds}",
          flush=True)

    meta = dict(source="Donghyun99/ISTD (HF parquet mirror of ISTD, Wang et al. CVPR 2018)",
                img=IMG, bounds=bounds)
    for s in SETTINGS:
        s_tr = [r for r in tr_recs if r["setting"] == s]
        s_va = [r for r in te_recs if r["setting"] == s]
        if not s_tr or not s_va:
            print(f"Missing image-deshadow REAL data for setting {s} "
                  f"(train={len(s_tr)} val={len(s_va)})", file=sys.stderr)
            sys.exit(1)
        (root / s).mkdir(parents=True, exist_ok=True)
        s_tr_shad, s_tr_clean, s_tr_mask = _stack(s_tr)
        s_va_shad, s_va_clean, s_va_mask = _stack(s_va)
        np.savez(root / s / "train.npz", shad=s_tr_shad, clean=s_tr_clean, mask=s_tr_mask)
        np.savez(root / s / "val.npz", shad=s_va_shad, clean=s_va_clean, mask=s_va_mask)
        floor = _psnr_masked(s_va_shad, s_va_clean, s_va_mask)
        meta[s] = dict(n_train=len(s_tr), n_val=len(s_va), copy_psnr_floor=floor)
        print(f"image-deshadow setting={s} ready: train={len(s_tr)} val={len(s_va)} "
              f"img={IMG}x{IMG} | val SHADOW-region copy PSNR floor={floor:.2f} dB", flush=True)

    (root / "meta.json").write_text(json.dumps(meta, indent=2))

    # final validation
    for s in SETTINGS:
        tr = np.load(root / s / "train.npz")
        va = np.load(root / s / "val.npz")
        if tr["shad"].shape[0] < 50 or va["shad"].shape[0] < 20:
            print(f"Suspiciously little image-deshadow data for setting {s}", file=sys.stderr)
            sys.exit(1)
    print(f"image-deshadow data ready: settings={list(SETTINGS)}")


if __name__ == "__main__":
    main()

"""Prepare data for the image-harmonization tasks from REAL iHarmony4 composites.

IMAGE HARMONIZATION: adjust the appearance (colour / brightness) of a COMPOSITED
FOREGROUND region so it is photometrically consistent with the surrounding background.
This is a DISTINCT restoration direction from inpainting (fill a hole), matting (extract
an alpha), colorization (grey -> colour) or dehazing (remove a global scattering
degradation): here the foreground content is ALREADY PRESENT and CORRECT in structure --
only its COLOUR STATISTICS are wrong (a pasted region whose white balance / exposure /
tint does not match the scene), and the task is to RECOLOUR that region to match the
background.

REAL DATA (iHarmony4, Cong et al., "DoveNet: Deep Image Harmonization via Domain
Verification", CVPR 2020 -- https://github.com/bcmi/Image-Harmonization-Dataset-iHarmony4):
iHarmony4 provides, for each of 4 real-photo sub-datasets, (composite, foreground-mask,
real-GT) triplets: a REAL photo J is the ground truth; a foreground region of J is colour-
transferred (Reinhard-style / histogram / IDT colour-transfer methods, applied by the
dataset authors using REAL reference photos, not a synthetic knob we invent) to produce a
composite whose foreground no longer matches the background, exactly mirroring the
`composite(x) = m(x)*T(J(x)) + (1-m(x))*J(x)` degradation model this harness was already
built around -- except now T is a REAL colour-transfer artifact and J is a REAL photo, not
a synthetic CIFAR patch + invented affine shift.

We use 3 of the 4 iHarmony4 sub-datasets as the THREE settings (mild/medium/strong), one
real PHOTOGRAPHIC DOMAIN each, ordered by their MEASURED foreground-region composite-vs-GT
PSNR "do-nothing" floor at the harness's fixed 64x64 working resolution (higher floor =
milder mismatch = easier setting), exactly the same floor statistic the harness reports as
`comp_fg_psnr`:
  mild   = HCOCO      (COCO-object composites, measured floor ~22-23dB @64px, the largest
                        and easiest sub-dataset -- color-transfer methods calibrated to
                        subtle, photograph-plausible mismatches)
  medium = Hday2night (day/night-swap composites, measured floor ~20-21dB @64px; the
                        smallest sub-dataset, so N_TRAIN/N_VAL below are capped by its size)
  strong = HFlickr     (Flickr composites, measured floor ~19-20dB @64px, the most severe
                        of the three at this working resolution)
We deliberately DROP HAdobe5k (the 4th sub-dataset): at ~40GB it is far larger than needed
for a 64x64 fixed-protocol harness and the 3 remaining sub-datasets already give a well-
separated severity spread with real photographic diversity (COCO objects / Flickr photos /
day-night-swap scenes).

Each sub-dataset's OFFICIAL train/test split (the `<Sub>_train.txt` / `<Sub>_test.txt`
id lists shipped inside the archive) is used verbatim for train/val, so there is no
leakage and the held-out foreground PSNR is a genuine generalisation measure on REAL,
disjoint composites (not just disjoint synthetic RNG streams). Composite/mask/real
filenames follow iHarmony4's documented convention: a composite
`<real_id>_<mask_id>_<variant>.jpg` maps to mask `<real_id>_<mask_id>.png` and real image
`<real_id>.jpg`. Degenerate masks (foreground area <2% or >90% of the frame after the
64x64 resize) are skipped, matching the harness's own tiny-mask guard.

SOURCE / STAGING: the official iHarmony4 README (bcmi/Image-Harmonization-Dataset-iHarmony4)
lists Baidu Netdisk (not scriptable from this environment: requires a China-based
non-programmatic client) and Dropbox (`dl=1` direct-download links, scriptable via curl
through the outbound proxy) mirrors. This script downloads the 3 sub-dataset zips from
their official Dropbox `dl=1` links (or reads them from `--raw-dir` if already staged, e.g.
on a shared moonfs mount) and extracts triplets directly from the zip (no unzip-to-disk
needed). Requires numpy + Pillow (no scipy dependency any more -- the synthetic depth-field
machinery is gone).
"""
import argparse
import io
import random
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np

IMG = 64
N_TRAIN = 2000            # capped by the smallest sub-dataset (Hday2night: 311 train ids)
N_VAL = 400                # capped by the smallest sub-dataset (Hday2night: 133 test ids)
MIN_FG_AREA = 0.02          # skip degenerate (near-empty) masks
MAX_FG_AREA = 0.90          # skip degenerate (near-full-frame) masks

# The 3 settings = 3 REAL iHarmony4 sub-datasets, in mismatch-SEVERITY order (measured
# foreground composite-vs-GT PSNR floor at 64x64, mild=highest floor .. strong=lowest).
# `zip_prefix` is the path prefix used INSIDE each official zip (HCOCO's zip is flat at
# the root; HFlickr/Hday2night nest everything under a `<Sub>/` folder).
SETTINGS = {
    "mild": dict(
        sub="HCOCO", zip_prefix="",
        url="https://www.dropbox.com/scl/fi/kehz4klw5bcggqoxeturs/HCOCO.zip"
            "?rlkey=z53zn5r9cdbvjwgqhaimaut9i&st=b42e4yra&dl=1",
    ),
    "medium": dict(
        sub="Hday2night", zip_prefix="Hday2night/",
        url="https://www.dropbox.com/scl/fi/ytcx9aoocpwkzika1o86i/Hday2night.zip"
            "?rlkey=qqfuludc7971rw2b7as4gr2wf&st=gsor5m2g&dl=1",
    ),
    "strong": dict(
        sub="HFlickr", zip_prefix="HFlickr/",
        url="https://www.dropbox.com/scl/fi/m8fs1nn7owatnutbnp8j5/HFlickr.zip"
            "?rlkey=e380vwzwq6bxml3iw6nzq64cy&st=hxcifhy3&dl=1",
    ),
}


def _ensure_zip(raw_dir: Path, sub: str, url: str) -> Path:
    zip_path = raw_dir / f"{sub}.zip"
    if zip_path.exists() and zipfile.is_zipfile(zip_path):
        return zip_path
    raw_dir.mkdir(parents=True, exist_ok=True)
    print(f"downloading iHarmony4/{sub} <- {url.split('?')[0]} -> {zip_path}", flush=True)
    subprocess.run(
        ["curl", "-sSL", "--retry", "8", "--retry-delay", "10", "-o", str(zip_path), url],
        check=True,
    )
    if not zipfile.is_zipfile(zip_path):
        raise RuntimeError(f"downloaded {sub}.zip is not a valid zip (proxy/auth issue?)")
    return zip_path


def _load_triplet(zf: zipfile.ZipFile, prefix: str, comp_name: str):
    """iHarmony4 naming convention: composite `<real>_<mask>_<variant>.jpg` ->
    mask `<real>_<mask>.png`, real image `<real>.jpg`."""
    from PIL import Image

    base = comp_name[len(prefix) + len("composite_images/"): -4]
    mask_id = base.rsplit("_", 1)[0]
    real_id = mask_id.rsplit("_", 1)[0]
    mask_name = f"{prefix}masks/{mask_id}.png"
    real_name = f"{prefix}real_images/{real_id}.jpg"

    with zf.open(comp_name) as f:
        comp = Image.open(io.BytesIO(f.read())).convert("RGB").resize((IMG, IMG), Image.BILINEAR)
    with zf.open(mask_name) as f:
        mask = Image.open(io.BytesIO(f.read())).convert("L").resize((IMG, IMG), Image.NEAREST)
    with zf.open(real_name) as f:
        real = Image.open(io.BytesIO(f.read())).convert("RGB").resize((IMG, IMG), Image.BILINEAR)

    comp = np.asarray(comp, dtype=np.float32) / 255.0
    real = np.asarray(real, dtype=np.float32) / 255.0
    mask = (np.asarray(mask, dtype=np.float32) / 255.0 > 0.5).astype(np.float32)
    return comp, real, mask


def _build_split(zf: zipfile.ZipFile, prefix: str, sub: str, split: str, n_max: int, seed: int):
    names = set(zf.namelist())
    split_txt = f"{prefix}{sub}_{split}.txt"
    with zf.open(split_txt) as f:
        ids = [line.decode().strip() for line in f if line.strip()]
    rng = random.Random(seed)
    rng.shuffle(ids)

    comps, reals, masks = [], [], []
    for cid in ids:
        comp_name = f"{prefix}composite_images/{cid}"
        if comp_name not in names:
            continue
        try:
            comp, real, mask = _load_triplet(zf, prefix, comp_name)
        except KeyError:
            continue
        if not (MIN_FG_AREA <= mask.mean() <= MAX_FG_AREA):
            continue
        comps.append(comp.transpose(2, 0, 1))       # (3,64,64)
        reals.append(real.transpose(2, 0, 1))
        masks.append(mask[None])                    # (1,64,64)
        if len(comps) >= n_max:
            break

    if not comps:
        raise RuntimeError(f"no valid {sub} {split} triplets found (bad zip contents?)")
    comp = np.stack(comps, 0).astype(np.float32)
    real = np.stack(reals, 0).astype(np.float32)
    mask = np.stack(masks, 0).astype(np.float32)
    return comp, real, mask


def _fg_psnr(comp, real, mask):
    """Foreground-region PSNR of the composite input vs the real GT (the do-nothing
    floor a harmonizer must beat)."""
    m = mask
    se = ((comp - real) ** 2) * m
    denom = m.sum(axis=(1, 2, 3)).clip(1.0)
    mse = se.sum(axis=(1, 2, 3)) / denom
    mse = mse.clip(1e-10)
    return float(np.mean(10.0 * np.log10(1.0 / mse)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--raw-dir", default=None, type=Path,
                     help="dir with pre-downloaded <Sub>.zip files (HCOCO.zip, "
                          "HFlickr.zip, Hday2night.zip); skips the Dropbox download")
    args = ap.parse_args()

    root = args.data_root.expanduser().resolve() / "image-harmonization"
    root.mkdir(parents=True, exist_ok=True)

    have_all = all((root / f"{sp}_{sev}.npz").exists()
                   for sp in ("train", "val") for sev in SETTINGS)
    if have_all:
        ok = True
        for sev in SETTINGS:
            tr = np.load(root / f"train_{sev}.npz"); va = np.load(root / f"val_{sev}.npz")
            if tr["comp"].shape[0] < 50 or va["comp"].shape[0] < 50:
                ok = False
        if ok:
            print(f"image-harmonization data ready (cached): {list(SETTINGS)}")
            return

    raw_dir = args.raw_dir if args.raw_dir else (root / "_raw")

    for sev, cfg in SETTINGS.items():
        sub, prefix, url = cfg["sub"], cfg["zip_prefix"], cfg["url"]
        print(f"preparing severity={sev} <- iHarmony4/{sub} ...", flush=True)
        zip_path = _ensure_zip(raw_dir, sub, url)
        with zipfile.ZipFile(zip_path) as zf:
            c_tr, r_tr, m_tr = _build_split(zf, prefix, sub, "train", N_TRAIN, seed=1000)
            c_va, r_va, m_va = _build_split(zf, prefix, sub, "test", N_VAL, seed=5000)
        np.savez(root / f"train_{sev}.npz", comp=c_tr, real=r_tr, mask=m_tr)
        np.savez(root / f"val_{sev}.npz", comp=c_va, real=r_va, mask=m_va)
        floor = _fg_psnr(c_va, r_va, m_va)
        print(f"  {sev} ({sub}): train={c_tr.shape[0]} val={c_va.shape[0]} img={IMG}x{IMG} | "
              f"val FOREGROUND composite-INPUT PSNR floor={floor:.2f} dB", flush=True)

    for sev in SETTINGS:
        va = np.load(root / f"val_{sev}.npz")
        if va["comp"].shape[0] < 50:
            print("Missing image-harmonization data", file=sys.stderr); sys.exit(1)
    print("image-harmonization data ready (REAL iHarmony4):", list(SETTINGS))


if __name__ == "__main__":
    main()

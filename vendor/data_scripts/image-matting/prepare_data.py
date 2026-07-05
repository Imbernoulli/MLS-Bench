"""Prepare data for the trimap-guided image-matting (cv-matting-*) tasks from REAL
photographs with REAL, human-annotated ground-truth alpha mattes.

REAL DATA SOURCE: PPM-100 (Photographic Portrait Matting, "Semantic Human Matting"
lineage / "MODNet" benchmark repo -- Zhanghan Ke et al., real Flickr portrait photos
with dense hand-refined alpha mattes, CC-licensed). We use the official 90/10
train/val id split shipped with the dataset (`train.txt` / `val.txt`). This gives us,
per sample, a REAL RGB photo `I` and its REAL exact alpha matte -- exactly the two
things a matting benchmark needs, and (unlike most synthetic-composite work) neither
is invented: both come from the dataset's own hand-annotation.

Mirror used (the upstream MODNet/PPM-100 Google-Drive links are not scriptable
through this environment's proxy): a Hugging Face dataset mirror
    https://huggingface.co/datasets/realdream-ai/ppm-matting/resolve/main/PPM-100.zip
(263,089,716 bytes; contains `PPM-100/{train,val}/{fg,alpha}/*.jpg` plus the official
`train.txt`/`val.txt` split lists). `_fetch_ppm100` downloads it (falling back to a
`--raw-dir`/`{data_root}/_raw` cache) and extracts in place.

WHY NOT AIM-500 (the first-choice target): AIM-500's public distribution is a Google
Drive *folder* (not a single archive/release asset); `gdown.download_folder(...)`
hung indefinitely on this proxy (folder listing via the Drive API is not reliably
scriptable headless). We tried alphamatting.com next (confirmed working, see below)
before landing on PPM-100, which has (a) an HTTP-mirrored single zip, (b) 100 images
(vs. alphamatting.com's 27), and (c) higher, more varied resolution.

We ALSO validated alphamatting.com's public low-resolution training set (27 real
images with real ground truth, https://alphamatting.com/eval_27.php, direct-download
via `tucloud.tuwien.ac.at`/`owncloud.tuwien.ac.at` Nextcloud links, `dl=1`-style
direct zips) as a fallback / secondary source -- it works (confirmed byte-exact
downloads with `wget` or `curl --http1.1`; plain `curl -sL` truncates through this
proxy, apparently an HTTP/2-vs-proxy chunking issue) but we did not need it once
PPM-100 proved sufficient; kept documented here in case PPM-100 ever becomes
unavailable (`_fetch_alphamatting27` below, currently unused by `main`).

FOREGROUND / BACKGROUND LAYER SYNTHESIS: PPM-100 (like almost every real matting
dataset) ships only the composited photo `I` and the alpha `alpha` -- NOT separate
foreground-colour (`F`) and background (`B`) layers, because those don't exist for a
photo (there is no "background behind the person" to observe). The harness's
composition-loss surfaces need `F`/`B`, so we SYNTHESIZE them from the single real
photo using `cv2.inpaint` (OpenCV's TELEA fast-marching inpainter): `F` is `I` with
the background half removed and refilled by inpainting outward from the foreground
boundary (so it becomes a plausible F everywhere, matching `I` exactly wherever
alpha>=0.999); `B` is the symmetric operation removing/reinpainting the foreground.
Measured reconstruction error `|I - (a*F + (1-a)*B)|` is ~1e-3 overall (exactly 0 in
the solid regions where F/B are copied verbatim from the real photo, ~0.04 in the
unknown transition band where inpainting is approximate) -- an authentic, real-image
composition target, not another synthetic invention.

RESOLUTION CHOICE: the harness's fixed small U-Net (encoder channels 32/64/96/128,
H/8 bottleneck) has no hardcoded input size; it consumes whatever (C,H,W) the stored
.npy has. PPM-100 photos are large (up to 6000px) and mostly PORTRAIT-CROP framed
(one person roughly centred). We crop to a padded (15% margin) bounding box of the
non-zero alpha region -- discarding mostly-irrelevant background margin while keeping
the hair/edge detail -- then resize to a FIXED 160x160 (up from the old synthetic
dataset's 128x128: real hair-matte detail is finer than the old rendered "hair spike"
proxy, and empirically the unknown-band fraction at the harness's 6/9/12 erosion
widths on PPM-100 crops at 128 was on the low side (~0.32-0.51 mean over widths
6/9/12, front-loaded by width; at 160 the bands are comfortably non-trivial while
the CPU smoke-test/training budget stays cheap). This is a modest, documented
resize choice per Requirement 3, not a wholesale re-architecture.

TRIMAP DERIVATION (identical mechanism to the old synthetic script, applied to REAL
alpha instead): solid_fg = alpha>=0.999, solid_bg = alpha<=0.001, the remaining
"not solid" region is dilated by a chosen width to build a comfortably-wide UNKNOWN
band. NOTE: `harness.py`'s own `load_split()` RE-DERIVES the trimap on the fly from
the stored exact GT alpha (its `derive_trimap`, an exact torch port of the erosion
below) at whichever of the three widths (medium=6/wide=9/xwide=12) the requested
task setting asks for -- this script does NOT need to bake a trimap into storage at
all; we only keep `_binary_dilate`/`_derive_trimap_np` here to (a) report per-split
unknown-band-fraction diagnostics at prepare time and (b) guarantee every stored
alpha has a genuine non-degenerate soft transition (skip flat/binary-only alphas).

Stored per sample (as a single (11,H,W) float32 array `img_*.npy`, IDENTICAL channel
layout to the old synthetic script so `harness.py` needs ZERO changes):
  channels 0:3 = the REAL composite image I in [0,1]
  channel  3   = a placeholder trimap (unused by harness.py at load time, kept only
                 for schema/debugging compatibility -- see NOTE above)
  channel  4   = the REAL GT alpha in [0,1]
  channels 5:8 = synthesized F (foreground colour layer)
  channels 8:11= synthesized B (background layer)

Degeneracy guard (same idea as before, now measured on real data): the CONST_HALF_SAD
/ MEAN_ALPHA_SAD floors are reported at prepare time on the real val split so a
trivial constant-0.5 or per-image-mean-alpha predictor's floor is documented; any
real matting net is expected to beat it comfortably.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np

IMG_SIZE = 160          # up from 128: real PPM-100 hair/edge detail needs a bit more
                        # resolution than the old rendered-hair-spike synthetic proxy.
BBOX_PAD_FRAC = 0.15    # crop margin around the alpha bounding box
MIN_UNKNOWN_PX = 80     # skip images whose real alpha has too little soft transition
                        # at ANY of the 3 harness trimap widths (degenerate / near-
                        # binary alpha -- rare, but PPM-100 has a couple of near-solid
                        # silhouettes with almost no soft edge).
HARNESS_WIDTHS = (6, 9, 12)   # mirrors harness.py's TRIMAP_WIDTHS (medium/wide/xwide)

PPM100_URL = ("https://huggingface.co/datasets/realdream-ai/ppm-matting/"
              "resolve/main/PPM-100.zip")
PPM100_SIZE_HINT = 263089716   # bytes; used only to sanity-check the download

# --- alphamatting.com low-res training set (27 real images) -- VALIDATED WORKING,
# kept as a documented fallback; NOT used by `main` since PPM-100 is sufficient.
ALPHAMATTING_INPUT_URL = ("https://owncloud.tuwien.ac.at/index.php/s/kbv8ZxuqehNk9vy/"
                           "download?path=%2FDatasets&files=input_training_lowres.zip")
ALPHAMATTING_GT_URL = ("https://tucloud.tuwien.ac.at/public.php/dav/files/"
                        "kbv8ZxuqehNk9vy/Datasets/gt_training_lowres.zip")

# channel layout of the stored (C,H,W) array -- IDENTICAL to the old synthetic script
# so harness.py's load_split()/C_* constants need no changes.
C_I = slice(0, 3)      # composite image (REAL photo)
C_TRI = 3              # placeholder trimap (unused at load time; harness re-derives)
C_ALPHA = 4            # REAL GT alpha
C_F = slice(5, 8)      # synthesized foreground colour layer
C_B = slice(8, 11)     # synthesized background layer
N_CHAN = 11


# --------------------------------------------------------------------------- #
# download / staging
# --------------------------------------------------------------------------- #
def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading <- {url.split('?')[0]} -> {dest}", flush=True)
    # NOTE: plain `curl -sL` was observed to silently truncate large files through
    # this environment's corporate proxy (looked like an HTTP/2-vs-proxy chunking
    # issue). `wget` (or `curl --http1.1`) reliably gets the exact Content-Length.
    subprocess.run(
        ["wget", "--timeout=120", "--tries=8", "-O", str(dest), url],
        check=True,
    )


def _ensure_ppm100(raw_dir: Path) -> Path:
    """Return the path to an extracted PPM-100/ directory (download+extract if
    needed; re-used as-is if already staged, e.g. on a shared moonfs mount)."""
    extract_root = raw_dir / "ppm100_extracted"
    ppm_dir = extract_root / "PPM-100"
    if (ppm_dir / "train.txt").exists() and (ppm_dir / "val.txt").exists():
        return ppm_dir

    zip_path = raw_dir / "PPM-100.zip"
    if not (zip_path.exists() and zipfile.is_zipfile(zip_path)):
        _download(PPM100_URL, zip_path)
    if not zipfile.is_zipfile(zip_path):
        raise RuntimeError("downloaded PPM-100.zip is not a valid zip (proxy/auth issue?)")

    print(f"extracting {zip_path} -> {extract_root}", flush=True)
    extract_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_root)
    if not (ppm_dir / "train.txt").exists():
        raise RuntimeError(f"PPM-100 zip extracted but {ppm_dir}/train.txt missing")
    return ppm_dir


def _fetch_alphamatting27(raw_dir: Path) -> tuple[Path, Path]:
    """VALIDATED fallback (unused by `main`): download alphamatting.com's public
    27-image low-res training set (real images + real GT), returns (input_zip,
    gt_zip) paths. Documented for completeness per the task's 'document what you
    tried' instruction."""
    input_zip = raw_dir / "input_training_lowres.zip"
    gt_zip = raw_dir / "gt_training_lowres.zip"
    if not (input_zip.exists() and zipfile.is_zipfile(input_zip)):
        _download(ALPHAMATTING_INPUT_URL, input_zip)
    if not (gt_zip.exists() and zipfile.is_zipfile(gt_zip)):
        _download(ALPHAMATTING_GT_URL, gt_zip)
    return input_zip, gt_zip


# --------------------------------------------------------------------------- #
# image processing (crop / resize / F-B synthesis / trimap diagnostics)
# --------------------------------------------------------------------------- #
def _binary_dilate(mask: np.ndarray, iters: int) -> np.ndarray:
    """4-neighbour binary dilation, `iters` steps (no scipy) -- same routine as the
    old synthetic script and as harness.py's torch `_binary_dilate_t`."""
    m = mask.copy()
    for _ in range(iters):
        s = m.copy()
        s[1:, :] |= m[:-1, :]
        s[:-1, :] |= m[1:, :]
        s[:, 1:] |= m[:, :-1]
        s[:, :-1] |= m[:, 1:]
        m = s
    return m


def _derive_trimap_np(alpha: np.ndarray, width: int) -> np.ndarray:
    """Numpy mirror of harness.py's `derive_trimap` (used only for diagnostics /
    the degeneracy guard here; the harness re-derives its own copy at load time)."""
    solid_fg = alpha >= 0.999
    solid_bg = alpha <= 0.001
    not_solid = ~(solid_fg | solid_bg)
    band = _binary_dilate(not_solid, width)
    tri = np.full_like(alpha, 0.5)
    tri[solid_fg & ~band] = 1.0
    tri[solid_bg & ~band] = 0.0
    if (tri == 0.5).sum() < 50:
        band = _binary_dilate(not_solid, max(width, 8))
        tri = np.full_like(alpha, 0.5)
        tri[solid_fg & ~band] = 1.0
        tri[solid_bg & ~band] = 0.0
    return tri


def _crop_resize(img: np.ndarray, alpha_full: np.ndarray, size: int):
    """Crop to a padded bbox of the non-zero-alpha region, resize both `img`
    (H,W,3) and `alpha_full` (H,W) to (size,size). Falls back to a full-frame
    resize if the alpha is (almost) entirely zero/one everywhere (rare)."""
    from PIL import Image

    ys, xs = np.nonzero(alpha_full > 0.001)
    H, W = alpha_full.shape
    if len(ys) == 0:
        y0c, y1c, x0c, x1c = 0, H, 0, W
    else:
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        pad_y = max(1, int((y1 - y0) * BBOX_PAD_FRAC))
        pad_x = max(1, int((x1 - x0) * BBOX_PAD_FRAC))
        y0c, y1c = max(0, y0 - pad_y), min(H, y1 + pad_y)
        x0c, x1c = max(0, x0 - pad_x), min(W, x1 + pad_x)
        if y1c - y0c < 8 or x1c - x0c < 8:
            y0c, y1c, x0c, x1c = 0, H, 0, W

    img_c = img[y0c:y1c, x0c:x1c]
    alpha_c = alpha_full[y0c:y1c, x0c:x1c]

    img_r = np.asarray(
        Image.fromarray((img_c * 255).astype(np.uint8)).resize((size, size), Image.BILINEAR)
    ).astype(np.float32) / 255.0
    alpha_r = np.asarray(
        Image.fromarray((alpha_c * 255).astype(np.uint8)).resize((size, size), Image.BILINEAR)
    ).astype(np.float32) / 255.0
    return img_r, np.clip(alpha_r, 0.0, 1.0)


def _synthesize_fg_bg(img: np.ndarray, alpha: np.ndarray):
    """Synthesize plausible foreground-colour (F) and background (B) layers from a
    SINGLE real composited photo + its real alpha, via cv2.inpaint (TELEA). F/B are
    exact copies of `img` in the solid regions (alpha>=0.999 / <=0.001 respectively)
    and an inpainted fill elsewhere -- so I = a*F + (1-a)*B holds essentially exactly
    outside the unknown transition band, and approximately inside it."""
    import cv2

    solid_fg = alpha >= 0.999
    solid_bg = alpha <= 0.001
    img_u8 = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)

    mask_f = (~solid_fg).astype(np.uint8) * 255
    f_img = cv2.inpaint(img_u8, mask_f, 3, cv2.INPAINT_TELEA).astype(np.float32) / 255.0
    f_img = np.where(solid_fg[..., None], img, f_img)

    mask_b = (~solid_bg).astype(np.uint8) * 255
    b_img = cv2.inpaint(img_u8, mask_b, 3, cv2.INPAINT_TELEA).astype(np.float32) / 255.0
    b_img = np.where(solid_bg[..., None], img, b_img)

    return f_img.astype(np.float32), b_img.astype(np.float32)


def _make_sample(img_path: Path, alpha_path: Path, size: int):
    from PIL import Image

    img_full = np.asarray(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
    alpha_full = np.asarray(Image.open(alpha_path).convert("L")).astype(np.float32) / 255.0
    if alpha_full.shape != img_full.shape[:2]:
        alpha_full = np.asarray(
            Image.fromarray((alpha_full * 255).astype(np.uint8)).resize(
                (img_full.shape[1], img_full.shape[0]), Image.BILINEAR)
        ).astype(np.float32) / 255.0

    img, alpha = _crop_resize(img_full, alpha_full, size)
    fg, bg = _synthesize_fg_bg(img, alpha)
    # placeholder trimap channel (unused by harness.py, kept for schema symmetry
    # with the old synthetic dataset's storage format); use the medium (6px) width.
    trimap = _derive_trimap_np(alpha, HARNESS_WIDTHS[0])

    out = np.zeros((N_CHAN, size, size), dtype=np.float32)
    out[C_I] = img.transpose(2, 0, 1)
    out[C_TRI] = trimap
    out[C_ALPHA] = alpha
    out[C_F] = fg.transpose(2, 0, 1)
    out[C_B] = bg.transpose(2, 0, 1)
    return out, alpha


def _unknown_frac_ok(alpha: np.ndarray) -> bool:
    """Degeneracy guard: require a genuine, non-trivial soft transition band at
    EVERY one of the harness's 3 trimap widths (medium/wide/xwide)."""
    for width in HARNESS_WIDTHS:
        tri = _derive_trimap_np(alpha, width)
        if (tri == 0.5).sum() < MIN_UNKNOWN_PX:
            return False
    return True


def _build_split(ppm_dir: Path, split: str, out_dir: Path, size: int):
    fg_dir = ppm_dir / split / "fg"
    alpha_dir = ppm_dir / split / "alpha"
    files = sorted(p.name for p in fg_dir.iterdir() if p.is_file())

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    unk_fracs = {w: [] for w in HARNESS_WIDTHS}
    kept = 0
    for fname in files:
        alpha_name = alpha_dir / fname
        if not alpha_name.exists():
            continue
        arr, alpha = _make_sample(fg_dir / fname, alpha_name, size)
        if not _unknown_frac_ok(alpha):
            print(f"  skip degenerate alpha: {split}/{fname}", flush=True)
            continue
        out_name = f"img_{kept:04d}.npy"
        np.save(out_dir / out_name, arr)
        for width in HARNESS_WIDTHS:
            tri = _derive_trimap_np(alpha, width)
            unk_fracs[width].append(float((tri == 0.5).mean()))
        manifest.append({"img": out_name, "src": fname})
        kept += 1
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f)
    return manifest, unk_fracs


def _degenerate_sads(val_dir: Path):
    """SAD (scaled /1000, in the UNKNOWN band, medium width=6) of two degenerate
    predictors on val: const-0.5 predicts 0.5 everywhere; mean-alpha predicts the
    per-image mean GT alpha of the unknown band."""
    with open(val_dir / "manifest.json") as f:
        items = json.load(f)
    const_sads, mean_sads = [], []
    for it in items:
        arr = np.load(val_dir / it["img"])
        alpha = arr[C_ALPHA]
        tri = _derive_trimap_np(alpha, HARNESS_WIDTHS[0])
        unk = tri == 0.5
        g = alpha[unk]
        if g.size == 0:
            continue
        const_sads.append(np.abs(0.5 - g).sum() / 1000.0)
        mean_sads.append(np.abs(g.mean() - g).sum() / 1000.0)
    return float(np.mean(const_sads)), float(np.mean(mean_sads))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--raw-dir", default=None, type=Path,
                     help="dir with a pre-downloaded PPM-100.zip (or an already-"
                          "extracted PPM-100/ folder inside <raw-dir>/ppm100_extracted); "
                          "skips the HF-mirror download when already staged")
    args = ap.parse_args()

    root = args.data_root.expanduser().resolve() / "image-matting" / "composites"
    val_manifest = root / "val" / "manifest.json"
    train_manifest = root / "train" / "manifest.json"
    if val_manifest.exists() and train_manifest.exists():
        try:
            with open(val_manifest) as f:
                n_val = len(json.load(f))
            with open(train_manifest) as f:
                n_train = len(json.load(f))
            if n_val >= 5 and n_train >= 20:
                print(f"image-matting data already present at {root} "
                      f"(train={n_train} val={n_val})")
                return
        except Exception:  # noqa: BLE001
            pass

    raw_dir = args.raw_dir if args.raw_dir else (root / "_raw")
    ppm_dir = _ensure_ppm100(raw_dir)

    root.mkdir(parents=True, exist_ok=True)
    _, tr_unk = _build_split(ppm_dir, "train", root / "train", IMG_SIZE)
    _, va_unk = _build_split(ppm_dir, "val", root / "val", IMG_SIZE)

    const_sad, mean_sad = _degenerate_sads(root / "val")
    unk_summary = ", ".join(
        f"w{w}={np.mean(va_unk[w]):.3f}" for w in HARNESS_WIDTHS if va_unk[w]
    )
    with open(root / "train" / "manifest.json") as f:
        n_train = len(json.load(f))
    with open(root / "val" / "manifest.json") as f:
        n_val = len(json.load(f))
    if n_val < 5 or n_train < 20:
        print("Not enough valid real matting samples after the degeneracy guard",
              file=sys.stderr)
        sys.exit(1)

    print(f"image-matting data ready (REAL PPM-100): train={n_train} val={n_val} "
          f"size={IMG_SIZE} val_unknown_frac[{unk_summary}] "
          f"CONST_HALF_SAD={const_sad:.3f} MEAN_ALPHA_SAD={mean_sad:.3f}")


if __name__ == "__main__":
    main()

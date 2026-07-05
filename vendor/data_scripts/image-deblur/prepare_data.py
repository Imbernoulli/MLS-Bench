"""Prepare data for the deblur-* image-deblurring tasks (THREE motion-blur SEVERITIES,
built from REAL blurry/sharp photograph pairs).

Produces, under {data_root}/image-deblur/<setting>/ for each of the THREE distinct
motion-blur SEVERITIES that the deblur-* tasks aggregate the score over:

    small/   train.npz  val.npz    # mildest real motion blur   (blurry-input floor highest)
    medium/  train.npz  val.npz    # medium real motion blur
    large/   train.npz  val.npz    # heaviest real motion blur  (blurry-input floor lowest)

Each npz has  blur (N,3,64,64) in [0,1]  and  sharp (N,3,64,64) in [0,1].

REAL DATA SOURCE: GoPro Large-Scale Blur Dataset (Nah, Kim & Lee, "Deep Multi-Scale
Convolutional Neural Network for Dynamic Scene Deblurring", CVPR 2017). GoPro captures
scenes with a 240fps GoPro camera; the BLURRY frame is a real average of consecutive
sharp frames (genuine camera+motion blur, not a synthetic kernel convolution) and the
SHARP frame is a real single high-speed exposure of the same instant -- both are
authentic photographs, aligned pixel-for-pixel. We use the author-hosted mirror
"snah/GOPRO_Large" on the HuggingFace Hub (streamed with `remotezip` so the ~9.5GB
archive is never downloaded in full; only a curated subset of sequences is pulled).

THREE SEVERITIES from REAL data (no synthetic severity knob exists for a real photograph
pair): every 64x64 tile's blur strength is a per-tile MEASURED quantity -- the PSNR of
the blurry tile against the co-located sharp tile:

    tile_psnr = 10*log10(1 / mean((blur_tile - sharp_tile)^2))     (higher = milder blur)

64x64 non-overlapping tiles are cut from each full-resolution (1280x720) frame pair;
near-uniform tiles (little content, sharp-tile std < MIN_STD on the 0-255 scale -- sky,
out-of-focus background) are dropped since they carry no real deblurring signal. The
kept tiles are bucketed into TERCILES of tile_psnr (pooled across the GoPro train +
test scenes, so the cut points are a property of the sampled corpus, not of one split)
into small (mildest real blur, PSNR tercile closest to the max) / medium / large
(heaviest real blur, PSNR tercile closest to the min) -- a single real, physically
measured severity ladder built from genuine motion-blur photographs.

Every one of the ORIGINAL fifteen synthetic "severity band" codes used by the ten
deblur-* tasks (small/medium/large, rs/rm/rl, ms/mm/ml, es/em/el, hs/hm/hl) is an ALIAS
onto this SAME three-bucket real ladder (see `_ALIAS`) -- the synthetic generator could
freely retune five independent severity bands (one per task family) because it invented
its own blur kernels; real photographs only offer one genuine measured severity axis, so
every task now shares the one real ladder instead of five fabricated ones. Tasks whose
design lever failed to stay monotone on this real ladder were dropped (see the CPU
smoke-test report), not force-fit with cherry-picked percentile windows.

Train/val WITHIN each severity bucket reuse GoPro's own disjoint-SCENE train/test split
(the four staged train sequences vs the four staged test sequences never share a scene),
so held-out PSNR still measures real cross-scene generalisation, exactly like the
train/val split used by the original synthetic generator (disjoint patch RNG streams).

Requires numpy + Pillow (+ remotezip only for the on-the-fly download path).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

IMG = 64
N_TRAIN = 2000
N_VAL = 400
MIN_STD = 8.0          # drop near-uniform tiles (0-255 scale std threshold)

# HF mirror of the official author-hosted GoPro Large dataset.
GOPRO_ZIP_URL = "https://hf-mirror.com/datasets/snah/GOPRO_Large/resolve/main/GOPRO_Large.zip"
# Curated subset (kept small: disk-constrained staging host has no spare local disk;
# moonfs has 1.6PB free, so this lives there, never on local disk). 60 frames/sequence.
TRAIN_SEQS = ["GOPR0372_07_00", "GOPR0374_11_00", "GOPR0477_11_00", "GOPR0857_11_00"]
TEST_SEQS = ["GOPR0384_11_00", "GOPR0396_11_00", "GOPR0862_11_00", "GOPR0881_11_01"]
FRAMES_PER_SEQ = 60

# The 3 real settings every task aggregates its score over.
SETTINGS = ("small", "medium", "large")

# Every legacy synthetic "severity band" code used by tasks/scripts is an ALIAS onto the
# ONE real tercile ladder above -- real photographs give one genuine severity axis, not
# five independently-tunable bands.
_ALIAS = {
    "small": "small", "medium": "medium", "large": "large",
    "rs": "small", "rm": "medium", "rl": "large",         # residual / edge-loss band
    "ms": "small", "mm": "medium", "ml": "large",         # multiscale/arch band
    "es": "small", "em": "small", "el": "large",          # loss-kind's em/el (+ medium)
    "hs": "small", "hm": "medium", "hl": "large",         # dilation "very-heavy" band
}


def _resolve_setting(blur_type: str) -> str:
    if blur_type not in _ALIAS:
        raise ValueError(f"unknown --blur-type {blur_type!r}; known: {sorted(_ALIAS)}")
    return _ALIAS[blur_type]


# --------------------------------------------------------------------------- #
# Raw frame staging: a pre-staged directory (GOPRO_RAW_DIR, e.g. the moonfs manifest
# already prepared on the B0 devmachine) is used first -- this is REQUIRED on
# network-restricted build hosts (e.g. k1) that cannot reach hf-mirror.com. If absent,
# fall back to streaming the needed frames directly out of the remote zip with
# `remotezip` (no full-archive download, safe even on disk-constrained hosts).
# --------------------------------------------------------------------------- #
def _raw_dir(cache_dir: Path) -> Path:
    pre = os.environ.get("GOPRO_RAW_DIR")
    if pre:
        p = Path(pre)
        if p.exists():
            return p
        print(f"GOPRO_RAW_DIR={pre!r} does not exist, falling back to streaming download",
              file=sys.stderr)
    out = cache_dir / "gopro_raw"
    _stream_download(out)
    return out


def _stream_download(out_root: Path) -> None:
    from remotezip import RemoteZip

    for split, seqs in (("train", TRAIN_SEQS), ("test", TEST_SEQS)):
        for seq in seqs:
            bdir = out_root / split / seq / "blur"
            sdir = out_root / split / seq / "sharp"
            done = (bdir.exists() and sdir.exists()
                    and len(list(bdir.glob("*.png"))) >= FRAMES_PER_SEQ
                    and len(list(sdir.glob("*.png"))) >= FRAMES_PER_SEQ)
            if done:
                print(f"[{split}/{seq}] raw frames already staged (cached)", flush=True)
                continue
            bdir.mkdir(parents=True, exist_ok=True)
            sdir.mkdir(parents=True, exist_ok=True)
            print(f"[{split}/{seq}] streaming {FRAMES_PER_SEQ} frame pairs from GoPro_Large "
                  f"via remotezip ...", flush=True)
            rz = RemoteZip(GOPRO_ZIP_URL)
            names = rz.namelist()
            blur_names = sorted(n for n in names
                                 if n.startswith(f"{split}/{seq}/blur/") and n.endswith(".png")
                                 )[:FRAMES_PER_SEQ]
            sharp_names = sorted(n for n in names
                                  if n.startswith(f"{split}/{seq}/sharp/") and n.endswith(".png")
                                  )[:FRAMES_PER_SEQ]
            for names_, outdir in ((blur_names, bdir), (sharp_names, sdir)):
                for n in names_:
                    outp = outdir / os.path.basename(n)
                    if outp.exists():
                        continue
                    data = rz.read(n)
                    outp.write_bytes(data)
            print(f"[{split}/{seq}] done: blur={len(blur_names)} sharp={len(sharp_names)}",
                  flush=True)


# --------------------------------------------------------------------------- #
# Tiling: cut every staged (blur, sharp) frame pair into non-overlapping 64x64 tiles,
# drop near-uniform ones, and keep the REAL blur/sharp pixel data + measured PSNR.
# --------------------------------------------------------------------------- #
def _psnr_tile(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float32) / 255.0 - b.astype(np.float32) / 255.0) ** 2))
    mse = max(mse, 1e-10)
    return 10.0 * float(np.log10(1.0 / mse))


def _tile_split(raw_root: Path, split: str, seqs: list[str]):
    """Return list of dicts: seq, blur(uint8 HWC), sharp(uint8 HWC), psnr."""
    from PIL import Image

    recs = []
    for seq in seqs:
        bfiles = sorted(glob.glob(str(raw_root / split / seq / "blur" / "*.png")))
        if len(bfiles) < FRAMES_PER_SEQ:
            print(f"Missing raw GoPro frames for {split}/{seq}: found {len(bfiles)}, "
                  f"need {FRAMES_PER_SEQ}. Set GOPRO_RAW_DIR to a pre-staged manifest "
                  f"(see vendor/image-deblur docs) if this host has no internet access.",
                  file=sys.stderr)
            sys.exit(1)
        for bf in bfiles:
            sf = bf.replace(f"{os.sep}blur{os.sep}", f"{os.sep}sharp{os.sep}")
            if not os.path.exists(sf):
                continue
            b_full = np.asarray(Image.open(bf).convert("RGB"))
            s_full = np.asarray(Image.open(sf).convert("RGB"))
            H, W = s_full.shape[:2]
            for y in range(0, H - IMG + 1, IMG):
                for x in range(0, W - IMG + 1, IMG):
                    st = s_full[y:y + IMG, x:x + IMG]
                    bt = b_full[y:y + IMG, x:x + IMG]
                    if float(st.astype(np.float32).std()) < MIN_STD:
                        continue
                    recs.append(dict(seq=seq, blur=bt, sharp=st, psnr=_psnr_tile(bt, st)))
        print(f"[{split}/{seq}] tiled: {len(bfiles)} frame pairs -> "
              f"{sum(1 for r in recs if r['seq'] == seq)} kept tiles", flush=True)
    return recs


def _bucket_settings(all_recs) -> dict:
    """Tercile-bucket ALL tiles (train+test pooled) by MEASURED tile PSNR into
    small (mildest real blur, PSNR tercile closest to max) / medium / large (heaviest
    real blur, PSNR tercile closest to min). Mutates each rec with a 'setting' key;
    returns {setting: (lo_psnr, hi_psnr)} for provenance."""
    psnrs = np.array([r["psnr"] for r in all_recs])
    q1, q2 = np.percentile(psnrs, [100.0 / 3.0, 200.0 / 3.0])
    for r, p in zip(all_recs, psnrs):
        if p > q2:
            r["setting"] = "small"
        elif p > q1:
            r["setting"] = "medium"
        else:
            r["setting"] = "large"
    return {
        "small": (float(q2), float(psnrs.max())),
        "medium": (float(q1), float(q2)),
        "large": (float(psnrs.min()), float(q1)),
    }


def _stack(recs, rng: np.random.RandomState, n: int):
    """Deterministically subsample `n` tiles (without replacement) and stack into
    (n,3,64,64) float32 [0,1] blur/sharp arrays."""
    if len(recs) < n:
        print(f"Only {len(recs)} tiles available, need {n}", file=sys.stderr)
        sys.exit(1)
    idx = rng.choice(len(recs), size=n, replace=False)
    blur = np.stack([recs[i]["blur"] for i in idx], 0).astype(np.float32) / 255.0
    sharp = np.stack([recs[i]["sharp"] for i in idx], 0).astype(np.float32) / 255.0
    return blur.transpose(0, 3, 1, 2).copy(), sharp.transpose(0, 3, 1, 2).copy()


def _psnr_batch(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a - b) ** 2, axis=(1, 2, 3)).clip(1e-10)
    return float(np.mean(10.0 * np.log10(1.0 / mse)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--blur-type", default="all",
                    choices=("all",) + tuple(SETTINGS) + tuple(_ALIAS.keys()),
                    help="(kept for CLI compatibility with the legacy band codes; every "
                         "code resolves to one of the 3 real settings -- see _ALIAS)")
    args = ap.parse_args()

    root = args.data_root.expanduser().resolve() / "image-deblur"
    root.mkdir(parents=True, exist_ok=True)

    ready = all((root / s / "train.npz").exists() and (root / s / "val.npz").exists()
                for s in SETTINGS)
    if ready:
        print(f"image-deblur data ready (cached): settings={list(SETTINGS)}")
        return

    cache_dir = root / "_raw"
    cache_dir.mkdir(parents=True, exist_ok=True)
    raw_root = _raw_dir(cache_dir)

    print("tiling REAL GoPro_Large (Nah et al. CVPR 2017) blurry/sharp frame pairs "
          "into 64x64 patches ...", flush=True)
    tr_recs = _tile_split(raw_root, "train", TRAIN_SEQS)
    te_recs = _tile_split(raw_root, "test", TEST_SEQS)
    print(f"usable tiles: train={len(tr_recs)} test={len(te_recs)}", flush=True)

    bounds = _bucket_settings(tr_recs + te_recs)
    print(f"tercile PSNR bounds (lo,hi dB; lower = heavier real blur): {bounds}", flush=True)

    meta = dict(source="snah/GOPRO_Large (HF mirror of GoPro Large, Nah et al. CVPR 2017)",
                img=IMG, min_std=MIN_STD, bounds=bounds, alias=_ALIAS)
    for s in SETTINGS:
        s_tr = [r for r in tr_recs if r["setting"] == s]
        s_va = [r for r in te_recs if r["setting"] == s]
        if len(s_tr) < N_TRAIN or len(s_va) < N_VAL:
            print(f"Not enough real image-deblur tiles for setting {s} "
                  f"(train={len(s_tr)}/{N_TRAIN} val={len(s_va)}/{N_VAL})", file=sys.stderr)
            sys.exit(1)
        (root / s).mkdir(parents=True, exist_ok=True)
        tr_rng = np.random.RandomState(hash(("train", s)) % (2 ** 31))
        va_rng = np.random.RandomState(hash(("val", s)) % (2 ** 31))
        b_tr, s_tr_ = _stack(s_tr, tr_rng, N_TRAIN)
        b_va, s_va_ = _stack(s_va, va_rng, N_VAL)
        np.savez(root / s / "train.npz", blur=b_tr, sharp=s_tr_)
        np.savez(root / s / "val.npz", blur=b_va, sharp=s_va_)
        floor = _psnr_batch(b_va, s_va_)
        meta[s] = dict(n_train_pool=len(s_tr), n_val_pool=len(s_va),
                        n_train=N_TRAIN, n_val=N_VAL, blurry_psnr_floor=floor)
        print(f"[{s}] ready: train={N_TRAIN}/{len(s_tr)} val={N_VAL}/{len(s_va)} "
              f"img={IMG}x{IMG} | val blurry-input PSNR floor={floor:.2f} dB", flush=True)

    (root / "meta.json").write_text(json.dumps(meta, indent=2))

    for s in SETTINGS:
        tr = np.load(root / s / "train.npz")
        va = np.load(root / s / "val.npz")
        if tr["blur"].shape[0] < N_TRAIN or va["blur"].shape[0] < N_VAL:
            print(f"Suspiciously little image-deblur data for setting {s}", file=sys.stderr)
            sys.exit(1)
    print(f"image-deblur data ready: settings={list(SETTINGS)}")


if __name__ == "__main__":
    main()

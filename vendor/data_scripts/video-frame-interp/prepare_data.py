"""Prepare data for the vfi-* VIDEO FRAME INTERPOLATION tasks (THREE motion-magnitude
SETTINGS) from REAL video triplets -- Vimeo-90K's temporal-frame-interpolation test set
(Xue, Chen, Wu, Wei & Freeman, "Video Enhancement with Task-Oriented Flow" / TOFlow,
IJCV 2019; project page http://toflow.csail.mit.edu/, code
https://github.com/anchen1011/toflow). This REPLACES the earlier fully-synthetic
two-layer-affine-warp generator: every frame below is a genuine decoded video frame,
not a procedurally rendered composite.

VIDEO FRAME INTERPOLATION (VFI): synthesize the MIDDLE frame that sits temporally
half-way between two given frames (frame0 at t=0 and frame2 at t=1); the target is the
REAL frame at t=0.5 that Vimeo-90K's clip actually contains (im2 of each triplet, the
middle of 3 consecutive decoded frames -- not something separately labelled or
interpolated: it is the genuine frame the camera captured between im1 and im3). This
is DISTINCT from every restoration direction (super-resolution / denoising basicsr-sr,
deblur image-deblur, dehaze image-dehaze, derain image-derain, inpaint
image-inpainting, colorize image-colorization, deshadow image-deshadow, HDR
hdr-reconstruction) -- nothing is degraded; the middle frame is simply withheld from
the model and must be SYNTHESIZED from the two neighbours. It is also DISTINCT from
optical-flow estimation (RAFT predicts a flow FIELD; VFI predicts an IMAGE): flow is a
means, the deliverable here is the interpolated frame itself. References: Super-SloMo
(Jiang et al., CVPR 2018), SepConv (Niklaus et al., ICCV 2017), RIFE (Huang et al.,
ECCV 2022).

DATA SOURCE: the Vimeo-90K "triplet dataset for temporal frame interpolation" test
split (3782 sequences, 448x256, extracted from real vimeo.com videos), downloaded
directly (no login/registration) from
    https://data.csail.mit.edu/tofu/testset/vimeo_interp_test.zip   (~2.99 GB)
License: released by the authors for non-commercial research use (see the dataset's
own readme.txt). Each sequence directory "<video>/<clip>" holds
    input/<video>/<clip>/im1.png   (frame0, t=0)
    input/<video>/<clip>/im3.png   (frame2, t=1)
    target/<video>/<clip>/im2.png  (the REAL middle frame, t=0.5 -- the target)
im1/im2/im3 are three CONSECUTIVE decoded frames of the original clip, so im2 is the
exact, camera-captured half-way frame -- there is no synthetic warping or compositing
anywhere in this pipeline.

SAMPLE VOLUME (tiling, not resizing): each 448x256 sequence is real but Vimeo-90K's
interpolation TEST split has only 3782 sequences total -- one 64x64 sample per sequence
(e.g. a center-crop-resize) yields far too few real samples per motion tercile to match
the original synthetic generator's per-setting volume (2000 train + 400 val). Instead,
we cut each real 448x256 frame into a grid of 4x7=28 NON-OVERLAPPING, UNRESIZED 64x64
tiles (genuine, untouched pixel sub-regions -- no synthesis, warping or resampling), so
one real triplet yields up to 28 real triplet-tiles. This multiplies the authentic
sample pool ~28x (up to 3782*28 ~= 105,896 tile-triplets) while every tile pixel remains
a byte-for-byte real decoded frame region.

THREE SETTINGS reconstructed from MEASURED real motion (not a synthetic control knob):
for every sequence we compute ONE dense optical flow field (Farneback, full 448x256
resolution) between im1 and im3, then for each of its 28 tiles take the mean flow
magnitude WITHIN that tile as the tile's motion score (so motion is measured locally,
not just once per whole frame -- a tile over a moving subject and a tile over a static
background of the SAME frame pair can land in different terciles). All tile-triplets
(pooled across all sequences) are sorted by this per-tile motion score and cut into
terciles -- named small / medium / large to match the ORIGINAL synthetic settings'
naming (same harness --motion flag, same task scripts/score_spec/leaderboard columns) --
giving a comparable difficulty ladder to the old synthetic version, but now derived from
genuinely real, locally-measured motion. Within each tercile, tiles are grouped by their
SOURCE VIDEO ID and shuffled (seed 42) into disjoint train/val subsets, so every tile
belonging to a given source clip lands in exactly one of train/val for that setting
(never split across train and val).

Produces, under {data_root}/video-frame-interp/<setting>/ for each of small/medium/large:
    train.npz  val.npz
Each npz holds (same schema the harness already expects)
    f0  (N,3,64,64)  frame at t=0     in [0,1]   (im1, real UNRESIZED 64x64 tile)
    f2  (N,3,64,64)  frame at t=1     in [0,1]   (im3, same tile region)
    gt  (N,3,64,64)  REAL middle frame at t=0.5  (im2, same tile region)     in [0,1]
64x64 keeps the exact tensor shapes / net architecture / compute budget the harness's
compact interpolation model and its 800-iteration training budget were built for; tiling
(rather than resizing) means every value is an untouched, real decoded pixel.

As before, the naive-blend PSNR floor (0.5*(f0+f2) vs gt) is reported at prepare time,
and should still fall with increasing measured motion, giving flow-warp / learned
synthesis strategies more (dis)occlusion headroom to earn on the harder settings.

Requires: numpy, pillow, opencv-python (cv2; only for the Farneback motion-magnitude
scan used to build the terciles -- not needed at harness train/eval time), and network
access on the HOST to fetch the ~2.99GB zip once (cached under a --cache-dir, default
{data-root}/../vfi-real-cache, so re-runs are offline). The task container itself stays
fully offline: it only ever reads the pre-built train.npz/val.npz produced here.
"""
from __future__ import annotations

import argparse
import io
import sys
import zipfile
from pathlib import Path
from urllib.request import urlopen, Request

import numpy as np

N_TRAIN = 2000
N_VAL = 400
IMG = 64
TILE_GRID_ROWS = 4   # 256 // 64
TILE_GRID_COLS = 7   # 448 // 64
SETTINGS = ("small", "medium", "large")

VIMEO_INTERP_TEST_URL = "https://data.csail.mit.edu/tofu/testset/vimeo_interp_test.zip"
VIMEO_INTERP_TEST_SIZE = 2992779759  # bytes, per the server Content-Length (sanity check)

# Tercile-split RNG seed (shuffling source videos within each bucket).
SPLIT_SEED = 42


# --------------------------------------------------------------------------- #
# Stage 0: fetch the Vimeo-90K interpolation test zip (once, cached).
# --------------------------------------------------------------------------- #
def _download(url: str, dst: Path, expected_size: int | None = None) -> None:
    if dst.exists() and (expected_size is None or dst.stat().st_size == expected_size):
        print(f"[download] cached: {dst} ({dst.stat().st_size} bytes)", flush=True)
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    print(f"[download] fetching {url} -> {dst} ...", flush=True)
    req = Request(url, headers={"User-Agent": "mls-bench-data-prep/1.0"})
    with urlopen(req, timeout=60) as resp, tmp.open("wb") as fh:
        total = resp.length or 0
        chunk = 1 << 20
        got = 0
        while True:
            buf = resp.read(chunk)
            if not buf:
                break
            fh.write(buf)
            got += len(buf)
            if total:
                print(f"\r[download] {got/1e6:.0f}/{total/1e6:.0f} MB", end="", flush=True)
    print(flush=True)
    tmp.rename(dst)
    if expected_size is not None and dst.stat().st_size != expected_size:
        print(f"[download] WARNING: size mismatch got={dst.stat().st_size} "
              f"expected={expected_size} (server file may have changed)", file=sys.stderr)


def _list_sequences(zf: zipfile.ZipFile) -> list[tuple[str, str]]:
    seqs = set()
    for name in zf.namelist():
        if name.startswith("vimeo_interp_test/target/") and name.endswith("im2.png"):
            parts = name.split("/")
            seqs.add((parts[2], parts[3]))
    return sorted(seqs)


def _read_rgb(zf: zipfile.ZipFile, path: str) -> np.ndarray:
    from PIL import Image
    with zf.open(path) as fh:
        img = Image.open(io.BytesIO(fh.read())).convert("RGB")
    return np.asarray(img)  # (H,W,3) uint8


def _tile_grid(h: int, w: int, size: int = IMG):
    """Yield (y0, x0) top-left corners of every non-overlapping size x size tile that
    fits inside an h x w frame (row-major)."""
    for y0 in range(0, h - size + 1, size):
        for x0 in range(0, w - size + 1, size):
            yield y0, x0


def _worker_scan(args):
    """Multiprocessing worker: decode one real triplet, compute ONE Farneback optical
    flow field between im1/im3 at full resolution, then cut all three frames (and the
    flow-magnitude map) into a grid of non-overlapping, UNRESIZED 64x64 tiles. Returns a
    list of per-tile records (video_id, clip_id, tile_idx, motion_magnitude, f0, f2, gt)
    -- f0/f2/gt are real uint8 (IMG,IMG,3) pixel tiles, never resized/blurred/warped."""
    import cv2

    zip_path, vid, clip = args
    zf = zipfile.ZipFile(zip_path)
    im1 = _read_rgb(zf, f"vimeo_interp_test/input/{vid}/{clip}/im1.png")
    im3 = _read_rgb(zf, f"vimeo_interp_test/input/{vid}/{clip}/im3.png")
    im2 = _read_rgb(zf, f"vimeo_interp_test/target/{vid}/{clip}/im2.png")

    g1 = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)
    g3 = cv2.cvtColor(im3, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(g1, g3, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)

    h, w, _ = im1.shape
    out = []
    for tile_idx, (y0, x0) in enumerate(_tile_grid(h, w, IMG)):
        f0_t = im1[y0:y0 + IMG, x0:x0 + IMG].copy()
        f2_t = im3[y0:y0 + IMG, x0:x0 + IMG].copy()
        gt_t = im2[y0:y0 + IMG, x0:x0 + IMG].copy()
        m = float(mag[y0:y0 + IMG, x0:x0 + IMG].mean())
        out.append((vid, clip, tile_idx, m, f0_t, f2_t, gt_t))
    return out


def _scan_all(zip_path: Path, workers: int):
    zf = zipfile.ZipFile(zip_path)
    seqs = _list_sequences(zf)
    print(f"[scan] {len(seqs)} Vimeo-90K interpolation triplets found in zip "
          f"(each yields up to {TILE_GRID_ROWS * TILE_GRID_COLS} real 64x64 tiles)",
          flush=True)
    jobs = [(str(zip_path), vid, clip) for vid, clip in seqs]
    import multiprocessing as mp
    results = []
    with mp.Pool(max(1, workers)) as pool:
        for i, tile_list in enumerate(pool.imap(_worker_scan, jobs, chunksize=4)):
            results.extend(tile_list)
            if (i + 1) % 500 == 0 or (i + 1) == len(jobs):
                print(f"[scan] {i + 1}/{len(jobs)} sequences processed "
                      f"({len(results)} tile-triplets so far)", flush=True)
    return results


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a - b) ** 2, axis=(1, 2, 3)).clip(1e-10)
    return float(np.mean(10.0 * np.log10(1.0 / mse)))


def _bucket_and_split(results, settings=SETTINGS):
    """Sort all per-tile (vid,clip,tile_idx,mag,...) records by MEASURED local motion
    magnitude, cut into terciles, and inside each tercile build a disjoint train/val
    split (seed 42), keeping every SOURCE VIDEO ID confined to a single split within
    that tercile -- so held-out val tiles never share a source clip with train."""
    results = sorted(results, key=lambda r: r[3])
    n = len(results)
    edges = [0, n // 3, 2 * n // 3, n]
    buckets = {}
    for i, name in enumerate(settings):
        buckets[name] = results[edges[i]:edges[i + 1]]

    out = {}
    rng = np.random.RandomState(SPLIT_SEED)
    for name in settings:
        rows = buckets[name]
        # Group by source video id so train/val never share a source clip.
        by_vid: dict[str, list] = {}
        for r in rows:
            by_vid.setdefault(r[0], []).append(r)
        vids = list(by_vid.keys())
        rng.shuffle(vids)
        train_rows, val_rows = [], []
        for v in vids:
            # Send each video's tiles to whichever split still needs more, so both
            # splits draw from many distinct source videos.
            target = train_rows if len(train_rows) <= len(val_rows) * (N_TRAIN / N_VAL) else val_rows
            target.extend(by_vid[v])
        rng.shuffle(train_rows)
        rng.shuffle(val_rows)
        out[name] = (train_rows[:N_TRAIN], val_rows[:N_VAL])
        mags = [r[3] for r in rows]
        print(f"[bucket] {name}: motion range "
              f"[{rows[0][3]:.3f}, {rows[-1][3]:.3f}] mean={np.mean(mags):.3f} "
              f"-> train={len(out[name][0])} val={len(out[name][1])} "
              f"(pool train={len(train_rows)} val={len(val_rows)}, "
              f"{len(vids)} distinct source videos, {len(rows)} tile-triplets total)",
              flush=True)
    return out


def _stack(rows) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    f0 = np.stack([r[4].transpose(2, 0, 1) for r in rows], 0).astype(np.float32) / 255.0
    f2 = np.stack([r[5].transpose(2, 0, 1) for r in rows], 0).astype(np.float32) / 255.0
    gt = np.stack([r[6].transpose(2, 0, 1) for r in rows], 0).astype(np.float32) / 255.0
    return f0, f2, gt


def _write_setting(root: Path, name: str, train_rows, val_rows) -> None:
    sub = root / name
    sub.mkdir(parents=True, exist_ok=True)
    if len(train_rows) < N_TRAIN or len(val_rows) < N_VAL:
        print(f"Not enough REAL tile-triplets for setting {name!r}: "
              f"train={len(train_rows)}/{N_TRAIN} val={len(val_rows)}/{N_VAL}",
              file=sys.stderr)
        sys.exit(1)
    f0_tr, f2_tr, gt_tr = _stack(train_rows[:N_TRAIN])
    f0_va, f2_va, gt_va = _stack(val_rows[:N_VAL])
    np.savez(sub / "train.npz", f0=f0_tr, f2=f2_tr, gt=gt_tr)
    np.savez(sub / "val.npz", f0=f0_va, f2=f2_va, gt=gt_va)
    blend = 0.5 * (f0_va + f2_va)
    print(f"[{name}] ready: train={f0_tr.shape[0]} val={f0_va.shape[0]} img={IMG}x{IMG} "
          f"| val blend PSNR floor={_psnr(blend, gt_va):.2f} dB "
          f"| mean measured motion train={np.mean([r[3] for r in train_rows[:N_TRAIN]]):.3f} "
          f"val={np.mean([r[3] for r in val_rows[:N_VAL]]):.3f}", flush=True)


def _already_built(root: Path, settings) -> bool:
    for name in settings:
        sub = root / name
        if not ((sub / "train.npz").exists() and (sub / "val.npz").exists()):
            return False
        tr = np.load(sub / "train.npz")
        va = np.load(sub / "val.npz")
        if tr["f0"].shape[0] < N_TRAIN or va["f0"].shape[0] < N_VAL:
            return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--motion", default="all", choices=("all",) + SETTINGS,
                     help="(kept for CLI back-compat; this script always builds/measures "
                          "all three terciles together since they share one motion scan, "
                          "but will skip work if all three are already cached)")
    ap.add_argument("--cache-dir", default=None, type=Path,
                     help="where to cache the downloaded Vimeo-90K zip "
                          "(default: <data-root>/../vfi-real-cache)")
    ap.add_argument("--url", default=VIMEO_INTERP_TEST_URL)
    ap.add_argument("--workers", type=int, default=32,
                     help="multiprocessing workers for the decode+flow scan")
    args = ap.parse_args()

    root = args.data_root.expanduser().resolve() / "video-frame-interp"
    root.mkdir(parents=True, exist_ok=True)

    if _already_built(root, SETTINGS):
        print("video-frame-interp REAL data already prepared for: "
              + ", ".join(SETTINGS), flush=True)
        return

    cache_dir = (args.cache_dir or (args.data_root.expanduser().resolve()
                                     / ".." / "vfi-real-cache")).resolve()
    zip_path = cache_dir / "vimeo_interp_test.zip"
    _download(args.url, zip_path, expected_size=VIMEO_INTERP_TEST_SIZE)

    results = _scan_all(zip_path, args.workers)
    buckets = _bucket_and_split(results, SETTINGS)
    for name in SETTINGS:
        train_rows, val_rows = buckets[name]
        _write_setting(root, name, train_rows, val_rows)

    print("video-frame-interp REAL data ready for: " + ", ".join(SETTINGS), flush=True)


if __name__ == "__main__":
    main()

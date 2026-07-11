"""Prepare REAL Kodak reference photos for the inr-* (implicit neural representation)
signal-fitting tasks.

The inr-signal-fitting harness benchmarks coordinate-MLP / Fourier-feature / SIREN
implicit representations by RECONSTRUCTION PSNR on three fixed target images spanning
LOW / MEDIUM / HIGH frequency content (task docs even cite literature PSNR numbers
measured on real "Kodak 256x256" images). The package used to substitute three
FULLY SYNTHETIC Gaussian-random-field images for those targets. This script instead
fetches three REAL photos from the classic 24-image Kodak Lossless True Color Image
Suite (same proven mirror already used by the pyiqa / compress-* packages) and bakes
them into fixed low/medium/high .npz targets, matching common.py's `load_signal`
expectations exactly:

    {data_root}/inr-signal-fitting/{low,medium,high}.npz
        coords: float64 [R*R, 2] in [-1, 1]   (identical grid as common._coord_grid)
        target: float64 [R*R, 3] in [0, 1]    (RGB, row-major = coords ravel order)

Image selection
----------------
We do NOT pick images by content description ("looks smooth" / "looks busy"); we
picked by measuring actual frequency content. All 24 Kodak reference images were
downloaded and center-cropped to CROP x CROP, then scored with two independent
frequency-content metrics on the luma channel:
  * Laplacian variance  (local high-frequency energy; higher = more detail)
  * Radial FFT high-frequency energy fraction (energy outside the inner half-radius
    of the 2D power spectrum; higher = more broadband/high-frequency content)
The two metrics do not induce the same complete ordering of all 24 images. Selection
therefore uses an explicit two-stage rule: choose the minimum and maximum Laplacian-
variance crops as endpoints, then choose the remaining crop nearest their geometric
midpoint jointly in log Laplacian variance and log FFT high-frequency fraction. This
produces three points that increase on both metrics (recomputed 2026-07-10):

    kodim10  lap_var= 227.6   hi_freq_frac=0.0010   <- LOW  (calm harbor/dock scene)
    kodim07  lap_var= 943.1   hi_freq_frac=0.0043   <- MEDIUM (flowers + lattice shutters)
    kodim13  lap_var=4740.0   hi_freq_frac=0.0245   <- HIGH (forest/rocks/mountain, dense fine texture)

The three IDs are pinned below. ``--recompute-ranking`` re-derives them with the stated
rule, and target digests fail closed if the mirror ever serves different pixels.

Requires network on the HOST (raw.githubusercontent.com); reuses the same mirror as
vendor/data_scripts/pyiqa/prepare_data.py:
    https://raw.githubusercontent.com/MohamedBakrAli/Kodak-Lossless-True-Color-Image-Suite
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import time
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np

BASE = "https://raw.githubusercontent.com/MohamedBakrAli/Kodak-Lossless-True-Color-Image-Suite/master/PhotoCD_PCD0992"
CROP = 256          # matches common._RES exactly (256x256 target resolution)
ALL_REF_IDS = list(range(1, 25))   # the full 24-image Kodak set, used only for ranking

# Pinned selection (see module docstring for the measured ranking that produced this).
SELECTED = {
    "low": 10,      # kodim10: calm dockside scene, lowest measured high-freq energy
    "medium": 7,    # kodim07: flowers + window lattice, mid-range detail
    "high": 13,     # kodim13: forest/rock/mountain texture, highest measured detail
}

EXPECTED_COORDS_SHA256 = "2ec4e6d2329db8d380428d6a411ff722a15715a05fbd067bf56540fedfd8996c"
EXPECTED_TARGET_SHA256 = {
    "low": "7a96bbe1ecfc6822cd92bfeec76690fbdee039905edbfeec89fee04726e2f59d",
    "medium": "e486c21f5e19429308bef5435f0234c6332aee0975d9e005d01d0f07580187be",
    "high": "a63555e868a232e07027295f99fd5f4aab00fda840219de9890c022eb03d2c79",
}


def _download_png(idx: int, retries: int = 4):
    """Fetch one Kodak PNG. Retries with backoff: the mirror's connection pool is
    flaky under parallel/high-latency fetches (observed empirically), but sequential
    single-shot retries reliably succeed."""
    from PIL import Image
    url = f"{BASE}/{idx:02d}.png"
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    last_exc = None
    for attempt in range(retries):
        try:
            with urlopen(req, timeout=90) as r:
                data = r.read()
            img = Image.open(io.BytesIO(data))
            img.load()
            return img.convert("RGB")
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"failed to download kodim{idx:02d} after {retries} attempts") from last_exc


def _center_crop(img, size: int):
    w, h = img.size
    left = (w - size) // 2
    top = (h - size) // 2
    return img.crop((left, top, left + size, top + size))


def _laplacian_var(gray: np.ndarray) -> float:
    """Local high-frequency energy via a discrete Laplacian (no scipy dependency)."""
    k = np.array([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
    h, w = gray.shape
    padded = np.pad(gray, 1, mode="reflect")
    out = np.zeros_like(gray)
    for dy in range(3):
        for dx in range(3):
            if k[dy, dx] == 0:
                continue
            out += k[dy, dx] * padded[dy:dy + h, dx:dx + w]
    return float(out.var())


def _radial_high_freq_frac(gray: np.ndarray) -> float:
    """Fraction of 2D FFT power outside the inner half-radius (broadband/HF energy)."""
    f = np.fft.fftshift(np.fft.fft2(gray))
    mag2 = np.abs(f) ** 2
    h, w = gray.shape
    cy, cx = h / 2.0, w / 2.0
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rmax = min(cy, cx)
    total = mag2.sum()
    if total <= 0:
        return 0.0
    return float(mag2[r > 0.5 * rmax].sum() / total)


def _to_gray(arr: np.ndarray) -> np.ndarray:
    return arr[..., 0] * 0.299 + arr[..., 1] * 0.587 + arr[..., 2] * 0.114


def _rank_candidates(cache_dir: Path) -> list[tuple[int, float, float]]:
    """Download all 24 Kodak refs, center-crop, and rank by frequency-content metrics.

    Only used for provenance / `--recompute-ranking`; the shipped pipeline uses the
    pinned SELECTED ids above and does not need to fetch all 24 images every run.
    """
    results = []
    for idx in ALL_REF_IDS:
        img = _center_crop(_download_png(idx), CROP)
        arr = np.asarray(img, dtype=np.float64)
        gray = _to_gray(arr)
        results.append((idx, _laplacian_var(gray), _radial_high_freq_frac(gray)))
    results.sort(key=lambda t: t[1])
    return results


def _derive_selection(
    ranking: list[tuple[int, float, float]],
) -> dict[str, int]:
    """Apply the documented endpoint-plus-log-midpoint selection rule."""
    if len(ranking) != len(ALL_REF_IDS):
        raise ValueError("selection requires measurements for all 24 Kodak references")
    ordered = sorted(ranking, key=lambda value: value[1])
    low = ordered[0]
    high = ordered[-1]
    lap_mid = math.sqrt(low[1] * high[1])
    fft_mid = math.sqrt(low[2] * high[2])

    def distance(value: tuple[int, float, float]) -> float:
        if value[1] <= 0.0 or value[2] <= 0.0:
            return math.inf
        return math.hypot(
            math.log(value[1] / lap_mid),
            math.log(value[2] / fft_mid),
        )

    medium = min(ordered[1:-1], key=distance)
    return {"low": low[0], "medium": medium[0], "high": high[0]}


def _coord_grid(res: int) -> np.ndarray:
    """Identical to common._coord_grid: [res*res, 2] normalized coords in [-1, 1]."""
    lin = np.linspace(-1.0, 1.0, res, dtype=np.float64)
    yy, xx = np.meshgrid(lin, lin, indexing="ij")
    return np.stack([xx.ravel(), yy.ravel()], axis=1)


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--recompute-ranking", action="store_true",
                     help="Fetch & rank all 24 Kodak images before staging "
                          "(slow; only for re-verifying the pinned selection).")
    args = ap.parse_args()

    out = Path(args.data_root) / "inr-signal-fitting"
    out.mkdir(parents=True, exist_ok=True)

    if args.recompute_ranking:
        ranking = _rank_candidates(out)
        print("Kodak 24 images sorted by Laplacian variance (FFT fraction also shown):")
        for idx, lv, hf in ranking:
            print(f"  kodim{idx:02d}  lap_var={lv:9.2f}  hi_freq_frac={hf:.4f}")
        derived = _derive_selection(ranking)
        if derived != SELECTED:
            raise RuntimeError(
                f"pinned selection {SELECTED} differs from recomputed selection {derived}"
            )
        print(f"Recomputed selection: {derived}")

    coords = _coord_grid(CROP)
    coords_sha256 = _array_sha256(coords)
    if coords_sha256 != EXPECTED_COORDS_SHA256:
        raise RuntimeError("generated coordinate grid does not match the anchored protocol")
    metrics = {}
    for name, idx in SELECTED.items():
        cropped = _center_crop(_download_png(idx), CROP)   # single fetch, reused below
        arr = np.asarray(cropped, dtype=np.float64)
        target = (arr / 255.0).reshape(-1, 3)               # row-major, matches coords order
        assert target.shape[0] == coords.shape[0]
        gray = _to_gray(arr)
        metrics[name] = {
            "kodim": idx,
            "laplacian_var": _laplacian_var(gray),
            "hi_freq_frac": _radial_high_freq_frac(gray),
            "target_sha256": _array_sha256(target),
        }
        if metrics[name]["target_sha256"] != EXPECTED_TARGET_SHA256[name]:
            raise RuntimeError(
                f"kodim{idx:02d} crop content does not match the anchored {name} target"
            )
        np.savez(out / f"{name}.npz", coords=coords, target=target)
        metrics[name]["npz_sha256"] = hashlib.sha256(
            (out / f"{name}.npz").read_bytes()
        ).hexdigest()
        print(f"Wrote {name}.npz <- kodim{idx:02d} "
              f"(lap_var={metrics[name]['laplacian_var']:.1f}, "
              f"hi_freq_frac={metrics[name]['hi_freq_frac']:.4f})", flush=True)

    # Sanity assertion: the pinned selection must reproduce the measured low<medium<high
    # ordering on BOTH metrics, or the mirror served different content than expected.
    order = ["low", "medium", "high"]
    lap = [metrics[n]["laplacian_var"] for n in order]
    hf = [metrics[n]["hi_freq_frac"] for n in order]
    assert lap[0] < lap[1] < lap[2], f"Laplacian-variance ordering broke: {lap}"
    assert hf[0] < hf[1] < hf[2], f"High-freq-fraction ordering broke: {hf}"

    manifest = {
        "source": "Kodak Lossless True Color Image Suite (PhotoCD PCD0992 mirror)",
        "base_url": BASE,
        "crop": CROP,
        "coords_sha256": coords_sha256,
        "selection_rule": (
            "Center-crop all 24 references and measure luma Laplacian variance plus "
            "radial-FFT high-frequency fraction. Use the minimum and maximum "
            "Laplacian-variance crops as endpoints; among the remaining crops, select "
            "the point nearest their geometric midpoint jointly in log Laplacian "
            "variance and log FFT fraction. Require the selected points to increase "
            "on both metrics."
        ),
        "selected": metrics,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"Wrote manifest.json -> {out / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()

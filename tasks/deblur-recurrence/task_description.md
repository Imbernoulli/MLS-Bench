# Image Deblurring: Within-scale recurrence depth (SRN)

## Objective

Design the **within-scale recurrence depth (srn)** of a compact motion-deblurring network so that it restores
sharp images from motion-blurred ones as accurately as possible, maximizing **deblur PSNR**
on a held-out set of blurry→sharp pairs under a fixed backbone, data, optimizer, and
evaluation protocol (only your chosen surface varies).

## Background

Single-image motion deblurring restores a sharp image from one degraded by camera
shake / object motion. The scale-recurrent refinement of SRN (Tao et al., CVPR 2018), applied without the multi-scale pyramid: the SAME weights are applied several times at full resolution, so each pass starts from a better-deblurred estimate and removes more of the (large) blur — at identical parameter count. A single pass under-deblurs a heavy streak; more passes progressively converge to a sharp image. This helps most in the heavy-blur band.

## Data & pipeline (FIXED)

- Tiny fixed set of blurry→sharp 64×64 pairs: **synthetic** natural-image-like sharp
  patches (2000 train / 400 val; smooth colour gradients + multi-octave texture + sharp
  edges, generated procedurally — no download) convolved with **known random motion-blur
  kernels** (random-walk trajectories; Boracchi & Foi 2012 / DeblurGAN). Train and val use
  **disjoint patches AND disjoint kernels**, so val PSNR is a clean generalisation measure.
- The score aggregates (geometric mean) over three HEAVY motion-blur severity settings (`ms`, `mm`, `ml` — long camera-shake streaks) where this lever is monotone.
- A compact residual encoder-decoder deblur net (global residual ON, Charbonnier+edge loss
  on the sharp target, fixed width/optimizer/iterations/seed). Only the **recurrence** surface
  varies; everything else is FIXED, so any change in the score is attributable to it.

## Implementation Contract

Edit **only** `get_recurrence_config()` in `image-deblur/solution/recurrence.py`:

```python
def get_recurrence_config():
    return {"n_recurrence": 3}
```

The default returns the **weak** choice (`{"n_recurrence": 1}`). A malformed / crashing return falls
back to the strong reference.

## Metric

The harness prints one line per run:

```
DEBLUR_METRICS surface=recurrence setting=<L> psnr=<..> psnr_gain=<..> blurry_psnr=<..> ssim=<..> mse=<..>
```

- **psnr** (PRIMARY, higher better): deblur PSNR (dB) of the **restored output vs the sharp
  GT** on the held-out split.
- **blurry_psnr**: PSNR of the **blurry input** vs the sharp GT — the identity ("do-nothing")
  floor. A net that copies its input scores `psnr == blurry_psnr`.
- **psnr_gain** `= psnr - blurry_psnr` (must be > 0 to beat the do-nothing floor).
- `ssim` / `mse` are diagnostics only.

## Validated baselines (real GPU, k1 H20, 1500 iters, seed 42)

The strong>weak partial-order is preserved in every setting, and both methods beat the blurry-input identity floor.

| setting | blurry floor | weak | strong |
|---------|-------------:|-----:|-------:|
| ms | 19.00 | 23.54 | **24.28** |
| mm | 18.40 | 21.04 | **22.66** |
| ml | 17.90 | 20.07 | **21.26** |


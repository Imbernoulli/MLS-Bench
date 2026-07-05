# Image Deblurring: Dilation / receptive field

## Objective

Design the **dilation / receptive field** of a compact motion-deblurring network so that it restores
sharp images from motion-blurred ones as accurately as possible, maximizing **deblur PSNR**
on a held-out set of blurry→sharp pairs under a fixed backbone, data, optimizer, and
evaluation protocol (only your chosen surface varies).

## Background

Single-image motion deblurring restores a sharp image from one degraded by camera
shake / object motion. A motion-blur kernel spreads each sharp pixel across a streak of neighbours; to invert it the network must see the whole streak at once. If the receptive field is smaller than the blur, the net cannot gather the information to deblur. Dilated convolutions (Yu & Koltun, 2016) widen the receptive field without extra parameters, so the net covers a larger streak. This helps most in the heavy-blur band.

## Data & pipeline (FIXED)

- Tiny fixed set of blurry→sharp 64×64 pairs: **synthetic** natural-image-like sharp
  patches (2000 train / 400 val; smooth colour gradients + multi-octave texture + sharp
  edges, generated procedurally — no download) convolved with **known random motion-blur
  kernels** (random-walk trajectories; Boracchi & Foi 2012 / DeblurGAN). Train and val use
  **disjoint patches AND disjoint kernels**, so val PSNR is a clean generalisation measure.
- The score aggregates (geometric mean) over three HEAVY motion-blur severity settings (`ms`, `mm`, `ml` — long camera-shake streaks) where this lever is monotone.
- A compact residual encoder-decoder deblur net (global residual ON, Charbonnier+edge loss
  on the sharp target, fixed width/optimizer/iterations/seed). Only the **dilation** surface
  varies; everything else is FIXED, so any change in the score is attributable to it.

## Implementation Contract

Edit **only** `get_arch_config()` in `image-deblur/solution/arch_dilation.py`:

```python
def get_arch_config():
    return {"dilation": 2}
```

The default returns the **weak** choice (`{"dilation": 1}`). A malformed / crashing return falls
back to the strong reference.

## Metric

The harness prints one line per run:

```
DEBLUR_METRICS surface=dilation setting=<L> psnr=<..> psnr_gain=<..> blurry_psnr=<..> ssim=<..> mse=<..>
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
| ms | 19.00 | 23.02 | **23.98** |
| mm | 18.40 | 22.79 | **24.12** |
| ml | 17.90 | 19.40 | **20.50** |


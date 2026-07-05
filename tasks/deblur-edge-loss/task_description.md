# Image Deblurring: Edge / gradient loss weight

## Objective

Design the **edge / gradient loss weight** of a compact motion-deblurring network so that it restores
sharp images from motion-blurred ones as accurately as possible, maximizing **deblur PSNR**
on a held-out set of blurry→sharp pairs under a fixed backbone, data, optimizer, and
evaluation protocol (only your chosen surface varies).

## Background

Single-image motion deblurring restores a sharp image from one degraded by camera
shake / object motion. Motion blur destroys high-frequency detail (edges); a plain reconstruction loss (L1 / Charbonnier) is dominated by the low-frequency bulk and under-penalises residual edge blur, so the output stays slightly soft. An edge / gradient loss that matches the image GRADIENTS of the restored and sharp images (LapSRN, Lai et al., CVPR 2017; MPRNet edge loss) explicitly rewards restoring sharp edges.

## Data & pipeline (FIXED)

- Tiny fixed set of blurry→sharp 64×64 pairs: **synthetic** natural-image-like sharp
  patches (2000 train / 400 val; smooth colour gradients + multi-octave texture + sharp
  edges, generated procedurally — no download) convolved with **known random motion-blur
  kernels** (random-walk trajectories; Boracchi & Foi 2012 / DeblurGAN). Train and val use
  **disjoint patches AND disjoint kernels**, so val PSNR is a clean generalisation measure.
- The score aggregates (geometric mean) over three motion-blur severity settings (`small`, `medium`, `large` — short → long camera-shake streaks).
- A compact residual encoder-decoder deblur net (global residual ON, Charbonnier+edge loss
  on the sharp target, fixed width/optimizer/iterations/seed). Only the **edge** surface
  varies; everything else is FIXED, so any change in the score is attributable to it.

## Implementation Contract

Edit **only** `get_loss_config()` in `image-deblur/solution/edge.py`:

```python
def get_loss_config():
    return {"kind": "charbonnier", "edge_weight": 0.1, "target_smooth": 0.0}
```

The default returns the **weak** choice (`{"kind": "charbonnier", "edge_weight": 0.0, "target_smooth": 0.0}`). A malformed / crashing return falls
back to the strong reference.

## Metric

The harness prints one line per run:

```
DEBLUR_METRICS surface=edge setting=<L> psnr=<..> psnr_gain=<..> blurry_psnr=<..> ssim=<..> mse=<..>
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
| small | 22.50 | 24.84 | **26.59** |
| medium | 19.80 | 24.11 | **25.14** |
| large | 18.30 | 20.08 | **20.93** |


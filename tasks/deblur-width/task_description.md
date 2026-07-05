# Image Deblurring: Channel Width (Backbone Capacity)

## Objective

Design the **channel width (backbone capacity)** of a compact motion-deblurring network so that it restores
sharp images from motion-blurred ones as accurately as possible, maximizing **deblur PSNR** on
a held-out set of blurry→sharp pairs under a fixed backbone, data, optimizer, and evaluation
protocol (only your chosen surface varies).

## Background

Single-image motion deblurring restores a sharp image from one degraded by camera shake /
object motion. The base CHANNEL WIDTH sets the number of feature channels in the backbone (deeper stages scale x2/x4 from it). A too-narrow backbone has too few channels to represent the spatially-varying deblur correction and under-fits, leaving the output blurry. Wider backbones (the width axis of EDSR, Lim et al. CVPR 2017, and MPRNet, Zamir et al. CVPR 2021) have the capacity to restore sharp detail. Width is the complementary capacity axis to depth, and matters most in the heavy-blur band.

## Data & pipeline (FIXED)

- Tiny fixed set of blurry→sharp 64×64 pairs: **synthetic** natural-image-like sharp patches
  (2000 train / 400 val; generated procedurally — no download) convolved with **known random
  motion-blur kernels** (random-walk trajectories; Boracchi & Foi 2012 / DeblurGAN). Train and
  val use **disjoint patches AND disjoint kernels**.
- The score aggregates (geometric mean) over the three severity settings **ms/mm/ml (heavy)**.
- Compact residual encoder-decoder deblur net (global residual ON, fixed width/optimizer/
  iterations/seed); only the **width** surface varies.

## Implementation Contract

Edit **only** `get_arch_config()` in `image-deblur/solution/arch_width.py`:

```python
def get_arch_config():
    return {"width": 32}
```

The default returns the **weak** choice (`{"width": 12}`). A malformed / crashing return falls
back to the strong reference.

## Metric

```
DEBLUR_METRICS surface=width setting=<L> psnr=<..> psnr_gain=<..> blurry_psnr=<..> ssim=<..> mse=<..>
```

- **psnr** (PRIMARY, higher better): deblur PSNR (dB), restored output vs sharp GT.
- **blurry_psnr**: identity floor (blurry input vs sharp GT); **psnr_gain** = psnr - blurry_psnr.
- `ssim` / `mse` are diagnostics.

## Validated baselines (real GPU, k1 H20, 1500 iters, seed 42)

The strong>weak partial-order is preserved in every setting, and both methods beat the blurry-input identity floor.

| setting | blurry floor | weak | strong |
|---------|-------------:|-----:|-------:|
| ms | 19.07 | 20.00 | **20.98** |
| mm | 18.36 | 19.26 | **19.96** |
| ml | 17.82 | 18.34 | **19.28** |


# Image Deblurring: Residual-block depth (network capacity)

## Objective

Design the **residual-block depth (network capacity)** of a compact motion-deblurring network so that it restores
sharp images from motion-blurred ones as accurately as possible, maximizing **deblur PSNR**
on a held-out set of blurry→sharp pairs under a fixed backbone, data, optimizer, and
evaluation protocol (only your chosen surface varies).

## Background

Single-image motion deblurring restores a sharp image from one degraded by camera
shake / object motion. The number of ResBlocks in each encoder/decoder stage sets the network's CAPACITY. Deblurring a large motion streak means inverting a wide, spatially-structured degradation; a too-shallow net under-fits it and leaves the output blurry. Deep ResBlock stacks are central to modern deblur nets (DeepDeblur, Nah et al., CVPR 2017; MPRNet, Zamir et al., CVPR 2021). Depth helps most in the heavy-blur band, where capacity is the binding constraint.

## Data & pipeline (FIXED)

- Tiny fixed set of blurry→sharp 64×64 pairs: **synthetic** natural-image-like sharp
  patches (2000 train / 400 val; smooth colour gradients + multi-octave texture + sharp
  edges, generated procedurally — no download) convolved with **known random motion-blur
  kernels** (random-walk trajectories; Boracchi & Foi 2012 / DeblurGAN). Train and val use
  **disjoint patches AND disjoint kernels**, so val PSNR is a clean generalisation measure.
- The score aggregates (geometric mean) over three HEAVY motion-blur severity settings (`ms`, `mm`, `ml` — long camera-shake streaks) where this lever is monotone.
- A compact residual encoder-decoder deblur net (global residual ON, Charbonnier+edge loss
  on the sharp target, fixed width/optimizer/iterations/seed). Only the **depth** surface
  varies; everything else is FIXED, so any change in the score is attributable to it.

## Implementation Contract

Edit **only** `get_arch_config()` in `image-deblur/solution/arch_depth.py`:

```python
def get_arch_config():
    return {"n_resblocks": 3}
```

The default returns the **weak** choice (`{"n_resblocks": 1}`). A malformed / crashing return falls
back to the strong reference.

## Metric

The harness prints one line per run:

```
DEBLUR_METRICS surface=depth setting=<L> psnr=<..> psnr_gain=<..> blurry_psnr=<..> ssim=<..> mse=<..>
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
| ms | 19.00 | 23.46 | **24.92** |
| mm | 18.40 | 21.37 | **22.13** |
| ml | 17.90 | 21.19 | **22.16** |


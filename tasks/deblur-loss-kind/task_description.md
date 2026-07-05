# Image Deblurring: Reconstruction Loss (L2 vs Charbonnier+Edge)

## Objective

Design the **reconstruction loss (l2 vs charbonnier+edge)** of a compact motion-deblurring network so that it restores
sharp images from motion-blurred ones as accurately as possible, maximizing **deblur PSNR** on
a held-out set of blurry→sharp pairs under a fixed backbone, data, optimizer, and evaluation
protocol (only your chosen surface varies).

## Background

Single-image motion deblurring restores a sharp image from one degraded by camera shake /
object motion. L2 (MSE) penalises errors quadratically, which over-smooths the restored image (the conditional-mean blur) and under-restores high-frequency detail. A robust Charbonnier loss (Lai et al., LapSRN; used by MPRNet, Zamir et al. CVPR 2021) plus an edge / gradient term is far less prone to over-smoothing and restores sharper edges. This is the loss FUNCTION choice, complementary to the loss-design task (which varies the training TARGET).

## Data & pipeline (FIXED)

- Tiny fixed set of blurry→sharp 64×64 pairs: **synthetic** natural-image-like sharp patches
  (2000 train / 400 val; generated procedurally — no download) convolved with **known random
  motion-blur kernels** (random-walk trajectories; Boracchi & Foi 2012 / DeblurGAN). Train and
  val use **disjoint patches AND disjoint kernels**.
- The score aggregates (geometric mean) over the three severity settings **small/medium/large**.
- Compact residual encoder-decoder deblur net (global residual ON, fixed width/optimizer/
  iterations/seed); only the **loss** surface varies.

## Implementation Contract

Edit **only** `get_loss_config()` in `image-deblur/solution/losskind.py`:

```python
def get_loss_config():
    return {"kind": "charbonnier", "edge_weight": 0.1, "target_smooth": 0.0}
```

The default returns the **weak** choice (`{"kind": "l2", "edge_weight": 0.0, "target_smooth": 0.0}`). A malformed / crashing return falls
back to the strong reference.

## Metric

```
DEBLUR_METRICS surface=loss setting=<L> psnr=<..> psnr_gain=<..> blurry_psnr=<..> ssim=<..> mse=<..>
```

- **psnr** (PRIMARY, higher better): deblur PSNR (dB), restored output vs sharp GT.
- **blurry_psnr**: identity floor (blurry input vs sharp GT); **psnr_gain** = psnr - blurry_psnr.
- `ssim` / `mse` are diagnostics.

## Validated baselines (real GPU, k1 H20, 1500 iters, seed 42)

The strong>weak partial-order is preserved in every setting, and both methods beat the blurry-input identity floor.

| setting | blurry floor | weak | strong |
|---------|-------------:|-----:|-------:|
| em | 24.70 | 29.02 | **30.02** |
| el | 22.23 | 26.10 | **26.29** |
| medium | 19.84 | 22.27 | **22.35** |


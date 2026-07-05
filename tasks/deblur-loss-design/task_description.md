# Image Deblurring: Reconstruction-Target / Loss Design

## Objective

Design what a compact motion-deblur network is **optimised toward** — the reconstruction
loss and, crucially, the **target** it is matched to — so that it restores sharp images
from blurred ones as accurately as possible, maximizing **deblur PSNR** on a held-out set
of blurry→sharp pairs under a fixed backbone, residual configuration, optimizer, data, and
evaluation protocol.

## Background

A deblur network can only be as sharp as the target it is trained to match. A plain **L2
(MSE)** loss against the sharp GT is minimised by the per-pixel **conditional mean**, an
**over-smoothed** image — L2 penalises the loss of high-frequency detail only softly. The
failure is even more explicit if you optimise toward a **low-pass (blurred) version of the
ground truth**: the network then learns to *reproduce* the blur, and deblur PSNR collapses,
often **below** the do-nothing blurry-input floor.

Two well-established fixes push the other way:

- **Optimise toward the true sharp target** (no over-smoothing) — the single most important
  choice: the target must contain the high-frequency detail you want restored.
- **Robust Charbonnier loss** `sqrt((pred − gt)² + ε²)` (Lai et al., LapSRN; used by MPRNet,
  Zamir et al., CVPR 2021) plus an **edge / gradient term** that explicitly rewards
  recovering high-frequency detail.

## Data & pipeline (FIXED)

- Tiny fixed set of blurry→sharp 64×64 pairs: **synthetic** natural-image-like sharp
  patches (2000 train / 400 val; generated procedurally — no download) convolved with
  **known random motion-blur kernels** (random-walk trajectories; Boracchi & Foi / DeblurGAN).
  Train and val use **disjoint patches AND disjoint kernels**. Sharp patch = exact GT.
- The score aggregates (geometric mean) over **THREE motion-blur severity settings** —
  `small`, `medium`, `large` (blurry-input PSNR floors ≈ 22.8 / 19.8 / 18.3 dB) — so the
  loss design is evaluated across a range of blur strengths.
- A compact residual encoder-decoder deblur net (global residual ON, single-scale — all
  FIXED), trained 1500 steps, fixed seed. Only your reconstruction target / loss varies.

## Implementation Contract

Edit **only** `get_loss_config()` in `image-deblur/solution/loss.py`. Return a dict:

```python
def get_loss_config():
    # kind:          'l2' (MSE) or 'charbonnier' (robust L1-like sqrt(e^2+eps^2))
    # edge_weight:   weight of an extra image-GRADIENT (edge) term (0 disables)
    # target_smooth: sigma of a Gaussian LOW-PASS applied to the sharp GT BEFORE the loss.
    #                >0 optimises toward an OVER-SMOOTHED target (low deblur PSNR); 0 = the
    #                true sharp target (high deblur PSNR).
    return {"kind": "charbonnier", "edge_weight": 0.1, "target_smooth": 0.0}
```

The default returns `{"kind": "charbonnier", "edge_weight": 0.1, "target_smooth": 1.2}`
(optimise toward an **over-smoothed** target — the net reproduces blur). A malformed /
crashing return falls back to the sharp target (`target_smooth=0.0`).

## Metric

The harness prints one line:

```
DEBLUR_METRICS surface=loss setting=<L> psnr=<..> psnr_gain=<..> blurry_psnr=<..> \
    ssim=<..> mse=<..>
```

- **psnr** (PRIMARY, higher better): deblur PSNR (dB) of the **restored output vs the
  sharp GT** on the held-out split.
- **blurry_psnr**: PSNR of the **blurry input** vs the sharp GT — the identity floor.
- **psnr_gain** `= psnr − blurry_psnr`: must be > 0 to beat passing the input through.
- `ssim` / `mse` are diagnostics only.

Optimising toward an over-smoothed target reproduces the blur (`psnr` at/below the floor);
optimising toward the true sharp target restores detail and lifts the score with clear
headroom.

## Validated baselines (real GPU, k1 H20, torch 2.4.1, 1500 iters, seed 42)

Reproduced deblur PSNR (dB) of the two real baselines across all three settings, with the
blurry-input floor. The strong>weak partial-order is preserved in every setting; the
over-smoothed baseline sits at/below the identity floor, while the sharp-target reference
(robust Charbonnier + edge, the LapSRN / MPRNet loss family) is clearly best.

| setting | blurry floor | over-smoothed target (weak) | sharp target (strong) |
|---------|-------------:|----------------------------:|----------------------:|
| small   | 22.79        | 21.35                       | **27.21**             |
| medium  | 19.84        | 20.19                       | **22.36**             |
| large   | 18.32        | 18.51                       | **19.01**             |

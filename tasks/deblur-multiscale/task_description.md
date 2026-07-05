# Image Deblurring: Multi-Scale Coarse-to-Fine Design

## Objective

Design the **number of coarse-to-fine scales** in a shared-weight motion-deblur
network so that it restores sharp images from blurred ones as accurately as possible,
maximizing **deblur PSNR** on a held-out set of blurry→sharp pairs under a fixed
backbone, residual configuration, loss, optimizer, data, and evaluation protocol.

## Background

The **coarse-to-fine** (multi-scale) strategy is central to deep deblurring
(DeepDeblur, Nah et al., CVPR 2017; SRN-DeblurNet, Tao et al., CVPR 2018). Large
motion blur spans many pixels and is hard to remove in a single full-resolution pass.
Instead, the network deblurs a **downsampled (coarse)** version first — where the same
blur covers fewer pixels and is easier to invert — then **upsamples that estimate** and
uses it to guide deblurring at each **finer** scale. SRN shares network weights across
all scales (a scale-**recurrent** net), so a multi-scale pyramid adds essentially no
parameters while making the optimisation much easier.

With `scales = 1` the net deblurs only at full resolution in one pass. With
`scales = 3` the same shared-weight net runs a 3-level coarse-to-fine pyramid,
progressively refining the restoration — sharper results and higher deblur PSNR on
large blur.

## Data & pipeline (FIXED)

- Tiny fixed set of blurry→sharp 64×64 pairs: **synthetic** natural-image-like
  sharp patches (2000 train / 400 val; smooth colour gradients + multi-octave
  texture + sharp edges, generated procedurally — no download) convolved with
  **known random motion-blur kernels** (random-walk trajectories; Boracchi & Foi / DeblurGAN). Train and val use
  **disjoint patches AND disjoint kernels**. Sharp patch = exact ground truth.
- The score aggregates (geometric mean) over **THREE heavy motion-blur severity
  settings** — `ms`, `mm`, `ml` (long camera-shake streaks; blurry-input PSNR floors
  ≈ 19.1 / 18.4 / 17.8 dB). Coarse-to-fine helps most when the blur is **large** relative
  to the patch, so the multiscale task stays in this heavy band where `single < multi`
  holds in all three settings.
- A compact deblur net with **shared weights across scales** (a scale-recurrent net:
  each finer scale sees the pristine blurry image at its own resolution plus the upsampled
  coarser deblurred estimate; global residual ON, sharp-target Charbonnier+edge loss — all
  FIXED), trained 1500 steps, fixed seed. Only your number of scales varies.

## Implementation Contract

Edit **only** `get_scale_config()` in `image-deblur/solution/multiscale.py`. Return:

```python
def get_scale_config():
    return {"scales": 3}     # 1 (single-scale) .. 3 (coarse-to-fine pyramid)
```

The default returns `{"scales": 1}` (single-scale — harder on large blur). A malformed
/ crashing return falls back to `scales=3` (values are clamped to `1..4`).

## Metric

The harness prints one line:

```
DEBLUR_METRICS surface=multiscale setting=<L> psnr=<..> psnr_gain=<..> blurry_psnr=<..> \
    ssim=<..> mse=<..>
```

- **psnr** (PRIMARY, higher better): deblur PSNR (dB) of the **restored output vs the
  sharp GT** on the held-out split.
- **blurry_psnr**: PSNR of the **blurry input** vs the sharp GT — the identity floor.
- **psnr_gain** `= psnr − blurry_psnr`: must be > 0 to beat passing the input through.
- `ssim` / `mse` are diagnostics only.

A single-scale net leaves large blur under-removed (lower `psnr`); a 3-scale
coarse-to-fine pyramid restores sharper images with clear PSNR headroom, well above
the blurry-input identity floor.

## Validated baselines (real GPU, k1 H20, torch 2.4.1, 1500 iters, seed 42)

Reproduced deblur PSNR (dB) of the two real baselines across all three heavy-blur
settings, with the blurry-input floor. The strong>weak partial-order is preserved in
every setting, and both beat the identity floor — the scale-recurrent coarse-to-fine
design of SRN-DeblurNet (Tao et al., CVPR 2018; cf. DeepDeblur, Nah et al., CVPR 2017) is
the strong reference here.

| setting | blurry floor | single-scale (weak) | 3-scale coarse-to-fine (strong) |
|---------|-------------:|--------------------:|--------------------------------:|
| ms      | 19.07        | 20.40               | **21.12**                       |
| mm      | 18.36        | 19.01               | **19.47**                       |
| ml      | 17.82        | 18.40               | **18.86**                       |

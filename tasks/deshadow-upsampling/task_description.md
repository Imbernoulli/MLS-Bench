# Shadow Removal: Decoder Upsampling (Transpose-Conv vs Bilinear-Resize + Conv)

## Objective

Design the **decoder upsampler** of the mask-guided residual deshadower — transpose
convolution (deconv), or bilinear-resize followed by a conv — so that it removes a cast shadow
to match the clean, shadow-free ground truth as accurately as possible, maximizing
**shadow-region PSNR** on a held-out set of shadowed→clean pairs under a fixed formulation,
loss, optimizer, data, and evaluation protocol.

## Background

Shadow removal recovers a shadow-free image from one on which a **cast shadow**
multiplicatively darkens a known, soft-edged region (SP+M-Net linear illumination model,
Le et al., ICCV 2019). The recovered image must reproduce the **smooth** soft penumbra falloff;
transpose-convolution upsampling is known to introduce periodic **checkerboard** artifacts that
are especially visible across large smooth regions, whereas resize-then-conv upsampling is
artifact-free.

- **weak (`up='transpose'`)** — transpose-convolution (deconv) upsampling, prone to checkerboard
  artifacts across the smooth soft shadow.
- **strong (`up='bilinear'`)** — BILINEAR-resize + conv upsampling: smooth, artifact-free, respects
  the soft penumbra falloff.

## Honesty note

This is the **weakest** of the shipped deshadow surfaces. The resize-conv upsampler helps
clearly on the harder **medium / heavy** settings (where transpose-conv checkerboard shows across
the larger smooth penumbra), but on the easy **light** setting the two are a near-tie. The
TASK-LEVEL geometric mean of shadow-region PSNR is monotone strong > weak on **both** validated
seeds, so the surface is a valid weak→strong lever at the score level; the medium/heavy gains
dominate the aggregate.

The pinned anchors below are still from the OLD SYNTHETIC cast-shadow data (pending a full GPU
re-anchor on the new REAL ISTD data -- see the data-source note further down and
`vendor/image-deshadow/anchors/README.md`). A CPU smoke-test re-check on the real ISTD triplets
(2 seeds, `vendor/image-deshadow/anchors/real_istd_cpu_smoke.log`) confirms the ordering
survives the data swap: the per-setting near-tie moved to **light** instead of being uniformly
positive (light is now the ONE setting that flips weak>strong on both seeds), but medium/heavy
gains are larger on real data and the task-level gmean stays monotone strong>weak on both seeds
(seed42 24.338→24.431 dB, delta +0.09; seed123 24.577→25.042 dB, delta +0.46) -- so this remains
the weakest shipped lever, now weaker still, and worth re-anchoring carefully once real GPU
numbers are available.

## Data & pipeline (FIXED)

- Tiny fixed set of REAL shadowed→shadow-free 64×64 photo triplets (+ a real shadow mask) from
  ISTD (Wang et al., CVPR 2018): the same outdoor scene photographed with and without a
  physical cast shadow. THREE severities (**light / medium / heavy**) are terciles of each
  triplet's MEASURED shadow attenuation (mean brightness ratio inside the mask). Train and val
  use ISTD's own **disjoint-scene train/test split**. The shadow-free photo is the ground
  truth, and the shadow **mask is provided as an input**.
- A FIXED mask-guided residual-learning deshadower, a fixed composite L1+SSIM loss up-weighted
  inside the shadow, a fixed optimizer, trained a few hundred steps, fixed seed. Only the
  decoder upsampler varies.

## Implementation Contract

Edit **only** `get_upsampling_config()` in `image-deshadow/solution/upsampling.py`. Return:

```python
def get_upsampling_config():
    # {'up': 'transpose' | 'bilinear'}
    return {"up": "bilinear"}
```

A malformed / crashing return falls back to `up='transpose'` (weak).

## Metric

```
DESHADOW_METRICS surface=upsampling setting=<L> psnr=<..> psnr_gain=<..> shadow_psnr=<..> \
    ssim=<..> mse=<..> full_psnr=<..>
```

- **psnr** (PRIMARY, higher better): SHADOW-REGION deshadow PSNR (dB), only over pixels the
  shadow touches. **shadow_psnr** is the copy floor; **psnr_gain** `= psnr − shadow_psnr` must
  be > 0. `full_psnr`, `ssim`, `mse` are diagnostics.

The resize-conv upsampler gives higher aggregate shadow-region PSNR than transpose-conv (most on
the larger medium/heavy shadows), and both clearly beat the shadowed-input identity floor.

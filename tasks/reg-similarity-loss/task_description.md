# Deformable Registration: Similarity Loss

## Research Question
In learned **deformable image registration** (a U-Net predicts a dense
displacement field that warps the MOVING image onto the FIXED image), the field
is trained *unsupervised* to maximise the **image similarity** between the
warped-moving and the fixed image. The choice of similarity term is central:

- **MSE** — mean-squared-error between warped-moving and fixed. Simple, but it
  assumes the two images differ *only* by geometry and is sensitive to any local
  intensity/contrast variation; its gradient weakens for large deformations.
- **NCC** — local normalized cross-correlation (the *VoxelMorph* default,
  Balakrishnan et al. CVPR 2018 / TMI 2019). Invariant to local
  intensity/contrast shifts and gives a robust gradient, so it holds up as the
  deformation grows.

With the registration method FIXED to the learned dense deformable field, **which
similarity term gives the best post-registration warped-moving vs fixed PSNR
across increasingly-large deformations?**

## Background
The intuition from the registration literature is that local NCC should be more
robust than MSE as the deformation grows (VoxelMorph's own default). Measured on
this task's REAL IXI2D brain-MRI data (k1 H20, 800 steps, both seed 42 and seed
123), however, the ordering is the OPPOSITE: **MSE consistently beats NCC** at
every deformation magnitude (small/medium/large), by a growing margin as the
warp increases (e.g. seed-avg PSNR: small 27.47 vs 26.67 dB, large 18.24 vs 15.69
dB). On this single, globally-consistent MRI intensity domain (one fixed image,
no cross-scanner/cross-contrast intensity variation), MSE's simpler, denser
gradient converges faster within the fixed 800-step budget than local NCC's
patch-normalized objective, which has more numerically-diffuse gradients at this
scale. This is an honest real-data finding, not a synthetic-data artifact (see
`vendor/deformable-registration/anchors/` for full provenance): the task is
scored with **MSE as the strong reference and NCC as the weaker baseline** — the
reverse of the literature's usual expectation, which is exactly why real-data
re-anchoring matters.

## Implementation Contract
Modify `build_similarity` in `deformable-registration/solution/similarity.py`:

```python
def build_similarity():
    # return "mse" or "ncc"
    return ...
```

A malformed / crashing return degrades to the harness default (`"ncc"`, the
weaker baseline on this real data -- see Background).

## Fixed Pipeline & Evaluation
- Data: a REAL T1-weighted brain-MRI FIXED image (IXI dataset, https://brain-development.org/ixi-dataset/, ~600 healthy subjects, Imperial College London, CC BY-SA 3.0; sourced via the no-login IXI2D 2-D-slice HTTP release https://huggingface.co/datasets/iamkzntsv/IXI2D) warped by an exactly-known smooth
  non-rigid deformation to make the MOVING image (exact GT deformation / PSNR /
  landmark TRE). Fixed seed-42 split (48 train / 24 val, 128x128).
- Settings (score aggregates over all three): `small`, `medium`, `large` — the
  same dataset with increasing deformation magnitude (peak displacement ~3 / ~7 /
  ~13 px).
- Method: FIXED learned dense deformable field (VoxelMorph U-Net), smoothness
  weight 0.05, AdamW + OneCycle, 800 steps.
- Metric (per setting, HIGHER better): `psnr_<setting>` — warped-moving vs fixed
  PSNR (dB). `tre_/folding_/ncc_<setting>` reported for diagnostics.
- Scoring is anchored on REAL measured IXI2D numbers (k1 H20, seeds 42+123) so
  MSE scores best at every magnitude while NCC trails, growing at medium/large.

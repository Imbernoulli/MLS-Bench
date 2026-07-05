# Deformable Registration: Regularizer Type

## Research Question
**Deformable (non-rigid) image registration** aligns a MOVING image to a FIXED
image by predicting a dense per-pixel displacement field `phi`, then warping the
moving image by `phi` so the warped-moving matches the fixed. A displacement-field regulariser keeps the deformation smooth and near-diffeomorphic. **Which regulariser TYPE best trades alignment against a valid (fold-free) field across increasingly-large deformations?** The gradient (L2 / diffusion) penalty and the bending-energy (thin-plate-spline / B-spline, Rueckert et al. 1999) penalty differ in what they penalise (field gradient vs curvature).

## Background
With no regulariser the field folds (non-diffeomorphic, TRE degrades, PSNR drops at large warps); a proper smoothness penalty (l2 or bending) keeps the field valid while aligning. The literature ordering is **none < {l2, bending}** once folding is scored.

## Implementation Contract
Modify `build_reg_type` in `deformable-registration/solution/regularization.py`:

```python
def build_reg_type():
    # return one of "none" | "l2" | "bending"
    # choices: `"none"` (no penalty) | `"l2"` (first-order diffusion, VoxelMorph default) | `"bending"` (second-order bending-energy / TPS)
    return ...
```

A malformed / crashing return degrades to the harness default (`"l2"`).

## Fixed Pipeline & Evaluation
- Data: a REAL T1-weighted brain-MRI FIXED image (IXI dataset, https://brain-development.org/ixi-dataset/, ~600 healthy subjects, Imperial College London, CC BY-SA 3.0; sourced via the no-login IXI2D 2-D-slice HTTP release https://huggingface.co/datasets/iamkzntsv/IXI2D) warped by an exactly-known smooth
  non-rigid deformation (a global affine + a low-frequency displacement) to make
  the MOVING image, so the per-pixel ground-truth deformation, the warped-moving
  vs fixed PSNR and per-landmark target-registration-error (TRE) are all EXACT.
  Fixed seed-42 split (48 train / 24 val pairs, 128x128).
- Settings (the score aggregates over all three): `small`, `medium`, `large` —
  the same dataset with an **increasing deformation magnitude** (peak
  displacement ~3 / ~7 / ~13 px).
- Metric: per setting the score COMBINES `psnr_<setting>` (warped-moving vs fixed PSNR in dB, HIGHER better, weight 0.7) and `folding_<setting>` (fraction of non-positive-Jacobian pixels, LOWER better, weight 0.3) — so a good field must both ALIGN the images AND stay near-diffeomorphic. `tre_<setting>` and `ncc_<setting>` are reported for diagnostics.
- Scoring combines post-registration PSNR with a FOLDING (diffeomorphism) penalty, anchored so the partial order is preserved across all three settings: **none ~weak < l2 / bending ~strong**.
- Everything else (data, U-Net, similarity, optimiser, schedule, seed, eval) is
  FIXED; only the regularizer type surface is under your control.

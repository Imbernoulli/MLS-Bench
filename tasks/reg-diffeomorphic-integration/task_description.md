# Deformable Registration: Diffeomorphic Integration (Scaling-and-Squaring)

## Research Question
**Deformable (non-rigid) image registration** aligns a MOVING image to a FIXED
image by predicting a dense per-pixel displacement field `phi`, then warping the
moving image by `phi` so the warped-moving matches the fixed. The plain VoxelMorph field can FOLD (non-positive Jacobian -> non-diffeomorphic, physically invalid). Interpreting the network output as a stationary VELOCITY field and integrating it by SCALING-AND-SQUARING yields a guaranteed-invertible (diffeomorphic) transform (Ashburner 2007; VoxelMorph-diff, Dalca et al. 2018). **How many integration steps best trade alignment against a fold-free field across increasingly-large deformations?**

## Background
Without integration the field develops folds that hurt landmark accuracy and validity, worst at large deformations; a handful of integration steps removes the folding while preserving alignment. The literature ordering is **no-integration < light < full integration** once folding is scored.

## Implementation Contract
Modify `build_integration_steps` in `deformable-registration/solution/integration.py`:

```python
def build_integration_steps():
    # return an int number of scaling-and-squaring steps in [0, 10]
    # choices: `0` (plain displacement, can fold) | `5`-`7` (integrate a velocity field into a fold-free diffeomorphism)
    return ...
```

A malformed / crashing return degrades to the harness default (`7`).

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
- Scoring combines post-registration PSNR with a FOLDING (diffeomorphism) penalty, anchored so the partial order is preserved across all three settings: **none ~weak < light ~mid < full ~strong**.
- Everything else (data, U-Net, similarity, optimiser, schedule, seed, eval) is
  FIXED; only the diffeomorphic integration (scaling-and-squaring) surface is under your control.

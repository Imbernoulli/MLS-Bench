# Deformable Registration: Inverse / Cycle Consistency

## Research Question
**Deformable (non-rigid) image registration** aligns a MOVING image to a FIXED
image by predicting a dense per-pixel displacement field `phi`, then warping the
moving image by `phi` so the warped-moving matches the fixed. A registration is INVERTIBLE when composing the forward field (moving->fixed) with the reverse field (fixed->moving) returns to the identity. Adding an inverse/cycle-consistency loss on that residual encourages a SYMMETRIC, invertible transform (inverse-consistency / SYMNet, Zhang 2018; Mok & Chung 2020), but this is a REGULARISER: it trades alignment fidelity for symmetry, and its weight must be chosen carefully. **What inverse-consistency weight best trades alignment against invertibility across increasingly-large deformations?**

## Background
Inverse-consistency behaves like a regularisation-strength knob. A LIGHT (or zero) weight leaves the field free to align the images at maximum PSNR; increasing the weight pulls the forward and reverse fields toward mutual inverses, which drives the folding to zero but progressively SACRIFICES alignment, and an over-strong weight collapses the deformation and destroys the PSNR. On this deterministic dataset (whose ground-truth deformation is already smooth and near-invertible) the plain one-directional field is already fold-controlled, so the ordering by post-registration quality is **over-constrained < strong < light/off**: too much cycle-consistency is the clear failure mode the agent must avoid.

## Implementation Contract
Modify `build_inverse_weight` in `deformable-registration/solution/inverse.py`:

```python
def build_inverse_weight():
    # return a float inverse-consistency weight in [0, 100]
    # choices: `0.0` (plain one-directional, best alignment) | `~1.0` (strong symmetry, some alignment loss) | `50.0` (over-constrained, collapses)
    return ...
```

A malformed / crashing return degrades to the harness default (`0.0`).

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
- Scoring combines post-registration PSNR with a FOLDING (diffeomorphism) penalty, anchored (k1 H20, real IXI2D data, cross-checked on seeds 42 and 123) so the partial order **over ~weak < on ~mid < off ~strong** holds at task level on both seeds: over ~0.38 < on ~0.47 < off = 0.50 (seed 42); over ~0.36 < on ~0.42 < off ~0.46 (seed 123).
- Everything else (data, U-Net, similarity, optimiser, schedule, seed, eval) is
  FIXED; only the inverse / cycle consistency surface is under your control.

# Deformable Registration: Smoothness Regularization Strength

## Research Question
A learned **deformable image registration** U-Net predicts a dense displacement
field that warps the MOVING image onto the FIXED image. Trained to *only*
maximise image similarity, the field can become jagged and **fold** (locally
non-invertible / non-diffeomorphic) — a classic registration failure. The remedy
is a **smoothness regulariser** on the displacement field:

```
loss = similarity(warped, fixed) + lambda * grad_smoothness(field)
```

The weight `lambda` trades off alignment against field regularity:

- `lambda = 0` (**none**) — no penalty: the field overfits the similarity but
  **folds** (large fraction of non-positive-Jacobian pixels), and at large
  deformations the PSNR actually drops.
- `lambda` too large (**high**) — the penalty dominates and collapses the field
  toward identity (perfectly smooth but useless), so alignment and PSNR collapse.
- a **moderate** `lambda` — both aligns the images AND keeps the field
  near-diffeomorphic (near-zero folding). This is the VoxelMorph regime.

**Which `lambda` maximises warped-moving vs fixed PSNR while keeping the field
smooth (low folding) across increasingly-large deformations?**

## Background
This is an inverted-U: too little regularisation folds, too much erases the
deformation. The ordering is **high < none < moderate** once field validity
(folding) is accounted for alongside alignment. The task's score combines
per-setting PSNR (weight 0.7) and a folding penalty (weight 0.3), so a good field
must both align AND stay near-diffeomorphic.

## Implementation Contract
Modify `build_smoothness_weight` in
`deformable-registration/solution/smoothness.py`:

```python
def build_smoothness_weight():
    # return a float lambda in [0, 100]
    return ...
```

A malformed / crashing return degrades to the harness default (`0.05`).

## Fixed Pipeline & Evaluation
- Data: a REAL T1-weighted brain-MRI FIXED image (IXI dataset, https://brain-development.org/ixi-dataset/, ~600 healthy subjects, Imperial College London, CC BY-SA 3.0; sourced via the no-login IXI2D 2-D-slice HTTP release https://huggingface.co/datasets/iamkzntsv/IXI2D), warped by an exactly-known smooth non-rigid deformation to make the MOVING image (exact GT deformation / PSNR / folding / landmark TRE — the standard "synthetic deformation of a real image" registration-benchmark protocol, e.g. Learn2Reg). Fixed seed-42 split (48 train / 24 val, 128x128) drawn from a curated 500-slice real-MRI pool.
- Settings (score aggregates over all three): `small`, `medium`, `large` — the
  same dataset with increasing deformation magnitude (peak displacement ~3 / ~7 /
  ~13 px).
- Method: FIXED learned dense deformable field (VoxelMorph U-Net) with local NCC
  similarity, AdamW + OneCycle, 800 steps.
- Metrics (per setting): `psnr_<setting>` (dB, HIGHER better, weight 0.7) and
  `folding_<setting>` (fraction of non-positive-Jacobian pixels, LOWER better,
  weight 0.3). `tre_/ncc_<setting>` reported for diagnostics.
- Scoring is anchored (k1 H20, real IXI2D data, cross-checked on seeds 42 and
  123) so a moderate weight wins at task level on both seeds: high ~0.39 <
  none ~0.46 < moderate ~0.50 (seed 42); high ~0.44 < none ~0.35 <
  moderate ~0.56 (seed 123, where no-regulariser folding is worse and PSNR
  collapses at `large`).

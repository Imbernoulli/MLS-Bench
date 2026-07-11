# Image Matting: Alpha Refinement

## Research Question

Design a differentiable refinement function applied to the fixed coarse alpha
prediction. Valid refinements are ranked only by terminal evaluation.

## Implementation Contract

Modify `refine(coarse_alpha, image, trimap)` in
`image-matting/solution/refine.py`. Return a finite tensor with the exact coarse
alpha shape, device, and dtype, with every value in `[0,1]`. The function executes
during both training and evaluation and must preserve a gradient path.

## Fixed Protocol

- Complete licensed Adobe Composition-1K: 43,100 train and 1,000 test composites.
- One model, 100,000 optimizer steps, batch 8, 320x320 crops, Adam 1e-4, seed 42.
- Every test item is evaluated with deterministic alpha-derived widths 6, 9, and 12;
  all three settings are scored.
- This full-inventory research protocol does not claim identity with the official
  single-trimap leaderboard.
- Alpha SAD is scored; MSE and finite-difference edge error are diagnostic.
- Invalid active output is never replaced by another implementation.

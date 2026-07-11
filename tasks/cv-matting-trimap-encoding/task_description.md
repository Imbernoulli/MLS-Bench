# Image Matting: Trimap Encoding

## Research Question

Design how the fixed trimap is encoded into feature planes before concatenation with
RGB. Valid encodings are ranked only by terminal evaluation.

## Implementation Contract

Modify `encode_trimap(trimap)` in `image-matting/solution/trimap.py`. For a finite
`(B,H,W)` trimap, return a finite floating-point `(B,K,H,W)` tensor on the same
device with `1 <= K <= 8`. Batch and spatial dimensions must be unchanged.

## Fixed Protocol

- Complete licensed Adobe Composition-1K: 43,100 train and 1,000 test composites.
- One model, 100,000 optimizer steps, batch 8, 320x320 crops, Adam 1e-4, seed 42.
- Every test item is evaluated with deterministic alpha-derived widths 6, 9, and 12;
  all three settings are scored.
- This full-inventory research protocol does not claim identity with the official
  single-trimap leaderboard.
- Alpha SAD is scored; MSE and finite-difference edge error are diagnostic.
- Invalid active output is never replaced by another implementation.

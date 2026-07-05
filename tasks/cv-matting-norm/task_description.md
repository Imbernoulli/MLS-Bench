# Image Matting: Design the Normalisation Layer (no-norm vs BatchNorm)

## Research Question
**Normalisation** stabilises and speeds up the matting net's short fine-tune. On this
synthetic composite data (recurring low-frequency fg/bg statistics across images):
no-norm (identity) < InstanceNorm (per-image) < **BatchNorm** (cross-image statistics,
which are informative because the composite statistics recur across the fixed set).
**Redesign the normalisation** used after each conv.

## Implementation Contract
Modify `make_norm(num_ch)` in `image-matting/solution/norm.py` to return a
`torch.nn.Module` normalisation layer (e.g. `nn.BatchNorm2d(num_ch)`). A malformed /
crashing norm falls back to BatchNorm.

## Fixed Pipeline & Evaluation
- Data: 100 train / 40 val **synthetic composites** (128x128), `I = a*F + (1-a)*B`
  with an **exact** soft GT alpha of a random blobby shape with fine hair-like detail.
- Network: a **fixed configurable matting U-Net** (encoder + skip-connection decoder),
  fed RGB + the trimap, trained a short fine-tune with a fixed alpha-L1 + composition
  loss. **Only your surface changes.**
- Three settings = three **trimap-width** difficulties: `medium` (band width 6),
  `wide` (band width 9), `xwide` (band width 12, thickest unknown band, hardest). The
  trimap is re-derived from the exact GT alpha at eval time; training always uses the
  medium band. The score is the **gmean over all three settings**.
- Metric (lower is better): **alpha SAD** in the trimap unknown band on the val split;
  MSE and gradient error are also recorded. A **constant-0.5 / copy-trimap predictor
  scores `CONST_HALF_SAD`** (far above any real net), so the metric is monotone in
  matting quality.

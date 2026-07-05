# Image Matting: Design the Bottleneck Dilation / Context (single 3x3 vs ASPP)

## Research Question
A **dilated multi-rate** block enlarges the receptive field WITHOUT losing resolution,
aggregating context across the wide unknown band (ASPP, Chen et al. 2017; dilated
context, Iizuka et al. 2017). **Redesign the bottleneck block**: a single 3x3 (limited
context) < a pointwise 1x1 < a **dilated multi-rate** block (parallel dilations 1/2/4,
fused) that aggregates the most context.

## Implementation Contract
Modify `build_dilation(ch)` in `image-matting/solution/dilation.py` to return a
`torch.nn.Module` mapping a bottleneck feature `(B,ch,H,W)` to the SAME shape (inserted
after the attention block). A malformed / wrong-shape module falls back to identity.

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

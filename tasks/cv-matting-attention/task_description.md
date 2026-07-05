# Image Matting: Design the Bottleneck Attention / Context (global-pool vs self-attention)

## Research Question
Matting must aggregate **context** so the unknown band knows which side of the matte it
belongs to. **Redesign the bottleneck block**: a **global-average-pool** collapses all
spatial context (worst); a **local 3x3 conv** aggregates only a neighbourhood; a
**non-local self-attention** block (contextual attention, Yu et al. 2018) aggregates
GLOBAL context (best).

## Implementation Contract
Modify `build_attention(ch)` in `image-matting/solution/attention.py` to return a
`torch.nn.Module` mapping a bottleneck feature `(B,ch,H,W)` to the SAME shape, inserted
at the stride-8 bottleneck. A malformed / wrong-shape module falls back to identity.

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

# Image Matting: Design the Decoder Upsampling (nearest vs learned)

## Research Question
The **upsampling** operator decides how much soft-edge detail survives the decoder:
**nearest-neighbour** (blocky, aliases the soft matte edges) < **bilinear** (smooth) <
a **learned / guided upsample** (transposed conv, or bilinear + a refine conv). **Redesign
the decoder upsampling** to reconstruct a smooth, sharp soft edge.

## Implementation Contract
Modify `build_upsampler(cin)` in `image-matting/solution/upsampling.py` to return a
`torch.nn.Module` mapping a decoder feature `(B,cin,H,W)` to `(B,cin,2H,2W)` (same
channels; the harness resizes to the exact skip size if needed). A malformed /
wrong-shape module falls back to bilinear.

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

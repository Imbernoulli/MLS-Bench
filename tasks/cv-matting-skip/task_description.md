# Image Matting: Design the Encoder-Decoder Skip Fusion (drop vs concat)

## Research Question
**Skip connections** (U-Net) inject the encoder's high-resolution, low-level features
into the decoder so the matte keeps sharp boundaries. **Redesign the skip fusion**:
dropping the skip loses that detail (blurry matte); a down-weighted partial skip
recovers some; the full-strength concat skip (standard U-Net fusion) recovers the most.

## Implementation Contract
Modify `fuse(dec_up, skip)` in `image-matting/solution/skip.py` to fuse an upsampled
decoder feature `dec_up (B,C_dec,H,W)` with the encoder skip `skip (B,C_skip,H,W)`; the
next decoder conv expects `C_dec + C_skip` channels (the default concat width). A
malformed / crashing fuse falls back to the full concat skip.

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

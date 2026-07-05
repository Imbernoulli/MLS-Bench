# Image Matting: Design the Whole Matting Network (plain encoder-decoder vs DIM deep-matting)

## Research Question
Trimap-guided **image matting** predicts a **soft foreground alpha** `a in [0,1]` per
pixel (NOT a hard mask) via `I = a*F + (1-a)*B`. The **network architecture** decides
whether the fine transition detail (hair, thin edges) survives. A **plain
encoder-decoder** downsamples to a bottleneck and bilinearly upsamples back with **no
skip connections and no refinement**, losing the high-frequency matte detail. The
matting-standard architecture is **Deep Image Matting** (DIM, Xu et al. 2017): an
encoder-decoder with **U-Net skip connections** (injecting the encoder's high-res
features into the decoder) followed by a **second refinement stage** (a shallow
residual net that sharpens the coarse matte). **Redesign the whole network** to recover
a sharp matte. This is the repo's strict-bar direction: the ordering
    copy-trimap / constant (degenerate) < plain encoder-decoder < DIM (skips + refine =
    SOTA)
holds across all three trimap-width settings.

## Implementation Contract
Modify `build_net(in_ch)` in `image-matting/solution/arch.py` to return a
`torch.nn.Module` whose forward `net(x, image=<B,3,H,W>, trimap=<B,H,W>)` maps
`x = concat(RGB, trimap-encoding)` (`in_ch` channels) to an alpha `(B,H,W)` in `[0,1]`
at full resolution. A malformed / crashing net falls back to the harness strong U-Net.

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

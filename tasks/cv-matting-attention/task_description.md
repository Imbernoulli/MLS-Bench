# Image Matting: Bottleneck Channel Gating

## Research Question

Design a channel-wise gate derived from the bottleneck and measure its effect on
soft-alpha reconstruction. The spatial receptive-field block is fixed by this task,
so the experiment isolates feature-channel selection rather than dilation design.

## Implementation Contract

Modify `build_attention(ch)` in `image-matting/solution/attention.py`. It must
return a `torch.nn.Module` mapping a finite `(B,ch,H,W)` bottleneck tensor to
finite gates of exact shape `(B,ch,1,1)` in `[0,1]`. The harness multiplies those
gates into the bottleneck before the fixed spatial receptive-field stage.

## Fixed Protocol

- Data: the complete licensed Adobe Composition-1K inventory: 43,100 training
  composites and all 1,000 test composites. Synthetic proxies and selected test
  slices do not participate.
- Training: one model for 100,000 optimizer steps, batch 8, random 320x320 crops,
  Adam at 1e-4 with cosine decay, seed 42.
- Evaluation: the trained model is evaluated on every test composite under
  deterministic alpha-derived trimap widths 6 (`medium`), 9 (`wide`), and 12
  (`xwide`). All three settings participate in scoring.
- This is an explicit full-inventory research protocol. Its three trimaps are
  derived from ground-truth alpha and are not presented as the official
  Composition-1K single-trimap leaderboard protocol.
- The scored metric is alpha SAD divided by 1000 after fixing known foreground and
  background pixels. Whole-image MSE and a deterministic finite-difference edge
  error are diagnostic only.
- Missing, crashing, wrong-shape, negative-loss, or non-finite editable output
  invalidates verification. The harness never substitutes another implementation.

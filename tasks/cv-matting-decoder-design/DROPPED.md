# DROPPED surface: cv-matting-decoder-design (NOT shipped)

This editable surface (deepest-feature bilinear decoder `bilinear` vs U-Net
skip-connection decoder `unet`) was DESIGNED and validated on a fully-synthetic
128x128 composite dataset with a single seed, but on the REAL PPM-100 cross-seed
re-anchor (B0 8xH200, torch 2.4.1, full budget 400 iters, seeds 42/123) the intended
weak(bilinear)<strong(unet) ordering (lower alpha SAD is better) robustly INVERTS on
1 of 3 trimap-width settings (not just noisy — a genuine, consistent margin on both
seeds):

Alpha SAD (unknown band, /1000, LOWER is better):
  medium: seed42  bilinear=0.4197 unet=0.3074 (unet wins,     Δ=+0.1123)
          seed123 bilinear=0.4031 unet=0.3420 (unet wins,     Δ=+0.0611)
          -- unet robustly wins both seeds (intended direction).
  wide:   seed42  bilinear=0.7042 unet=0.7413 (bilinear wins, Δ=-0.0371)
          seed123 bilinear=0.7528 unet=0.8235 (bilinear wins, Δ=-0.0707)
          -- bilinear robustly wins BOTH seeds (inverts the intended ordering,
             real margin, not noise).
  xwide:  seed42  bilinear=1.2971 unet=1.2805 (unet wins,     Δ=+0.0166)
          seed123 bilinear=1.3986 unet=1.3059 (unet wins,     Δ=+0.0927)
          -- unet robustly wins both seeds (intended direction).

2/3 settings are robustly monotone in the intended direction, but the "wide" setting
robustly inverts on BOTH seeds with a real, non-trivial margin (-0.037 to -0.071 SAD)
-- this is the same structural pattern as the `image-deblur` repo's
`deblur-loss-kind` drop earlier in this re-anchor pass (2/3 settings robust + 1
setting robustly inverted is treated as a genuine per-setting failure, not
overridable by a favourable task-level gmean). The task-level gmean is robust in the
intended direction both seeds (unet beats bilinear at the gmean level), but per the
project's per-setting cross-seed robustness bar (never HP-sweep to force
monotonicity; a single robustly-inverted setting disqualifies shipping as-is or a
clean relabel in either direction, since reversing the label would then break the
other 2 settings), this surface is dropped for consistency with that precedent.
Plausibly, the "wide" trimap band sits at an intermediate difficulty where the
U-Net decoder's extra skip-connection capacity mildly overfits the short 400-iter
fine-tune on real photographic texture (vs. the synthetic blobby shapes used in the
original single-seed design), while the bilinear decoder's simpler upsampling path is
more stable there specifically. The surface code remains in
`image-matting/solution/decoder.py` (lines 53-70) + this task's
`edits/`/`scripts/`/`task_description.md` for provenance, but no task is shipped for
it.

# DROPPED surface: cv-matting-upsampling (NOT shipped)

This editable surface (nearest-neighbour upsample `nearest` vs learned upsample
`learned`, bilinear + refine conv) was DESIGNED and validated on a fully-synthetic
128x128 composite dataset with a single seed, but on the REAL PPM-100 cross-seed
re-anchor (B0 8xH200, torch 2.4.1, full budget 400 iters, seeds 42/123) the intended
weak(nearest)<strong(learned) ordering (lower alpha SAD is better) is robust on only
2 of 3 trimap-width settings; the third is a degenerate near-tie that flips sign
across seeds:

Alpha SAD (unknown band, /1000, LOWER is better):
  medium: seed42  nearest=0.2812 learned=0.2750 (learned wins, Δ=+0.0062)
          seed123 nearest=0.2530 learned=0.2426 (learned wins, Δ=+0.0104)
          -- learned robustly wins both seeds (intended direction).
  wide:   seed42  nearest=0.7289 learned=0.6602 (learned wins, Δ=+0.0687)
          seed123 nearest=0.6524 learned=0.6186 (learned wins, Δ=+0.0338)
          -- learned robustly wins both seeds (intended direction).
  xwide:  seed42  nearest=1.1381 learned=1.0088 (learned wins, Δ=+0.1293)
          seed123 nearest=1.1309 learned=1.1484 (nearest wins, Δ=-0.0175)
          -- correct direction at seed42 with a large margin, but flips sign (small
             margin) at seed123: a degenerate near-tie / seed-unstable cell, not a
             robust inversion.

2/3 settings (medium, wide) are cleanly robust in the intended direction on both
seeds. The third (xwide) is the "degenerate near-tie / seed-unstable" case explicitly
called out in the project mandate -- structurally the same category as
`mono3d-height-source`'s seed-flipping "easy" tier, which was dropped rather than
shipped with a caveat. The task-level gmean is robust in the intended direction both
seeds (seed42 gmean nearest=0.6156>learned=0.5679; seed123 gmean
nearest=0.5715>learned=0.5565), but per the same per-setting cross-seed robustness
bar applied to `cv-matting-decoder-design` (2/3 robust + 1 seed-unstable/inverted
setting is not overridable by a favourable task-level gmean, for consistency with
the `deblur-loss-kind` precedent), this surface is dropped rather than shipped with
a caveat. Plausibly, at the thickest xwide trimap band the unknown region is largest
and the short 400-iter fine-tune's learned-upsample refine conv has the least signal
to specialize on, making its advantage over plain nearest-neighbour upsampling
indistinguishable from run-to-run noise there specifically. The surface code remains
in `image-matting/solution/upsampling.py` (lines 31-38) + this task's
`edits/`/`scripts/`/`task_description.md` for provenance, but no task is shipped for
it.

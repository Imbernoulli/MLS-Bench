# DROPPED surface: cv-matting-attention (NOT shipped)

This editable surface (bottleneck context aggregation: global-average-pool `gap` vs
non-local self-attention `selfattn`, contextual attention Yu et al. 2018) was DESIGNED
and validated on a fully-synthetic 128x128 composite dataset with a single seed, but on
the REAL PPM-100 cross-seed re-anchor (B0 8xH200, torch 2.4.1, full budget 400 iters,
seeds 42/123) the intended weak(gap)<strong(selfattn) ordering (lower alpha SAD is
better) is robust on only 1 of 3 trimap-width settings, and flips sign or robustly
inverts on the other 2:

Alpha SAD (unknown band, /1000, LOWER is better):
  medium: seed42  gap=0.2713 selfattn=0.2617 (selfattn wins, Δ=+0.0096)
          seed123 gap=0.2493 selfattn=0.2633 (gap wins,      Δ=-0.0140)
          -- SIGN FLIPS across seeds; not robust either direction.
  wide:   seed42  gap=0.6809 selfattn=0.6457 (selfattn wins, Δ=+0.0352)
          seed123 gap=0.6129 selfattn=0.5751 (selfattn wins, Δ=+0.0378)
          -- selfattn robustly wins both seeds (the ONLY robust setting).
  xwide:  seed42  gap=1.0248 selfattn=1.0838 (gap wins,      Δ=-0.0590)
          seed123 gap=1.0240 selfattn=1.0637 (gap wins,      Δ=-0.0397)
          -- gap robustly wins both seeds (inverts the intended ordering).

The intended ordering (selfattn > gap) holds robustly on only 1/3 settings (wide);
medium sign-flips between seeds and xwide robustly inverts on both seeds. The
task-level gmean itself is seed-unstable (seed42 gmean favours selfattn, seed123 gmean
favours gap), so this is not a clean "honest relabel" candidate either (gap does not
robustly beat selfattn on all 3 settings). Per the project mandate ("if it's just a
degenerate near-tie / seed-unstable... DROP it honestly", never HP-sweep to force
monotonicity), this surface is dropped. Plausibly, on real PPM-100 photos (vs. the
synthetic blobby-shape composites used for the original design) the self-attention
block's global-context benefit is much smaller relative to its added
capacity/optimization noise at this short 400-iter budget, particularly on the
thickest xwide trimap band where the unknown region is largest and noisiest. The
surface code remains in `image-matting/solution/attention.py` (lines 33-46) + this
task's `edits/`/`scripts/`/`task_description.md` for provenance, but no task is
shipped for it.

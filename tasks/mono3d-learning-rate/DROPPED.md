# DROPPED surface: mono3d-learning-rate (NOT shipped)

This editable surface (depth-head LR multiplier: tiny 0.01x vs well-tuned ~1.0x) was DESIGNED
and GPU-VALIDATED on a fully-synthetic dataset, but on the REAL KITTI 3D Object Detection
dataset cross-seed re-anchor (B0 8xH200, torch 2.4.1, 1200 steps, seeds 42/123) the intended
weak(lr_tiny)<strong(lr_tuned) ordering holds robustly on only 1 of 3 difficulty tiers; the
other 2 tiers flip sign at seed 123:

AP3D@0.25:
  easy:     seed42 lr_tiny=0.315328 lr_tuned=0.334307 (tuned wins, Δ=+0.0190)
            seed123 lr_tiny=0.328467 lr_tuned=0.349635 (tuned wins, Δ=+0.0212)
            -- ROBUST, both seeds: tuned > tiny.
  moderate: seed42 lr_tiny=0.172061 lr_tuned=0.172061 (EXACT TIE)
            seed123 lr_tiny=0.166383 lr_tuned=0.157865 (tiny WINS, Δ=+0.0085)
            -- tied at seed42, INVERTED at seed123.
  hard:     seed42 lr_tiny=0.172579 lr_tuned=0.185043 (tuned wins, Δ=+0.0125)
            seed123 lr_tiny=0.191755 lr_tuned=0.171620 (tiny WINS, Δ=+0.0201)
            -- INVERTED sign flip between seeds.

Only `easy` is robustly monotone in the intended direction across both seeds; `moderate` is an
exact tie at seed42 and inverts at seed123; `hard` inverts outright between seeds (a >0.02
AP3D swing with opposite sign). Per project mandate (never HP-sweep to force monotonicity, and
"drop it honestly" on a degenerate/seed-unstable signal), 1/3 robustly-monotone settings does
not meet the required 3/3-settings, cross-seed-robust bar, so this surface is dropped.
Plausibly, at this fixed 1200-step budget the depth head's multiplicative residual has already
mostly converged for both LR multipliers on the harder/noisier tiers (where absolute AP3D is
low and dominated by occlusion/truncation noise), so which LR "wins" on those tiers is
effectively a coin flip across seeds, while on the cleaner `easy` tier the tuned LR's faster
convergence still shows through. The surface code remains in
`mono3d-detection/solution/lr_mult.py` (lines 25-28) + this task's
`edits/`/`scripts/`/`task_description.md` for provenance, but no task is shipped for it.

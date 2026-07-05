# DROPPED surface: deblur-residual-depth (NOT shipped)

This editable surface (shallow vs deep residual-block stack) was DESIGNED and
GPU-VALIDATED on CPU-synthetic motion-blur data, but on the REAL GoPro Large-Scale Blur
Dataset (Nah, CVPR'17) cross-seed re-anchor (B0 8xH200, torch 2.4.1, 400 iters, seeds
42/123) it is NOT robustly monotone — it is seed-dependent.

Reason: at this compact-net / mild-real-blur operating point the backbone is already close
to its PSNR ceiling (blurry floors 36.26 / 27.71 / 21.32 dB for small/medium/large), so
extra depth is a second-order lever, and on seed 123 EVERY setting inverts (the shallow net
beats the deep one), while seed 42 is flat/mixed:

Deblur PSNR (dB), weak=depth_shallow, strong=depth_deep:
  small : seed42 weak=36.0061 strong=36.0329 delta=+0.0268 | seed123 weak=36.1647 strong=36.0322 delta=-0.1325
  medium: seed42 weak=27.8779 strong=27.8837 delta=+0.0058 | seed123 weak=27.8856 strong=27.8425 delta=-0.0431
  large : seed42 weak=21.4699 strong=21.4520 delta=-0.0179 | seed123 weak=21.4537 strong=21.4479 delta=-0.0058

Seed 42 gives a near-flat (<0.03 dB) mixed signal; seed 123 robustly and consistently
inverts on all three settings (deeper = worse at 400 iters, likely undertrained). Per
project mandate (never HP-sweep to force monotonicity), this surface is dropped. The
surface code remains in `vendor/image-deblur/solution/arch_depth.py` + this task's
`edits/`/`scripts/`/`task_description.md` for provenance, but no task is shipped for it.

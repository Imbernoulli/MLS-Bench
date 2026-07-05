# DROPPED surface: deblur-width (NOT shipped)

This editable surface (narrow vs wide channel width) was DESIGNED and GPU-VALIDATED on
CPU-synthetic motion-blur data, but on the REAL GoPro Large-Scale Blur Dataset (Nah,
CVPR'17) cross-seed re-anchor (B0 8xH200, torch 2.4.1, 400 iters, seeds 42/123) it is NOT
robustly monotone.

Reason: at this compact-net / mild-real-blur operating point the backbone is already close
to its PSNR ceiling (blurry floors 36.26 / 27.71 / 21.32 dB for small/medium/large).
`medium`/`large` favour the wider net on both seeds but with tiny margins (<0.09 dB); the
`small` setting robustly and cross-seed-flip INVERTS with a large, seed-dependent swing:

Deblur PSNR (dB), weak=width_narrow, strong=width_wide:
  small : seed42 weak=36.1961 strong=35.9807 delta=-0.2154 | seed123 weak=35.8582 strong=36.1521 delta=+0.2939
  medium: seed42 weak=27.8155 strong=27.8758 delta=+0.0603 | seed123 weak=27.8051 strong=27.8889 delta=+0.0838
  large : seed42 weak=21.4555 strong=21.4718 delta=+0.0163 | seed123 weak=21.4249 strong=21.4319 delta=+0.0070

The `small` setting swings by >0.5 dB between seeds with opposite signs — a seed-dependent
inversion, not a genuine effect — so the 3/3-settings, cross-seed-robust bar fails. Per
project mandate (never HP-sweep to force monotonicity), this surface is dropped. The
surface code remains in `vendor/image-deblur/solution/arch_width.py` + this task's
`edits/`/`scripts/`/`task_description.md` for provenance, but no task is shipped for it.

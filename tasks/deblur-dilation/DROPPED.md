# DROPPED surface: deblur-dilation (NOT shipped)

This editable surface (dilated-conv receptive field narrow/wide) was DESIGNED and
GPU-VALIDATED on CPU-synthetic motion-blur data, but on the REAL GoPro Large-Scale Blur
Dataset (Nah, CVPR'17) cross-seed re-anchor (B0 8xH200, torch 2.4.1, 400 iters, seeds
42/123) it is NOT robustly monotone.

Reason: at this compact-net / mild-real-blur operating point the backbone is already close
to its PSNR ceiling (blurry floors 36.26 / 27.71 / 21.32 dB for small/medium/large), so a
wider dilation is a second-order lever whose effect (<0.03 dB either way) is within
cross-seed noise. The small setting flips sign across seeds:

Deblur PSNR (dB), weak=dil_narrow, strong=dil_wide:
  small : seed42 weak=36.0439 strong=36.0737 delta=+0.0298 | seed123 weak=36.1902 strong=36.1800 delta=-0.0102
  medium: seed42 weak=27.8790 strong=27.8872 delta=+0.0082 | seed123 weak=27.8776 strong=27.8911 delta=+0.0135
  large : seed42 weak=21.4639 strong=21.4774 delta=+0.0135 | seed123 weak=21.4050 strong=21.4282 delta=+0.0232

While medium/large are directionally consistent, the margins (<0.025 dB) are far smaller
than the cross-seed noise floor observed elsewhere on this harness (~0.05-0.3 dB), and the
small setting outright inverts on seed 123. Per project mandate (never HP-sweep to force
monotonicity), this surface is dropped rather than shipped on a noise-level signal. The
surface code remains in `vendor/image-deblur/solution/arch_dilation.py` + this task's
`edits/`/`scripts/`/`task_description.md` for provenance, but no task is shipped for it.

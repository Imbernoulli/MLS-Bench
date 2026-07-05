# DROPPED surface: deblur-multiscale (NOT shipped)

This editable surface (single full-res pass vs 3-scale coarse-to-fine pyramid) was
DESIGNED and GPU-VALIDATED on CPU-synthetic heavy motion-blur data, but on the REAL GoPro
Large-Scale Blur Dataset (Nah, CVPR'17) cross-seed re-anchor (B0 8xH200, torch 2.4.1, 400
iters, seeds 42/123) it is NOT robustly monotone.

Reason: at this compact-net / mild-real-blur operating point the backbone is already close
to its PSNR ceiling (blurry floors 36.26 / 27.71 / 21.32 dB for small/medium/large), so the
extra coarse-to-fine scales are a second-order lever whose effect (<0.03 dB) is within
cross-seed noise, and the sign flips across seeds on every setting:

Deblur PSNR (dB), weak=single, strong=multiscale:
  small : seed42 weak=36.0901 strong=36.0604 delta=-0.0297 | seed123 weak=36.1528 strong=36.1584 delta=+0.0056
  medium: seed42 weak=27.9117 strong=27.9105 delta=-0.0012 | seed123 weak=27.8626 strong=27.8764 delta=+0.0138
  large : seed42 weak=21.4822 strong=21.4610 delta=-0.0212 | seed123 weak=21.4379 strong=21.4368 delta=-0.0011

No setting is robustly monotone (strong>weak) across both seeds; all margins are far
smaller than typical cross-seed noise. Per project mandate (never HP-sweep to force
monotonicity), this surface is dropped. The surface code remains in
`vendor/image-deblur/solution/multiscale.py` + this task's `edits/`/`scripts/`/
`task_description.md` for provenance, but no task is shipped for it.

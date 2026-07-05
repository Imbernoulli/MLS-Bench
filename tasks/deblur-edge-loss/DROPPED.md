# DROPPED surface: deblur-edge-loss (NOT shipped)

This editable surface (edge/gradient loss term on/off) was DESIGNED and GPU-VALIDATED on
CPU-synthetic motion-blur data, but on the REAL GoPro Large-Scale Blur Dataset (Nah,
CVPR'17) cross-seed re-anchor (B0 8xH200, torch 2.4.1, 400 iters, seeds 42/123) it is NOT
robustly monotone.

Reason: at this compact-net / mild-real-blur operating point the backbone is already close
to its PSNR ceiling (blurry floors 36.26 / 27.71 / 21.32 dB for small/medium/large), so the
edge term is a second-order lever whose effect is within cross-seed noise, and the large
setting robustly INVERTS (edge_off beats edge_on on both seeds):

Deblur PSNR (dB), weak=edge_off, strong=edge_on:
  small : seed42 weak=36.0861 strong=36.2031 delta=+0.1170 | seed123 weak=36.2304 strong=36.2749 delta=+0.0445
  medium: seed42 weak=27.9261 strong=27.9153 delta=-0.0108 | seed123 weak=27.8698 strong=27.9000 delta=+0.0302
  large : seed42 weak=21.4497 strong=21.4172 delta=-0.0325 | seed123 weak=21.4191 strong=21.4092 delta=-0.0099

`small` is robustly positive both seeds, but `medium` flips sign across seeds and `large` is
robustly NEGATIVE (edge_off > edge_on) on both seeds — not the 3/3-settings monotone bar.
Per project mandate (never HP-sweep to force monotonicity), this surface is dropped rather
than shipped with a forced ordering. The surface code remains in
`vendor/image-deblur/solution/edge.py` + this task's `edits/`/`scripts/`/`task_description.md`
for provenance, but no task is shipped for it.

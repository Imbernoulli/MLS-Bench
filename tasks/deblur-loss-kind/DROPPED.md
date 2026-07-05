# DROPPED surface: deblur-loss-kind (NOT shipped)

This editable surface (plain L2 vs robust Charbonnier+edge loss) was DESIGNED and
GPU-VALIDATED on CPU-synthetic motion-blur data, but on the REAL GoPro Large-Scale Blur
Dataset (Nah, CVPR'17) cross-seed re-anchor (B0 8xH200, torch 2.4.1, 400 iters, seeds
42/123) it is NOT robustly monotone across all three settings.

Reason: the small (em) and medium settings robustly favour Charbonnier+edge (weak=loss_l2,
strong=loss_charbonnier_edge) on both seeds, but the large (el) setting robustly INVERTS
(plain L2 beats Charbonnier+edge) on both seeds — this real-data blur band is dominated by
low-frequency displacement where the robust loss's edge term adds no benefit and slightly
hurts optimisation at only 400 iters:

Deblur PSNR (dB), weak=loss_l2, strong=loss_charbonnier_edge:
  small (em) : seed42 weak=36.0471 strong=36.1819 delta=+0.1348 | seed123 weak=36.1250 strong=36.2406 delta=+0.1156
  medium     : seed42 weak=27.8803 strong=27.9118 delta=+0.0315 | seed123 weak=27.8654 strong=27.9046 delta=+0.0392
  large (el) : seed42 weak=21.4832 strong=21.4419 delta=-0.0413 | seed123 weak=21.4504 strong=21.4113 delta=-0.0391

2/3 settings monotone (strong>weak, both seeds) but the third (large) robustly and
cross-seed inverts — not the required 3/3-settings bar. Per project mandate (never
HP-sweep to force monotonicity, never drop/relabel just the inconvenient setting to save
the task), the whole surface is dropped. The surface code remains in
`vendor/image-deblur/solution/losskind.py` + this task's `edits/`/`scripts/`/
`task_description.md` for provenance, but no task is shipped for it.

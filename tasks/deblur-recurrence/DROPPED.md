# DROPPED surface: deblur-recurrence (NOT shipped)

This editable surface (1 vs 3 SRN recurrence unrolls) was DESIGNED and GPU-VALIDATED on
CPU-synthetic motion-blur data, but on the REAL GoPro Large-Scale Blur Dataset (Nah,
CVPR'17) cross-seed re-anchor (B0 8xH200, torch 2.4.1, 400 iters, seeds 42/123) it is NOT
robustly monotone.

Reason: at this compact-net / mild-real-blur operating point the backbone is already close
to its PSNR ceiling (blurry floors 36.26 / 27.71 / 21.32 dB for small/medium/large), so
extra recurrence unrolls are a second-order lever whose effect (<0.03 dB) is within
cross-seed noise, and the sign flips across seeds on the small and medium settings:

Deblur PSNR (dB), weak=recur_one, strong=recur_three:
  small : seed42 weak=36.1233 strong=36.1413 delta=+0.0180 | seed123 weak=36.1529 strong=36.1516 delta=-0.0013
  medium: seed42 weak=27.9125 strong=27.9069 delta=-0.0056 | seed123 weak=27.8615 strong=27.8784 delta=+0.0169
  large : seed42 weak=21.4271 strong=21.4564 delta=+0.0293 | seed123 weak=21.4162 strong=21.4329 delta=+0.0167

Only `large` is directionally consistent both seeds, and even there the margin (<0.03 dB)
is at the cross-seed noise floor; `small` and `medium` flip sign across seeds. Not the
required 3/3-settings, cross-seed-robust bar. Per project mandate (never HP-sweep to force
monotonicity), this surface is dropped. The surface code remains in
`vendor/image-deblur/solution/recurrence.py` + this task's `edits/`/`scripts/`/
`task_description.md` for provenance, but no task is shipped for it.

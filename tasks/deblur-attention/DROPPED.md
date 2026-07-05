# DROPPED surface: deblur-attention (NOT shipped)

This editable surface (channel attention SE/CAB on/off) was DESIGNED and GPU-VALIDATED on
CPU-synthetic motion-blur data (see the old anchors in `task_description.md` / the prior
`score_spec.py`), but on the REAL GoPro Large-Scale Blur Dataset (Nah, CVPR'17) cross-seed
re-anchor (B0 8xH200, torch 2.4.1, 400 iters, seeds 42/123) it is NOT robustly monotone.

Reason: at this compact-net / mild-real-blur operating point the backbone is already close
to its PSNR ceiling (blurry floors 36.26 / 27.71 / 21.32 dB for small/medium/large), so
channel attention is a second-order lever whose effect (<0.12 dB either way) is within
cross-seed noise. The sign of `attn_on - attn_off` flips across settings AND across seeds:

Deblur PSNR (dB), weak=attn_off, strong=attn_on:
  small : seed42 weak=36.1370 strong=36.0214 delta=-0.1156 | seed123 weak=36.1244 strong=36.1645 delta=+0.0401
  medium: seed42 weak=27.9067 strong=27.8768 delta=-0.0299 | seed123 weak=27.8072 strong=27.8745 delta=+0.0673
  large : seed42 weak=21.4824 strong=21.4686 delta=-0.0138 | seed123 weak=21.4676 strong=21.4370 delta=-0.0306

No setting is robustly monotone (strong>weak) across both seeds. Per project mandate (never
HP-sweep to force monotonicity), this surface is dropped rather than shipped with a forced
ordering. The surface code remains in `vendor/image-deblur/solution/arch_attention.py` +
this task's `edits/`/`scripts/`/`task_description.md` for provenance, but no task is shipped
for it (see `deblur-global-residual` / `deblur-loss-design` for the two shipped, robustly
cross-seed-monotone surfaces on the real GoPro data).
